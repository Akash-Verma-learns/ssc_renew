"""
core/metadata_extractor.py
---------------------------
Single-shot extraction of opportunity_name and client_name.

Uses requests (not the ollama Python client) to avoid the Windows/httpx
socket stall bug that causes the pipeline to hang after the LLM finishes.

Also uses a fresh, short-lived DB session pattern consistent with
extractor.py so it never holds a connection across a long Ollama call.
"""

import json
import re
import requests
from core.vector_store import retrieve

OLLAMA_MODEL    = "llama3.2"
OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TIMEOUT  = 120   # metadata extraction is fast — 2 min ceiling

METADATA_QUERIES = [
    "name of assignment project title",
    "request for proposal title heading",
    "procuring entity client organization name",
    "invitation to tender subject",
    "issued by ministry department authority organization",
    "NITI Aayog ministry department board corporation authority",
    "employer client funding agency",
    "background introduction project overview",
    "about the project program scheme",
]

METADATA_PROMPT = """You are reading excerpts from an Indian government or MDB RFP/tender document.

RULES:
1. client_name: Find the REAL organisation name — Ministry, Department, Board, Corporation,
   Agency, or a funding body (ADB / World Bank / UNDP). Do NOT return "the Authority",
   "the Client", "the Employer", or any other placeholder. Look harder for the actual name.
2. opportunity_name: The core project/assignment description. Strip any leading
   "RFP for", "EOI for", "Tender for", "Selection of" prefix.

EXCERPTS:
{context}

Return ONLY valid JSON — no markdown, no explanation:
{{"opportunity_name": "clean project name without RFP/EOI prefix", "client_name": "actual organisation name"}}

If you genuinely cannot find the real organisation name, set client_name to null.
If you genuinely cannot find the project name, set opportunity_name to null."""


_PLACEHOLDERS = {
    "the authority", "authority", "the client", "client",
    "the employer", "employer", "the owner", "owner",
    "null", "none", "n/a", "unknown", "the procuring entity", "procuring entity",
}

_STRIP_PREFIXES = re.compile(
    r"^(?:RFP|EOI|Tender|Request for Proposal|Expression of Interest"
    r"|Invitation|Selection of|Appointment of|Hiring of)\s+(?:for\s+)?",
    re.IGNORECASE,
)


def _clean_json(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    return text[start:end] if start >= 0 and end > start else ""


def _clean(opp, clt):
    if opp:
        opp = _STRIP_PREFIXES.sub("", opp).strip()
        opp = opp if len(opp) >= 5 else None
    if clt and str(clt).strip().lower() in _PLACEHOLDERS:
        clt = None
    if clt and len(str(clt).strip()) < 3:
        clt = None
    return opp or None, clt or None


def extract_metadata(doc_name: str, model: str = OLLAMA_MODEL) -> dict:
    """Single-shot metadata extraction. Never raises."""
    seen, chunks = set(), []
    for query in METADATA_QUERIES:
        for c in retrieve(query, doc_name=doc_name, top_k=3):
            key = c["clause_ref"] + str(c["page_no"])
            if key not in seen and c["score"] > 0.18:
                seen.add(key)
                chunks.append(c)

    chunks.sort(key=lambda x: x["score"], reverse=True)
    chunks = chunks[:10]

    if not chunks:
        return {
            "opportunity_name": None,
            "client_name": None,
            "error": "No chunks found in vector store.",
        }

    top = sorted(chunks[:8], key=lambda x: (x["page_no"] or 999, -x["score"]))
    context = "\n\n---\n\n".join(
        f"[Page {c['page_no']} | {c['section_heading']}]\n{c['text']}"
        for c in top
    )

    print(f"[MetaExtractor] Attempt 1/3...")
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": METADATA_PROMPT.format(context=context),
                    }
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
    except Exception as e:
        return {
            "opportunity_name": None,
            "client_name": None,
            "error": f"Ollama error: {e}",
        }

    if not raw or not raw.strip():
        return {
            "opportunity_name": None,
            "client_name": None,
            "error": "Empty LLM response.",
        }

    cleaned = _clean_json(raw)
    if not cleaned:
        return {
            "opportunity_name": None,
            "client_name": None,
            "error": f"No JSON in response: {raw[:120]}",
        }

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "opportunity_name": None,
            "client_name": None,
            "error": f"JSON parse error: {e}",
        }

    opp, clt = _clean(data.get("opportunity_name"), data.get("client_name"))

    print(f"[MetaExtractor] opportunity_name = {opp!r}")
    print(f"[MetaExtractor] client_name      = {clt!r}")

    return {
        "opportunity_name": opp,
        "client_name":      clt,
        "error":            None if (opp or clt) else "LLM returned no usable values.",
    }