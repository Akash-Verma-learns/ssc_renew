"""
core/pageindex_retriever.py
============================
PageIndex Integration — Vectorless, Reasoning-Based Retrieval
===============================================================

PageIndex replaces ChromaDB semantic similarity search with LLM-driven
tree-search retrieval. Instead of "find the most similar chunk", it builds
a hierarchical Table-of-Contents index and reasons about which sections
are relevant to a query — exactly how a human expert would navigate a document.

Why this matters for RFP/tender documents:
  - Clause 4.3 Liability may appear in a 200-page RFP — similarity search
    will find "liability" in wrong sections (preamble, definitions, etc.)
  - PageIndex builds: Document → Section → Sub-section → Clause → Paragraph
    and reasons: "Limitation of liability → most likely in Section 4 Contract
    Terms → Sub-section 4.3 → retrieve pages 47-51"
  - Result: the right clause text, not the most semantically similar fragment

Architecture:
  LOCAL mode (no API key):
    - Build in-memory tree index from chunks using LLM (Gemini/Ollama)
    - Perform tree-search retrieval locally
    - Results equivalent to PageIndex but self-hosted

  API mode (PAGEINDEX_API_KEY set):
    - Upload document to PageIndex cloud
    - Use their enhanced OCR + tree building pipeline
    - Best quality — 98.7% accuracy on FinanceBench

Usage:
    from core.pageindex_retriever import PageIndexRetriever

    retriever = PageIndexRetriever(doc_name="tender.pdf")
    retriever.build_index(chunks)  # one-time per document

    results = retriever.retrieve("limitation of liability clause", top_k=5)
    # returns: [{"text": ..., "page_no": ..., "section": ..., "score": ...}]
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PAGEINDEX_API_KEY  = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_BASE_URL = os.getenv("PAGEINDEX_BASE_URL", "https://api.pageindex.ai")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2")

# ─────────────────────────────────────────────────────────────────────────────
# Tree node for the in-memory index
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    """One node in the PageIndex hierarchical tree."""
    node_id:     str
    title:       str
    summary:     str
    start_page:  int
    end_page:    int
    level:       int                          # 0=root, 1=chapter, 2=section, 3=clause
    children:    List["TreeNode"] = field(default_factory=list)
    chunk_refs:  List[int]        = field(default_factory=list)  # indices into chunks list
    parent_id:   Optional[str]    = None

    def to_dict(self) -> dict:
        return {
            "node_id":    self.node_id,
            "title":      self.title,
            "summary":    self.summary,
            "start_page": self.start_page,
            "end_page":   self.end_page,
            "level":      self.level,
            "children":   [c.to_dict() for c in self.children],
        }


# ─────────────────────────────────────────────────────────────────────────────
# PageIndex Retriever
# ─────────────────────────────────────────────────────────────────────────────

class PageIndexRetriever:
    """
    Vectorless retrieval using a hierarchical tree index.

    Mimics how a human expert navigates a document:
      1. Scan Table of Contents → identify relevant chapter
      2. Within chapter → identify relevant section
      3. Within section → read the specific clause

    This is fundamentally more accurate than semantic similarity for
    professional documents where context and structure matter.
    """

    def __init__(self, doc_name: str = "", cache_dir: str = "./pageindex_cache"):
        self.doc_name  = doc_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.root:     Optional[TreeNode] = None
        self.chunks:   List[dict]         = []
        self._index_built                 = False
        self._api_doc_id: Optional[str]   = None

    # ── Build Index ──────────────────────────────────────────────────────────

    def build_index(self, chunks: list, force_rebuild: bool = False) -> bool:
        """
        Build the hierarchical tree index from parsed chunks.

        Tries:
          1. Load from cache (fast)
          2. PageIndex API (best quality, requires API key)
          3. Local LLM tree builder (good quality, free)

        Returns True if index was built successfully.
        """
        # Store chunks as plain dicts for serialisation
        self.chunks = [
            {
                "text":    getattr(c, "text", ""),
                "page_no": getattr(c, "page_no", 0),
                "section": getattr(c, "section_heading", ""),
                "ref":     getattr(c, "clause_ref", ""),
                "is_table":getattr(c, "is_table", False),
            }
            for c in chunks
        ]

        # Try cache first
        cache_file = self.cache_dir / f"{self._safe_name(self.doc_name)}.json"
        if not force_rebuild and cache_file.exists():
            try:
                self._load_from_cache(cache_file)
                print(f"[PageIndex] Loaded index from cache: {cache_file.name}")
                self._index_built = True
                return True
            except Exception as e:
                print(f"[PageIndex] Cache load failed: {e} — rebuilding")

        # Try PageIndex API
        if PAGEINDEX_API_KEY:
            try:
                self._build_via_api()
                self._save_to_cache(cache_file)
                self._index_built = True
                return True
            except Exception as e:
                print(f"[PageIndex] API build failed: {e} — using local builder")

        # Local tree builder
        try:
            self._build_local_index()
            self._save_to_cache(cache_file)
            self._index_built = True
            print(f"[PageIndex] Local index built: {self._count_nodes(self.root)} nodes")
            return True
        except Exception as e:
            print(f"[PageIndex] Local build failed: {e}")
            return False

    def _build_local_index(self):
        """
        Build tree index locally using the LLM to summarise sections.
        Groups chunks by section heading, then creates a 3-level hierarchy.
        """
        if not self.chunks:
            return

        # Level 1: Group chunks by major section
        sections: Dict[str, List[dict]] = {}
        for chunk in self.chunks:
            heading = chunk.get("section", "") or "Preamble"
            # Normalise heading to top-level (remove sub-numbering)
            top = re.split(r"\s*[:\-–]\s*", heading)[0].strip()
            top = re.sub(r"^\d+\.\d+[\.\d]*\s+", "", top)  # strip sub-numbers
            sections.setdefault(top, []).append(chunk)

        # Build root node
        all_pages = [c["page_no"] for c in self.chunks if c["page_no"]]
        root_page_start = min(all_pages) if all_pages else 1
        root_page_end   = max(all_pages) if all_pages else 1

        self.root = TreeNode(
            node_id    = "root",
            title      = self.doc_name or "Document",
            summary    = f"Document with {len(self.chunks)} sections across pages {root_page_start}-{root_page_end}",
            start_page = root_page_start,
            end_page   = root_page_end,
            level      = 0,
        )

        # Build level-1 section nodes
        for sec_idx, (section_title, sec_chunks) in enumerate(sections.items()):
            pages     = [c["page_no"] for c in sec_chunks if c["page_no"]]
            p_start   = min(pages) if pages else 0
            p_end     = max(pages) if pages else 0

            # Generate summary using LLM
            sample_text = " ".join(c["text"][:200] for c in sec_chunks[:3])
            summary     = self._llm_summarise(section_title, sample_text)

            section_node = TreeNode(
                node_id    = f"sec_{sec_idx:04d}",
                title      = section_title,
                summary    = summary,
                start_page = p_start,
                end_page   = p_end,
                level      = 1,
                parent_id  = "root",
                chunk_refs = list(range(
                    self.chunks.index(sec_chunks[0]),
                    self.chunks.index(sec_chunks[0]) + len(sec_chunks),
                )) if sec_chunks else [],
            )
            self.root.children.append(section_node)

            # Level 2: Sub-sections (group by clause_ref patterns)
            sub_groups: Dict[str, List[dict]] = {}
            for c in sec_chunks:
                ref = c.get("ref", "") or "general"
                sub_groups.setdefault(ref, []).append(c)

            for sub_idx, (ref, sub_chunks) in enumerate(sub_groups.items()):
                if len(sub_chunks) < 2:
                    continue
                sub_pages = [c["page_no"] for c in sub_chunks if c["page_no"]]
                sub_text  = " ".join(c["text"][:150] for c in sub_chunks[:2])
                sub_node  = TreeNode(
                    node_id    = f"sub_{sec_idx:04d}_{sub_idx:03d}",
                    title      = ref or section_title,
                    summary    = sub_text[:300],
                    start_page = min(sub_pages) if sub_pages else p_start,
                    end_page   = max(sub_pages) if sub_pages else p_end,
                    level      = 2,
                    parent_id  = section_node.node_id,
                )
                section_node.children.append(sub_node)

    def _build_via_api(self):
        """
        Upload document to PageIndex API and use cloud-built tree.
        """
        import requests

        # Prepare context representation for API
        doc_content = "\n\n".join(
            f"[Page {c['page_no']}] {c['section']}\n{c['text'][:500]}"
            for c in self.chunks[:100]  # first 100 chunks for index building
        )

        resp = requests.post(
            f"{PAGEINDEX_BASE_URL}/v1/index/build",
            headers={"Authorization": f"Bearer {PAGEINDEX_API_KEY}",
                     "Content-Type": "application/json"},
            json={"document_name": self.doc_name, "content": doc_content},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        self._api_doc_id = data.get("doc_id")
        # Parse tree from API response
        if "tree" in data:
            self.root = self._parse_api_tree(data["tree"])

    def _parse_api_tree(self, tree_dict: dict, level: int = 0) -> TreeNode:
        """Recursively parse PageIndex API tree response."""
        node = TreeNode(
            node_id    = tree_dict.get("node_id", f"n{level}"),
            title      = tree_dict.get("title", ""),
            summary    = tree_dict.get("summary", ""),
            start_page = tree_dict.get("start_index", 0),
            end_page   = tree_dict.get("end_index", 0),
            level      = level,
        )
        for child_dict in tree_dict.get("nodes", []):
            node.children.append(self._parse_api_tree(child_dict, level + 1))
        return node

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:   str,
        top_k:   int   = 5,
        use_api: bool  = False,
    ) -> List[dict]:
        """
        Retrieve the most relevant chunks using tree-search reasoning.

        This is fundamentally different from vector similarity:
          1. Ask LLM: "Given this Table of Contents, which sections likely
             contain the answer to: {query}?"
          2. Navigate those branches of the tree
          3. Within those sections, ask: "Which specific paragraph answers {query}?"
          4. Return the verbatim text from those paragraphs

        Returns list of {text, page_no, section, score, retrieval_path}.
        """
        if not self._index_built:
            return []

        # PageIndex API retrieval (cloud)
        if PAGEINDEX_API_KEY and self._api_doc_id and use_api:
            try:
                return self._api_retrieve(query, top_k)
            except Exception as e:
                print(f"[PageIndex] API retrieve failed: {e} — using local")

        # Local tree-search retrieval
        return self._local_tree_retrieve(query, top_k)

    def _local_tree_retrieve(self, query: str, top_k: int) -> List[dict]:
        """
        Local tree-search: uses LLM to navigate the tree, then fetches chunks.
        """
        if self.root is None:
            return []

        # Step 1: Ask LLM which top-level sections are relevant
        toc_str = self._tree_to_toc_string(self.root, max_depth=2)
        relevant_sections = self._llm_select_sections(query, toc_str)

        # Step 2: Collect chunks from relevant sections
        candidates = []
        for section_title in relevant_sections:
            section_chunks = self._find_chunks_in_section(section_title)
            candidates.extend(section_chunks)

        # Step 3: If API mode found good candidates, score and return
        if not candidates:
            # Fallback: keyword scoring across all chunks
            candidates = self._keyword_score_chunks(query)

        # Step 4: Score by query relevance
        scored = self._llm_score_candidates(query, candidates, top_k)
        return scored[:top_k]

    def _llm_select_sections(self, query: str, toc: str) -> List[str]:
        """
        Ask LLM which sections in the TOC are most likely to contain
        the answer to the query. Returns list of section titles.
        """
        prompt = f"""You are navigating a document's Table of Contents to find information.

QUERY: {query}

TABLE OF CONTENTS:
{toc}

Which sections (by title) are MOST LIKELY to contain information about: "{query}"?
List up to 3 section titles, one per line. Output ONLY the titles, nothing else."""

        response = self._call_llm(prompt, max_tokens=200)
        lines    = [l.strip() for l in response.strip().split("\n") if l.strip()]
        # Clean up (remove numbering, bullets)
        titles = [re.sub(r"^[\d\.\-\*\)]\s*", "", l) for l in lines[:3]]
        return [t for t in titles if t]

    def _llm_score_candidates(
        self, query: str, candidates: List[dict], top_k: int
    ) -> List[dict]:
        """
        Ask LLM to rank candidates by relevance to the query.
        Returns candidates with added 'score' field.
        """
        if not candidates:
            return []

        # Build numbered candidate list
        items = []
        for i, c in enumerate(candidates[:15]):
            items.append(f"[{i+1}] Page {c.get('page_no', '?')}: {c.get('text', '')[:300]}")
        candidates_str = "\n\n".join(items)

        prompt = f"""You are evaluating document excerpts for relevance to a query.

QUERY: {query}

EXCERPTS:
{candidates_str}

Rank these excerpts by relevance to the query. Output ONLY a comma-separated list
of excerpt numbers in order from most to least relevant. Example: 3, 1, 5, 2, 4"""

        response = self._call_llm(prompt, max_tokens=100)
        nums     = re.findall(r"\d+", response)

        # Reorder candidates by LLM ranking
        ranked = []
        seen   = set()
        for n in nums:
            idx = int(n) - 1
            if 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                c = dict(candidates[idx])
                c["score"]           = 1.0 - (len(ranked) * 0.1)
                c["retrieval_method"] = "pageindex-tree"
                ranked.append(c)

        # Add any unranked items at the end
        for i, c in enumerate(candidates):
            if i not in seen and len(ranked) < top_k * 2:
                ranked.append({**c, "score": 0.3, "retrieval_method": "pageindex-fallback"})

        return ranked

    def _api_retrieve(self, query: str, top_k: int) -> List[dict]:
        """Use PageIndex API for cloud retrieval."""
        import requests
        resp = requests.post(
            f"{PAGEINDEX_BASE_URL}/v1/retrieve",
            headers={"Authorization": f"Bearer {PAGEINDEX_API_KEY}"},
            json={"doc_id": self._api_doc_id, "query": query, "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "text":             item.get("text", ""),
                "page_no":          item.get("page_no", 0),
                "section":          item.get("section", ""),
                "score":            item.get("score", 0.5),
                "retrieval_method": "pageindex-api",
            })
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tree_to_toc_string(self, node: TreeNode, depth: int = 0, max_depth: int = 2) -> str:
        if depth > max_depth:
            return ""
        indent = "  " * depth
        lines  = [f"{indent}{node.title} (p{node.start_page}-{node.end_page}): {node.summary[:100]}"]
        for child in node.children[:10]:  # limit breadth
            child_str = self._tree_to_toc_string(child, depth + 1, max_depth)
            if child_str:
                lines.append(child_str)
        return "\n".join(lines)

    def _find_chunks_in_section(self, section_title: str) -> List[dict]:
        """Find chunks whose section heading fuzzy-matches section_title."""
        title_lower = section_title.lower()
        results     = []
        for c in self.chunks:
            section = (c.get("section", "") or "").lower()
            if (title_lower in section or section in title_lower or
                    self._token_overlap(title_lower, section) > 0.5):
                results.append(c)
        return results[:20]

    def _keyword_score_chunks(self, query: str) -> List[dict]:
        """Simple keyword fallback scoring."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored       = []
        for c in self.chunks:
            text_tokens = set(re.findall(r"\w+", (c.get("text", "") or "").lower()))
            overlap     = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            if overlap > 0:
                scored.append((overlap, id(c), c))   # id(c) breaks dict comparison ties
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, _, c in scored[:20]]

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        ta = set(a.split())
        tb = set(b.split())
        return len(ta & tb) / max(len(ta | tb), 1)

    def _llm_summarise(self, title: str, text: str) -> str:
        """Generate a one-sentence summary of a section."""
        if not text.strip():
            return title
        prompt = (f"Summarise this document section titled '{title}' in ONE sentence "
                  f"(max 100 chars):\n{text[:400]}")
        summary = self._call_llm(prompt, max_tokens=80)
        return summary.strip()[:200] or title

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Call the best available LLM."""
        # Try Gemini first
        if GEMINI_API_KEY:
            try:
                return self._call_gemini(prompt, max_tokens)
            except Exception:
                pass
        # Ollama fallback
        return self._call_ollama(prompt, max_tokens)

    def _call_gemini(self, prompt: str, max_tokens: int) -> str:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=GEMINI_API_KEY)
        config   = types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_tokens)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt, config=config)
        return response.text or ""

    def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        import requests as req
        try:
            r = req.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False,
                      "options": {"temperature": 0.0, "num_predict": max_tokens}},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["message"]["content"] or ""
        except Exception:
            return ""

    def _count_nodes(self, node: Optional[TreeNode]) -> int:
        if node is None:
            return 0
        return 1 + sum(self._count_nodes(c) for c in node.children)

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:60]

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _save_to_cache(self, path: Path):
        data = {
            "doc_name": self.doc_name,
            "chunks":   self.chunks,
            "tree":     self.root.to_dict() if self.root else None,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load_from_cache(self, path: Path):
        data         = json.loads(path.read_text())
        self.chunks  = data.get("chunks", [])
        tree_dict    = data.get("tree")
        if tree_dict:
            self.root = self._parse_api_tree(tree_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacement for vector_store.retrieve()
# ─────────────────────────────────────────────────────────────────────────────

# Global registry: doc_name → PageIndexRetriever
_retrievers: Dict[str, PageIndexRetriever] = {}


def get_retriever(doc_name: str) -> Optional[PageIndexRetriever]:
    return _retrievers.get(doc_name)


def register_retriever(doc_name: str, retriever: PageIndexRetriever):
    _retrievers[doc_name] = retriever


def pageindex_retrieve(query: str, doc_name: str, top_k: int = 5) -> List[dict]:
    """
    Drop-in replacement for vector_store.retrieve().
    Returns chunks in the same format expected by extractor.py.
    """
    retriever = _retrievers.get(doc_name)
    if retriever is None:
        return []

    raw_results = retriever.retrieve(query, top_k=top_k)

    # Normalise to the shape expected by extractor.py
    return [
        {
            "text":            r.get("text", ""),
            "page_no":         r.get("page_no", 0),
            "section_heading": r.get("section", ""),
            "clause_ref":      r.get("ref", f"p{r.get('page_no', 0)}"),
            "score":           r.get("score", 0.5),
        }
        for r in raw_results
    ]
