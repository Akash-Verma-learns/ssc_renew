"""
core/llm_client.py
------------------
Unified LLM client: Google Gemini (primary) + Ollama (fallback).

Fallback: if GEMINI_API_KEY is not set or Gemini fails, Ollama is used.

Usage:
    from core.llm_client import call_llm, extract_json

    raw   = call_llm("Extract the max marks from this table: ...")
    data  = extract_json(raw)   # -> dict | list | None
"""

from __future__ import annotations

import json
import os
import re
import time
import requests
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# 4.1 s between Gemini calls → ≤14.6 RPM, safely under the 15 RPM free limit
_GEMINI_INTERVAL   = 4.1
_last_gemini_ts: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    prompt:    str,
    ctx:       int  = 4096,      # used only by Ollama fallback
    json_mode: bool = False,     # ask Gemini to respond in JSON MIME type
    label:     str  = "",        # debug label shown in logs
) -> str:
    """
    Send a prompt to the best available LLM.
    Returns the raw text response. Returns "" on total failure — never raises.
    """
    tag = f"[{label}] " if label else ""

    if GEMINI_API_KEY:
        try:
            result = _call_gemini(prompt, json_mode=json_mode)
            if result:
                return result
            print(f"[LLM] {tag}Gemini returned empty — trying Ollama fallback")
        except Exception as e:
            print(f"[LLM] {tag}Gemini error: {e} — trying Ollama fallback")

    return _call_ollama(prompt, ctx=ctx, label=label)


def extract_json(text: str) -> Any:
    """
    Robustly extract a JSON object or array from raw LLM output.
    Handles markdown fences, trailing commas, preamble text.
    Returns the parsed Python object (dict/list), or None on failure.
    """
    if not text:
        return None

    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find outermost JSON object or array
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if esc:                   esc = False; continue
            if ch == "\\" and in_str: esc = True;  continue
            if ch == '"':             in_str = not in_str; continue
            if in_str:                continue
            if ch == open_ch:         depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = re.sub(r",\s*([}\]])", r"\1", text[start:i + 1])
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        break

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Gemini (google-genai SDK — NOT the deprecated google-generativeai)
# Install: pip install google-genai
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, json_mode: bool = False, retries: int = 3) -> str:
    global _last_gemini_ts

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    config_kwargs: dict = {
        "temperature":       0.0,
        "max_output_tokens": 4096,
    }
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    config = types.GenerateContentConfig(**config_kwargs)

    for attempt in range(retries):
        elapsed = time.time() - _last_gemini_ts
        if elapsed < _GEMINI_INTERVAL:
            time.sleep(_GEMINI_INTERVAL - elapsed)
        _last_gemini_ts = time.time()

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)
                print(f"[LLM] Gemini rate-limited (attempt {attempt+1}) — waiting {wait}s")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(5)
            else:
                raise

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Ollama fallback
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL,
                 ctx: int = 4096, label: str = "") -> str:
    tag = f"[{label}] " if label else ""
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model":    model,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.0, "num_ctx": ctx},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"] or ""
    except Exception as e:
        print(f"[LLM] {tag}Ollama error: {e}")
        return ""