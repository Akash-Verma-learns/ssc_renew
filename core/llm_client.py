"""
core/llm_client.py  —  v3  Anthropic-First Architecture
=========================================================

PROVIDER PRIORITY
─────────────────
  0. Anthropic Claude (claude-sonnet-4-20250514)  ← NEW PRIMARY
     Best structured-JSON accuracy for table extraction.
     Set ANTHROPIC_API_KEY=sk-ant-...
     pip install anthropic

  1. Groq  (GROQ_API_KEY)   — llama-3.3-70b-versatile, free 30 RPM
  2. Gemini (GEMINI_API_KEY) — gemini-2.0-flash
  3. Ollama (always local)   — final fallback

WHY ANTHROPIC FIRST
────────────────────
Claude excels at:
  • Extracting tables from verbatim PDF text (the RFP scoring table)
  • Returning valid JSON even on complex multi-page layouts
  • NOT hallucinating criteria that don't exist
  • NOT merging the live-assessment row into document criteria

RATE LIMITING
─────────────
Anthropic Tier-1: ~50 RPM on Sonnet — circuit opens on 429
"""

from __future__ import annotations

import collections
import json
import os
import re
import threading
import time
import requests
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_TIMEOUT", "600"))

_DEFAULT_COOLDOWN = 180.0


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter + circuit breaker (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self, rpm: int, window: float = 60.0, name: str = "LLM"):
        self._rpm        = rpm
        self._window     = window
        self._name       = name
        self._lock       = threading.Lock()
        self._times: collections.deque = collections.deque()
        self._open_until: float = 0.0

    def is_open(self) -> bool:
        return time.time() < self._open_until

    def remaining_open(self) -> float:
        return max(0.0, self._open_until - time.time())

    def wait_for_slot(self) -> bool:
        if self.is_open():
            rem = self.remaining_open()
            print(f"[{self._name}] circuit open — {rem:.0f}s remaining — skipping provider")
            return False
        while True:
            with self._lock:
                now = time.time()
                while self._times and self._times[0] < now - self._window:
                    self._times.popleft()
                if len(self._times) < self._rpm:
                    return True
                oldest    = self._times[0]
                wait_secs = (oldest + self._window) - now + 0.3
            print(f"[{self._name}] RPM cap ({self._rpm}/min) — waiting {wait_secs:.1f}s")
            time.sleep(max(wait_secs, 0.3))

    def record_success(self) -> None:
        with self._lock:
            self._times.append(time.time())

    def record_429(self, retry_after: float = _DEFAULT_COOLDOWN) -> None:
        open_duration = max(retry_after, 10.0)
        with self._lock:
            self._open_until = time.time() + open_duration
        print(f"[{self._name}] 429 received — circuit open for {open_duration:.0f}s")


_anthropic_lim = _RateLimiter(rpm=45, name="Anthropic")
_groq_lim      = _RateLimiter(rpm=20, name="Groq")
_gemini_lim    = _RateLimiter(rpm=8,  name="Gemini")


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    prompt:    str,
    ctx:       int  = 8192,
    json_mode: bool = False,
    label:     str  = "",
) -> str:
    """
    Call best available LLM.  Priority: Anthropic → Groq → Gemini → Ollama.
    Returns raw text.  Never raises — returns "" on total failure.
    """
    tag = f"[{label}] " if label else ""

    # ── 0. Anthropic Claude (most accurate for structured JSON) ────────────
    if ANTHROPIC_API_KEY and not _anthropic_lim.is_open():
        try:
            result = _call_anthropic(prompt, json_mode=json_mode, label=label)
            if result:
                return result
            print(f"[LLM] {tag}Anthropic returned empty — trying Groq")
        except Exception as e:
            print(f"[LLM] {tag}Anthropic exception: {e} — trying Groq")

    # ── 1. Groq ────────────────────────────────────────────────────────────
    if GROQ_API_KEY and not _groq_lim.is_open():
        try:
            result = _call_groq(prompt, json_mode=json_mode, label=label)
            if result:
                return result
            print(f"[LLM] {tag}Groq returned empty — trying Gemini")
        except Exception as e:
            print(f"[LLM] {tag}Groq exception: {e} — trying Gemini")

    # ── 2. Gemini ──────────────────────────────────────────────────────────
    if GEMINI_API_KEY and not _gemini_lim.is_open():
        try:
            result = _call_gemini(prompt, json_mode=json_mode, label=label)
            if result:
                return result
            print(f"[LLM] {tag}Gemini returned empty — falling back to Ollama")
        except Exception as e:
            print(f"[LLM] {tag}Gemini exception: {e} — falling back to Ollama")

    # ── 3. Ollama (always-available local fallback) ────────────────────────
    return _call_ollama(prompt, ctx=ctx, label=label)


def extract_json(text: str) -> Any:
    """Robustly extract JSON from raw LLM output."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        end = -1
        for i, ch in enumerate(text[start:], start):
            if esc:                    esc = False;         continue
            if ch == "\\" and in_str:  esc = True;          continue
            if ch == '"':              in_str = not in_str;  continue
            if in_str:                 continue
            if ch == open_ch:          depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            candidate = re.sub(r",\s*([}\]])", r"\1", text[start:end])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = _repair_json(candidate)
                if repaired:
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass
        break
    return None


def _repair_json(text: str) -> str:
    text = re.sub(r',\s*"[^"]*$', '', text.rstrip())
    text = re.sub(r',\s*\{[^{}]*$', '', text)
    text = re.sub(r',\s*$',         '', text)
    stack: list[str] = []
    in_str = esc = False
    for ch in text:
        if esc:                    esc = False;         continue
        if ch == "\\" and in_str:  esc = True;          continue
        if ch == '"':              in_str = not in_str;  continue
        if in_str:                 continue
        if ch in "{[":             stack.append(ch)
        elif ch in "}]" and stack: stack.pop()
    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[c] for c in reversed(stack))


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic  (pip install anthropic)
# ─────────────────────────────────────────────────────────────────────────────

def _call_anthropic(prompt: str, json_mode: bool = False,
                    label: str = "", retries: int = 2) -> str:
    """
    Call Anthropic Claude.  Uses direct REST to avoid SDK version conflicts.
    json_mode: prefills '{"' to force JSON output (Claude supports prefill).
    """
    tag = f"[{label}] " if label else ""

    for attempt in range(retries):
        if not _anthropic_lim.wait_for_slot():
            return ""

        messages = [{"role": "user", "content": prompt}]
        # Prefill assistant turn to force JSON — very reliable with Claude
        body: dict = {
            "model":       ANTHROPIC_MODEL,
            "max_tokens":  4096,
            "temperature": 0,
            "messages":    messages,
        }
        if json_mode:
            body["messages"].append({"role": "assistant", "content": "{"})

        try:
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json=body,
                timeout=180,
            )

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", _DEFAULT_COOLDOWN))
                _anthropic_lim.record_429(retry_after=retry_after)
                return ""

            if resp.status_code in (500, 502, 503, 504):
                wait = 5 * (attempt + 1)
                print(f"[Anthropic] {tag}{resp.status_code} — waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data    = resp.json()
            content = data.get("content", [])
            text    = "".join(
                block.get("text", "") for block in content
                if block.get("type") == "text"
            )
            # If we prefilled "{", prepend it back
            if json_mode and text:
                text = "{" + text
            _anthropic_lim.record_success()
            print(f"[Anthropic] {tag}OK ({len(text)} chars)")
            return text

        except requests.exceptions.Timeout:
            print(f"[Anthropic] {tag}timeout (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"[Anthropic] {tag}error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                break

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Groq  (OpenAI-compatible REST)
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str, json_mode: bool = False,
               label: str = "", retries: int = 2) -> str:
    tag = f"[{label}] " if label else ""
    for attempt in range(retries):
        if not _groq_lim.wait_for_slot():
            return ""
        payload: dict = {
            "model":       GROQ_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type":  "application/json"},
                json=payload, timeout=120,
            )
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", _DEFAULT_COOLDOWN))
                _groq_lim.record_429(retry_after=retry_after)
                return ""
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
            _groq_lim.record_success()
            return content
        except requests.exceptions.Timeout:
            if attempt < retries - 1: time.sleep(5)
        except Exception as e:
            print(f"[Groq] {tag}error: {e}")
            if attempt < retries - 1: time.sleep(5)
            else: break
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Gemini  (google-genai SDK)
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, json_mode: bool = False,
                 label: str = "", retries: int = 2) -> str:
    tag = f"[{label}] " if label else ""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    config_kwargs: dict = {"temperature": 0.0, "max_output_tokens": 8192}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    config = types.GenerateContentConfig(**config_kwargs)
    for attempt in range(retries):
        if not _gemini_lim.wait_for_slot():
            return ""
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config)
            _gemini_lim.record_success()
            return response.text or ""
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                _gemini_lim.record_429(retry_after=_DEFAULT_COOLDOWN)
                return ""
            elif "503" in err or "UNAVAILABLE" in err:
                time.sleep(15 * (attempt + 1))
            elif attempt < retries - 1:
                time.sleep(5)
            else:
                raise
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Ollama  (local)
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL,
                 ctx: int = 8192, label: str = "") -> str:
    tag = f"[{label}] " if label else ""
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "options": {"temperature": 0.0, "num_ctx": ctx}},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"] or ""
    except requests.exceptions.Timeout:
        print(f"[Ollama] {tag}timeout after {OLLAMA_TIMEOUT}s")
        return ""
    except Exception as e:
        print(f"[Ollama] {tag}error: {e}")
        return ""


def provider_status() -> dict:
    return {
        "anthropic": {
            "configured":    bool(ANTHROPIC_API_KEY),
            "model":         ANTHROPIC_MODEL,
            "circuit_open":  _anthropic_lim.is_open(),
            "open_for_secs": round(_anthropic_lim.remaining_open(), 1),
        },
        "groq": {
            "configured":    bool(GROQ_API_KEY),
            "circuit_open":  _groq_lim.is_open(),
            "open_for_secs": round(_groq_lim.remaining_open(), 1),
        },
        "gemini": {
            "configured":    bool(GEMINI_API_KEY),
            "circuit_open":  _gemini_lim.is_open(),
            "open_for_secs": round(_gemini_lim.remaining_open(), 1),
        },
        "ollama": {
            "configured": True,
            "circuit_open": False,
            "open_for_secs": 0,
        },
    }