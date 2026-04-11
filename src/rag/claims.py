#!/usr/bin/env python3
"""
Phase 3 — Claim Extraction
Turns a post's text into 1–3 atomic claims for retrieval/NLI.

USAGE (programmatic):
    from src.rag.claims import extract_claims

    claims = extract_claims(
        text=post_text,
        cfg=cfg_dict,                 # parsed from configs/rag.yaml
        llm_call_fn=None              # optional; if provided and cfg.claims.backend == "llm"
    )

CONFIG KEYS USED (from configs/rag.yaml):
    claims:
      backend: "rule_based" | "llm"
      max_claims: int
      min_chars: int
      max_chars: int
      dedupe: bool
      llm:
        provider: str       # informational; we don't import SDKs here
        model: str
        temperature: float  # should be 0.0 for determinism
        max_tokens: int

DESIGN GOALS
- Deterministic by default (seed=42; rule-based path has no randomness).
- LLM path is opt-in via cfg; we *inject* the call function so this module has no vendor lock-in.
- Clean, readable text processing; robust to headlines and short blurbs.

NOTE
- We only consume Phase-1 schema fields (`text` content) and Phase-3 config;
  nothing else from Phase 2 is required here.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional
import re

# --------- public API --------- #

def extract_claims(
    text: str,
    cfg: Dict,
    llm_call_fn: Optional[Callable[[str, Dict], str]] = None,
) -> List[str]:
    """
    Extract up to cfg['claims']['max_claims'] atomic claims from the input text.

    Args:
        text: The post headline/body from Phase-1 manifests (cfg.data.text_field).
        cfg:  Parsed dict from configs/rag.yaml.
        llm_call_fn: Optional callable used only when backend == "llm".
                     Signature: (prompt:str, llm_cfg:Dict) -> raw_text_response

    Returns:
        List[str]: Cleaned, deduplicated, length-bounded claims (1..max_claims), deterministic.

    Behavior:
      - If cfg.claims.backend == "rule_based": use the deterministic splitter.
      - If cfg.claims.backend == "llm":
           * If llm_call_fn is provided → use it with temperature=0 and parse the response.
           * Else → fallback to rule_based (keeps pipeline runnable without LLM credentials).
    """
    claims_cfg = cfg.get("claims", {})
    backend = (claims_cfg.get("backend") or "rule_based").lower()

    if backend == "llm" and llm_call_fn is not None:
        raw = _llm_generate_claims(text, claims_cfg, llm_call_fn)
        candidates = _parse_llm_numbered_list(raw)
    else:
        # Fallback path if backend != 'llm' OR llm_call_fn not provided.
        candidates = _rule_based_candidates(text)

    # sanitize → filter → clip → dedupe → top-k
    return _postprocess_candidates(candidates, claims_cfg)


# --------- rule-based extractor (deterministic) --------- #

_SENT_SPLIT_RE = re.compile(
    r"""          # split on:
    [\.\?\!；;：:]+      # sentence-ending punctuation
    | \n+               # hard line breaks
    | \s*[•·\-–—]\s+    # list bullets/dashes
    """,
    re.VERBOSE,
)

_WS_RE = re.compile(r"\s+")

def _rule_based_candidates(text: str) -> List[str]:
    """
    Deterministic splitter for headlines/short blurbs:
      1) Split on sentence endings / newlines / bullets.
      2) Keep non-empty fragments.
      3) Merge very short neighboring fragments where possible.

    Rationale:
      - Headline-like inputs often lack terminal punctuation.
      - Bulleted/stacked claims are common in misinformation posts.
    """
    # Normalize whitespace; keep text as-is otherwise
    norm = _WS_RE.sub(" ", (text or "")).strip()
    if not norm:
        return []

    # Primary segmentation
    parts = [p.strip(" \"'“”‘’()[]") for p in _SENT_SPLIT_RE.split(norm)]
    parts = [p for p in parts if p]  # drop empties

    # Optionally: join extremely short fragments to next piece (avoid trivial claims)
    merged: List[str] = []
    buf = ""
    for p in parts:
        if len(p) < 20:  # heuristic: short fragment, try merging
            buf = f"{buf} {p}".strip()
            continue
        if buf:
            merged.append(f"{buf} {p}".strip())
            buf = ""
        else:
            merged.append(p)
    if buf:
        merged.append(buf)

    return merged


# --------- LLM backend (optional; deterministic with temperature=0) --------- #

_LLM_SYSTEM = (
    "You are a precise fact-checking assistant. "
    "Rewrite the user text as 1–3 atomic factual claims, each concise and verifiable."
)
_LLM_USER_TEMPLATE = """Text:
{TEXT}

Rules:
- Return a numbered list, each item a single factual claim.
- Keep each claim stand-alone and <= 280 characters.
- Do not include opinions or vague language.
- If the text has <=3 obvious claims, list only those.
"""

def _llm_generate_claims(text: str, claims_cfg: Dict, llm_call_fn) -> str:
    """
    Calls the injected LLM function with temperature=0 for determinism.
    We don't import any SDKs here — the caller wires their own function.

    llm_call_fn is expected to **synchronously** return the assistant's text.
    """
    llm_cfg = {
        "provider": claims_cfg.get("llm", {}).get("provider", "openai"),
        "model": claims_cfg.get("llm", {}).get("model", "gpt-4o-mini"),
        "temperature": 0.0,  # force determinism
        "max_tokens": int(claims_cfg.get("llm", {}).get("max_tokens", 256)),
        "system_prompt": _LLM_SYSTEM,
    }
    prompt = _LLM_USER_TEMPLATE.replace("{TEXT}", text or "")
    return llm_call_fn(prompt, llm_cfg)


_NUM_LIST_RE = re.compile(r"^\s*(?:\d+[\.\)]|-|\*)\s+(.*)$")

def _parse_llm_numbered_list(raw: str) -> List[str]:
    """
    Parses a numbered/bulleted list into plain claim strings.
    Falls back to a single-item list if parsing fails.
    """
    if not raw:
        return []
    lines = [l.strip() for l in raw.splitlines()]
    items: List[str] = []
    for l in lines:
        if not l:
            continue
        m = _NUM_LIST_RE.match(l)
        if m:
            items.append(m.group(1).strip(" \"'“”‘’()[]"))
        else:
            # accept bare lines if they look like a sentence/claim
            items.append(l.strip(" \"'“”‘’()[]"))
    # remove empties and over-verbose artifacts
    items = [i for i in items if i]
    # If the LLM returned a paragraph, treat the whole thing as one claim
    return items or [raw.strip()]


# --------- shared post-processing --------- #

def _postprocess_candidates(cands: List[str], claims_cfg: Dict) -> List[str]:
    """
    Applies: strip, de-duplication, length bounds, cap to max_claims.
    """
    max_claims = int(claims_cfg.get("max_claims", 3))
    min_chars  = int(claims_cfg.get("min_chars", 20))
    max_chars  = int(claims_cfg.get("max_chars", 280))
    dedupe     = bool(claims_cfg.get("dedupe", True))

    cleaned: List[str] = []
    seen_norm = set()

    for c in cands:
        c0 = _WS_RE.sub(" ", (c or "")).strip().strip(" .")
        if not c0:
            continue
        if len(c0) < min_chars:
            continue
        if len(c0) > max_chars:
            c0 = c0[:max_chars].rstrip()  # clip long claims conservatively
        if dedupe:
            key = c0.lower()
            if key in seen_norm:
                continue
            seen_norm.add(key)
        cleaned.append(c0)
        if len(cleaned) >= max_claims:
            break

    return cleaned
# ------------------------------ simple CLI ---------------------------------- #
if __name__ == "__main__":
    """
    Quick smoke run for claim extraction.

    Examples:
      python -m src.rag.claims --config configs/rag.yaml --limit 10
      python src/rag/claims.py --config configs/rag.yaml --limit 25 --out artifacts/rag/claims_preview.tsv

    It reads cfg.data.input_path (Phase-1 JSONL), extracts claims, and writes a small TSV preview:
      id \t claim_index \t claim_text
    """
    import argparse
    import json
    from pathlib import Path
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/rag.yaml")
    ap.add_argument("--limit", type=int, default=20, help="number of posts to process")
    ap.add_argument("--out", type=str, default="artifacts/rag/claims_preview.tsv")
    args = ap.parse_args()

    # load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    input_path = Path(cfg["data"]["input_path"])
    text_field = cfg["data"]["text_field"]
    id_field = cfg["data"]["id_field"]

    # ensure output dir exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_posts = 0
    n_claims = 0
    rows = []

    # read JSONL deterministically; limit for a quick preview
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if n_posts >= args.limit:
                break
            obj = json.loads(line)
            text = obj.get(text_field, "") or ""
            pid = obj.get(id_field, "")
            claims = extract_claims(text=text, cfg=cfg, llm_call_fn=None)
            for idx, c in enumerate(claims):
                rows.append((pid, idx, c))
            n_posts += 1
            n_claims += len(claims)

    # write TSV
    with out_path.open("w", encoding="utf-8") as wf:
        wf.write("id\tclaim_idx\tclaim\n")
        for pid, idx, c in rows:
            wf.write(f"{pid}\t{idx}\t{c}\n")

    print(f"[claims] processed posts: {n_posts}, total claims: {n_claims}")
    print(f"[claims] wrote preview → {out_path}")
