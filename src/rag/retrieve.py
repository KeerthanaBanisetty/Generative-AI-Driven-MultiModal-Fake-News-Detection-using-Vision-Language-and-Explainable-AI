#!/usr/bin/env python3
"""
Phase 3 — Retrieval (Serper / Google Search JSON)

What this module does:
  • Given an atomic claim, call Serper's Google Search JSON API and collect top-k
    substantial snippets (title, url, date, snippet, score, source="serper").
  • Deterministic ordering by result rank.
  • Honors optional domain allow/block lists.
  • Robust cleaning and min length filtering to keep useful evidence only.

Public entrypoint (used by pipeline code):
    from src.rag.retrieve import retrieve
    snippets = retrieve(claim="Mars has two moons named Phobos and Deimos.", cfg=cfg)

Returned shape (list of dicts, ordered by relevance):
    {
      "source":  "serper",
      "title":   str,
      "url":     str,
      "date":    str | None,
      "snippet": str,
      "score":   float  # rank-based in (0,1], 1.0 = top hit
    }

CLI (smoke runs):
  # Single-claim (no configs needed)
  PYTHONPATH=. python -m src.rag.retrieve --claim "Mars has two moons named Phobos and Deimos." --top_k 5 --verbose

  # Config-driven (reads configs/rag.yaml and Phase-1 JSONL; writes TSV preview)
  PYTHONPATH=. python -m src.rag.retrieve \
      --config configs/rag.yaml --limit 20 --top_k 5 \
      --out artifacts/rag/retrieval_preview.tsv --verbose

Config keys used (configs/rag.yaml):
  retrieve:
    provider: serper
    top_k: 5
    min_chars: 80
    serper:
      api_key_env: SERPER_API_KEY     # optional env override; hardcoded key used by default
      country: "in"                   # "us", "in", etc. (maps to gl)
      num: 10                         # how many results to request before filtering
      allow_domains: []               # keep only these domains if non-empty
      block_domains: []               # drop these domains if non-empty
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import os
import time
import html
import re
import json
import urllib.parse
import requests


# ------------------------------- Serper setup --------------------------------

# Your provided key (hardcoded default); can be overridden by env var.
# If you DON'T want env override, delete the os.getenv(...) line below.
_DEFAULT_SERPER_API_KEY = "55d4c25d043357659abb9af404d76c54852d9acc"

_SERPER_ENDPOINT = "https://google.serper.dev/search"


# ------------------------------- Data shapes -------------------------------- #

@dataclass
class Snippet:
    source: str         # "serper"
    title: str
    url: str
    date: Optional[str]
    snippet: str
    score: float

    def to_dict(self) -> Dict:
        return asdict(self)


# ------------------------------ Text cleaners --------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")

def _strip_html(txt: str) -> str:
    return _HTML_TAG_RE.sub("", txt or "")

def _clean_text(txt: str) -> str:
    if not txt:
        return ""
    t = _strip_html(txt)
    t = html.unescape(t)
    t = _MULTI_WS_RE.sub(" ", t).strip()
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    # bound extreme length for previews (NLI will truncate again)
    if len(t) > 900:
        cut = t[:900]
        dot = cut.rfind(".")
        t = cut[:dot+1] if dot > 600 else cut
    return t


def _hostname(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return ""


# --------------------------- Provider dispatcher ----------------------------- #

def retrieve(
    claim: str,
    cfg: Dict,
    *,
    top_k: Optional[int] = None,
    min_chars: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict]:
    """
    Dispatch to the configured provider; here we implement 'serper'.
    """
    r_cfg = cfg.get("retrieve", {}) if cfg else {}
    provider = (r_cfg.get("provider") or "serper").lower()
    k = int(top_k if top_k is not None else r_cfg.get("top_k", 5))
    mchars = int(min_chars if min_chars is not None else r_cfg.get("min_chars", 80))

    if provider == "serper":
        s_cfg = r_cfg.get("serper", {}) if r_cfg else {}
        return _retrieve_serper(claim, k, mchars, s_cfg, verbose)

    # Unknown provider → default to serper
    s_cfg = r_cfg.get("serper", {}) if r_cfg else {}
    return _retrieve_serper(claim, k, mchars, s_cfg, verbose)


# --------------------------------- Serper ----------------------------------- #

def _retrieve_serper(
    query: str,
    top_k: int,
    min_chars: int,
    serper_cfg: Dict,
    verbose: bool = False,
) -> List[Dict]:
    """
    Calls Serper's /search endpoint (POST) and normalizes results.
    """
    # Key: env override (if present) else hardcoded default supplied by user
    api_key = os.getenv(serper_cfg.get("api_key_env", "SERPER_API_KEY"), _DEFAULT_SERPER_API_KEY)

    # Request params/body
    country = serper_cfg.get("country", "in")  # maps to 'gl'
    num = int(serper_cfg.get("num", 10))
    allow_domains = set([d.lower() for d in serper_cfg.get("allow_domains", []) if d])
    block_domains = set([d.lower() for d in serper_cfg.get("block_domains", []) if d])

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "gl": country,
        "num": num,
    }

    try:
        resp = requests.post(_SERPER_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=15)
        resp.raise_for_status()
        js = resp.json()
    except Exception as e:
        if verbose:
            print(f"[retrieve][serper][ERR] API call failed: {e}")
        return []

    organic = js.get("organic", []) or []
    if verbose:
        print(f"[retrieve][serper] query='{query[:80]}{'…' if len(query)>80 else ''}' hits={len(organic)}")

    # rank and filter results
    results: List[Snippet] = []
    kept = 0
    for rank, item in enumerate(organic, start=1):
        title = _clean_text(item.get("title") or "")
        url = item.get("link") or item.get("url") or ""
        date = item.get("date") or item.get("dateUtc") or None
        snippet_raw = item.get("snippet") or item.get("description") or ""
        snippet = _clean_text(snippet_raw)

        if not title or not url or not snippet:
            continue

        host = _hostname(url)
        if block_domains and any(b in host for b in block_domains):
            continue
        if allow_domains and not any(a in host for a in allow_domains):
            continue

        # If snippet is still short, try merging 'snippet + title' to help NLI context.
        if len(snippet) < min_chars:
            snippet = _clean_text(f"{title}. {snippet}")

        # Keep even if < min_chars (to avoid totally empty TSVs), but log it.
        too_short = len(snippet) < min_chars

        score = 1.0 - (rank - 1) / max(1, len(organic))  # simple rank-based score
        results.append(Snippet(
            source="serper",
            title=title,
            url=url,
            date=date,
            snippet=snippet,
            score=float(score),
        ))
        kept += 1
        if verbose:
            msg = f"  - kept [{rank}] {host}  len={len(snippet)}  score={score:.2f}"
            if too_short:
                msg += "  [<min_chars]"
            print(msg)

        if kept >= top_k:
            break

        time.sleep(0.05)  # gentle pacing (usually not needed, but polite)

    return [r.to_dict() for r in results]


# ----------------------------------- CLI ------------------------------------ #

if __name__ == "__main__":
    """
    Examples:
      # Single-claim sanity (no configs needed)
      PYTHONPATH=. python -m src.rag.retrieve --claim "Mars has two moons named Phobos and Deimos." --top_k 5 --verbose

      # Config-driven mode (reads Phase-1 JSONL and writes TSV preview)
      PYTHONPATH=. python -m src.rag.retrieve \
        --config configs/rag.yaml --limit 20 --top_k 5 \
        --out artifacts/rag/retrieval_preview.tsv --verbose
    """
    import argparse
    import yaml
    from pathlib import Path
    from src.rag.claims import extract_claims

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="configs/rag.yaml (multi-post mode)")
    ap.add_argument("--claim", type=str, default=None, help="single claim text (overrides --config)")
    ap.add_argument("--top_k", type=int, default=None, help="override top_k")
    ap.add_argument("--min_chars", type=int, default=None, help="override min_chars")
    ap.add_argument("--limit", type=int, default=20, help="num posts to process (config mode)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default="artifacts/rag/retrieval_preview.tsv")
    args = ap.parse_args()

    # Single-claim mode
    if args.claim:
        # Minimal cfg stub for single-claim sanity
        cfg = {
            "retrieve": {
                "provider": "serper",
                "top_k": args.top_k or 5,
                "min_chars": args.min_chars or 80,
                "serper": {
                    "api_key_env": "SERPER_API_KEY",
                    "country": "in",
                    "num": 10,
                    "allow_domains": [],
                    "block_domains": [],
                }
            }
        }
        out = retrieve(args.claim, cfg, top_k=args.top_k, min_chars=args.min_chars, verbose=args.verbose)
        print(f"[retrieve][serper] claim: {args.claim}")
        if not out:
            print("[retrieve][serper] no results (check API key / quota / network)")
        for i, s in enumerate(out, 1):
            print(f"  [{i}] {s['title']}  ({s['score']:.2f})")
            print(f"      {s['url']}")
            print(f"      {s['snippet'][:240]}{'…' if len(s['snippet'])>240 else ''}")
        raise SystemExit(0)

    # Config-driven mode (multi-post)
    if not args.config:
        raise SystemExit("Either provide --claim OR provide --config for multi-post mode.")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # force provider to serper to avoid stale configs
    cfg.setdefault("retrieve", {})["provider"] = "serper"

    # Honor CLI overrides
    if args.top_k is not None:
        cfg["retrieve"]["top_k"] = int(args.top_k)
    if args.min_chars is not None:
        cfg["retrieve"]["min_chars"] = int(args.min_chars)

    # Default Serper section if missing
    cfg["retrieve"].setdefault("serper", {
        "api_key_env": "SERPER_API_KEY",
        "country": "in",
        "num": 10,
        "allow_domains": [],
        "block_domains": [],
    })

    # IO paths from config (Phase-1)
    input_path = Path(cfg["data"]["input_path"])
    text_field = cfg["data"]["text_field"]
    id_field = cfg["data"]["id_field"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_posts = 0
    n_claims = 0
    n_snippets = 0

    with input_path.open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as wf:
        wf.write("id\tclaim_idx\tclaim\ttitle\turl\tdate\tscore\tsnippet\n")
        for line in f:
            if n_posts >= args.limit:
                break
            obj = json.loads(line)
            text = obj.get(text_field, "") or ""
            pid = obj.get(id_field, "")

            claims = extract_claims(text=text, cfg=cfg, llm_call_fn=None)
            if args.verbose:
                print(f"[retrieve][serper] id={pid}  claims={len(claims)}")

            for c_idx, claim in enumerate(claims):
                n_claims += 1
                snippets = retrieve(claim, cfg, verbose=args.verbose)
                if args.verbose and not snippets:
                    print("  - no snippets kept for this claim")
                for s in snippets:
                    n_snippets += 1
                    wf.write(
                        f"{pid}\t{c_idx}\t{claim}\t{s['title']}\t{s['url']}\t"
                        f"{s.get('date','')}\t{s.get('score',0.0):.3f}\t{s['snippet'].replace('\t',' ').replace('\n',' ')}\n"
                    )

            n_posts += 1

    print(f"[retrieve][serper] posts processed: {n_posts}, claims extracted: {n_claims}, snippets written: {n_snippets}")
    print(f"[retrieve][serper] wrote preview → {out_path}")
