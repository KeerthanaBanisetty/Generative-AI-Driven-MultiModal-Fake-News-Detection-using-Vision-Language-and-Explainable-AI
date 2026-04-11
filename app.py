"""
Fake News Detector — Streamlit App  (Enhanced UI)
==================================================
Models that run LIVE in real-time:
  ✅ RoBERTa (your trained text classifier)  →  main verdict
  ✅ Advanced vision analysis                →  image–text consistency check
  ✅ Sentiment analysis (TextBlob)           →  tone of writing
  ✅ Clickbait scorer (rule-based)           →  sensational language check
  ✅ XGBoost fusion brain                   →  multi-signal cross-check

Run:
    streamlit run app.py
"""

import streamlit as st
import json
import logging
import os
import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import base64
import html as html_mod
import tempfile
import time

# ─────────────────────────────────────────────────────────
# Phase 5
# ─────────────────────────────────────────────────────────
from src.framing.sentiment import analyze_sentiment
from src.framing.clickbait import clickbait_score

# ─────────────────────────────────────────────────────────
# Phase 6
# ─────────────────────────────────────────────────────────
from src.fusion.infer import predict_sample

# ─────────────────────────────────────────────────────────
# Phase 2 / 7  — RoBERTa  (loaded lazily + cached)
# ─────────────────────────────────────────────────────────
ROBERTA_OK   = False
_tok = _mdl = _DEVICE = explain_text = None
_roberta_err = ""

def _load_roberta():
    """Cached loader — called once after set_page_config."""
    from src.xai.text import (
        tokenizer,
        model,
        DEVICE,
        explain_text,
    )
    return tokenizer, model, DEVICE, explain_text

_ROBERTA_LABELS = ["real", "fake", "uncertain"]

# ─────────────────────────────────────────────────────────
# Phase 7 — XAI helpers
# ─────────────────────────────────────────────────────────
try:
    from src.xai.counterfactuals import run_counterfactuals
    from src.xai.card_builder    import build_evidence_card
    XAI_OK = True
except Exception:
    XAI_OK = False

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR  = PROJECT_ROOT / "artifacts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "artifacts" / "cards").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

# ─────────────────────────────────────────────────────────
# Vision checker (silent helper)
# ─────────────────────────────────────────────────────────
try:
    from src.vision.image_checker import (
        check_image_text       as _check_image_text,
        analyse_image_standalone as _analyse_image_standalone,
    )
    _VISION_CHECKER_AVAILABLE = True
except Exception:
    _VISION_CHECKER_AVAILABLE = False

# ─────────────────────────────────────────────────────────
# Finetuned Qwen2-VL (10% signal weight in hybrid blend)
# ─────────────────────────────────────────────────────────
_qwen_infer_mod = None
try:
    from src.vlm import infer as _qwen_infer_mod
    from src.vlm.infer import (
        predict_consistency as _qwen_consistency,
        predict_standalone  as _qwen_standalone,
    )
    _QWEN_IMPORT_OK = True
except Exception:
    _QWEN_IMPORT_OK = False
    def _qwen_consistency(*a, **kw):
        return {"label": "uncertain", "consistency_score": 0.5,
                "probs": {}, "entropy": 0, "available": False}
    def _qwen_standalone(*a, **kw):
        return {"label": "uncertain", "fake_score": 0.5,
                "probs": {}, "entropy": 0, "available": False}

def _is_qwen_available() -> bool:
    """Check live status (model loads lazily on first call)."""
    if _qwen_infer_mod is None:
        return False
    return getattr(_qwen_infer_mod, "QWEN_AVAILABLE", False)

# ─────────────────────────────────────────────────────────
# RAG — live web search fact-check (Serper + Groq)
# ─────────────────────────────────────────────────────────
try:
    from src.rag.claims   import extract_claims as _extract_claims
    from src.rag.retrieve import retrieve       as _retrieve
    _RAG_AVAILABLE = True
except Exception:
    _RAG_AVAILABLE = False

_RAG_CFG = {
    "claims": {
        "backend"   : "rule_based",
        "max_claims": 2,
        "min_chars" : 20,
        "max_chars" : 280,
        "dedupe"    : True,
    },
    "retrieve": {
        "provider" : "serper",
        "top_k"    : 3,
        "min_chars": 60,
        "serper"   : {
            "api_key_env"   : "SERPER_API_KEY",
            "country"       : "in",
            "num"           : 5,
            "allow_domains" : [],
            "block_domains" : [],
        },
    },
}

def analyse_image(image_path: str, text: str) -> dict:
    if image_path and _VISION_CHECKER_AVAILABLE:
        return _check_image_text(image_path, text, fallback_score=0.5)
    return {
        "consistency_score": 0.5, "verdict": "uncertain",
        "image_summary": "", "text_claim": text,
        "match_analysis": "", "mismatch_reason": "",
        "confidence": 0.0, "available": False,
    }


# ─────────────────────────────────────────────────────────
# Hybrid VLM blending  (90 % Groq + 10 % finetuned Qwen2-VL)
# ─────────────────────────────────────────────────────────
_GROQ_W  = 0.90
_QWEN_W  = 0.10

def blend_vision_signals(groq_result: dict, qwen_result: dict) -> dict:
    """Blend Groq + Qwen2-VL consistency scores for image+text mode."""
    if not qwen_result.get("available"):
        return groq_result
    if not groq_result.get("available"):
        return groq_result          # keep Groq metadata even if it failed

    groq_sc = groq_result["consistency_score"]
    qwen_sc = qwen_result["consistency_score"]
    blended = _GROQ_W * groq_sc + _QWEN_W * qwen_sc

    if   blended > 0.70: verdict = "consistent"
    elif blended < 0.35: verdict = "mismatch"
    elif blended < 0.50: verdict = "unrelated"
    else:                verdict = "uncertain"

    return {
        **groq_result,
        "consistency_score": round(blended, 4),
        "verdict":           verdict,
        # originals preserved for debug view
        "_groq_score":  groq_sc,
        "_qwen_score":  qwen_sc,
        "_qwen_label":  qwen_result.get("label", "unknown"),
        "_qwen_probs":  qwen_result.get("probs", {}),
    }

def blend_standalone_signals(groq_standalone: dict, qwen_result: dict) -> dict:
    """Blend Groq + Qwen2-VL fake scores for image-only mode."""
    if not qwen_result.get("available") or not groq_standalone.get("available"):
        return groq_standalone

    groq_f  = groq_standalone["fake_score"]
    qwen_f  = qwen_result.get("fake_score", 0.5)
    blended = round(max(0.0, min(1.0, _GROQ_W * groq_f + _QWEN_W * qwen_f)), 4)

    return {
        **groq_standalone,
        "fake_score":       blended,
        "_groq_fake_score": groq_f,
        "_qwen_fake_score": qwen_f,
        "_qwen_label":      qwen_result.get("label", "unknown"),
    }


def _get_groq_client():
    """Return a Groq client using the key from .env / environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception:
        return None


_RAG_GROQ_SYSTEM = (
    "You are a precise fact-checking assistant. "
    "Given a news claim and web search results, determine if the evidence "
    "supports or contradicts the claim. Return ONLY valid JSON."
)

def live_rag_check(text: str) -> dict:
    """
    Fact-check `text` against live web results.
    1. Extract atomic claims (rule-based).
    2. Retrieve top Google snippets via Serper API.
    3. Ask Groq (Llama-4-Scout) to judge stance.
    Returns real rag_support / rag_contradict scores plus human-readable summary.
    """
    _FALLBACK = {
        "rag_support"    : 0.33,
        "rag_contradict" : 0.33,
        "verdict"        : "unverifiable",
        "evidence_summary": "",
        "key_sources"    : [],
        "available"      : False,
    }

    if not _RAG_AVAILABLE or not text.strip():
        return _FALLBACK

    try:
        claims = _extract_claims(text, _RAG_CFG)
        if not claims:
            return _FALLBACK

        # Collect snippets for up to 2 claims (keep it fast)
        all_snippets = []
        for claim in claims[:2]:
            snippets = _retrieve(claim, _RAG_CFG)
            for s in snippets[:3]:
                all_snippets.append({"claim": claim, "snippet": s})

        if not all_snippets:
            return _FALLBACK

        # Format evidence block for Groq
        evidence_lines = []
        for i, item in enumerate(all_snippets[:5], 1):
            s = item["snippet"]
            url     = s.get("url", "")
            snippet = s.get("snippet", "")
            title   = s.get("title", "")
            evidence_lines.append(f"[{i}] {title} ({url})\n    {snippet}")
        evidence_block = "\n\n".join(evidence_lines)

        prompt = (
            f'News claim: "{text}"\n\n'
            f"Web search results:\n{evidence_block}\n\n"
            "Analyse whether these results support or contradict the claim.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "support_score": <float 0.0-1.0>,\n'
            '  "contradict_score": <float 0.0-1.0>,\n'
            '  "verdict": "<supported|contradicted|mixed|unverifiable>",\n'
            '  "evidence_summary": "<1-2 sentences: what does the web say>",\n'
            '  "key_sources": ["<domain1>", "<domain2>"]\n'
            "}"
        )

        client = _get_groq_client()
        if not client:
            return _FALLBACK

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": _RAG_GROQ_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=350,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return _FALLBACK

        result = json.loads(json_match.group())
        result["rag_support"]    = float(max(0.0, min(1.0, result.get("support_score",    0.33))))
        result["rag_contradict"] = float(max(0.0, min(1.0, result.get("contradict_score", 0.33))))
        result.setdefault("verdict",          "unverifiable")
        result.setdefault("evidence_summary", "")
        result.setdefault("key_sources",      [])
        result["available"] = True
        return result

    except Exception as ex:
        logging.warning(f"live_rag_check failed: {ex}")
        return _FALLBACK

# ─────────────────────────────────────────────────────────
# ML helpers
# ─────────────────────────────────────────────────────────
def roberta_predict(text: str) -> dict:
    if not ROBERTA_OK:
        return {"real": 0.33, "fake": 0.33, "uncertain": 0.33}
    enc = _tok(text, return_tensors="pt", truncation=True, max_length=256).to(_DEVICE)
    with torch.no_grad():
        logits = _mdl(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    return {label: round(probs[i], 4) for i, label in enumerate(_ROBERTA_LABELS)}

def roberta_verdict(probs: dict) -> tuple:
    f, r = probs["fake"], probs["real"]
    if f > 0.60:               return "fake",      f,    False
    elif r > 0.60:             return "real",      r,    False
    elif f > 0.40 and f >= r:  return "fake",      f,    True
    elif r > 0.40 and r > f:   return "real",      r,    True
    else:
        best = max(probs, key=probs.get)
        return "uncertain", probs[best], False

def build_features(text, image_path, roberta_probs, vision_result, rag_result=None):
    sent     = analyze_sentiment(text)
    click    = clickbait_score(text)
    vlm_cons = vision_result["consistency_score"]
    vlm_entr = float(1.0 - abs(vlm_cons - 0.5) * 2.0)
    rag_sup  = rag_result["rag_support"]    if rag_result and rag_result.get("available") else 0.33
    rag_con  = rag_result["rag_contradict"] if rag_result and rag_result.get("available") else 0.33
    return {
        "text_cls_score"  : roberta_probs["fake"],
        "rag_support"     : rag_sup,
        "rag_contradict"  : rag_con,
        "vlm_consistency" : vlm_cons,
        "vlm_entropy"     : vlm_entr,
        "ocr_match"       : 0.0,
        "sentiment"       : sent["sentiment"],
        "subjectivity"    : sent["subjectivity"],
        "clickbait"       : click["clickbait"],
        "text_length_norm": min(len(text) / 300.0, 1.0),
        "has_image"       : 1 if image_path else 0,
    }

def combined_verdict(r_probs, rag_result, standalone_result, image_mode):
    """
    Produce the final verdict by weighing ALL available signals.
    Web evidence is treated as the strongest external signal — if credible
    sources say it's real, that overrides a noisy text-classifier.
    """
    # ── 1. Primary fake probability ──
    if image_mode == "image_only" and standalone_result.get("available"):
        vlm_fake = standalone_result["fake_score"]
        rob_fake = r_probs["fake"]
        fake_prob = vlm_fake * 0.65 + rob_fake * 0.35
    else:
        fake_prob = r_probs["fake"]

    # ── 2. Web evidence modifier (strongest signal) ──
    if rag_result.get("available"):
        web_verdict = rag_result.get("verdict", "unverifiable")
        web_sup = rag_result.get("rag_support", 0.33)
        web_con = rag_result.get("rag_contradict", 0.33)

        if web_verdict == "supported":
            # ANY "supported" verdict dampens fake — the LLM already judged stance
            dampen = 0.25 + 0.20 * (1.0 - web_sup)   # range 0.25–0.45
            fake_prob = fake_prob * dampen
        elif web_verdict == "contradicted":
            boost = 0.35 * max(web_con, 0.5)
            fake_prob = fake_prob + (1.0 - fake_prob) * boost
        elif web_verdict == "mixed":
            # Slight dampen — web found something, so pure-fake is less likely
            fake_prob = fake_prob * 0.85

    fake_prob = max(0.0, min(1.0, fake_prob))
    real_prob = 1.0 - fake_prob

    # ── 3. Threshold-based verdict ──
    if fake_prob > 0.60:
        return "fake",      fake_prob,  False
    elif real_prob > 0.60:
        return "real",      real_prob,  False
    elif fake_prob > 0.40 and fake_prob >= real_prob:
        return "fake",      fake_prob,  True
    elif real_prob > 0.40:
        return "real",      real_prob,  True
    else:
        return "uncertain", max(fake_prob, real_prob), False


def _cf_plain_english(original_label, cf_result):
    labels = {
        "remove_sensational"  : "If the writing style were calm and neutral",
        "remove_contradiction": "If web evidence did not contradict the claims",
        "remove_image_signal" : "If we ignored the image",
    }
    lines = []
    for key, pred in cf_result.get("counterfactuals", {}).items():
        new_label = pred["label"]
        scenario  = labels.get(key, key)
        flipped   = new_label != original_label
        lines.append((scenario, new_label, flipped))
    return lines


# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load RoBERTa (cached — only loads once across reruns) ─────
@st.cache_resource(show_spinner="Loading RoBERTa model…")
def _cached_load_roberta():
    return _load_roberta()

try:
    _tok, _mdl, _DEVICE, explain_text = _cached_load_roberta()
    ROBERTA_OK = True
except Exception as _e:
    ROBERTA_OK   = False
    _roberta_err = str(_e)

# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base / Light theme ───────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f0f4f8;
    color: #1e293b;
}

.main { background-color: #f0f4f8; }

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
    background-color: #f0f4f8;
}

/* ── Header ──────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 60%, #2563eb 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    border: none;
    text-align: center;
    box-shadow: 0 8px 32px rgba(37,99,235,0.25);
}
.app-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -1px;
}
.app-header p {
    color: rgba(255,255,255,0.82);
    font-size: 1.05rem;
    margin: 0;
}
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #ffffff;
    margin-bottom: 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Verdict banner ──────────────────────────── */
.verdict-wrap {
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.verdict-fake {
    background: linear-gradient(135deg, #b91c1c 0%, #dc2626 50%, #ef4444 100%);
    border: none;
    box-shadow: 0 8px 32px rgba(220,38,38,0.35);
}
.verdict-real {
    background: linear-gradient(135deg, #15803d 0%, #16a34a 50%, #22c55e 100%);
    border: none;
    box-shadow: 0 8px 32px rgba(22,163,74,0.35);
}
.verdict-uncertain {
    background: linear-gradient(135deg, #b45309 0%, #d97706 50%, #f59e0b 100%);
    border: none;
    box-shadow: 0 8px 32px rgba(217,119,6,0.35);
}
.verdict-label {
    font-size: 2.4rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.verdict-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.9);
    margin-bottom: 1.2rem;
    line-height: 1.5;
}
.verdict-conf {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.22);
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 0.95rem;
    font-weight: 600;
    color: #fff;
}
.prob-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.prob-pill {
    background: rgba(0,0,0,0.18);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.85rem;
    color: #ffffff;
    font-weight: 500;
}
.prob-pill span {
    font-weight: 800;
    font-size: 1rem;
}

/* ── Section header ──────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.3px;
}
.section-divider {
    height: 1px;
    background: linear-gradient(to right, rgba(0,0,0,0.1), transparent);
    margin: 2rem 0;
}

/* ── Signal cards ────────────────────────────── */
.sig-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    height: 100%;
    transition: box-shadow 0.2s, border-color 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.sig-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    border-color: rgba(0,0,0,0.14);
}
.sig-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}
.sig-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.8rem;
    line-height: 1.2;
}
.sig-value.red   { color: #dc2626; }
.sig-value.green { color: #16a34a; }
.sig-value.amber { color: #d97706; }
.sig-value.blue  { color: #2563eb; }

/* ── Progress bar ────────────────────────────── */
.pbar-wrap {
    background: rgba(0,0,0,0.07);
    border-radius: 100px;
    height: 7px;
    margin-bottom: 0.4rem;
    overflow: hidden;
}
.pbar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}
.pbar-red    { background: linear-gradient(to right, #dc2626, #f87171); }
.pbar-green  { background: linear-gradient(to right, #15803d, #4ade80); }
.pbar-amber  { background: linear-gradient(to right, #b45309, #fbbf24); }
.pbar-blue   { background: linear-gradient(to right, #1d4ed8, #60a5fa); }
.pbar-label  {
    font-size: 0.76rem;
    color: #94a3b8;
    text-align: right;
}

/* ── Token pills ─────────────────────────────── */
.token-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 0.5rem;
}
.token-pill {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 14px;
    border-radius: 10px;
    font-size: 0.88rem;
    font-weight: 600;
    gap: 3px;
    min-width: 70px;
}
.token-fake {
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
    color: #b91c1c;
}
.token-real {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    color: #15803d;
}
.token-score {
    font-size: 0.72rem;
    font-weight: 500;
    opacity: 0.7;
}

/* ── Cause card ──────────────────────────────── */
.cause-card {
    background: #ffffff;
    border-left: 4px solid;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin-bottom: 0.7rem;
    font-size: 0.95rem;
    color: #334155;
    line-height: 1.5;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.cause-red    { border-color: #dc2626; background: #fef2f2; color: #7f1d1d; }
.cause-amber  { border-color: #d97706; background: #fffbeb; color: #78350f; }
.cause-blue   { border-color: #2563eb; background: #eff6ff; color: #1e3a8a; }
.cause-purple { border-color: #7c3aed; background: #f5f3ff; color: #4c1d95; }
.cause-gray   { border-color: #94a3b8; background: #f8fafc; color: #334155; }

/* ── Counterfactual cards ────────────────────── */
.cf-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.cf-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 2px; }
.cf-body  { flex: 1; }
.cf-scenario { font-size: 0.88rem; color: #64748b; margin-bottom: 4px; }
.cf-result   { font-size: 0.95rem; font-weight: 600; color: #1e293b; }
.cf-flipped  { color: #d97706; }
.cf-same     { color: #94a3b8; }

/* ── Web evidence card ───────────────────────── */
.web-ev-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.web-ev-verdict {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 100px;
    padding: 5px 16px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.web-ev-supported   { background:#f0fdf4; color:#15803d; border:1.5px solid #86efac; }
.web-ev-contradicted{ background:#fef2f2; color:#b91c1c; border:1.5px solid #fca5a5; }
.web-ev-mixed       { background:#fffbeb; color:#b45309; border:1.5px solid #fde68a; }
.web-ev-unverifiable{ background:#f8fafc; color:#64748b; border:1.5px solid #e2e8f0; }
.web-ev-summary {
    font-size: 0.95rem;
    color: #334155;
    line-height: 1.6;
    margin-bottom: 0.8rem;
}
.web-ev-sources {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.web-ev-source-pill {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #475569;
    font-weight: 500;
}

/* ── Fake signal pill ────────────────────────── */
.signal-pill {
    display: inline-block;
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 0.82rem;
    color: #b91c1c;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}

/* ── Download button ─────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100%;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.9 !important; }

/* ── Analyse button ──────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    box-shadow: 0 4px 18px rgba(37,99,235,0.4) !important;
    transition: opacity 0.2s, transform 0.1s !important;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(0,0,0,0.08) !important;
}
section[data-testid="stSidebar"] * { color: #1e293b !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #334155 !important; }

/* ── Streamlit native text override ──────────── */
p, li, span, label, div { color: #1e293b; }
.stTextArea textarea {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
    font-size: 0.97rem !important;
}
.stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
.stFileUploader {
    background: #ffffff !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 10px !important;
}

/* ── Spinner ─────────────────────────────────── */
.stSpinner > div { color: #2563eb !important; }

/* ── Streamlit info / warning / error ────────── */
div[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Hide Streamlit branding ─────────────────── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Feature capability cards ─────────────────── */
.feature-cards-row {
    display: flex;
    gap: 12px;
    margin: 1.2rem 0 1.5rem;
    flex-wrap: wrap;
}
.feature-card {
    flex: 1;
    min-width: 180px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.feature-card:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    border-color: #cbd5e1;
}
.feature-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.feature-info { flex: 1; }
.feature-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 3px;
}
.feature-desc {
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.4;
}

/* ── Analysis Results card ────────────────────── */
.results-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    overflow: hidden;
    margin: 1.5rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.results-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #e2e8f0;
    background: #f8fafc;
}
.results-body {
    display: flex;
    gap: 2rem;
    padding: 1.8rem 1.8rem 1.5rem;
    align-items: flex-start;
}
.results-left {
    flex: 1;
    min-width: 0;
}
.results-right {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
}

/* ── Warning banner ───────────────────────────── */
.warning-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0 1.5rem;
}

/* ── App footer ───────────────────────────────── */
.app-footer {
    text-align: center;
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 2.5rem;
    padding: 1.2rem 0;
    border-top: 1px solid #e2e8f0;
}

/* ── File uploader styling ────────────────────── */
.stFileUploader > div {
    background: #ffffff !important;
}
.stFileUploader label {
    font-weight: 600 !important;
    color: #1e293b !important;
}

/* ── Feature cards equal height ──────────────── */
.feature-cards-row > .feature-card {
    min-height: 80px;
}

/* ── Input labels ────────────────────────────── */
.stTextArea label, .stFileUploader label {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}

/* ── Responsive: stack results on mobile ──────── */
@media (max-width: 768px) {
    .results-body { flex-direction: column; }
    .feature-cards-row { flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    show_debug = st.checkbox("Show detailed analysis")
    st.markdown("---")
    st.markdown("### 🔧 System Status")

    def _status(ok, label):
        icon = "🟢" if ok else "🔴"
        return f"{icon} {label}"

    st.markdown(_status(ROBERTA_OK,              "Text AI (RoBERTa)"))
    st.markdown(_status(True,                    "Sentiment & Clickbait"))
    st.markdown(_status(_VISION_CHECKER_AVAILABLE, "Image Analysis (Groq)"))
    st.markdown(_status(_QWEN_IMPORT_OK,           "Finetuned VLM (Qwen2-VL)"))
    st.markdown(_status(True,                    "XGBoost Fusion"))
    st.markdown(_status(XAI_OK,                  "XAI / Evidence Card"))

    if not _QWEN_IMPORT_OK:
        st.caption("Qwen2-VL module not available.")
    else:
        st.caption("Qwen2-VL: loads on first image (10 % weight).")

    if not ROBERTA_OK:
        st.error(f"RoBERTa error: {_roberta_err}")

# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="header-badge">🛡️ AI-POWERED</div>
    <h1>Fake News Detector</h1>
    <p>Paste any news headline, social media post, or claim. Our AI analyses<br>
    text, image, sentiment, and writing style to detect misinformation instantly.</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  INPUT SECTION
# ═══════════════════════════════════════════════════════════════
col_img, col_txt = st.columns([1, 2], gap="medium")

with col_img:
    uploaded = st.file_uploader(
        "🖼️ Attach Image (Optional)",
        type=["jpg", "jpeg", "png", "9dg"],
        help="Upload an image that accompanies the post to check if it matches the text.",
    )

with col_txt:
    post_text = st.text_area(
        "📝 Enter News Headline or Post Text",
        height=148,
        placeholder="Enter a news headline, social media post, or claim to analyze...",
    )

go = st.button("🔍  Run Analysis", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  FEATURE CAPABILITY CARDS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="feature-cards-row">
    <div class="feature-card">
        <div class="feature-icon">📄</div>
        <div class="feature-info">
            <div class="feature-title">Text Analysis</div>
            <div class="feature-desc">Detect suspicious claims and misleading content.</div>
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🖼️</div>
        <div class="feature-info">
            <div class="feature-title">Image Verification</div>
            <div class="feature-desc">Checks uploaded images for manipulation cues.</div>
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💛</div>
        <div class="feature-info">
            <div class="feature-title">Sentiment Detection</div>
            <div class="feature-desc">Measures emotional tone used in the content.</div>
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">✍️</div>
        <div class="feature-info">
            <div class="feature-title">Writing Style Analysis</div>
            <div class="feature-desc">Analyses framing, clickbait, and subjectivity patterns.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════
if go:
    if not post_text.strip() and not uploaded:
        st.error("⚠️  Please enter some text or upload an image before analysing.")
        st.stop()

    t0 = time.time()

    # ── Save uploaded image ────────────────────────────────────
    image_path = None
    if uploaded:
        tmp_path = Path(tempfile.gettempdir()) / uploaded.name
        tmp_path.write_bytes(uploaded.read())
        image_path = str(tmp_path)

    # ── Determine analysis mode ────────────────────────────────
    image_mode = (
        "image_only"  if image_path and not post_text.strip() else
        "image_text"  if image_path and post_text.strip()     else
        "text_only"
    )

    prog = st.progress(0, text="Starting analysis…")

    # ── IMAGE-ONLY: extract text + fake signals via Groq VLM ──
    standalone_result = {}
    qwen_result = {}
    if image_mode == "image_only":
        prog.progress(10, text="Analysing image with AI vision model…")
        if _VISION_CHECKER_AVAILABLE:
            standalone_result = _analyse_image_standalone(image_path)

        # Run finetuned Qwen2-VL (10 % weight)
        try:
            qwen_result = _qwen_standalone(image_path)
        except Exception as _qex:
            logging.warning(f"Qwen standalone failed: {_qex}")
            qwen_result = {"available": False}

        # Blend standalone results (90 % Groq + 10 % Qwen)
        if standalone_result and qwen_result.get("available"):
            standalone_result = blend_standalone_signals(standalone_result, qwen_result)

        # Build synthetic text from image: prefer extracted text, fallback to description
        extracted_text = standalone_result.get("extracted_text", "").strip()
        image_desc     = standalone_result.get("image_description", "").strip()
        post_text      = extracted_text if len(extracted_text) > 20 else image_desc
        if extracted_text:
            st.info(f"📄 **Text extracted from image:** {extracted_text}")
        elif image_desc:
            st.info(f"🖼️ **AI image description:** {image_desc}")

    prog.progress(20, text="Running text AI model…")
    r_probs = roberta_predict(post_text) if post_text.strip() else {"real": 0.33, "fake": 0.33, "uncertain": 0.33}

    prog.progress(38, text="Analysing framing signals…")
    sent_out  = analyze_sentiment(post_text) if post_text.strip() else {"sentiment": 0.5, "subjectivity": 0.0}
    click_out = clickbait_score(post_text)   if post_text.strip() else {"clickbait": 0.0}

    prog.progress(52, text="Checking image…")
    if image_mode == "image_only" and standalone_result.get("available"):
        # Map standalone fake_score → vision_result shape so fusion stays consistent
        fs = standalone_result["fake_score"]
        vision_result = {
            "consistency_score": round(1.0 - fs, 4),
            "verdict"          : "mismatch" if fs > 0.55 else "consistent" if fs < 0.35 else "uncertain",
            "image_summary"    : standalone_result.get("image_description", ""),
            "text_claim"       : post_text,
            "match_analysis"   : "",
            "mismatch_reason"  : standalone_result.get("manipulation_reason", ""),
            "confidence"       : standalone_result.get("confidence", 0.0),
            "available"        : True,
        }
    else:
        vision_result = analyse_image(image_path, post_text)

        # Run finetuned Qwen2-VL for image+text consistency (10 % weight)
        if image_path:
            try:
                qwen_result = _qwen_consistency(image_path, post_text)
            except Exception as _qex:
                logging.warning(f"Qwen consistency failed: {_qex}")
                qwen_result = {"available": False}

            # Blend (90 % Groq + 10 % Qwen)
            if qwen_result.get("available"):
                vision_result = blend_vision_signals(vision_result, qwen_result)

    prog.progress(65, text="Searching web for fact-check…")
    rag_result = live_rag_check(post_text) if post_text.strip() else {
        "rag_support": 0.33, "rag_contradict": 0.33,
        "verdict": "unverifiable", "evidence_summary": "",
        "key_sources": [], "available": False,
    }

    # ── Combined verdict: weighs RoBERTa + VLM + web evidence ────
    r_label, r_conf, r_leaning = combined_verdict(
        r_probs, rag_result, standalone_result, image_mode,
    )

    prog.progress(78, text="Running fusion model…")
    features = build_features(post_text, image_path, r_probs, vision_result, rag_result)
    xgb_out  = predict_sample(features)

    prog.progress(86, text="Computing word attributions…")
    token_pairs = []
    if ROBERTA_OK:
        try:
            xai  = explain_text(post_text, target_label="fake")
            skip = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"}
            def _is_real_word(t):
                if t in skip: return False
                return any("a" <= c.lower() <= "z" for c in t)
            token_pairs = [
                (t, s) for t, s in
                sorted(zip(xai["tokens"], xai["attributions"]),
                       key=lambda x: abs(x[1]), reverse=True)
                if _is_real_word(t)
            ][:8]
        except Exception as ex:
            logging.warning(f"XAI failed: {ex}")

    prog.progress(94, text="Building evidence card…")
    card, cf_lines = {}, []
    if XAI_OK:
        try:
            card = build_evidence_card({
                "id": f"live_{int(time.time())}",
                "image_path": image_path,
                "post_text" : post_text,
                "features"  : features,
            })
            cf_lines = _cf_plain_english(r_label, card.get("counterfactuals", {}))
        except Exception as ex:
            logging.warning(f"Card builder failed: {ex}")

    prog.progress(100, text="Done!")
    time.sleep(0.3)
    prog.empty()

    # ── Persist all results in session_state so they survive reruns ──
    st.session_state["analysis_done"] = True
    st.session_state["r_label"] = r_label
    st.session_state["r_conf"] = r_conf
    st.session_state["r_leaning"] = r_leaning
    st.session_state["r_probs"] = r_probs
    st.session_state["sent_out"] = sent_out
    st.session_state["click_out"] = click_out
    st.session_state["vision_result"] = vision_result
    st.session_state["rag_result"] = rag_result
    st.session_state["standalone_result"] = standalone_result
    st.session_state["qwen_result"] = qwen_result
    st.session_state["features"] = features
    st.session_state["xgb_out"] = xgb_out
    st.session_state["token_pairs"] = token_pairs
    st.session_state["card"] = card
    st.session_state["cf_lines"] = cf_lines
    st.session_state["image_path"] = image_path
    st.session_state["post_text_used"] = post_text
    st.session_state["image_mode"] = image_mode
    st.session_state["elapsed"] = time.time() - t0

# ═══════════════════════════════════════════════════════════════
#  DISPLAY RESULTS (reads from session_state so they persist)
# ═══════════════════════════════════════════════════════════════
if st.session_state.get("analysis_done"):
    # ── Restore variables from session_state ──────────────
    r_label = st.session_state["r_label"]
    r_conf = st.session_state["r_conf"]
    r_leaning = st.session_state["r_leaning"]
    r_probs = st.session_state["r_probs"]
    sent_out = st.session_state["sent_out"]
    click_out = st.session_state["click_out"]
    vision_result = st.session_state["vision_result"]
    rag_result = st.session_state["rag_result"]
    standalone_result = st.session_state["standalone_result"]
    qwen_result = st.session_state["qwen_result"]
    features = st.session_state["features"]
    xgb_out = st.session_state["xgb_out"]
    token_pairs = st.session_state["token_pairs"]
    card = st.session_state["card"]
    cf_lines = st.session_state["cf_lines"]
    image_path = st.session_state["image_path"]
    post_text = st.session_state["post_text_used"]
    image_mode = st.session_state["image_mode"]
    elapsed = st.session_state["elapsed"]

    # ── Verdict label & icon ──────────────────────────────
    _verdict_icon = {
        "fake"     : "❌",
        "real"     : "✅",
        "uncertain": "⚠️",
    }.get(r_label, "⚠️")

    _verdict_display = {
        ("fake",  False): "Fake News",
        ("fake",  True ): "Likely Fake News",
        ("real",  False): "Real News",
        ("real",  True ): "Likely Real News",
        ("uncertain", False): "Uncertain",
    }.get((r_label, r_leaning), "Uncertain")

    _verdict_color = {
        "fake"     : "#dc2626",
        "real"     : "#16a34a",
        "uncertain": "#d97706",
    }.get(r_label, "#d97706")

    _result_header_color = {
        "fake"     : "#dc2626",
        "real"     : "#16a34a",
        "uncertain": "#d97706",
    }.get(r_label, "#d97706")

    # ── Build "Why this was flagged" bullet points ────────
    _why_bullets = []
    if r_probs["fake"] > 0.45:
        _why_bullets.append("The headline uses emotionally charged language designed to create fear or urgency.")
    if image_path and vision_result.get("available"):
        _vv = vision_result.get("verdict", "")
        if _vv in ("mismatch", "unrelated"):
            _why_bullets.append("The attached image shows signs of manipulation or misleading visual context.")
        elif _vv == "consistent" and r_label == "fake":
            _why_bullets.append("Despite a matching image, the text contains misleading claims.")
    if click_out.get("clickbait", 0) > 0.3:
        _why_bullets.append("Several keywords match patterns commonly found in misinformation content.")
    if sent_out.get("subjectivity", 0) > 0.5:
        _why_bullets.append("The writing style uses exaggeration and lacks trustworthy supporting evidence.")
    if rag_result.get("available"):
        _wv = rag_result.get("verdict", "")
        if _wv == "contradicted":
            _why_bullets.append("Credible web sources contradict the claims made in this content.")
        elif _wv == "supported" and r_label == "real":
            _why_bullets.append("Credible web sources support the claims made in this content.")
        elif _wv == "mixed":
            _why_bullets.append("Web sources show mixed signals about the claims in this content.")
    if r_label == "real" and not _why_bullets:
        _why_bullets.append("The language matches credible factual reporting patterns.")
        _why_bullets.append("No significant red flags were detected in the content.")
    if not _why_bullets:
        _why_bullets.append("The AI detected subtle patterns that suggest this content needs further verification.")

    _why_html = "".join(
        f'<li style="margin-bottom:8px;color:#334155;font-size:0.92rem;line-height:1.5;">{b}</li>'
        for b in _why_bullets
    )

    # ── Sentiment display ─────────────────────────────────
    _sent_pct = int(sent_out.get("sentiment", 0.5) * 100)
    _sent_label_text = "Positive" if _sent_pct > 55 else "Negative" if _sent_pct < 45 else "Neutral"

    # ── Verdict stamp values ─────────────────────────────
    if r_label == "fake":
        _stamp_text = "FAKE"
        _stamp_color = "rgba(220,38,38,0.85)"
        _stamp_border = "4px solid rgba(220,38,38,0.9)"
    elif r_label == "real":
        _stamp_text = "REAL"
        _stamp_color = "rgba(22,163,74,0.85)"
        _stamp_border = "4px solid rgba(22,163,74,0.9)"
    else:
        _stamp_text = "UNVERIFIED"
        _stamp_color = "rgba(217,119,6,0.85)"
        _stamp_border = "4px solid rgba(217,119,6,0.9)"

    # ── Image section for right column ────────────────────
    _image_review_html = ""
    if image_path:
        with open(image_path, "rb") as _imgf:
            _img_b64 = base64.b64encode(_imgf.read()).decode()
        _img_ext = image_path.rsplit(".", 1)[-1].lower()
        _img_mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(_img_ext, "jpeg")

        _image_review_html = f"""
            <div style="position:relative;margin-bottom:1rem;overflow:hidden;border-radius:12px;
                        box-shadow:0 2px 12px rgba(0,0,0,0.12);">
                <img src="data:image/{_img_mime};base64,{_img_b64}"
                     style="width:100%;display:block;border-radius:12px;" />
                <div style="position:absolute;top:0;left:0;right:0;bottom:0;display:flex;
                            align-items:center;justify-content:center;
                            background:rgba(0,0,0,0.15);border-radius:12px;">
                    <span style="font-size:3rem;font-weight:900;color:{_stamp_color};
                                 letter-spacing:6px;transform:rotate(-18deg);
                                 text-shadow:2px 2px 8px rgba(0,0,0,0.4);
                                 border:{_stamp_border};border-radius:8px;
                                 padding:8px 28px;background:rgba(255,255,255,0.15);
                                 backdrop-filter:blur(2px);">
                        {_stamp_text}
                    </span>
                </div>
            </div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.6rem;">
                <span style="font-size:1.1rem;">🖼️</span>
                <span style="font-size:0.95rem;font-weight:700;color:#1e293b;">Attached Image Review</span>
            </div>
            <div style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px;padding:0.9rem 1.1rem;
                        display:flex;align-items:flex-start;gap:10px;">
                <span style="font-size:1.1rem;flex-shrink:0;">🔍</span>
                <span style="font-size:0.85rem;color:#475569;line-height:1.5;">
                    The uploaded image was checked for visual inconsistencies, misleading overlays, and context mismatch.
                </span>
            </div>
        """

    # ── Text preview for right column (when no image) ────
    if not image_path:
        _escaped_text = html_mod.escape(post_text[:500])
        _text_preview_html = f'<div style="position:relative;margin-bottom:1rem;overflow:hidden;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.12);min-height:200px;background:#f8fafc;"><div style="padding:1.2rem;font-size:0.85rem;color:#94a3b8;line-height:1.6;max-height:220px;overflow:hidden;word-break:break-word;">{_escaped_text}</div><div style="position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,0.55);border-radius:12px;backdrop-filter:blur(1px);"><span style="font-size:3rem;font-weight:900;color:{_stamp_color};letter-spacing:6px;transform:rotate(-18deg);text-shadow:2px 2px 8px rgba(0,0,0,0.4);border:{_stamp_border};border-radius:8px;padding:8px 28px;background:rgba(255,255,255,0.15);backdrop-filter:blur(2px);">{_stamp_text}</span></div></div>'
    else:
        _text_preview_html = ""

    _right_col_html = _image_review_html.strip() if image_path else _text_preview_html

    # ── Build the full Analysis Results card ──────────────
    _why_bullets_html = "".join(
        f'<li style="margin-bottom:8px;color:#334155;font-size:0.92rem;line-height:1.5;position:relative;padding-left:14px;"><span style="position:absolute;left:0;color:{_result_header_color};font-size:0.7rem;top:5px;">&#9679;</span>{b}</li>'
        for b in _why_bullets
    )

    st.markdown(f"""<div class="results-card">
<div class="results-header">
<span style="font-size:1.1rem;">⚠️</span>
<span style="font-size:1.1rem;font-weight:700;color:#1e293b;">Analysis Results</span>
</div>
<div class="results-body">
<div class="results-left">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
<span style="font-size:1.5rem;">{_verdict_icon}</span>
<span style="font-size:1.6rem;font-weight:800;color:{_verdict_color};">{_verdict_display}</span>
</div>
<div style="margin-bottom:0.6rem;">
<span style="font-size:0.92rem;color:#64748b;">Confidence Score</span>
<span style="font-size:1.5rem;font-weight:800;color:#1e293b;margin-left:8px;">{r_conf*100:.0f}%</span>
</div>
<div style="height:1px;background:#e2e8f0;margin:0.6rem 0;"></div>
<div style="margin-bottom:1.2rem;">
<span style="font-size:0.92rem;color:#64748b;">Sentiment</span>
<span style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-left:8px;">{_sent_pct}%</span>
<span style="font-size:0.88rem;color:#64748b;margin-left:4px;">{_sent_label_text}</span>
<span style="font-size:0.75rem;color:#94a3b8;margin-left:4px;cursor:help;" title="Sentiment analysis measures the emotional tone of the text">&#9432;</span>
</div>
<div style="font-size:1rem;font-weight:700;color:#1e293b;margin-bottom:0.8rem;">
Why this was flagged
</div>
<ul style="margin:0;padding-left:1.2rem;list-style:none;">
{_why_bullets_html}
</ul>
</div>
<div class="results-right">
{_right_col_html}
</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── Warning banner ────────────────────────────────────
    if r_label == "fake":
        _warn_text = "This content is likely <strong>fake</strong> and may <strong>spread misinformation</strong>. Verify it with trusted sources before sharing."
        _warn_bg = "linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%)"
        _warn_border = "#fca5a5"
        _warn_icon_bg = "#dc2626"
    elif r_label == "real":
        _warn_text = "This content appears to be <strong>credible</strong>. However, always cross-check with multiple trusted sources."
        _warn_bg = "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)"
        _warn_border = "#86efac"
        _warn_icon_bg = "#16a34a"
    else:
        _warn_text = "This content could not be conclusively verified. <strong>Exercise caution</strong> and verify with trusted sources before sharing."
        _warn_bg = "linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)"
        _warn_border = "#fde68a"
        _warn_icon_bg = "#d97706"

    st.markdown(f"""
    <div class="warning-banner" style="background:{_warn_bg};border:1px solid {_warn_border};">
        <div style="background:{_warn_icon_bg};border-radius:50%;width:28px;height:28px;display:flex;
                    align-items:center;justify-content:center;flex-shrink:0;">
            <span style="color:#fff;font-size:0.85rem;font-weight:800;">!</span>
        </div>
        <span style="font-size:0.92rem;color:#1e293b;line-height:1.5;">{_warn_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    #  WEB FACT-CHECK (always visible — most useful signal)
    # ═══════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🌐 Web Fact-Check</div>', unsafe_allow_html=True)

    if rag_result.get("available"):
        _rv = rag_result["verdict"]
        _rv_map = {
            "supported"    : ("web-ev-supported",    "✅ Supported by web sources"),
            "contradicted" : ("web-ev-contradicted",  "🚨 Contradicted by web sources"),
            "mixed"        : ("web-ev-mixed",         "⚠️ Mixed signals from web"),
            "unverifiable" : ("web-ev-unverifiable",  "🔍 Could not verify online"),
        }
        _rv_css, _rv_label = _rv_map.get(_rv, ("web-ev-unverifiable", "🔍 Could not verify"))

        sources_html = "".join(
            f'<span class="web-ev-source-pill">🔗 {src}</span>'
            for src in rag_result.get("key_sources", [])
        )
        st.markdown(f"""
        <div class="web-ev-card">
            <div class="web-ev-verdict {_rv_css}">{_rv_label}</div>
            <div class="web-ev-summary">{rag_result.get("evidence_summary", "")}</div>
            <div class="web-ev-sources">{sources_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="web-ev-card">
            <div class="web-ev-verdict web-ev-unverifiable">🔍 Web check unavailable</div>
            <div class="web-ev-summary" style="color:#94a3b8">
                Live fact-check requires text input and Serper / Groq API access.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Download evidence card ─────────────────────────
    if card:
        dl_col, _ = st.columns([1, 2])
        with dl_col:
            st.download_button(
                label="📥  Download Evidence Card (JSON)",
                data=json.dumps(card, indent=2),
                file_name="evidence_card.json",
                mime="application/json",
            )

    # ═══════════════════════════════════════════════════════
    #  EXPLAINABILITY AI  (always visible)
    # ═══════════════════════════════════════════════════════
    _has_xai = token_pairs or (card and card.get("root_causes")) or cf_lines
    if _has_xai:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
                    border-radius:16px;padding:1.8rem 2rem 0.6rem;margin:2rem 0 1.5rem;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.3rem;">
                <span style="font-size:1.6rem;">🧠</span>
                <span style="font-size:1.25rem;font-weight:800;color:#fff;letter-spacing:-0.5px;">
                    Explainable AI — Why This Verdict?
                </span>
            </div>
            <p style="color:rgba(255,255,255,0.6);font-size:0.85rem;margin:0 0 1.2rem 0;">
                Our AI doesn't just give a verdict — it shows its reasoning so you can decide whether to trust it.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── 1. Word Influence (Integrated Gradients) ──────
        if token_pairs:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;margin:1rem 0 0.6rem;">
                <span style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:8px;
                             padding:4px 10px;font-size:0.8rem;font-weight:700;color:#1d4ed8;">
                    STEP 1</span>
                <span style="font-size:1rem;font-weight:700;color:#1e293b;">
                    🔤 Word Influence <span style="font-weight:400;color:#64748b;font-size:0.85rem;">
                    — Integrated Gradients Attribution</span>
                </span>
            </div>
            <div style="background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:14px;
                        padding:1.2rem 1.4rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-bottom:0.5rem;">
                <p style="font-size:0.82rem;color:#64748b;margin:0 0 0.8rem 0;line-height:1.5;">
                    Each word is scored by how much it pushed the AI toward <strong style="color:#dc2626;">fake</strong>
                    (red, positive) or <strong style="color:#16a34a;">real</strong> (green, negative).
                    The algorithm gradually "fades in" each word and measures the model's reaction.
                </p>
            """, unsafe_allow_html=True)

            pills_html = ""
            for tok, score in token_pairs:
                if score > 0:
                    intensity = min(255, int(80 + abs(score) * 175))
                    pills_html += (
                        f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
                        f'padding:8px 14px;border-radius:10px;font-size:0.88rem;font-weight:600;'
                        f'gap:3px;min-width:70px;background:rgba(239,68,68,{abs(score)*0.18 + 0.06:.2f});'
                        f'border:1.5px solid rgba(239,68,68,0.3);color:#b91c1c;">'
                        f'{tok}'
                        f'<span style="font-size:0.72rem;font-weight:500;opacity:0.7;">⬆ {score:+.3f}</span>'
                        f'</span> '
                    )
                else:
                    pills_html += (
                        f'<span style="display:inline-flex;flex-direction:column;align-items:center;'
                        f'padding:8px 14px;border-radius:10px;font-size:0.88rem;font-weight:600;'
                        f'gap:3px;min-width:70px;background:rgba(34,197,94,{abs(score)*0.18 + 0.06:.2f});'
                        f'border:1.5px solid rgba(34,197,94,0.3);color:#15803d;">'
                        f'{tok}'
                        f'<span style="font-size:0.72rem;font-weight:500;opacity:0.7;">{score:+.3f}</span>'
                        f'</span> '
                    )
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:10px;">{pills_html}</div></div>',
                unsafe_allow_html=True,
            )

        # ── 2. Root Causes ─────────────────────────────────
        root_causes = card.get("root_causes", []) if card else []
        if root_causes:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;margin:1.5rem 0 0.6rem;">
                <span style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:8px;
                             padding:4px 10px;font-size:0.8rem;font-weight:700;color:#1d4ed8;">
                    STEP 2</span>
                <span style="font-size:1rem;font-weight:700;color:#1e293b;">
                    🔍 Root Causes <span style="font-weight:400;color:#64748b;font-size:0.85rem;">
                    — Why was this flagged?</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            _rc_icons = {
                "sensational_framing"    : ("📣", "Sensational or clickbait writing style detected", "cause-red"),
                "image_text_mismatch"    : ("🖼️", "The image does not match the accompanying text", "cause-red"),
                "external_contradiction" : ("🌐", "Credible web sources contradict this claim", "cause-red"),
                "high_subjectivity"      : ("✍️", "The language is highly opinion-based, not factual", "cause-amber"),
                "no_strong_signal"       : ("🤔", "No single strong red flag — verdict based on subtle patterns", "cause-gray"),
            }
            for rc in root_causes:
                icon, desc, css = _rc_icons.get(rc, ("❓", rc, "cause-gray"))
                st.markdown(f"""
                <div class="cause-card {css}" style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:1.3rem;flex-shrink:0;">{icon}</span>
                    <span>{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── 3. Counterfactuals ─────────────────────────────
        if cf_lines:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;margin:1.5rem 0 0.6rem;">
                <span style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:8px;
                             padding:4px 10px;font-size:0.8rem;font-weight:700;color:#1d4ed8;">
                    STEP 3</span>
                <span style="font-size:1rem;font-weight:700;color:#1e293b;">
                    🔄 What Would Change the Verdict? <span style="font-weight:400;color:#64748b;font-size:0.85rem;">
                    — Counterfactual Analysis</span>
                </span>
            </div>
            <p style="font-size:0.82rem;color:#64748b;margin:0 0 0.8rem 0;line-height:1.5;">
                We simulate "what if" scenarios by removing one signal at a time and
                re-running the AI to see if the verdict flips.
            </p>
            """, unsafe_allow_html=True)

            for scenario, new_label, flipped in cf_lines:
                if flipped:
                    _cf_icon = "⚡"
                    _cf_color = "#d97706"
                    _cf_bg = "#fffbeb"
                    _cf_border = "#fde68a"
                    _cf_text = f"→ Verdict <strong>would change</strong> to <strong>{new_label.upper()}</strong>"
                else:
                    _cf_icon = "🔒"
                    _cf_color = "#94a3b8"
                    _cf_bg = "#f8fafc"
                    _cf_border = "#e2e8f0"
                    _cf_text = f"→ Verdict stays <strong>{new_label.upper()}</strong> (no change)"
                st.markdown(f"""
                <div style="background:{_cf_bg};border:1px solid {_cf_border};border-radius:12px;
                            padding:1rem 1.3rem;margin-bottom:0.6rem;display:flex;
                            align-items:flex-start;gap:12px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                    <span style="font-size:1.3rem;flex-shrink:0;margin-top:2px;">{_cf_icon}</span>
                    <div>
                        <div style="font-size:0.85rem;color:#64748b;margin-bottom:4px;">
                            {scenario}...</div>
                        <div style="font-size:0.95rem;font-weight:600;color:{_cf_color};">
                            {_cf_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    #  DETAILED ANALYSIS (hidden by default — toggle in sidebar)
    # ═══════════════════════════════════════════════════════
    if show_debug:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Signal Breakdown</div>', unsafe_allow_html=True)

        sc1, sc2, sc3, sc4 = st.columns(4, gap="small")

        # Card 1 — Text AI
        fake_pct = r_probs["fake"] * 100
        txt_color = "red" if fake_pct > 60 else "green" if fake_pct < 40 else "amber"
        txt_bar   = "pbar-red" if fake_pct > 60 else "pbar-green" if fake_pct < 40 else "pbar-amber"
        with sc1:
            st.markdown(f"""
            <div class="sig-card">
                <div class="sig-title">🤖 Text AI (RoBERTa)</div>
                <div class="sig-value {txt_color}">{fake_pct:.0f}% fake</div>
                <div class="pbar-wrap"><div class="pbar-fill {txt_bar}" style="width:{fake_pct:.0f}%"></div></div>
                <div class="pbar-label">fake probability</div>
            </div>
            """, unsafe_allow_html=True)

        # Card 2 — Sentiment
        sent_val = sent_out["sentiment"]
        subj_val = sent_out["subjectivity"]
        if sent_val < 0.4:
            sent_label, sent_color = "😡 Negative", "red"
            sent_bar = "pbar-red"
        elif sent_val > 0.6:
            sent_label, sent_color = "😊 Positive", "green"
            sent_bar = "pbar-green"
        else:
            sent_label, sent_color = "😐 Neutral", "blue"
            sent_bar = "pbar-blue"
        with sc2:
            st.markdown(f"""
            <div class="sig-card">
                <div class="sig-title">✍️ Tone of Writing</div>
                <div class="sig-value {sent_color}">{sent_label}</div>
                <div class="pbar-wrap"><div class="pbar-fill {sent_bar}" style="width:{subj_val*100:.0f}%"></div></div>
                <div class="pbar-label">subjectivity {subj_val*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Card 3 — Clickbait
        cb_val = click_out["clickbait"] * 100
        cb_color = "red" if cb_val > 40 else "green" if cb_val < 20 else "amber"
        cb_bar   = "pbar-red" if cb_val > 40 else "pbar-green" if cb_val < 20 else "pbar-amber"
        cb_label = "🔥 High" if cb_val > 40 else "✅ Low" if cb_val < 20 else "⚠️ Medium"
        with sc3:
            st.markdown(f"""
            <div class="sig-card">
                <div class="sig-title">📣 Clickbait Score</div>
                <div class="sig-value {cb_color}">{cb_label} ({cb_val:.0f}%)</div>
                <div class="pbar-wrap"><div class="pbar-fill {cb_bar}" style="width:{cb_val:.0f}%"></div></div>
                <div class="pbar-label">sensationalism level</div>
            </div>
            """, unsafe_allow_html=True)

        # Card 4 — Image
        if image_path and vision_result["available"]:
            v_score = vision_result["consistency_score"] * 100
            v_verd  = vision_result["verdict"]
            v_color = {"consistent": "green", "mismatch": "red", "unrelated": "red"}.get(v_verd, "amber")
            v_bar   = {"consistent": "pbar-green", "mismatch": "pbar-red", "unrelated": "pbar-red"}.get(v_verd, "pbar-amber")
            v_icon  = {"consistent": "✅", "mismatch": "🚨", "unrelated": "⚠️"}.get(v_verd, "🤔")
            v_text  = {"consistent": "Matches text", "mismatch": "Does not match", "unrelated": "Unrelated"}.get(v_verd, "Inconclusive")
            with sc4:
                st.markdown(f"""
                <div class="sig-card">
                    <div class="sig-title">🖼️ Image vs Text</div>
                    <div class="sig-value {v_color}">{v_icon} {v_text}</div>
                    <div class="pbar-wrap"><div class="pbar-fill {v_bar}" style="width:{v_score:.0f}%"></div></div>
                    <div class="pbar-label">consistency {v_score:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with sc4:
                st.markdown("""
                <div class="sig-card">
                    <div class="sig-title">🖼️ Image vs Text</div>
                    <div class="sig-value blue">— No image</div>
                    <div class="pbar-wrap"><div class="pbar-fill pbar-blue" style="width:0%"></div></div>
                    <div class="pbar-label">upload an image to enable</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Hybrid VLM detail (Groq + Qwen2-VL) ──────────
        _has_blend_consistency = vision_result.get("_groq_score") is not None
        _has_blend_standalone  = standalone_result.get("_groq_fake_score") is not None
        if _has_blend_consistency or _has_blend_standalone:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🧠 Hybrid VLM Signal Sources</div>', unsafe_allow_html=True)
        if _has_blend_consistency:
            _gs = vision_result["_groq_score"]
            _qs = vision_result["_qwen_score"]
            _ql = vision_result.get("_qwen_label", "n/a")
            _bl = vision_result["consistency_score"]
            st.markdown(f"""
            <div class="sig-card">
                <div style="font-size:0.88rem; color:#334155; line-height:2;">
                    <strong>Groq Llama-4-Scout (90 %):</strong> consistency = {_gs:.2f}<br>
                    <strong>Finetuned Qwen2-VL (10 %):</strong> consistency = {_qs:.2f}
                    &nbsp;<span style="color:#64748b;">(label: {_ql})</span><br>
                    <strong>Blended score:</strong> {_bl:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif _has_blend_standalone:
            _gf = standalone_result["_groq_fake_score"]
            _qf = standalone_result.get("_qwen_fake_score", 0.5)
            _ql = standalone_result.get("_qwen_label", "n/a")
            _bf = standalone_result["fake_score"]
            st.markdown(f"""
            <div class="sig-card">
                <div style="font-size:0.88rem; color:#334155; line-height:2;">
                    <strong>Groq Llama-4-Scout (90 %):</strong> fake_score = {_gf:.2f}<br>
                    <strong>Finetuned Qwen2-VL (10 %):</strong> fake_score = {_qf:.2f}
                    &nbsp;<span style="color:#64748b;">(label: {_ql})</span><br>
                    <strong>Blended fake_score:</strong> {_bf:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Raw Data (JSON) ────────────────────────────────
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("🗃️ Raw technical data (JSON)"):
            st.json({
                "roberta_probs": r_probs,
                "features": features,
                "xgboost": xgb_out,
                "vision": vision_result,
                "qwen_vlm": qwen_result if qwen_result else None,
                "rag": rag_result,
                "standalone_vlm": standalone_result if image_mode == "image_only" else None,
            })

    # ── Footer ─────────────────────────────────────────────
    st.markdown(
        f"""<div class="app-footer">
            <div>AI-powered misinformation detection system for text and image analysis</div>
            <div style="font-size:0.72rem;margin-top:4px;color:#94a3b8;">Analysis completed in {elapsed:.2f}s</div>
        </div>""",
        unsafe_allow_html=True,
    )
    logging.info(
        f"label={r_label} conf={r_conf:.2f} "
        f"fake_prob={r_probs['fake']:.3f} clickbait={features['clickbait']:.3f}"
    )
