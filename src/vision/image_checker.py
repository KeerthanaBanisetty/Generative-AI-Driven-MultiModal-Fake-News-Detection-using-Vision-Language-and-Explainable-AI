"""
Image–Text Consistency Checker
================================
Uses a large vision model to analyse whether an image genuinely
matches the text that accompanies it.

Returns a structured result with:
  - consistency_score : float [0.0 – 1.0]
      1.0 = image perfectly matches the text
      0.0 = image is completely unrelated or contradicts the text
  - verdict          : "consistent" | "mismatch" | "unrelated" | "uncertain"
  - image_summary    : what is actually visible in the image
  - text_claim       : what the text claims
  - mismatch_reason  : if inconsistent, why (else empty string)
  - confidence       : float [0.0 – 1.0]  model's own certainty

All external dependencies are loaded lazily so importing this module
has zero startup cost.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────
def _load_key() -> Optional[str]:
    """Read the API key from the environment (loaded from .env if present)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    except ImportError:
        pass
    return os.environ.get("GROQ_API_KEY")


# ─────────────────────────────────────────────────────────
# Vision model config
# ─────────────────────────────────────────────────────────
_MODEL          = "meta-llama/llama-4-scout-17b-16e-instruct"
_MAX_TOKENS     = 512
_IMAGE_MAX_PX   = 1024   # downscale if larger (saves tokens)


# ─────────────────────────────────────────────────────────
# Image encoding
# ─────────────────────────────────────────────────────────
def _encode_image(image_path: str) -> tuple[str, str]:
    """
    Load image, optionally resize, and return (base64_string, mime_type).
    """
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")

    # Downscale if very large — keeps API cost low
    w, h = img.size
    if max(w, h) > _IMAGE_MAX_PX:
        scale = _IMAGE_MAX_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


# ─────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert fact-checker and media forensics analyst specialising in \
detecting fake news, image manipulation, and misleading media.

Your task: carefully examine the provided image and the accompanying text, \
then determine whether the image genuinely supports, matches, or is relevant \
to what the text claims.

Think step by step:
1. What does the image actually show? (objects, people, setting, text visible \
   in the image, logos, dates, locations if visible)
2. What does the accompanying text claim or describe?
3. Do the image and text refer to the same event, place, person, or topic?
4. Are there any red flags?
   - Mismatched location (image from country A, text claims country B)
   - Mismatched time (old archival image used for recent event)
   - Mismatched subject (image of person X used for story about person Y)
   - Misleading cropping or context
   - Text overlaid on image that contradicts the caption
   - Stock photo used as if it were a real news photo
   - Document/graphic content that supports or contradicts the text

Respond ONLY with a JSON object — no extra text before or after — in this exact format:
{
  "image_summary": "<one concise sentence describing what is genuinely in the image>",
  "text_claim": "<one concise sentence of what the text is claiming>",
  "match_analysis": "<2–4 sentences of your step-by-step reasoning>",
  "verdict": "<one of: consistent | mismatch | unrelated | uncertain>",
  "mismatch_reason": "<if verdict is mismatch or unrelated, explain the specific discrepancy; else empty string>",
  "consistency_score": <float 0.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>
}

Scoring guide for consistency_score:
  0.9 – 1.0  : Image directly and accurately illustrates the text
  0.7 – 0.89 : Image is broadly related but may be generic/stock
  0.5 – 0.69 : Partial match; some elements align, some do not
  0.3 – 0.49 : Weak or coincidental connection
  0.0 – 0.29 : Image contradicts, is completely unrelated, or is from a different event/context
"""

_USER_TEMPLATE = """\
Please analyse whether the image matches this text:

TEXT: "{text}"

Examine the image carefully and respond with the JSON object as instructed."""


# ─────────────────────────────────────────────────────────
# Standalone image analysis (image-only mode, no text)
# ─────────────────────────────────────────────────────────
_STANDALONE_SYSTEM_PROMPT = """\
You are an expert fake news detector and media forensics analyst.
Analyse the provided image to determine if it is being used to DECEIVE or spread misinformation.

CRITICAL — "digitally designed" does NOT mean "fake news":
- Celebratory posters / team graphics / victory banners for REAL events → NOT fake (score LOW)
- Official infographics, promotional material, event announcements → NOT fake (score LOW)
- Memes that are clearly humour, satire, or entertainment → NOT fake (score LOW)
- Professional photos with filters, colour grading, or cropping → NOT fake (score LOW)
- News article screenshots from REAL publications about REAL events → NOT fake (score LOW)

ONLY flag as fake when the image is DESIGNED TO DECEIVE:
- Doctored / photoshopped photos with altered faces, text, or scene elements
- Fake screenshots fabricated to look like real news sites, tweets, or messages
- Real photos used with FALSE context to mislead (out-of-context misuse)
- AI-generated images presented as real photographs of events that did not happen
- Misleading charts with manipulated scales, axes, or cherry-picked data
- Fabricated official documents, IDs, or letterheads

Think step by step:
1. What does this image show?
2. Extract ALL visible text verbatim (headlines, captions, overlaid text, watermarks, dates)
3. Is this a REAL event / legitimate content that is simply designed well? If yes → low fake_score.
4. Or is this DECEPTIVELY manipulated to spread false information? If yes → high fake_score.

Respond ONLY with a JSON object — no extra text before or after:
{
  "image_description": "<one concise sentence describing what is in the image>",
  "extracted_text": "<ALL visible text verbatim, empty string if none>",
  "content_type": "<one of: news_screenshot | meme | photo | graphic | chart | other>",
  "fake_signals": ["<only DECEPTIVE signals — not normal design choices>"],
  "is_manipulated": <true ONLY if deceptively manipulated to mislead>,
  "manipulation_reason": "<specific deception reason if manipulated, else empty string>",
  "fake_score": <float 0.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>
}

fake_score guide:
  0.0 – 0.25: Legitimate — real event, real content, clearly designed graphic
  0.25 – 0.45: Minor concerns — probably real but has unusual elements
  0.45 – 0.70: Suspicious — possible deception, ambiguous intent
  0.70 – 1.0 : Deceptive — clear signs of fabrication or misleading manipulation
"""

_STANDALONE_USER_PROMPT = (
    "Analyse this image for fake news signals and extract any visible text. "
    "Respond with the JSON object as instructed."
)


def analyse_image_standalone(image_path: str) -> dict:
    """
    Analyse an image independently (no accompanying text) for fake news signals.
    Uses Groq vision LLM to describe the image, extract visible text, and score
    fake news indicators.

    Returns
    -------
    dict with keys:
        image_description   str
        extracted_text      str   (verbatim text visible in image)
        content_type        str
        fake_signals        list[str]
        is_manipulated      bool
        manipulation_reason str
        fake_score          float [0, 1]
        confidence          float [0, 1]
        available           bool
    """
    _FALLBACK = {
        "image_description" : "",
        "extracted_text"    : "",
        "content_type"      : "other",
        "fake_signals"      : [],
        "is_manipulated"    : False,
        "manipulation_reason": "",
        "fake_score"        : 0.5,
        "confidence"        : 0.0,
        "available"         : False,
    }

    api_key = _load_key()
    if not api_key:
        log.warning("Image standalone: API key not found — returning fallback.")
        return _FALLBACK

    try:
        from groq import Groq
    except ImportError:
        log.warning("Image standalone: groq package not installed — returning fallback.")
        return _FALLBACK

    try:
        b64, mime = _encode_image(image_path)
    except Exception as e:
        log.warning(f"Image standalone: could not encode image ({e}) — returning fallback.")
        return _FALLBACK

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _STANDALONE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": _STANDALONE_USER_PROMPT,
                        },
                    ],
                },
            ],
            max_tokens=_MAX_TOKENS,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {raw[:200]}")

        result = json.loads(json_match.group())

        result["fake_score"]  = float(max(0.0, min(1.0, result.get("fake_score",  0.5))))
        result["confidence"]  = float(max(0.0, min(1.0, result.get("confidence", 0.5))))
        result.setdefault("image_description",  "")
        result.setdefault("extracted_text",     "")
        result.setdefault("content_type",       "other")
        result.setdefault("fake_signals",       [])
        result.setdefault("is_manipulated",     False)
        result.setdefault("manipulation_reason","")
        result["available"] = True

        log.info(
            f"Image standalone: fake_score={result['fake_score']:.2f} "
            f"confidence={result['confidence']:.2f} "
            f"manipulated={result['is_manipulated']}"
        )
        return result

    except Exception as e:
        log.warning(f"Image standalone: model call failed ({e}) — returning fallback.")
        return _FALLBACK


# ─────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────
def check_image_text(
    image_path: str,
    text: str,
    fallback_score: float = 0.5,
) -> dict:
    """
    Check whether the image is consistent with the text.

    Parameters
    ----------
    image_path    : path to the image file (jpg / png / webp)
    text          : the news headline or post text
    fallback_score: returned as consistency_score if the model is unavailable

    Returns
    -------
    dict with keys:
        consistency_score  float [0, 1]
        verdict            str
        image_summary      str
        text_claim         str
        match_analysis     str
        mismatch_reason    str
        confidence         float [0, 1]
        available          bool   (False if model could not be reached)
    """
    _FALLBACK = {
        "consistency_score": fallback_score,
        "verdict"          : "uncertain",
        "image_summary"    : "",
        "text_claim"       : text,
        "match_analysis"   : "",
        "mismatch_reason"  : "",
        "confidence"       : 0.0,
        "available"        : False,
    }

    api_key = _load_key()
    if not api_key:
        log.warning("Image checker: API key not found — returning fallback.")
        return _FALLBACK

    try:
        from groq import Groq
    except ImportError:
        log.warning("Image checker: groq package not installed — returning fallback.")
        return _FALLBACK

    try:
        b64, mime = _encode_image(image_path)
    except Exception as e:
        log.warning(f"Image checker: could not encode image ({e}) — returning fallback.")
        return _FALLBACK

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": _USER_TEMPLATE.format(text=text),
                        },
                    ],
                },
            ],
            max_tokens=_MAX_TOKENS,
            temperature=0.1,   # low temp → deterministic, factual responses
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON even if the model adds extra prose
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {raw[:200]}")

        result = json.loads(json_match.group())

        # Sanitise and clamp numeric fields
        result["consistency_score"] = float(
            max(0.0, min(1.0, result.get("consistency_score", fallback_score)))
        )
        result["confidence"] = float(
            max(0.0, min(1.0, result.get("confidence", 0.5)))
        )
        result.setdefault("verdict",         "uncertain")
        result.setdefault("image_summary",   "")
        result.setdefault("text_claim",      text)
        result.setdefault("match_analysis",  "")
        result.setdefault("mismatch_reason", "")
        result["available"] = True

        log.info(
            f"Image checker: verdict={result['verdict']} "
            f"score={result['consistency_score']:.2f} "
            f"confidence={result['confidence']:.2f}"
        )
        return result

    except Exception as e:
        log.warning(f"Image checker: model call failed ({e}) — returning fallback.")
        _FALLBACK["consistency_score"] = fallback_score
        return _FALLBACK
