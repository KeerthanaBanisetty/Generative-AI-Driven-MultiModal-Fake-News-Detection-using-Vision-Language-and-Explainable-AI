"""
Phase 5 – Clickbait / Sensational Framing
-----------------------------------------
Rule-based clickbait scoring for news titles and social posts.

Output:
- clickbait_score: float in [0,1]
"""

import re
from typing import Dict


# Common clickbait phrases (expandable)
CLICKBAIT_PHRASES = [
    "you won't believe",
    "shocking",
    "what happens next",
    "this is why",
    "top",
    "reasons why",
    "exposed",
    "goes viral",
    "will blow your mind",
    "can't believe",
]

SUPERLATIVES = [
    "best", "worst", "most", "least",
    "never", "always", "everyone", "no one"
]


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _excess_punct(text: str) -> float:
    return 1.0 if re.search(r"[!?]{2,}", text) else 0.0


def _contains_phrases(text: str) -> float:
    t = text.lower()
    return float(any(p in t for p in CLICKBAIT_PHRASES))


def _superlative_count(text: str) -> float:
    t = text.lower()
    return min(1.0, sum(1 for w in SUPERLATIVES if w in t) / 2.0)


def _is_question(text: str) -> float:
    return 1.0 if text.strip().endswith("?") else 0.0


def clickbait_score(text: str) -> Dict[str, float]:
    """
    Compute clickbait score.

    Args:
        text (str): title / headline / post

    Returns:
        dict with:
          - clickbait: float [0,1]
          - breakdown: individual signals
    """
    if not text or not text.strip():
        return {
            "clickbait": 0.0,
            "caps_ratio": 0.0,
            "excess_punct": 0.0,
            "phrases": 0.0,
            "superlatives": 0.0,
            "question": 0.0,
        }

    caps = _caps_ratio(text)
    punct = _excess_punct(text)
    phr = _contains_phrases(text)
    sup = _superlative_count(text)
    q = _is_question(text)

    # Weighted sum (tuned for news titles)
    score = (
        0.25 * caps +
        0.20 * punct +
        0.25 * phr +
        0.20 * sup +
        0.10 * q
    )

    return {
        "clickbait": round(min(score, 1.0), 4),
        "caps_ratio": round(caps, 4),
        "excess_punct": punct,
        "phrases": phr,
        "superlatives": round(sup, 4),
        "question": q,
    }


# -------------------------------
# Quick sanity test
# -------------------------------
def main():
    examples = [
        "Government releases annual budget report",
        "SHOCKING truth they don't want you to know!!!",
        "Top 10 reasons why this will blow your mind",
        "Is this the worst disaster ever?",
        "Scientists discover new particle",
        "",
    ]

    print("\n[Phase 5] Clickbait sanity check\n" + "-" * 40)
    for txt in examples:
        out = clickbait_score(txt)
        print(f"\nTEXT: {repr(txt)}")
        for k, v in out.items():
            print(f"  {k:14s}: {v}")


if __name__ == "__main__":
    main()
