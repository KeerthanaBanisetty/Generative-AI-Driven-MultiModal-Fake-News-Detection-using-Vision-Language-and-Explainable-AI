"""
Phase 5 – Sentiment & Subjectivity
---------------------------------
Lightweight framing signals for misinformation detection.

Outputs:
- sentiment: [0,1]  (mapped polarity)
- subjectivity: [0,1]
"""

from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment polarity and subjectivity.

    Args:
        text (str): input text (title / claim / caption)

    Returns:
        dict with keys:
          - polarity_raw: float in [-1,1]
          - sentiment: float in [0,1]
          - subjectivity: float in [0,1]
    """
    if not text or not text.strip():
        return {
            "polarity_raw": 0.0,
            "sentiment": 0.5,
            "subjectivity": 0.0,
        }

    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    # Map polarity [-1,1] → [0,1]
    sentiment = (polarity + 1.0) / 2.0

    return {
        "polarity_raw": round(polarity, 4),
        "sentiment": round(sentiment, 4),
        "subjectivity": round(subjectivity, 4),
    }


# -------------------------------
# Quick sanity test
# -------------------------------
def main():
    examples = [
        "Government releases annual budget report",
        "SHOCKING truth they don't want you to know!!!",
        "This disgusting act proves how corrupt the system is",
        "Scientists discover water on Mars",
        "",
    ]

    print("\n[Phase 5] Sentiment sanity check\n" + "-" * 40)
    for txt in examples:
        out = analyze_sentiment(txt)
        print(f"\nTEXT: {repr(txt)}")
        for k, v in out.items():
            print(f"  {k:14s}: {v}")


if __name__ == "__main__":
    main()
