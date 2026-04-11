"""
Phase 7.3 — Counterfactual Explanations

We simulate "what-if" changes on features
and observe if final label flips.
"""

import copy
import json

from src.fusion.infer import predict_sample


# -----------------------------
# Counterfactual rules
# -----------------------------
def apply_counterfactuals(features: dict):
    variants = {}

    # 1️⃣ Remove sensational framing
    f1 = copy.deepcopy(features)
    f1["clickbait"] = 0.0
    f1["subjectivity"] = 0.3
    variants["remove_sensational"] = f1

    # 2️⃣ Remove contradiction evidence
    f2 = copy.deepcopy(features)
    f2["rag_contradict"] = 0.0
    variants["remove_contradiction"] = f2

    # 3️⃣ Remove image influence
    f3 = copy.deepcopy(features)
    f3["vlm_consistency"] = 0.5
    f3["vlm_entropy"] = 1.0
    variants["remove_image_signal"] = f3

    return variants


# -----------------------------
# Run CF analysis
# -----------------------------
def run_counterfactuals(features: dict):

    base = predict_sample(features)

    variants = apply_counterfactuals(features)

    results = {
        "original": base,
        "counterfactuals": {}
    }

    for name, feat in variants.items():
        pred = predict_sample(feat)
        results["counterfactuals"][name] = pred

    return results


# -----------------------------
# CLI test
# -----------------------------
def main():
    print("\n[Phase 7.3] Counterfactual sanity check")
    print("-" * 50)

    sample_features = {
        "text_cls_score": 0.25,
        "rag_support": 0.10,
        "rag_contradict": 0.75,
        "vlm_consistency": 0.40,
        "vlm_entropy": 0.60,
        "ocr_match": 0.05,
        "sentiment": 0.20,
        "subjectivity": 0.80,
        "clickbait": 0.70,
        "text_length_norm": 0.20,
        "has_image": 1,
    }

    results = run_counterfactuals(sample_features)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
