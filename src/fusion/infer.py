"""
Phase 6 – Fusion Inference
---------------------------
Loads trained XGBoost fusion model and performs prediction
on a single feature dictionary.

Used by:
- API / inference scripts
- Phase 7 counterfactuals
- Evidence card generation
"""

import json
import numpy as np
import xgboost as xgb
from pathlib import Path
import pandas as pd
# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = Path("artifacts/fusion/xgb_fusion.json")
CALIB_PATH = Path("artifacts/fusion/calibration.json")

# ------------------------------------------------------------
# Global cache
# ------------------------------------------------------------

_MODEL = None
_TEMPERATURE = 1.0

# ------------------------------------------------------------
# Feature Order (MUST match training order)
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "text_cls_score",
    "rag_support",
    "rag_contradict",
    "vlm_consistency",
    "vlm_entropy",
    "ocr_match",
    "sentiment",
    "subjectivity",
    "clickbait",
    "text_length_norm",
    "has_image",
]

CLASS_NAMES = ["fake", "real", "uncertain"]

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def _load_model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Fusion model not found at {MODEL_PATH}")

        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        _MODEL = booster

    return _MODEL


def _load_temperature():
    global _TEMPERATURE

    if CALIB_PATH.exists():
        with open(CALIB_PATH, "r") as f:
            data = json.load(f)
            _TEMPERATURE = float(data.get("temperature", 1.0))

    return _TEMPERATURE


def _softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / exp.sum()


# ------------------------------------------------------------
# Main Prediction Function
# ------------------------------------------------------------


def predict_sample(features: dict):
    """
    features: dict containing all required fusion features
    returns: dict with label + calibrated probabilities
    """

    model = _load_model()
    temperature = _load_temperature()

    # Ensure correct feature ordering
    X_df = pd.DataFrame(
    [[float(features.get(f, 0.0)) for f in FEATURE_COLUMNS]],
    columns=FEATURE_COLUMNS
)

    dmatrix = xgb.DMatrix(X_df)

    # Raw logits from XGBoost
    raw_preds = model.predict(dmatrix)

    # XGBoost multiclass returns probabilities directly
    probs = raw_preds[0]

    # Apply temperature scaling (if calibrated)
    if temperature != 1.0:
        logits = np.log(probs + 1e-12)
        scaled_logits = logits / temperature
        probs = _softmax(scaled_logits)

    pred_idx = int(np.argmax(probs))

    return {
        "label": CLASS_NAMES[pred_idx],
        "probs": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        },
        "confidence": float(np.max(probs)),
    }


# ------------------------------------------------------------
# CLI Test
# ------------------------------------------------------------

def main():
    print("\n[Phase 6] Fusion Inference Sanity Check")
    print("-" * 50)

    sample_features = {
        "text_cls_score": 0.7,
        "rag_support": 0.8,
        "rag_contradict": 0.1,
        "vlm_consistency": 0.9,
        "vlm_entropy": 0.2,
        "ocr_match": 0.7,
        "sentiment": 0.4,
        "subjectivity": 0.3,
        "clickbait": 0.1,
        "text_length_norm": 0.2,
        "has_image": 1,
    }

    result = predict_sample(sample_features)

    print("Prediction:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
