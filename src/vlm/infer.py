"""
Qwen2-VL + LoRA inference wrapper for the Streamlit app.
=========================================================
Provides two public functions:
  - predict_consistency(image_path, post_text, ocr_text) -> dict
  - predict_standalone(image_path) -> dict

And a module-level flag:
  - QWEN_AVAILABLE : bool (True once model loads successfully)

The model is loaded lazily on the first call (4-bit NF4 quantised,
~5 GB VRAM).  If loading fails (no GPU, OOM, missing libs) the
flag stays False and every call returns a safe fallback dict.

Adapted from artifacts/vlm/stage_b/release/inference.py (the
reference batch-inference script used during evaluation).
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

ADAPTER_DIR   = str(_PROJECT_ROOT / "artifacts" / "vlm" / "stage_b" / "adapter")
PROCESSOR_DIR = str(_PROJECT_ROOT / "artifacts" / "vlm" / "stage_b" / "processor")
TAU_FILE      = str(_PROJECT_ROOT / "artifacts" / "vlm" / "stage_b" / "release" / "tau.json")
DEFAULT_TAU   = 1.022          # val 75th-percentile; ~25 % abstain
MODEL_NAME    = "Qwen/Qwen2-VL-7B-Instruct"

LABELS = ["consistent", "mismatched", "uncertain"]

# Help CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ─────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────
_model: Any     = None
_processor: Any = None
_abc_ids: Optional[Tuple[int, int, int]] = None
_tau: float     = DEFAULT_TAU
_load_attempted = False

QWEN_AVAILABLE  = False        # flipped to True after successful load


# ─────────────────────────────────────────────────────────
# Fallback dicts (returned when model is unavailable)
# ─────────────────────────────────────────────────────────
_FALLBACK_CONSISTENCY: Dict[str, Any] = {
    "label":             "uncertain",
    "consistency_score":  0.5,
    "probs":             {"consistent": 0.33, "mismatched": 0.33, "uncertain": 0.34},
    "entropy":            math.log(3),
    "available":          False,
}

_FALLBACK_STANDALONE: Dict[str, Any] = {
    "label":      "uncertain",
    "fake_score":  0.5,
    "probs":      {"consistent": 0.33, "mismatched": 0.33, "uncertain": 0.34},
    "entropy":     math.log(3),
    "available":   False,
}


# ─────────────────────────────────────────────────────────
# Lazy model loading
# ─────────────────────────────────────────────────────────
def _load_tau() -> float:
    """Read tau from the release JSON, or fall back to default."""
    try:
        data = json.loads(Path(TAU_FILE).read_text())
        return float(data.get("tau", DEFAULT_TAU))
    except Exception:
        return DEFAULT_TAU


def _try_load_model() -> bool:
    """
    Attempt to load Qwen2-VL + LoRA adapter (4-bit NF4).
    Returns True on success, False on any failure.
    Sets module-level _model, _processor, _abc_ids, _tau.
    """
    global _model, _processor, _abc_ids, _tau, _load_attempted, QWEN_AVAILABLE
    _load_attempted = True

    # ── Pre-checks ────────────────────────────────────────
    try:
        import torch
    except ImportError:
        log.warning("Qwen VLM: torch not installed")
        return False

    if not torch.cuda.is_available():
        log.info("Qwen VLM: no CUDA GPU — skipping load")
        return False

    # Check minimum free VRAM (~4 GB needed for 4-bit)
    try:
        free, _ = torch.cuda.mem_get_info()
        if free < 3.5 * (1024 ** 3):
            log.warning("Qwen VLM: insufficient VRAM (%.1f GB free)", free / (1024**3))
            return False
    except Exception:
        pass  # proceed optimistically

    if not Path(ADAPTER_DIR).exists():
        log.warning("Qwen VLM: adapter dir not found at %s", ADAPTER_DIR)
        return False

    # ── Load processor ────────────────────────────────────
    try:
        from transformers import AutoProcessor
        proc_path = PROCESSOR_DIR if Path(PROCESSOR_DIR).exists() else MODEL_NAME
        _processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
        log.info("Qwen VLM: processor loaded from %s", proc_path)
    except Exception as e:
        log.warning("Qwen VLM: processor load failed: %s", e)
        return False

    # ── Load base model (4-bit quantised) ─────────────────
    try:
        from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
        dtype = torch.bfloat16

        quant_cfg = None
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        except Exception:
            log.info("Qwen VLM: bitsandbytes unavailable, loading in bf16")

        base = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
            quantization_config=quant_cfg,
        )
        log.info("Qwen VLM: base model loaded (%s)", MODEL_NAME)
    except Exception as e:
        log.warning("Qwen VLM: base model load failed: %s", e)
        _processor = None
        return False

    # ── Attach LoRA adapter ───────────────────────────────
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, ADAPTER_DIR, device_map="auto")
        model.eval()
        torch.set_grad_enabled(False)
        _model = model
        log.info("Qwen VLM: LoRA adapter loaded from %s", ADAPTER_DIR)
    except Exception as e:
        log.warning("Qwen VLM: LoRA adapter load failed: %s", e)
        _processor = None
        return False

    # ── Resolve A/B/C token IDs ───────────────────────────
    try:
        tok = _processor.tokenizer
        tokA = tok(" A", add_special_tokens=False).input_ids[-1]
        tokB = tok(" B", add_special_tokens=False).input_ids[-1]
        tokC = tok(" C", add_special_tokens=False).input_ids[-1]
        _abc_ids = (tokA, tokB, tokC)
    except Exception as e:
        log.warning("Qwen VLM: token ID lookup failed: %s", e)
        _model = _processor = None
        return False

    _tau = _load_tau()
    QWEN_AVAILABLE = True
    log.info("Qwen VLM: ready  (tau=%.3f)", _tau)
    return True


def _ensure_loaded() -> bool:
    """Load model on first call.  No-op after the first attempt."""
    global _load_attempted
    if _load_attempted:
        return QWEN_AVAILABLE
    try:
        return _try_load_model()
    except Exception as e:
        log.warning("Qwen VLM: unexpected error during load: %s", e)
        _load_attempted = True
        return False


# ─────────────────────────────────────────────────────────
# Core inference helpers
# ─────────────────────────────────────────────────────────
def _make_prompt(post_text: str, ocr_text: str = "") -> str:
    """Exact prompt template matching training / evaluation."""
    return (
        f"POST: {post_text}\n"
        f"OCR: {ocr_text}\n"
        "Q: Is the image consistent with the text? Choose one:\n"
        "A. consistent\n"
        "B. mismatched\n"
        "C. uncertain\n"
        "Answer: "
    )


def _entropy(probs) -> float:
    """Natural-log entropy (nats) of a 1-D probability tensor."""
    import torch
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def _first_token_scores(img, post_text: str, ocr_text: str = ""):
    """
    Run a single forward pass and extract softmax probs over
    the A / B / C choice tokens.

    Returns (probs_tensor[3], raw_logprobs_dict).
    """
    import torch

    prompt = _make_prompt(post_text, ocr_text)

    # Build chat-style input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt},
            ],
        },
    ]
    chat = _processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    enc = _processor(text=chat, images=img, return_tensors="pt")

    # Move to model device
    device = next(_model.parameters()).device
    enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v
           for k, v in enc.items()}

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        out = _model(**enc)

    next_logits = out.logits[:, -1, :]
    logprobs = torch.log_softmax(next_logits, dim=-1).squeeze(0)

    tokA, tokB, tokC = _abc_ids
    vals = torch.stack([logprobs[tokA], logprobs[tokB], logprobs[tokC]])
    probs = torch.softmax(vals, dim=0)

    raw = {"A": float(vals[0]), "B": float(vals[1]), "C": float(vals[2])}
    return probs, raw


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────
def predict_consistency(
    image_path: str,
    post_text: str,
    ocr_text: str = "",
) -> Dict[str, Any]:
    """
    Check image-text consistency using the finetuned Qwen2-VL.

    Returns
    -------
    dict with keys:
        label              : "consistent" | "mismatched" | "uncertain"
        consistency_score  : float [0, 1]   (= prob of "consistent")
        probs              : {consistent, mismatched, uncertain}
        entropy            : float (nats)
        available          : bool
    """
    if not _ensure_loaded():
        return dict(_FALLBACK_CONSISTENCY)

    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")

        probs, _ = _first_token_scores(img, post_text, ocr_text)
        ent = _entropy(probs)

        # Entropy-based abstention
        if ent > _tau:
            label = "uncertain"
        else:
            import torch
            label = LABELS[int(torch.argmax(probs))]

        probs_dict = {
            "consistent": float(probs[0]),
            "mismatched": float(probs[1]),
            "uncertain":  float(probs[2]),
        }

        return {
            "label":             label,
            "consistency_score": round(probs_dict["consistent"], 4),
            "probs":             probs_dict,
            "entropy":           round(ent, 4),
            "available":         True,
        }

    except Exception as e:
        log.warning("Qwen VLM predict_consistency failed: %s", e)
        return dict(_FALLBACK_CONSISTENCY)


def predict_standalone(image_path: str) -> Dict[str, Any]:
    """
    Analyse an image without accompanying text (image-only mode).

    Uses a generic prompt since the model was trained on image+text
    pairs.  Derives a ``fake_score`` from the mismatched probability.

    Returns
    -------
    dict with keys:
        label      : "consistent" | "mismatched" | "uncertain"
        fake_score : float [0, 1]
        probs      : {consistent, mismatched, uncertain}
        entropy    : float
        available  : bool
    """
    if not _ensure_loaded():
        return dict(_FALLBACK_STANDALONE)

    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB")

        # Generic prompt for image-only analysis
        probs, _ = _first_token_scores(img, "[image content]", "")
        ent = _entropy(probs)

        if ent > _tau:
            label = "uncertain"
        else:
            import torch
            label = LABELS[int(torch.argmax(probs))]

        probs_dict = {
            "consistent": float(probs[0]),
            "mismatched": float(probs[1]),
            "uncertain":  float(probs[2]),
        }

        # Derive fake_score: high mismatched → high fake
        fake_score = probs_dict["mismatched"] * 0.7 + probs_dict["uncertain"] * 0.3

        return {
            "label":      label,
            "fake_score": round(max(0.0, min(1.0, fake_score)), 4),
            "probs":      probs_dict,
            "entropy":    round(ent, 4),
            "available":  True,
        }

    except Exception as e:
        log.warning("Qwen VLM predict_standalone failed: %s", e)
        return dict(_FALLBACK_STANDALONE)
