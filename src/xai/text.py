"""
Phase 7.1 — Text Explainability (Integrated Gradients)

Explains why the TEXT signal influenced the final decision.
Uses the Phase-2 text classifier.
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients


# -----------------------------
# Config
# -----------------------------
MODEL_DIR = Path("artifacts/text_bas/hf_model")  # Phase-2 model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["fake", "real", "uncertain"]


# -----------------------------
# Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()


# -----------------------------
# Forward function for Captum
# -----------------------------
def forward_func(inputs_embeds, attention_mask):
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    return outputs.logits


# -----------------------------
# Explain text
# -----------------------------
def explain_text(
    text: str,
    target_label: str,
    max_length: int = 256,
) -> Dict:
    """
    Returns token-level attributions using Integrated Gradients.
    """

    assert target_label in LABELS

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # Get embeddings
    if hasattr(model, "roberta"):
        embeddings_module = model.roberta.embeddings.word_embeddings
    elif hasattr(model, "bert"):
        embeddings_module = model.bert.embeddings.word_embeddings
    else:
        # Fallback for other models, try to find word_embeddings
        embeddings_module = model.get_input_embeddings()

    input_embeds = embeddings_module(input_ids)

    ig = IntegratedGradients(forward_func)

    target_idx = LABELS.index(target_label)

    attributions, _ = ig.attribute(
        inputs=input_embeds,
        additional_forward_args=(attention_mask,),
        target=target_idx,
        return_convergence_delta=True,
    )

    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions.detach().cpu().numpy()

    # Use tokenizer.convert_ids_to_tokens and format nicely
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Advanced cleanup: 
    # RoBERTa uses 'Ġ' for space and '<s>'/'</s>' for special tokens.
    # We strip 'Ġ' to show the actual word/subword.
    readable_tokens = []
    for t in tokens:
        if t == "<s>":
            readable_tokens.append("[CLS]")
        elif t == "</s>":
            readable_tokens.append("[SEP]")
        elif t.startswith("Ġ"):
            readable_tokens.append(t[1:])
        else:
            readable_tokens.append(t)

    # Normalize for visualization
    max_abs = np.max(np.abs(attributions)) + 1e-6
    scores = (attributions / max_abs).tolist()

    return {
        "text": text,
        "target_label": target_label,
        "tokens": readable_tokens,
        "attributions": scores,
    }


# -----------------------------
# Sanity test
# -----------------------------
if __name__ == "__main__":
    print("\n[Phase 7.1] Text XAI sanity check")
    print("-" * 50)

    sample_texts = [
        "Government releases annual budget report",
        "SHOCKING truth they don't want you to know!!!",
    ]

    for txt in sample_texts:
        result = explain_text(txt, target_label="fake")
        top = sorted(
            zip(result["tokens"], result["attributions"]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:6]

        print(f"\nTEXT: {txt}")
        print("Top contributing tokens:")
        for tok, score in top:
            print(f"  {tok:>12s}  {score:+.3f}")
