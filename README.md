# AI-NLP-FAKE-NEWS-DETECTION

A multimodal fake news detection system that combines text analysis, image-text consistency checking, web evidence retrieval, and explainable AI into a single end-to-end pipeline. The system accepts a news post (text + optional image) and classifies it as **Fake**, **Real**, or **Uncertain**, along with human-readable explanations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Repository Structure](#4-repository-structure)
5. [Core Files Reference](#5-core-files-reference)
6. [Pipeline — Phase by Phase](#6-pipeline--phase-by-phase)
7. [Installation & Setup](#7-installation--setup)
8. [Data Preparation](#8-data-preparation)
9. [Running Each Phase](#9-running-each-phase)
10. [Running the Streamlit App](#10-running-the-streamlit-app)
11. [Artifacts Produced](#11-artifacts-produced)
12. [Known Issues & Limitations](#12-known-issues--limitations)
13. [FAQ for Beginners](#13-faq-for-beginners)

---

## 1. Project Overview

This project detects fake news using **seven sequential phases**:

| Phase | Name | What It Does |
|-------|------|-------------|
| 1 | Data Manifests | Downloads & standardizes LIAR and Fakeddit datasets into JSONL format |
| 2 | Text Baseline | Fine-tunes RoBERTa-base on LIAR for 3-class text classification |
| 3 | RAG Verification | Extracts claims from text, retrieves web evidence via Google Search, runs NLI stance scoring |
| 4 | VLM (Vision-Language) | Fine-tunes Qwen2-VL-7B on Fakeddit to detect image–text mismatches |
| 5 | Framing Signals | Computes sentiment, subjectivity, and clickbait scores from post text |
| 6 | Fusion Brain | Trains an XGBoost meta-learner on all signals (text + RAG + VLM + framing) |
| 7 | Explainability | Produces token attributions, Grad-CAM heatmaps, counterfactuals, and evidence cards |

The **Streamlit app** (`app.py`) integrates Phases 4–7 into a real-time web interface.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT (app.py)                         │
│              Post Text (required) + Image (optional)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
  │  PHASE 5    │   │   PHASE 4    │   │ Defaults (0.5)   │
  │  Framing    │   │     VLM      │   │ text/RAG signals │
  │  Signals    │   │  Qwen2-VL-7B │   │ (if not computed)│
  │             │   │   LoRA       │   └──────────────────┘
  │ • Sentiment │   │ • consistency│
  │ • Clickbait │   │ • entropy    │
  └──────┬──────┘   └──────┬───────┘
         │                 │
         └────────┬────────┘
                  ▼
         ┌────────────────┐
         │    PHASE 6     │
         │  Fusion Brain  │
         │  XGBoost +     │
         │  Calibration   │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │    PHASE 7     │
         │ Explainability │
         │ • Token XAI    │
         │ • Counterfacts │
         │ • Evidence Card│
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │   PREDICTION   │
         │  FAKE / REAL / │
         │   UNCERTAIN    │
         └────────────────┘


OFFLINE TRAINING PIPELINE:
────────────────────────────────────────────────────────────────────
Raw Data → [Phase 1] → JSONL Manifests
         → [Phase 2] → RoBERTa Model  ──────────────────────┐
         → [Phase 3] → RAG Scores     ────────────────────┐ │
         → [Phase 4] → VLM Adapter    ──────────────────┐ │ │
         → [Phase 5] → (inline, no artifact)            │ │ │
                                                         ▼ ▼ ▼
                                                     [Phase 6]
                                                     XGBoost Fusion
                                                         │
                                                         ▼
                                                   xgb_fusion.json
                                                   calibrator.json
```

---

## 3. Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.30 | Web interface (app.py) |
| `transformers` | ≥ 4.40 | RoBERTa (Phase 2), roberta-large-mnli (Phase 3), Qwen2-VL-7B (Phase 4) |
| `peft` | ≥ 0.10 | LoRA fine-tuning of Qwen2-VL (Phase 4) |
| `bitsandbytes` | ≥ 0.43 | QLoRA 4-bit quantization (Phase 4, GPU only) |
| `torch` + `torchvision` | ≥ 2.1 | All neural network operations |
| `xgboost` | ≥ 2.0 | Meta-learner fusion classifier (Phase 6) |
| `captum` | ≥ 0.7 | Integrated Gradients for token XAI (Phase 7) |
| `textblob` | ≥ 0.17 | Sentiment analysis (Phase 5) |
| `paddleocr` + `paddlepaddle` | ≥ 2.7 | OCR text extraction from images (Phase 4) |
| `sentence-transformers` | ≥ 2.6 | Semantic similarity (Phase 4, optional) |
| `opencv-python` | ≥ 4.9 | Grad-CAM overlay image generation (Phase 7) |
| `scikit-learn` | ≥ 1.4 | Dataset splitting, metrics, LabelEncoder |
| `pandas` + `numpy` | standard | Data manipulation throughout |
| `PyYAML` | ≥ 6.0 | Reading all config files |
| `requests` | standard | Serper/Google Search API calls (Phase 3) |
| `Pillow` | ≥ 10.0 | Image loading throughout |
| `matplotlib` | ≥ 3.8 | Confusion matrix plots (Phase 2) |
| `tqdm` | ≥ 4.66 | Training progress bars |

---

## 4. Repository Structure

```
AI-NLP-FAKE-NEWS-DETECTION/
│
├── app.py                          ← Streamlit web app (main entry point)
│
├── configs/                        ← YAML configuration files for each phase
│   ├── data.yaml                   ← Phase 1: dataset paths + label maps
│   ├── text_baseline.yaml          ← Phase 2: RoBERTa training settings
│   ├── rag.yaml                    ← Phase 3: RAG pipeline settings
│   ├── ocr.yaml                    ← Phase 4: PaddleOCR settings
│   ├── vlm_stage_a.yaml            ← Phase 4: Stage A VLM config
│   ├── vlm_stage_b.yaml            ← Phase 4: Stage B VLM config
│   └── vlm_stage_b_qwen2vl7b.yaml  ← Phase 4: Qwen2-VL-7B LoRA config (active)
│
├── src/                            ← All source code
│   ├── common/
│   │   └── io.py                   ← Shared utilities: YAML reader, JSONL reader/writer
│   ├── dataio/
│   │   ├── build_manifests.py      ← Phase 1: builds JSONL manifests from raw CSVs
│   │   └── label_maps.py           ← Normalizes labels to {fake, real, uncertain}
│   ├── text/
│   │   ├── dataset.py              ← Phase 2: PyTorch Dataset for JSONL text data
│   │   ├── train.py                ← Phase 2: RoBERTa fine-tuning trainer
│   │   └── explain.py              ← Phase 2: Integrated Gradients HTML heatmaps
│   ├── rag/
│   │   ├── claims.py               ← Phase 3: extracts factual claims from text
│   │   ├── retrieve.py             ← Phase 3: Serper API web search
│   │   ├── nli.py                  ← Phase 3: NLI stance scoring (supports/contradicts)
│   │   ├── aggregate.py            ← Phase 3: combines stance scores per post
│   │   └── run_rag.py              ← Phase 3: one-command pipeline orchestrator
│   ├── vlm/
│   │   ├── data_stage_b.py         ← Phase 4: Fakeddit multimodal dataset loader
│   │   ├── model_lora.py           ← Phase 4: CLIP-BERT backbone with LoRA
│   │   ├── train_lora.py           ← Phase 4: Stage B trainer
│   │   └── ocr.py                  ← Phase 4: PaddleOCR cache generator
│   ├── framing/
│   │   ├── sentiment.py            ← Phase 5: TextBlob sentiment & subjectivity
│   │   └── clickbait.py            ← Phase 5: rule-based clickbait scorer
│   ├── fusion/
│   │   ├── build_features.py       ← Phase 6: builds 11-feature CSV for training
│   │   ├── train_meta.py           ← Phase 6: trains XGBoost on fusion features
│   │   ├── calibrate.py            ← Phase 6: temperature scaling calibration
│   │   └── infer.py                ← Phase 6: real-time single-sample inference
│   └── xai/
│       ├── text.py                 ← Phase 7: token-level Integrated Gradients
│       ├── vision.py               ← Phase 7: Grad-CAM heatmap for VLM
│       ├── counterfactuals.py      ← Phase 7: what-if feature analysis
│       └── card_builder.py         ← Phase 7: builds structured evidence card JSON
│
├── scripts/                        ← CLI entry points
│   ├── make_data.py                ← Runs Phase 1 (calls build_manifests.py)
│   └── run_rag.py                  ← Runs Phase 3 (calls rag/run_rag.py)
│
├── notebooks/
│   ├── 04A_stageA_alignment_qwen2vl7b.ipynb  ← Phase 4 Stage A training
│   └── 04B_fakeddit_mismatch_qwen2vl7b.ipynb ← Phase 4 Stage B training
│
├── experiments/
│   └── eda.ipynb                   ← Exploratory data analysis
│
├── data/                           ← (gitignored — you must download manually)
│   ├── raw/
│   │   ├── liar/                   ← LIAR dataset TSV files
│   │   ├── fakeddit/               ← Fakeddit TSV + image folder
│   │   └── coco2017/               ← COCO images (Phase 4 Stage A only)
│   ├── processed/                  ← Generated by Phase 1
│   │   ├── liar/
│   │   └── fakeddit/
│   └── cache/
│       ├── ocr/                    ← Generated by Phase 4 OCR step
│       └── retrieval/              ← Generated by Phase 3 RAG step
│
├── artifacts/                      ← All trained models and output files
│   ├── text_bas/                   ← Phase 2 outputs (RoBERTa)
│   ├── rag/                        ← Phase 3 outputs (TSV previews)
│   ├── vlm/
│   │   ├── stage_a/                ← Phase 4 Stage A LoRA adapter
│   │   └── stage_b/
│   │       ├── adapter/            ← Phase 4 Stage B LoRA adapter weights
│   │       ├── processor/          ← Qwen2-VL processor/tokenizer files
│   │       ├── preds/              ← JSONL prediction outputs
│   │       └── release/
│   │           └── inference.py    ← Production VLM inference script
│   ├── fusion/
│   │   ├── xgb_fusion.json         ← Trained XGBoost model (required by app)
│   │   ├── features_*.csv          ← Feature matrices per split
│   │   └── calibration/
│   │       └── calibrator.json     ← Temperature scaling value (required by app)
│   ├── cards/                      ← Evidence card JSONs (generated at runtime)
│   └── xai/                        ← Grad-CAM images (generated at runtime)
│
└── tests/                          ← Unit tests for each phase
```

---

## 5. Core Files Reference

Below is a precise breakdown of every core file: what it does, what inputs it needs, and what it outputs.

---

### `app.py` — Streamlit Web App

**What it does:** The main user-facing application. Takes a news post as input, runs the analysis pipeline, and displays predictions with explanations.

| Input | Type | Description |
|-------|------|-------------|
| Post text | Text area | The news article or social media post to analyze (required) |
| Image | File upload | Optional JPG/PNG image attached to the post |
| Enable VLM | Checkbox | Whether to use the Vision-Language Model (needs GPU + VLM artifacts) |
| Show Debug Logs | Checkbox | Displays raw feature dict and fusion output JSON |

**Outputs displayed in browser:**
- Prediction label: `FAKE` / `REAL` / `UNCERTAIN` with confidence score
- Framing signals: sentiment polarity, subjectivity, clickbait score
- Root causes: human-readable explanation list
- Top-5 important tokens with attribution scores
- Counterfactual analysis JSON
- Downloadable `evidence_card.json`

**Artifacts needed to run app.py:**
- `artifacts/fusion/xgb_fusion.json`
- `artifacts/fusion/calibration/calibrator.json`
- `artifacts/text_bas/hf_model/` (for token XAI)

---

### `src/common/io.py` — Shared I/O Utilities

**What it does:** Provides helper functions used across all phases.

| Function | Input | Output |
|----------|-------|--------|
| `read_yaml(path)` | YAML file path | Python dict |
| `read_jsonl(path)` | JSONL file path | List of dicts |
| `write_jsonl(records, path)` | List of dicts + output path | Writes JSONL file atomically |
| `read_tsv(path)` | TSV file path | `pd.DataFrame` |
| `ensure_dir(path)` | Directory path | Creates directory if missing |

---

### `src/dataio/build_manifests.py` — Phase 1: Data Manifests

**What it does:** Converts raw dataset files into a standardized JSONL format that all downstream phases consume.

| Input | Type | Description |
|-------|------|-------------|
| `configs/data.yaml` | YAML config | Paths to raw files, split ratios (80/10/10), label mappings |
| `data/raw/liar/train.tsv` | TSV | LIAR train split (14 columns, column 2 = statement, column 1 = label) |
| `data/raw/liar/valid.tsv` | TSV | LIAR validation split |
| `data/raw/liar/test.tsv` | TSV | LIAR test split |
| `data/raw/fakeddit/subset_meta_ok.tsv` | TSV | Fakeddit metadata (id, title, label, image path) |
| `data/raw/fakeddit/images/` | Directory | Fakeddit image files (named by post id) |

**Output JSONL record format:**
```json
{
  "id": "unique-post-id",
  "text": "the news headline or post text",
  "image_path": "data/raw/fakeddit/images/abc123.jpg",
  "label": "fake",
  "meta": {"source": "fakeddit", "split": "train"}
}
```

**Output files:**
- `data/processed/liar/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/processed/fakeddit/train.jsonl`, `val.jsonl`, `test.jsonl`

---

### `src/dataio/label_maps.py` — Label Normalizer

**What it does:** Converts dataset-specific labels into the 3-class target set used by the entire project.

| Source Dataset | Raw Labels | Normalized To |
|----------------|------------|---------------|
| LIAR | `true`, `mostly-true` | `real` |
| LIAR | `pants-fire`, `false`, `barely-true` | `fake` |
| LIAR | `half-true` | `uncertain` |
| Fakeddit | `0` (real) | `real` |
| Fakeddit | `1` (fake) | `fake` |
| Fakeddit | `2` (no-factual-content) | `uncertain` |

---

### `src/text/train.py` — Phase 2: RoBERTa Text Classifier

**What it does:** Fine-tunes `roberta-base` on the LIAR dataset for 3-class fake news classification. Uses FP16 mixed precision, AdamW optimizer with linear warmup scheduler, and early stopping on validation macro-F1.

| Input | Type | Description |
|-------|------|-------------|
| `configs/text_baseline.yaml` | YAML config | All training hyperparameters |
| `data/processed/liar/train.jsonl` | JSONL | Training samples |
| `data/processed/liar/val.jsonl` | JSONL | Validation samples |
| `data/processed/liar/test.jsonl` | JSONL | Test samples |

**Key config options in `configs/text_baseline.yaml`:**

```yaml
model_name: roberta-base
max_length: 256
batch_size: 32
learning_rate: 2.0e-5
num_epochs: 3
output_dir: artifacts/text_bas
```

**Outputs:**
- `artifacts/text_bas/best.ckpt` — Best PyTorch checkpoint by val F1
- `artifacts/text_bas/hf_model/` — HuggingFace-compatible model + tokenizer (used by app)
- `artifacts/text_bas/metrics.json` — Accuracy, macro-F1 per split
- `artifacts/text_bas/confusion_matrix.png` — Confusion matrix plot
- `artifacts/text_bas/label_order.json` — Class index order
- `artifacts/text_bas/sample_explanations/*.html` — IG token heatmaps for 10 samples

---

### `src/text/explain.py` — Phase 2 Explainability

**What it does:** Generates Integrated Gradients (IG) token attribution HTML heatmaps using the Captum library. Shows which words the RoBERTa model focuses on when making a prediction.

| Input | Type | Description |
|-------|------|-------------|
| HF model | `nn.Module` | Loaded RoBERTa sequence classification model |
| Tokenizer | HF tokenizer | Corresponding RoBERTa tokenizer |
| Dataset | `JsonlTextDataset` | Phase 2 dataset object |
| Sample indices | List[int] | Which samples to explain |
| Output dir | Path | Where to save HTML files |

**Output:** HTML files with color-coded token importance scores.

---

### `src/rag/claims.py` — Phase 3: Claim Extraction

**What it does:** Extracts 1–3 short, atomic factual claims from a news post. Uses sentence splitting rules by default.

| Input | Type | Description |
|-------|------|-------------|
| `post_text` | str | The raw news post text |
| `configs/rag.yaml` | YAML config | Max claims, min sentence length |
| `llm_fn` (optional) | callable | Custom LLM function for better claim extraction |

**Output:** `List[str]` — a list of extracted claim strings

**Example:**
```python
from src.rag.claims import extract_claims
claims = extract_claims("The president signed a bill banning all imports from China.")
# → ["The president signed a bill.", "The bill bans all imports from China."]
```

---

### `src/rag/retrieve.py` — Phase 3: Web Evidence Retrieval

**What it does:** Calls the Serper API (Google Search JSON API) to retrieve top-K web snippets for a given claim.

| Input | Type | Description |
|-------|------|-------------|
| `claim` | str | A single factual claim string |
| `configs/rag.yaml` | YAML config | `top_k` (default 5), API key location |
| `SERPER_API_KEY` | env var | Your Serper API key (get free at serper.dev) |

**Output:** `List[dict]` — each dict has `{source, title, url, date, snippet, score}`

> **Important:** You must set the `SERPER_API_KEY` environment variable. The default hardcoded key in the source code is for demo only and may be expired.

---

### `src/rag/nli.py` — Phase 3: NLI Stance Scoring

**What it does:** Uses `roberta-large-mnli` to score how well each retrieved web snippet supports or contradicts a claim. Downloads the model automatically on first run (≈1.4 GB).

| Input | Type | Description |
|-------|------|-------------|
| `pairs` | List[Tuple[str, str]] | List of (claim, evidence_snippet) pairs |
| `configs/rag.yaml` | YAML config | Batch size, model name |

**Output:** `np.ndarray` of shape `[N, 3]` — columns are `[supports, neutral, contradicts]` probabilities

---

### `src/rag/aggregate.py` — Phase 3: Score Aggregation

**What it does:** Reads the stance TSV and computes per-post aggregate support/contradict scores plus top-K citations.

| Input | Type | Description |
|-------|------|-------------|
| `artifacts/rag/stance_preview.tsv` | TSV | Output from nli.py |
| `configs/rag.yaml` | YAML config | Aggregation thresholds |

**Outputs:**
- `data/cache/retrieval/retrieval_cache.jsonl` — Per-post evidence cache used by fusion
- `artifacts/rag/examples.tsv` — Human-readable summary

---

### `src/rag/run_rag.py` — Phase 3: One-Command Pipeline

**What it does:** Chains all four RAG steps (claims → retrieve → nli → aggregate) in order by calling each as a subprocess.

| Input | Description |
|-------|-------------|
| `--config configs/rag.yaml` | Config file path |
| `--limit N` | Process only first N posts (use small N for testing) |
| `--top_k N` | Number of web snippets per claim |
| `--stop_after step` | Stop after `claims`, `retrieve`, `nli`, or `aggregate` |

---

### `src/vlm/ocr.py` — Phase 4: OCR Cache Generator

**What it does:** Runs PaddleOCR on every image in a JSONL manifest and caches the extracted text to disk. This is run once before VLM training and inference to avoid repeated GPU work.

| Input | Type | Description |
|-------|------|-------------|
| `configs/ocr.yaml` | YAML config | OCR language, confidence threshold |
| Input JSONL | Path | JSONL with `image_path` field |
| `images_root` | Path | Base directory for relative image paths |
| `--max_count N` | CLI flag | Limit to N images (optional) |

**Output:** One JSON file per image at `data/cache/ocr/{split}/{item_id}.json`
```json
{"id": "abc123", "text": "BREAKING NEWS", "lines": ["BREAKING", "NEWS"], "meta": {}}
```

---

### `src/vlm/data_stage_b.py` — Phase 4: Fakeddit Dataset Loader

**What it does:** PyTorch Dataset class for loading Fakeddit multimodal samples. Joins images with OCR cache, builds structured prompts, and computes numeric features.

| Input | Type | Description |
|-------|------|-------------|
| JSONL path | Path | Fakeddit split manifest |
| `StageBConfig` | dataclass | VLM config with paths and thresholds |
| OCR root | Path | `data/cache/ocr/{split}/` |
| Images root | Path | `data/raw/fakeddit/images/` |

**Output per batch:** `{images, labels, prompts, ids, image_ok, features, raw}`

---

### `src/vlm/train_lora.py` — Phase 4: VLM Stage B Trainer

**What it does:** Trains a LoRA-adapted model for image-text mismatch detection on Fakeddit. Supports two modes: `dummy` (fast 4-feature MLP for CI/testing) and `real` (full CLIP-BERT backbone).

| Input | Type | Description |
|-------|------|-------------|
| `configs/vlm_stage_b_qwen2vl7b.yaml` | YAML | All training hyperparameters |
| `data/processed/fakeddit_stage_b/train.jsonl` | JSONL | Training split |
| `data/processed/fakeddit_stage_b/val.jsonl` | JSONL | Validation split |

**Key config settings:**
```yaml
base_model: Qwen/Qwen2-VL-7B-Instruct
lora_r: 32
use_qlora: true          # 4-bit quantization (needs CUDA GPU)
micro_batch_size: 2
gradient_accumulation: 16
num_epochs: 1
output_dir: artifacts/vlm/stage_b
```

**Outputs:**
- `artifacts/vlm/stage_b/best.ckpt`
- `artifacts/vlm/stage_b/last.ckpt`
- `artifacts/vlm/stage_b/metrics.json`

---

### `artifacts/vlm/stage_b/release/inference.py` — Phase 4: Production VLM Inference

**What it does:** Loads the trained Qwen2-VL-7B + LoRA adapter and runs first-token scoring over choices A (consistent), B (mismatched), C (uncertain). Uses entropy-based abstention with a tuned threshold `tau`.

| Input | Type | Description |
|-------|------|-------------|
| `image_path` | str | Path to the post image |
| `post_text` | str | The news post text |
| `ocr_text` | str | Pre-extracted OCR text from the image (can be `""`) |

**Output dict:**
```python
{
  "label": "consistent",          # or "mismatched" or "uncertain"
  "probs": {
    "consistent": 0.72,
    "mismatched": 0.19,
    "uncertain": 0.09
  },
  "entropy": 0.84,
  "tau": 1.1,
  "first_token_logprobs": {...}
}
```

**Artifacts needed:**
- `artifacts/vlm/stage_b/adapter/` — LoRA weights
- `artifacts/vlm/stage_b/processor/` — Tokenizer/processor files
- `artifacts/vlm/stage_b/release/tau.json` — Abstention threshold

---

### `src/framing/sentiment.py` — Phase 5: Sentiment Analysis

**What it does:** Uses TextBlob to compute sentiment polarity (positive/negative) and subjectivity (factual vs. opinionated) of a text string.

| Input | Type | Description |
|-------|------|-------------|
| `text` | str | Any text string |

**Output dict:**
```python
{
  "polarity_raw": -0.3,        # raw TextBlob value (-1 to 1)
  "sentiment": 0.35,           # remapped to [0, 1] (0=negative, 1=positive)
  "subjectivity": 0.72         # [0=factual, 1=highly subjective]
}
```

---

### `src/framing/clickbait.py` — Phase 5: Clickbait Scoring

**What it does:** Detects clickbait characteristics using rule-based heuristics: excessive capitalization, punctuation abuse, superlatives, question formats, and known clickbait phrases.

| Input | Type | Description |
|-------|------|-------------|
| `text` | str | Any text string |

**Output dict:**
```python
{
  "clickbait": 0.65,       # composite score [0, 1]
  "caps_ratio": 0.4,       # fraction of CAPS words
  "excess_punct": 0.2,     # score for !!! or ??? patterns
  "phrases": 0.5,          # matched clickbait phrases score
  "superlatives": 0.3,     # words like "best", "worst", "never"
  "question": 1.0          # 1 if text ends with "?"
}
```

---

### `src/fusion/build_features.py` — Phase 6: Feature Matrix Builder

**What it does:** Joins all upstream signals into a single 11-column feature CSV for XGBoost training. Reads from LIAR manifests, text classifier predictions, RAG outputs, and VLM predictions.

| Input File | Description |
|------------|-------------|
| `data/processed/liar/{train,val,test}.jsonl` | Gold labels |
| `artifacts/text_baseline/preds_{val,test}.jsonl` | RoBERTa softmax scores |
| `artifacts/rag/rag_outputs.jsonl` | Per-post RAG support/contradict scores |
| `artifacts/vlm/stage_b/preds/preds_*_cal.jsonl` | VLM consistency/entropy scores |

**Output columns in `artifacts/fusion/features_{split}.csv`:**

| Feature | Description |
|---------|-------------|
| `text_cls_score` | RoBERTa fake-class probability |
| `rag_support` | Fraction of evidence that supports the claim |
| `rag_contradict` | Fraction of evidence that contradicts the claim |
| `vlm_consistency` | VLM image-text consistency score |
| `vlm_entropy` | VLM prediction entropy (uncertainty) |
| `ocr_match` | Overlap between headline and OCR text |
| `sentiment` | TextBlob sentiment [0, 1] |
| `subjectivity` | TextBlob subjectivity [0, 1] |
| `clickbait` | Clickbait score [0, 1] |
| `text_length_norm` | Post length normalized to [0, 1] |
| `has_image` | Binary: 1 if image present, 0 otherwise |

---

### `src/fusion/train_meta.py` — Phase 6: XGBoost Meta-Learner

**What it does:** Trains a 3-class XGBoost classifier on the 11 fusion features. Evaluates on the validation set and exports probability scores for calibration.

| Input | Type | Description |
|-------|------|-------------|
| `artifacts/fusion/features_full.csv` | CSV | Combined train+val feature matrix |

**Outputs:**
- `artifacts/fusion/xgb_fusion.json` — Saved XGBoost model **(required by app.py)**
- `artifacts/fusion/feature_importance.csv` — Feature gain scores
- `artifacts/fusion/val_probs.csv` — Val set predicted probabilities (input to calibrate.py)

---

### `src/fusion/calibrate.py` — Phase 6: Temperature Calibration

**What it does:** Performs temperature scaling on the XGBoost output probabilities to make confidence scores better calibrated (i.e., a 70% confidence should mean it is correct ~70% of the time).

| Input | Type | Description |
|-------|------|-------------|
| `artifacts/fusion/val_probs.csv` | CSV | Validation predicted probabilities from XGBoost |

**Outputs:**
- `artifacts/fusion/calibration/calibrator.json` — `{"temperature": 1.23}` **(required by app.py)**
- `artifacts/fusion/calibration/reliability_curve.csv` — Calibration curve data
- `artifacts/fusion/calibration/calibration_report.txt` — Pre/post calibration comparison

---

### `src/fusion/infer.py` — Phase 6: Real-Time Inference

**What it does:** Loads the trained XGBoost + temperature calibrator and runs a single feature dict through the full prediction pipeline. Used live in app.py.

| Input | Type | Description |
|-------|------|-------------|
| `features` | dict | Dict with all 11 feature keys (see table above) |

**Output dict:**
```python
{
  "label": "fake",                 # predicted class
  "probs": {
    "fake": 0.78,
    "real": 0.12,
    "uncertain": 0.10
  },
  "confidence": 0.78               # max probability
}
```

**Artifacts loaded at import time:**
- `artifacts/fusion/xgb_fusion.json`
- `artifacts/fusion/calibration/calibrator.json`

---

### `src/xai/text.py` — Phase 7: Token Attribution

**What it does:** Uses Captum's Layer Integrated Gradients to compute per-token importance scores explaining why the text model made its prediction. Returns a ranked list of (token, score) pairs.

| Input | Type | Description |
|-------|------|-------------|
| `text` | str | The news post text to explain |
| `target_label` | str (optional) | Which label to explain (default: model's top prediction) |

**Output:** `List[Tuple[str, float]]` — tokens sorted by attribution magnitude

**Model loaded at import:** `artifacts/text_bas/hf_model/`

---

### `src/xai/vision.py` — Phase 7: Grad-CAM Heatmaps

**What it does:** Generates Gradient-weighted Class Activation Maps (Grad-CAM) to highlight which image regions the VLM focuses on. Saves a color overlay PNG.

| Input | Type | Description |
|-------|------|-------------|
| `image_path` | str | Path to the input image |
| `prompt` | str | Text prompt used for VLM inference |
| `model` | PEFT model | Loaded Qwen2-VL + LoRA adapter |
| `output_path` | str | Where to save the overlay PNG |

**Output:** PNG file at `artifacts/xai/gradcam_{id}.png`

---

### `src/xai/counterfactuals.py` — Phase 7: What-If Analysis

**What it does:** Simulates three counterfactual scenarios by modifying the feature vector and re-running the fusion model. Reports whether the prediction label changes.

| Input | Type | Description |
|-------|------|-------------|
| `features` | dict | The original 11-feature dict |

**Three scenarios tested:**

| Scenario | What Changes | Question Asked |
|----------|-------------|----------------|
| `remove_sensational` | Sets `clickbait=0, sentiment=0.5, subjectivity=0` | Would prediction change if writing style were neutral? |
| `remove_contradiction` | Sets `rag_contradict=0, rag_support=0.5` | Would prediction change if web evidence were neutral? |
| `remove_image_signal` | Sets `vlm_consistency=0.5, vlm_entropy=1.0, has_image=0` | Would prediction change without image analysis? |

**Output dict:**
```python
{
  "original": {"label": "fake", "probs": {...}, "confidence": 0.78},
  "counterfactuals": {
    "remove_sensational": {"label": "uncertain", "flipped": true, ...},
    "remove_contradiction": {"label": "fake", "flipped": false, ...},
    "remove_image_signal": {"label": "uncertain", "flipped": true, ...}
  }
}
```

---

### `src/xai/card_builder.py` — Phase 7: Evidence Card Builder

**What it does:** Aggregates all signals (fusion prediction, framing signals, token XAI, counterfactuals, Grad-CAM path) into a single structured JSON evidence card. Also generates human-readable root cause strings.

| Input | Type | Description |
|-------|------|-------------|
| `sample` | dict | Must have keys: `id`, `image_path`, `post_text`, `features` |

**Output dict (also saved as JSON):**
```json
{
  "id": "post_id",
  "prediction": {"label": "fake", "confidence": 0.78, "probs": {...}},
  "framing": {"sentiment": 0.35, "subjectivity": 0.72, "clickbait": 0.65},
  "root_causes": ["High clickbait score detected", "Strong contradicting evidence found"],
  "token_attributions": [["shocking", 0.91], ["claim", 0.74]],
  "counterfactuals": {...},
  "gradcam_path": "artifacts/xai/gradcam_post_id.png",
  "timestamp": "2024-01-01T12:00:00"
}
```

**Saved to:** `artifacts/cards/{id}_card.json`

---

## 6. Pipeline — Phase by Phase

This diagram shows the **exact sequence** you must follow to train the system from scratch:

```
STEP 1 ── scripts/make_data.py ──────────────────────────────────────
          Input:  data/raw/liar/*.tsv  +  data/raw/fakeddit/*
          Output: data/processed/liar/*.jsonl
                  data/processed/fakeddit/*.jsonl

STEP 2 ── src/text/train.py ─────────────────────────────────────────
          Input:  data/processed/liar/*.jsonl
          Output: artifacts/text_bas/hf_model/
                  artifacts/text_bas/best.ckpt

STEP 3a ── src/vlm/ocr.py ───────────────────────────────────────────
           Input:  data/processed/fakeddit/train.jsonl + images
           Output: data/cache/ocr/train/{id}.json

STEP 3b ── notebooks/04A (Jupyter) ──────────────────────────────────
           Input:  data/raw/coco2017/ + Qwen2-VL-7B base
           Output: artifacts/vlm/stage_a/ (LoRA adapter)

STEP 3c ── notebooks/04B (Jupyter) ──────────────────────────────────
           Input:  data/processed/fakeddit_stage_b/*.jsonl
                   data/cache/ocr/
                   artifacts/vlm/stage_a/ (warm-start adapter)
           Output: artifacts/vlm/stage_b/adapter/
                   artifacts/vlm/stage_b/processor/

STEP 4 ── scripts/run_rag.py ────────────────────────────────────────
          Input:  data/processed/liar/val.jsonl
          Output: data/cache/retrieval/retrieval_cache.jsonl
                  artifacts/rag/stance_preview.tsv

STEP 5 ── src/fusion/build_features.py ──────────────────────────────
          Input:  data/processed/liar/*.jsonl
                  artifacts/text_bas/preds_*.jsonl    (from Step 2)
                  artifacts/rag/rag_outputs.jsonl     (from Step 4)
                  artifacts/vlm/stage_b/preds/*.jsonl (from Step 3c)
          Output: artifacts/fusion/features_full.csv

STEP 6 ── src/fusion/train_meta.py ──────────────────────────────────
          Input:  artifacts/fusion/features_full.csv
          Output: artifacts/fusion/xgb_fusion.json
                  artifacts/fusion/val_probs.csv

STEP 7 ── src/fusion/calibrate.py ───────────────────────────────────
          Input:  artifacts/fusion/val_probs.csv
          Output: artifacts/fusion/calibration/calibrator.json

DONE ──── streamlit run app.py ──────────────────────────────────────
          Needs:  artifacts/fusion/xgb_fusion.json         ✓
                  artifacts/fusion/calibration/calibrator.json ✓
                  artifacts/text_bas/hf_model/             ✓
                  artifacts/vlm/stage_b/ (optional, for VLM) ✓
```

---

## 7. Installation & Setup

### Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU with ≥ 24 GB VRAM (required for Phase 4 VLM training; Phases 2, 5, 6, 7 can run on CPU)
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/AI-NLP-FAKE-NEWS-DETECTION.git
cd AI-NLP-FAKE-NEWS-DETECTION
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes xgboost captum
pip install textblob streamlit scikit-learn pandas numpy PyYAML requests Pillow matplotlib tqdm
pip install sentence-transformers opencv-python
pip install paddlepaddle-gpu paddleocr     # GPU version; use paddlepaddle for CPU-only
```

> **Note:** A `requirements.txt` is not currently provided. Install libraries as listed above.

### Step 4 — Download NLTK data (needed by TextBlob)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Step 5 — Set your Serper API key (Phase 3 only)

```bash
export SERPER_API_KEY="your-key-here"
```

Get a free key at [serper.dev](https://serper.dev). Without this, Phase 3 (RAG) will fail.

---

## 8. Data Preparation

### LIAR Dataset

1. Download from [https://www.cs.ucsb.edu/~william/data/liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
2. Extract and place files:
   ```
   data/raw/liar/train.tsv
   data/raw/liar/valid.tsv
   data/raw/liar/test.tsv
   ```

### Fakeddit Dataset

1. Download from [https://github.com/entitize/Fakeddit](https://github.com/entitize/Fakeddit)
2. Place metadata file at:
   ```
   data/raw/fakeddit/subset_meta_ok.tsv
   ```
3. Download images and place at:
   ```
   data/raw/fakeddit/images/{post_id}.jpg
   ```

### COCO 2017 (Phase 4 Stage A only)

1. Download from [https://cocodataset.org/#download](https://cocodataset.org/#download)
2. Place images and annotations at:
   ```
   data/raw/coco2017/images/
   data/raw/coco2017/annotations/
   ```

---

## 9. Running Each Phase

### Phase 1 — Build data manifests

```bash
python scripts/make_data.py --config configs/data.yaml
# To process only one dataset:
python scripts/make_data.py --config configs/data.yaml --only liar
python scripts/make_data.py --config configs/data.yaml --only fakeddit
```

### Phase 2 — Train text classifier

```bash
python -m src.text.train --config configs/text_baseline.yaml
```

### Phase 3 — Run RAG pipeline

```bash
# Full pipeline (slow — makes real web search API calls)
python scripts/run_rag.py --config configs/rag.yaml

# Test with only 10 posts, stop after claim extraction
python scripts/run_rag.py --config configs/rag.yaml --limit 10 --stop_after claims
```

### Phase 4 — VLM (requires GPU ≥ 24 GB)

```bash
# Step 1: Generate OCR cache
python -m src.vlm.ocr --config configs/ocr.yaml --split train

# Step 2: Stage A training (open notebook in Jupyter)
jupyter notebook notebooks/04A_stageA_alignment_qwen2vl7b.ipynb

# Step 3: Stage B training (open notebook in Jupyter)
jupyter notebook notebooks/04B_fakeddit_mismatch_qwen2vl7b.ipynb
```

### Phase 5 — Framing signals (no separate script — runs inside app.py)

```python
# Test manually:
from src.framing.sentiment import analyze_sentiment
from src.framing.clickbait import analyze_clickbait

print(analyze_sentiment("SHOCKING: Government LIES about everything!!!"))
print(analyze_clickbait("You WON'T BELIEVE what happened next!!!"))
```

### Phase 6 — Train fusion model

```bash
# Build feature matrix (requires artifacts from Phases 2, 3, 4)
python -m src.fusion.build_features

# Train XGBoost
python -m src.fusion.train_meta

# Calibrate probabilities
python -m src.fusion.calibrate
```

### Phase 7 — Explainability (runs automatically in app.py)

```python
# Test manually:
from src.xai.counterfactuals import run_counterfactuals
features = {
    "text_cls_score": 0.8, "rag_support": 0.2, "rag_contradict": 0.7,
    "vlm_consistency": 0.3, "vlm_entropy": 1.5, "ocr_match": 0.1,
    "sentiment": 0.2, "subjectivity": 0.8, "clickbait": 0.9,
    "text_length_norm": 0.5, "has_image": 1
}
print(run_counterfactuals(features))
```

---

## 10. Running the Streamlit App

The app requires the XGBoost model and calibration file to be present before starting.

**Minimum required artifacts:**
```
artifacts/fusion/xgb_fusion.json
artifacts/fusion/calibration/calibrator.json
artifacts/text_bas/hf_model/
```

**Start the app:**
```bash
streamlit run app.py
```

The browser will open at `http://localhost:8501`.

**How to use the app:**
1. (Optional) Upload a JPG/PNG image in the left column
2. Enter the news post text in the right column
3. Click **Analyze**
4. View the prediction, framing signals, root causes, and token importance
5. Download the evidence card JSON using the download button

**Sidebar options:**
- **Show Debug Logs** — Displays raw feature values and fusion model output
- **Enable Vision Model (Phase 4)** — Uses the Qwen2-VL-7B model for image analysis (requires VLM artifacts and GPU)

---

## 11. Artifacts Produced

After running all phases, the `artifacts/` directory contains:

```
artifacts/
├── text_bas/
│   ├── hf_model/               ← RoBERTa model (used by app.py for XAI)
│   ├── best.ckpt               ← Best PyTorch checkpoint
│   ├── metrics.json            ← Test accuracy + F1 scores
│   ├── confusion_matrix.png    ← Visual confusion matrix
│   └── sample_explanations/    ← 10 HTML token heatmap files
│
├── rag/
│   ├── claims_preview.tsv      ← Extracted claims sample
│   ├── retrieval_preview.tsv   ← Web snippets sample
│   ├── stance_preview.tsv      ← NLI stance scores sample
│   └── examples.tsv            ← Human-readable RAG examples
│
├── vlm/stage_b/
│   ├── adapter/                ← Qwen2-VL-7B LoRA weights
│   ├── processor/              ← Tokenizer/processor files
│   └── preds/                  ← JSONL prediction files per split
│
├── fusion/
│   ├── xgb_fusion.json         ← Trained XGBoost model ← app.py needs this
│   ├── features_*.csv          ← Feature matrices
│   ├── feature_importance.csv  ← Top features by gain
│   └── calibration/
│       ├── calibrator.json     ← Temperature value ← app.py needs this
│       └── reliability_curve.csv
│
├── cards/                      ← Evidence card JSONs (generated at runtime)
├── xai/                        ← Grad-CAM PNG overlays (generated at runtime)
└── logs/
    └── app.log                 ← Streamlit app log
```

---

## 12. Known Issues & Limitations

| Issue | Location | Impact | Status |
|-------|----------|--------|--------|
| `explain_text_tokens` imported but function is named `explain_text` | `app.py:29` vs `src/xai/text.py` | App crashes when Analyze is clicked | Bug — rename function in `src/xai/text.py` |
| `AutoTokenizer` loads at import time | `src/xai/text.py` top-level | Import fails if `artifacts/text_bas/hf_model/` is missing | Expected — run Phase 2 first |
| Serper API key hardcoded in source | `src/rag/retrieve.py:64` | Security risk — key exposed in git history | Use env var `SERPER_API_KEY` instead |
| Hardcoded absolute path `/teamspace/studios/this_studio` | `src/fusion/build_features.py` | Breaks on any other machine | Change to relative path using `Path(__file__)` |
| 7 script files are empty stubs | `scripts/train_text.py`, etc. | Cannot run these scripts | Use module-level commands (`python -m src.text.train`) |
| `src/fusion/aggregate_signals.py`, `src/vlm/infer.py`, `src/vlm/cams.py` | Various | These modules raise `NotImplementedError` | Placeholder — not used in main pipeline |
| Phase 4 VLM requires large GPU | `notebooks/04B` | Cannot run on CPU or small GPU | Use Google Colab Pro+ or A100 instance |
| No `requirements.txt` | Root directory | Manual dependency installation | Create from pip freeze after setup |

---

## 13. FAQ for Beginners

**Q: I just want to run the demo. What is the minimum setup?**

You need the trained artifacts. If they are already present in `artifacts/fusion/` and `artifacts/text_bas/hf_model/`, run:
```bash
pip install streamlit transformers torch xgboost textblob captum Pillow
streamlit run app.py
```

---

**Q: What is a JSONL file?**

JSONL (JSON Lines) is a text file where each line is a valid JSON object. Example:
```
{"id": "001", "text": "Vaccine causes autism", "label": "fake"}
{"id": "002", "text": "Earth orbits the Sun", "label": "real"}
```
This format is used throughout the project for datasets.

---

**Q: What does "Phase 1 must run before Phase 2" mean?**

Phase 2 reads files that Phase 1 creates. If you skip Phase 1, Phase 2 will crash with a `FileNotFoundError` because `data/processed/liar/train.jsonl` does not exist yet. Always follow the numbered phase order.

---

**Q: Do I need a GPU?**

- **Phase 2 (RoBERTa training):** Runs on CPU but is slow. A GPU speeds it up significantly.
- **Phase 4 (VLM training):** Requires a CUDA GPU with at least 24 GB VRAM (e.g., A100, H100). Cannot run on CPU.
- **App (app.py):** Runs on CPU if VLM is disabled. Enable VLM only if you have a compatible GPU.

---

**Q: The app crashes with `ImportError: cannot import name 'explain_text_tokens'`. How do I fix it?**

This is a known bug. Open `src/xai/text.py` and rename the function from `explain_text` to `explain_text_tokens`. Then restart the app.

---

**Q: What are `artifacts/` — can I delete them?**

No. The `artifacts/` folder contains trained model weights and configuration files that the app requires to run. Deleting it means you must retrain all phases from scratch. Only delete artifacts if you intend to retrain.

---

**Q: What is temperature calibration in Phase 6?**

XGBoost outputs raw probabilities that are sometimes overconfident. Temperature scaling divides the raw logits by a constant `T` before the softmax. If `T > 1`, probabilities become less extreme; if `T < 1`, they become more extreme. The optimal `T` is found by minimizing log-loss on the validation set.

---

**Q: How is the final prediction made?**

The app computes 11 numeric features from the post (framing, VLM scores, etc.) and feeds them to the trained XGBoost classifier. XGBoost outputs class probabilities for `{fake, real, uncertain}`. These are temperature-scaled and the highest probability class is returned as the prediction.

---

**Q: Where do I get a Serper API key?**

Go to [serper.dev](https://serper.dev), create a free account, and copy your API key. Set it as an environment variable:
```bash
export SERPER_API_KEY="your-key-here"
```
The free tier provides 2,500 searches per month.

---

## License

See [LICENSE](LICENSE) for details.
