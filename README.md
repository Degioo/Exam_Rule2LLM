# Rule2LLM — Rule-Guided LLMs for Administrative Compliance Checking

> **NLP Exam Project** · University of Milan (La Statale) · A.Y. 2025–26

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Research Question

> *Can Large Language Models, when explicitly provided with formal administrative rules expressed in natural language, outperform traditional rule-based and semantic baselines in compliance checking of noisy, real-world healthcare documents?*

---

## Task Definition

**Compliance checking** is the task of deciding, for each pair *(document, rule)*, whether the document satisfies or violates the administrative rule. This project focuses on Italian public-healthcare prescription documents (OCR-extracted text), which are often noisy and ambiguous.

Each rule defines a required field or linguistic condition (e.g., "expiry date must be present", "anti-doping warning must appear"). A system assigns one of three labels:

| Label | Meaning |
|-------|---------|
| `OK` | Document satisfies the rule |
| `VIOLATED` | Document fails to satisfy the rule |
| `UNCERTAIN` | System abstains (used only by LLM-based approaches) |

---

## Repository Structure

```
Rule2LLM/
├── data/
│   ├── docs/               # 15 OCR-extracted Italian prescription documents (DOC_001–DOC_015)
│   └── gold_labels.json    # Ground-truth compliance labels for all documents × rules
├── models/                 # Pre-trained model binaries (NOT committed — see Setup)
├── notebooks/
│   └── 01_experiments.ipynb  # Main experiment notebook — run this to reproduce results
├── paper/                  # Manuscript / report (if present)
├── prompts/
│   └── rule2llm_prompt.txt   # System prompt for the LLM compliance-checking pipeline
├── results/
│   ├── run_results.csv     # Per-prediction results (model, doc, rule, status, evidence)
│   └── summary.csv         # Macro-averaged precision / recall / F1 per model
├── rules/
│   └── rules.json          # 8 administrative rules with IDs and natural-language descriptions
├── src/
│   ├── baselines/
│   │   ├── fasttext_05a.py     # FastText semantic-similarity baseline (Rule 05A only)
│   │   ├── logistic_baseline.py  # TF-IDF + Logistic Regression baseline
│   │   └── tfidf_ir.py         # TF-IDF Information-Retrieval baseline
│   ├── common/
│   │   ├── io.py           # Data loading utilities (docs, rules, gold labels)
│   │   └── schema.py       # Dataclasses: Document, Rule, Prediction, RuleStatus
│   ├── eval/
│   │   └── metrics.py      # Precision, Recall, F1, Coverage computation
│   └── llm/
│       ├── base.py             # Abstract LLM client interface
│       ├── ollama_client.py    # Ollama (local) LLM client
│       └── openai_client.py    # OpenAI API client
├── requirements.txt
└── README.md
```

---

## Dataset

- **15 documents** (`DOC_001` – `DOC_015`): real-world Italian public-healthcare prescriptions, converted from scanned PDFs via OCR. Documents range from clean to heavily noisy and include adversarial edge cases designed to challenge each rule.
- **8 administrative rules** defined in `rules/rules.json`, covering: expiry dates, doctor signatures, pharmacy send dates, special-need statements (05A), ATC codes, dosage/frequency, anti-doping warnings, and cost fields.
- **Gold labels** in `data/gold_labels.json`: manually annotated ground truth for every (document, rule) pair.

---

## Baselines Implemented

### 1. TF-IDF Information Retrieval (`TFIDF-IR`)
Computes cosine similarity between the TF-IDF representation of a document and the textual description of each rule. A document is classified as `OK` if similarity exceeds a threshold (default: 0.10), `VIOLATED` otherwise.

**Limitation:** treats the rule description as a query; entirely insensitive to negation or document structure.

### 2. TF-IDF + Logistic Regression (`TFIDF+LogReg`)
Trains one binary classifier per rule on TF-IDF bigram features using the available labeled documents (closed-book, leave-none-out setting). With only 15 training examples this is deliberately illustrative rather than performance-oriented.

**Limitation:** very small data size makes generalisation unreliable; acts as an upper-bound sanity check.

### 3. FastText Semantic Similarity (`fastText-sim-05A`)
Targets **Rule 05A** ("special needs / resistance to therapy") specifically. For each document token, computes max cosine similarity against a set of conceptual keywords (e.g., *refrattario*, *resistente*) using pre-trained Italian FastText vectors (`cc.it.300.bin`). Classified as `OK` if the best token-level similarity exceeds 0.55.

**Limitation:** cannot handle negation — the phrase "non è refrattario" would score high despite meaning the opposite.

### 4. Rule2LLM (LLM-based, main contribution)
Uses a prompted LLM (Ollama local or OpenAI API) that receives both the document text and the full rule description. The model reasons about compliance and outputs a structured verdict (`OK` / `VIOLATED` / `UNCERTAIN`) along with a natural-language evidence snippet.

---

## Evaluation Metrics

For each (model, rule) pair:

| Metric | Formula |
|--------|---------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-score** | 2 · P · R / (P + R) |
| **Coverage** | Fraction of non-UNCERTAIN predictions |

Macro-averaged scores across all rules are reported in `results/summary.csv`.

---

## Setup & Reproduction

### Prerequisites

- Python ≥ 3.9
- (Optional) [Ollama](https://ollama.com/) for local LLM inference
- (Optional) OpenAI API key for GPT-based inference

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FastText Italian Vectors

The FastText baseline requires the pre-trained Italian binary vectors (~4.5 GB compressed, ~7 GB unpacked). They are **not** committed to this repo.

```bash
# Create the models directory
mkdir -p models

# Download (PowerShell)
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz" `
    -OutFile "models/cc.it.300.bin.gz"

# Extract with 7-Zip (Windows) or gunzip (Linux/macOS)
& "C:\Program Files\7-Zip\7z.exe" x "models/cc.it.300.bin.gz" -o"models"
# -- OR --
gunzip models/cc.it.300.bin.gz
```

Place the unpacked `cc.it.300.bin` in `models/`.

### 3. (Optional) Configure LLM Backend

**Ollama (local):**
```bash
ollama pull mistral   # or any other model
# Ollama listens on http://localhost:11434 by default
```

**OpenAI:**
```bash
export OPENAI_API_KEY="sk-..."   # Linux/macOS
$env:OPENAI_API_KEY="sk-..."     # PowerShell
```

### 4. Run Experiments

```bash
jupyter notebook notebooks/01_experiments.ipynb
```

Run all cells top-to-bottom. Results are saved automatically to `results/run_results.csv` and `results/summary.csv`.

---

## Results Overview

| Model | Macro-P | Macro-R | Macro-F1 | Notes |
|-------|---------|---------|----------|-------|
| TFIDF-IR | 0.240 | 0.750 | 0.337 | High recall, low precision — classifies most docs as OK |
| TFIDF+LogReg | 0.250 | 0.250 | 0.250 | Illustrative only — too few training examples |
| fastText-sim-05A | 0.750 | 0.600 | 0.667 | Rule 05A only — best single-rule baseline |
| Rule2LLM (LLM) | — | — | — | Requires local Ollama or OpenAI API key |

*Full per-rule breakdown available in [`results/summary.csv`](results/summary.csv).*

---

## AI Usage Disclaimer

Parts of this project were developed with the assistance of AI tools (OpenAI GPT-4, Google Gemini) for ideation, boilerplate code generation, and documentation drafting. All outputs were reviewed, validated, and integrated by the author, who takes full responsibility for the final content.

---

## License

This project is released under the [MIT License](LICENSE) for academic purposes.
