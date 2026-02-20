# Rule2LLM — Rule-Guided LLMs for Administrative Compliance Checking

> **NLP Exam Project** · University of Milan (La Statale) · A.Y. 2025–26

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
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
| `UNCERTAIN` | System abstains (LLM-only; used when rule cannot be evaluated) |

---

## Repository Structure

```
Rule2LLM/
├── configs/
│   └── models.yaml             # Model matrix — all evaluation runs
├── data/
│   ├── docs/                   # 15 OCR-extracted Italian prescription documents
│   ├── gold_labels.json        # Ground-truth compliance labels
│   └── rules.json              # 8 administrative rules (canonical path)
├── models/                     # Pre-trained binaries (NOT committed — see Setup)
├── notebooks/
│   └── 01_experiments.ipynb    # Main experiment notebook
├── prompts/
│   └── rule2llm_prompt.txt     # Legacy flat prompt (superseded by src/llm/prompt.py)
├── results/
│   ├── predictions/            # Per-model JSONL outputs (not committed)
│   └── metrics/                # Per-rule & overall CSV results
├── rules/                      # Original rules location (rules.json also in data/)
├── scripts/
│   └── run_all_models.py       # Run all models → aggregate summary.csv
├── src/
│   ├── baselines/
│   │   ├── fasttext_05a.py
│   │   ├── logistic_baseline.py
│   │   └── tfidf_ir.py
│   ├── common/
│   │   ├── io.py               # Data loaders (docs, rules, gold)
│   │   └── schema.py           # Dataclasses: Document, Rule, Prediction, RuleStatus
│   ├── eval/
│   │   ├── eval_llm.py         # LLM evaluation CLI
│   │   └── metrics.py          # P/R/F1 computation
│   └── llm/
│       ├── base.py             # Abstract LLMClient interface
│       ├── prompt.py           # Prompt builder (system + user messages)
│       ├── runner.py           # Main CLI runner
│       ├── schema.py           # JSON output schema + validation + retry
│       └── providers/
│           ├── __init__.py     # Provider factory (get_provider)
│           ├── openai_provider.py
│           └── ollama_provider.py
├── .env.example                # Secrets template — copy to .env
├── .gitignore
├── environment.yml             # Conda environment (Python 3.10)
├── requirements.txt
└── README.md
```

---

## Dataset

- **15 documents** (`DOC_001` – `DOC_015`): real-world Italian public-healthcare prescriptions converted from scanned PDFs via OCR. Documents range from clean to heavily noisy.
- **8 administrative rules** in `data/rules.json`: expiry dates, doctor signatures, pharmacy send dates, special-need statements (05A), ATC codes, dosage/frequency, anti-doping warnings, cost fields.
- **Gold labels** in `data/gold_labels.json`: manually annotated ground truth for every (document, rule) pair.

---

## Baselines

| Model | Description |
|-------|-------------|
| `TFIDF-IR` | Cosine similarity between doc and rule TF-IDF vectors |
| `TFIDF+LogReg` | One binary classifier per rule on TF-IDF bigrams |
| `fastText-sim-05A` | FastText semantic similarity; Rule 05A only |
| `Rule2LLM` | Prompted LLM (OpenAI / Ollama) with structured JSON output |

---

## Setup & Reproduction

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate rule2llm
```

Or with pip only:
```bash
pip install -r requirements.txt
```

### 2. Configure Secrets

```bash
# Copy the template:
cp .env.example .env       # Linux/macOS
copy .env.example .env     # Windows

# Edit .env and fill in your values:
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest
RUN_TEMPERATURE=0
RUN_SEED=42
MAX_TOKENS=800
```

> ⚠️ **Never commit `.env` to git.** It is already listed in `.gitignore`.

### 3. Set Up Ollama (for local models)

```bash
# 1. Install Ollama from https://ollama.com/

# 2. Pull the models you want to evaluate:
ollama pull mistral:latest
ollama pull llama3:8b-instruct
ollama pull qwen2.5:7b-instruct

# 3. Start the Ollama server (runs at http://localhost:11434 by default):
ollama serve
```

### 4. (Optional) Download FastText Vectors — for baseline only

The Rule 05A FastText baseline requires the Italian binary vectors (~7 GB unpacked):

```powershell
# Windows PowerShell:
mkdir models
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz" `
    -OutFile "models/cc.it.300.bin.gz"
& "C:\Program Files\7-Zip\7z.exe" x "models/cc.it.300.bin.gz" -o"models"
```

---

## Running Evaluation

### a) Single model — OpenAI

```bash
python -m src.llm.runner \
    --provider openai \
    --model gpt-4.1 \
    --docs data/docs \
    --rules data/rules.json \
    --out results/predictions/openai_gpt41.jsonl
```

### b) Single model — Ollama (local)

```bash
python -m src.llm.runner \
    --provider ollama \
    --model mistral:latest \
    --docs data/docs \
    --rules data/rules.json \
    --out results/predictions/ollama_mistral.jsonl
```

**Runner options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | required | `openai` or `ollama` |
| `--model` | from `.env` | Override model name |
| `--docs` | `data/docs` | Directory of `.txt` documents |
| `--rules` | `data/rules.json` | Rules JSON file |
| `--out` | auto | Output `.jsonl` file |
| `--max-retries` | `2` | Format-fix retries per doc |
| `--temperature` | from `.env` | Generation temperature |
| `--dry-run` | off | Test loading without API calls |

### c) Evaluate predictions

```bash
python -m src.eval.eval_llm \
    --predictions results/predictions/ollama_mistral.jsonl \
    --gold data/gold_labels.json \
    --out-dir results/metrics \
    --model-tag ollama_mistral
```

Outputs:
- `results/metrics/ollama_mistral_per_rule.csv`
- `results/metrics/ollama_mistral_overall.csv`

### d) Run all models and produce summary.csv

```bash
# Full run (requires API key / Ollama running):
python scripts/run_all_models.py

# Dry-run (no API calls):
python scripts/run_all_models.py --dry-run

# Only evaluate existing predictions:
python scripts/run_all_models.py --eval-only
```

Aggregated results → `results/metrics/summary.csv`

---

## Evaluation Metrics

For each (model, rule) pair:

| Metric | Formula |
|--------|---------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-score** | 2·P·R / (P+R) |
| **Coverage** | Fraction of non-UNCERTAIN predictions |
| **Abstention Rate** | Fraction of UNCERTAIN predictions |

Macro-averaged scores across all rules in `results/metrics/summary.csv`.

---

## Results Overview

| Model | Macro-P | Macro-R | Macro-F1 | Abstention | Notes |
|-------|---------|---------|----------|------------|-------|
| TFIDF-IR | 0.240 | 0.750 | 0.337 | — | High recall, low precision |
| TFIDF+LogReg | 0.250 | 0.250 | 0.250 | — | Illustrative only (tiny dataset) |
| fastText-sim-05A | 0.750 | 0.600 | 0.667 | — | Rule 05A only |
| **Rule2LLM (OpenAI gpt-4.1)** | **0.613** | **0.729** | **0.633** | **0.0%** | Best overall; 15/15 docs processed |
| Rule2LLM (Ollama mistral:latest) | 0.065 | 0.030 | 0.037 | 8.9% | Hallucinated rule IDs — see analysis |

> **Analysis note — Mistral instruction-following failure:** Mistral returned its own rule IDs (`R1`–`R11`, `rule_1`–`rule_12`) instead of the canonical identifiers (`01`, `03`, `05A` …) specified in the prompt. Evaluation correctly matched only canonical IDs against gold labels, driving F1 to near-zero. This is an interpretable error: Mistral disregards explicit identifier constraints in the prompt — a finding discussed in the paper.

Full per-rule breakdowns: `results/metrics/openai_gpt41_per_rule.csv`, `results/metrics/ollama_mistral_per_rule.csv`.

---

## Security Notes

- Do **NOT** commit `.env` or any file containing API keys.
- Do **NOT** commit `models/*.bin` / `*.gz` (large binaries, gitignored).
- Do **NOT** commit predictions JSONL files if they contain sensitive document text.
- Only commit: code, configs, small CSVs, gold labels, documents (if permitted by your institution).

---

## Git Quick-Start

```bash
git add .
git commit -m "feat: add multi-LLM evaluation pipeline"
git push origin main
```

---

## AI Usage Disclaimer

Parts of this project were developed with the assistance of AI tools (OpenAI GPT, Google Gemini) for ideation, boilerplate code generation, and documentation drafting. All outputs were reviewed, validated, and integrated by the author, who takes full responsibility for the final content.

---

## License

This project is released under the [MIT License](LICENSE) for academic purposes.
