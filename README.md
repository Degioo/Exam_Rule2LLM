# Rule2LLM – Rule-Guided LLMs for Administrative Compliance Checking

This repository contains the code, data, and experimental results for the project
**Rule2LLM**, developed as part of the Natural Language Processing course.

## Project Overview
The project investigates whether Large Language Models (LLMs), when explicitly guided
by administrative rules expressed in natural language, can support compliance checking
of public healthcare documents more robustly than traditional deterministic rule-based systems.

## Repository Structure
- `data/`: OCR-extracted document texts and gold labels
- `rules/`: Administrative rules (difformità) used in the experiments
- `prompts/`: Prompt used to guide the LLM
- `results/`: Experimental results
- `notebooks/`: Demonstration notebook reproducing the experiments

## Experimental Setup
The experiments are conducted on three realistic OCR-extracted document scenarios:
1. A standard case with clear violations
2. A case affected by OCR noise and vague formulations
3. A borderline semantic case designed to induce false positives in rule-based systems

The proposed Rule2LLM approach is compared against a PA-style deterministic baseline.

## Reproducibility
All experiments are documented and reproducible using the provided documents,
rules, prompts, and result files.

## AI Usage Disclaimer
Parts of this project have been developed with the assistance of OpenAI's GPT-5.
The AI was used to support ideation, prompt design, and drafting of descriptive text.
All outputs were carefully reviewed, validated, and integrated by the author,
who takes full responsibility for the final content.
