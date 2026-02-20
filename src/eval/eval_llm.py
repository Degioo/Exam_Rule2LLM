"""
src/eval/eval_llm.py
─────────────────────
CLI evaluator: loads a .jsonl prediction file, computes P/R/F1 and abstention
rate, and saves per-rule and overall CSVs.

Usage
-----
  python -m src.eval.eval_llm \\
      --predictions results/predictions/ollama_mistral_latest.jsonl \\
      --gold data/gold_labels.json \\
      --out-dir results/metrics \\
      --model-tag ollama_mistral
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..common.io import load_gold
from ..common.schema import Prediction, RuleStatus
from .metrics import evaluate


def load_predictions(jsonl_path: str) -> list[Prediction]:
    """
    Parse a .jsonl file written by runner.py into a flat list of Prediction
    objects (one per rule per document).
    Records with an "error" key are skipped.
    """
    preds: list[Prediction] = []
    path = Path(jsonl_path)

    if not path.exists():
        print(f"[ERROR] Predictions file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as ex:
                print(f"[WARN] line {lineno}: JSON decode error — {ex}", file=sys.stderr)
                continue

            if "error" in record and "rules" not in record:
                print(
                    f"[WARN] skipping failed record for {record.get('document_id', '?')} — {record['error']}",
                    file=sys.stderr,
                )
                continue

            doc_id = record.get("document_id", "UNKNOWN")
            model  = record.get("model", "unknown")

            for rule_entry in record.get("rules", []):
                try:
                    status = RuleStatus(rule_entry["status"])
                except (KeyError, ValueError):
                    status = RuleStatus.UNCERTAIN

                preds.append(Prediction(
                    doc_id=doc_id,
                    model=model,
                    rule_id=rule_entry.get("id", "?"),
                    status=status,
                    evidence=rule_entry.get("evidence", ""),
                ))

    return preds


def abstention_rate(preds: list[Prediction]) -> float:
    """Fraction of predictions labelled UNCERTAIN."""
    if not preds:
        return 0.0
    n_uncertain = sum(1 for p in preds if p.status == RuleStatus.UNCERTAIN)
    return n_uncertain / len(preds)


def run_eval(
    predictions_path: str,
    gold_path: str,
    out_dir: str,
    model_tag: str,
) -> None:
    preds = load_predictions(predictions_path)
    gold  = load_gold(gold_path)

    if not preds:
        print("[ERROR] No usable predictions found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(preds)} predictions for model '{model_tag}'.")

    results   = evaluate(preds, gold)
    per_rule  = results["per_rule"]
    overall   = results["overall"]
    abst_rate = abstention_rate(preds)

    # ── per-rule CSV ───────────────────────────────────────────────────────────
    per_rule_rows = []
    for rule_id, metrics in sorted(per_rule.items()):
        per_rule_rows.append({
            "model":     model_tag,
            "rule_id":   rule_id,
            "tp":        metrics["tp"],
            "fp":        metrics["fp"],
            "fn":        metrics["fn"],
            "precision": round(metrics["precision"], 4),
            "recall":    round(metrics["recall"],    4),
            "f1":        round(metrics["f1"],        4),
            "coverage":  round(metrics["coverage"],  4),
        })
    df_per_rule = pd.DataFrame(per_rule_rows)

    # ── overall CSV ────────────────────────────────────────────────────────────
    overall_row = {
        "model":           model_tag,
        "macro_precision": round(overall["macro_precision"], 4),
        "macro_recall":    round(overall["macro_recall"],    4),
        "macro_f1":        round(overall["macro_f1"],        4),
        "abstention_rate": round(abst_rate, 4),
        "n_predictions":   len(preds),
    }
    df_overall = pd.DataFrame([overall_row])

    # ── save ───────────────────────────────────────────────────────────────────
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_rule_csv = out / f"{model_tag}_per_rule.csv"
    overall_csv  = out / f"{model_tag}_overall.csv"

    df_per_rule.to_csv(per_rule_csv, index=False)
    df_overall.to_csv(overall_csv,   index=False)

    print(f"\nOverall results for '{model_tag}':")
    print(df_overall.to_string(index=False))
    print(f"\nPer-rule results:\n{df_per_rule.to_string(index=False)}")
    print(f"\nSaved:\n  {per_rule_csv}\n  {overall_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.eval_llm",
        description="Rule2LLM — evaluate LLM predictions against gold labels.",
    )
    parser.add_argument("--predictions", required=True,
                        help="Path to .jsonl predictions file.")
    parser.add_argument("--gold",        default="data/gold_labels.json",
                        help="Path to gold_labels.json.")
    parser.add_argument("--out-dir",     default="results/metrics",
                        help="Directory to save CSV results.")
    parser.add_argument("--model-tag",   required=True,
                        help="Short identifier for this model run (used in filenames).")
    args = parser.parse_args()

    run_eval(
        predictions_path=args.predictions,
        gold_path=args.gold,
        out_dir=args.out_dir,
        model_tag=args.model_tag,
    )


if __name__ == "__main__":
    main()
