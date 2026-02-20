"""
scripts/run_all_models.py
──────────────────────────
Read configs/models.yaml and sequentially:
  1. Run python -m src.llm.runner  (prediction)
  2. Run python -m src.eval.eval_llm (evaluation)
for every configured model, then aggregate overall metrics into
results/metrics/summary.csv.

Usage
-----
  # Full run (requires API keys / Ollama running):
  python scripts/run_all_models.py

  # Dry-run (tests loading, no API calls):
  python scripts/run_all_models.py --dry-run

  # Skip models already predicted (JSONL exists):
  python scripts/run_all_models.py --skip-existing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
SUMMARY_PATH = METRICS_DIR / "summary.csv"


def load_config() -> list[dict]:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg.get("runs", [])


def run_cmd(cmd: list[str], dry_run: bool = False) -> int:
    """Run a command as a subprocess; return exit code."""
    display = " ".join(str(c) for c in cmd)
    print(f"\n>>> {display}")
    if dry_run:
        print("    [DRY-RUN] skipped.")
        return 0
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule2LLM — run all models defined in configs/models.yaml"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip prediction step if JSONL already exists.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip prediction step, only run evaluation.")
    args = parser.parse_args()

    runs = load_config()
    if not runs:
        print("[ERROR] No runs found in configs/models.yaml", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(runs)} run(s) from {CONFIG_PATH}")
    overall_records: list[dict] = []
    failed: list[str] = []

    for run in runs:
        tag      = run["tag"]
        provider = run["provider"]
        model    = run.get("model", "")
        docs     = run.get("docs", "data/docs")
        rules    = run.get("rules", "data/rules.json")
        gold     = run.get("gold",  "data/gold_labels.json")
        out_path = run.get("out",   f"results/predictions/{tag}.jsonl")
        temp     = str(run.get("temperature", 0))
        retries  = str(run.get("max_retries", 2))

        print(f"\n{'='*60}")
        print(f"  Model: {tag}  ({provider} / {model})")
        print(f"{'='*60}")

        # ── Step 1: prediction ─────────────────────────────────────────────────
        skip_pred = args.eval_only or (
            args.skip_existing and (REPO_ROOT / out_path).exists()
        )
        if skip_pred:
            print(f"  Skipping prediction step (output exists or eval-only).")
        else:
            runner_cmd = [
                sys.executable, "-m", "src.llm.runner",
                "--provider", provider,
                "--model",    model,
                "--docs",     docs,
                "--rules",    rules,
                "--out",      out_path,
                "--max-retries", retries,
                "--temperature", temp,
            ]
            if args.dry_run:
                runner_cmd.append("--dry-run")

            rc = run_cmd(runner_cmd, dry_run=False)  # subprocess handles dry_run flag
            if rc != 0:
                print(f"[WARN] Runner exited with code {rc} for '{tag}'. Continuing.")
                failed.append(f"{tag} (runner)")

        # ── Step 2: evaluation ─────────────────────────────────────────────────
        if args.dry_run:
            print(f"  [DRY-RUN] Would run eval for '{tag}'.")
            continue

        out_absolute = REPO_ROOT / out_path
        if not out_absolute.exists():
            print(f"[WARN] Predictions file not found for '{tag}': {out_path}. Skipping eval.")
            failed.append(f"{tag} (eval — no predictions)")
            continue

        eval_cmd = [
            sys.executable, "-m", "src.eval.eval_llm",
            "--predictions", out_path,
            "--gold",        gold,
            "--out-dir",     "results/metrics",
            "--model-tag",   tag,
        ]
        rc = run_cmd(eval_cmd)
        if rc != 0:
            print(f"[WARN] Eval exited with code {rc} for '{tag}'.")
            failed.append(f"{tag} (eval)")
            continue

        # Load overall CSV for aggregation
        overall_csv = METRICS_DIR / f"{tag}_overall.csv"
        if overall_csv.exists():
            df = pd.read_csv(overall_csv)
            overall_records.append(df.iloc[0].to_dict())

    # ── Aggregate summary ──────────────────────────────────────────────────────
    if overall_records:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(overall_records)
        # Sort by macro_f1 descending
        if "macro_f1" in summary.columns:
            summary = summary.sort_values("macro_f1", ascending=False)
        summary.to_csv(SUMMARY_PATH, index=False)
        print(f"\n{'='*60}")
        print("SUMMARY (all models):")
        print(summary.to_string(index=False))
        print(f"\nSaved to {SUMMARY_PATH}")
    elif not args.dry_run:
        print("\n[INFO] No completed evaluations to aggregate.")

    if failed:
        print(f"\n[WARN] Failed steps: {failed}")
        sys.exit(1)

    print("\nAll done.")


if __name__ == "__main__":
    main()
