"""
src/llm/runner.py
──────────────────
CLI runner: iterates documents, calls an LLM provider, validates and repairs
the JSON output, and saves one JSON line per document to a .jsonl file.

Usage examples
--------------
# OpenAI:
  python -m src.llm.runner \\
      --provider openai \\
      --docs data/docs \\
      --rules data/rules.json \\
      --out results/predictions/openai_gpt41.jsonl

# Ollama (mistral):
  python -m src.llm.runner \\
      --provider ollama \\
      --model mistral:latest \\
      --docs data/docs \\
      --rules data/rules.json \\
      --out results/predictions/ollama_mistral.jsonl

# Dry-run (structure check, no API calls):
  python -m src.llm.runner --dry-run --provider ollama
"""
from __future__ import annotations

from typing import Optional

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from .providers import get_provider
from .prompt import build_prompt
from .schema import validate_output, FORMAT_FIX_PROMPT
from ..common.io import load_docs, load_rules

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def _make_error_record(doc_id: str, model_tag: str, raw: str, error: str) -> dict:
    return {
        "document_id": doc_id,
        "model": model_tag,
        "raw_response": raw,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run(
    provider_name: str,
    model_override: Optional[str],
    docs_dir: str,
    rules_path: str,
    out_path: str,
    max_retries: int = 2,
    temperature: Optional[float] = None,
    dry_run: bool = False,
) -> None:
    # ── load data ──────────────────────────────────────────────────────────────
    docs  = load_docs(docs_dir)
    rules = load_rules(rules_path)

    if not docs:
        log.error("No documents found in %s", docs_dir)
        sys.exit(1)
    if not rules:
        log.error("No rules found in %s", rules_path)
        sys.exit(1)

    log.info("Loaded %d docs | %d rules", len(docs), len(rules))

    # ── provider ───────────────────────────────────────────────────────────────
    kwargs: dict = {}
    if model_override:
        kwargs["model"] = model_override
    if temperature is not None:
        kwargs["temperature"] = temperature

    if not dry_run:
        provider = get_provider(provider_name, **kwargs)
        model_tag = f"{provider_name}_{getattr(provider, 'model', 'unknown')}"
    else:
        provider = None
        model_tag = f"{provider_name}_{model_override or 'default'}"
        log.info("DRY-RUN mode — no API calls will be made.")

    # ── output file ────────────────────────────────────────────────────────────
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_invalid = n_failed = 0

    with out_file.open("w", encoding="utf-8") as fh:
        for doc in docs:
            log.info("Processing %s …", doc.doc_id)

            if dry_run:
                log.info("  [DRY-RUN] skipping API call for %s", doc.doc_id)
                continue

            system_msg, user_msg = build_prompt(doc, rules)
            raw = ""
            parsed = None
            error_msg = None

            # ── initial call ───────────────────────────────────────────────
            try:
                raw = provider.generate(system_msg, user_msg)
            except Exception as exc:
                log.error("  API error for %s: %s", doc.doc_id, exc)
                fh.write(json.dumps(_make_error_record(doc.doc_id, model_tag, "", str(exc))) + "\n")
                n_failed += 1
                continue

            parsed, error_msg = validate_output(raw)

            # ── retries with format-fix prompt ────────────────────────────
            attempt = 0
            while parsed is None and attempt < max_retries:
                attempt += 1
                log.warning(
                    "  Validation failed for %s (attempt %d/%d): %s",
                    doc.doc_id, attempt, max_retries, error_msg,
                )
                fix_user = (
                    f"Previous attempt for document {doc.doc_id} failed.\n"
                    f"Error: {error_msg}\n\n"
                    f"{FORMAT_FIX_PROMPT}\n\n"
                    f"Original document text:\n\"\"\"\n{doc.text.strip()}\n\"\"\""
                )
                try:
                    raw = provider.generate(system_msg, fix_user)
                    parsed, error_msg = validate_output(raw)
                except Exception as exc:
                    error_msg = str(exc)
                    break

            # ── write result ───────────────────────────────────────────────
            if parsed is not None:
                record = dict(
                    list(parsed.items()) + list({
                        "model": model_tag,
                        "raw_response": raw,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }.items())
                )
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_ok += 1
                log.info("  ✓ %s — saved", doc.doc_id)
            else:
                log.error("  ✗ %s — all retries failed: %s", doc.doc_id, error_msg)
                fh.write(json.dumps(_make_error_record(doc.doc_id, model_tag, raw, error_msg or "")) + "\n")
                n_invalid += 1

    log.info("Done. OK=%d  invalid=%d  failed=%d → %s", n_ok, n_invalid, n_failed, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.llm.runner",
        description="Rule2LLM — run LLM compliance checker on all documents.",
    )
    parser.add_argument("--provider",    required=True, choices=["openai", "ollama"],
                        help="LLM provider to use.")
    parser.add_argument("--model",       default=None,
                        help="Model name override (e.g. 'mistral:latest', 'gpt-4.1').")
    parser.add_argument("--docs",        default="data/docs",
                        help="Directory containing DOC_*.txt files.")
    parser.add_argument("--rules",       default="data/rules.json",
                        help="Path to rules JSON file.")
    parser.add_argument("--out",         default=None,
                        help="Output .jsonl file path.")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max format-fix retries per document (default: 2).")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override generation temperature.")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Load everything and print info but make no API calls.")
    args = parser.parse_args()

    # Default output path based on provider+model
    if args.out is None:
        model_slug = (args.model or "default").replace(":", "_").replace("/", "_")
        args.out = f"results/predictions/{args.provider}_{model_slug}.jsonl"

    run(
        provider_name=args.provider,
        model_override=args.model,
        docs_dir=args.docs,
        rules_path=args.rules,
        out_path=args.out,
        max_retries=args.max_retries,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
