"""
src/llm/schema.py
─────────────────
JSON Schema definition and validation for Rule2LLM model outputs.

Expected model output structure:
  {
    "document_id": "DOC_001",
    "rules": [
      {"id": "01", "status": "OK",        "evidence": "..."},
      {"id": "05A", "status": "VIOLATED", "evidence": "..."}
    ],
    "summary": {
      "violated":  ["05A"],
      "ok":        ["01"],
      "uncertain": []
    }
  }
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

# ──────────────────────────────────────────────────────────────────────────────
# JSON Schema (strict)
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["document_id", "rules", "summary"],
    "additionalProperties": False,
    "properties": {
        "document_id": {"type": "string", "pattern": r"^DOC_\d+$"},
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "status", "evidence"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["OK", "VIOLATED", "UNCERTAIN"],
                    },
                    "evidence": {"type": "string"},
                },
            },
        },
        "summary": {
            "type": "object",
            "required": ["violated", "ok", "uncertain"],
            "additionalProperties": False,
            "properties": {
                "violated":  {"type": "array", "items": {"type": "string"}},
                "ok":        {"type": "array", "items": {"type": "string"}},
                "uncertain": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

_VALIDATOR = jsonschema.Draft7Validator(OUTPUT_SCHEMA)


# ──────────────────────────────────────────────────────────────────────────────
# Format-fix prompt appended on retry
# ──────────────────────────────────────────────────────────────────────────────

FORMAT_FIX_PROMPT = """\
Your previous response could not be parsed as valid JSON or did not match the \
required schema.

Please reply with ONLY a valid JSON object matching exactly this structure \
(no markdown, no code fences, no extra text):

{
  "document_id": "<DOC_XXX>",
  "rules": [
    {"id": "<rule_id>", "status": "OK|VIOLATED|UNCERTAIN", "evidence": "<text>"}
  ],
  "summary": {
    "violated":  ["<rule_id>", ...],
    "ok":        ["<rule_id>", ...],
    "uncertain": ["<rule_id>", ...]
  }
}

Allowed values for status: OK, VIOLATED, UNCERTAIN.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_json(raw: str) -> Optional[str]:
    """
    Try to strip markdown fences and return the first JSON object found.
    Returns the cleaned string, or None if no '{' found.
    """
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    return text[start : end + 1]


def _normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce common model quirks before strict schema validation:
    - evidence as list  → join with '; '
    - extra top-level keys (model, raw_response, timestamp) → strip them
      so additionalProperties:false doesn't reject re-parsed JSONL records
    """
    # Strip runner-added metadata keys that might be present if re-validating
    allowed_top = {"document_id", "rules", "summary"}
    data = {k: v for k, v in data.items() if k in allowed_top}

    # Normalise each rule entry
    for rule in data.get("rules", []):
        ev = rule.get("evidence", "")
        if isinstance(ev, list):
            rule["evidence"] = "; ".join(str(x) for x in ev)
        elif not isinstance(ev, str):
            rule["evidence"] = str(ev)

    return data


def validate_output(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse, normalise, and validate raw model text against OUTPUT_SCHEMA.

    Returns
    -------
    (parsed_dict, None)      on success
    (None, error_message)    on failure
    """
    cleaned = extract_json(raw)
    if cleaned is None:
        return None, "No JSON object found in model response."

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"

    # Normalise before strict schema check
    data = _normalize_data(data)

    errors = list(_VALIDATOR.iter_errors(data))
    if errors:
        msgs = "; ".join(e.message for e in errors[:3])
        return None, f"Schema validation failed: {msgs}"

    return data, None
