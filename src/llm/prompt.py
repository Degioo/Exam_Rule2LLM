"""
src/llm/prompt.py
─────────────────
Prompt builder for the Rule2LLM compliance-checking pipeline.

Usage:
    from src.llm.prompt import build_prompt
    system_msg, user_msg = build_prompt(doc, rules)
"""

from __future__ import annotations

from typing import List, Tuple

from ..common.schema import Document, Rule

# ──────────────────────────────────────────────────────────────────────────────
# System prompt (conservative/strict PA auditor persona)
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a strict public-administration (PA) compliance auditor specialised in \
Italian healthcare prescription documents.

Your task is to evaluate whether each compliance rule listed below is satisfied \
by the provided OCR-extracted prescription text.

IMPORTANT CONSTRAINTS:
1. Return ONLY a valid JSON object — no prose, no markdown, no code fences.
2. Use "UNCERTAIN" when the rule cannot be reliably evaluated from the text \
(e.g. severe OCR corruption, ambiguous wording). Do NOT guess.
3. Do NOT hallucinate fields, dates, codes, or names not present in the text.
4. Tolerate minor OCR noise (e.g. "N02BG1O" vs "N02BG10", missing spaces).
5. Evidence must be an exact short quote from the document, or an observation \
   in Italian if no direct quote is possible. Keep it ≤ 80 characters.
6. Populate the "summary" lists consistently with the rule statuses above.
"""

# ──────────────────────────────────────────────────────────────────────────────
# One-shot worked example (stabilises output format)
# ──────────────────────────────────────────────────────────────────────────────

_EXAMPLE = """\
### EXAMPLE (do not use this data — for format reference only)

Document text:
  "Utilizzare entro 31/01/2025. Dr. Rossi [timbro]. ATC: N02BG10. \
Tassa 1.50€. Quantità 1. ATTENZIONE avvisare il paziente."

Expected output:
{
  "document_id": "DOC_000",
  "rules": [
    {"id": "01",  "status": "OK",        "evidence": "Utilizzare entro 31/01/2025"},
    {"id": "02",  "status": "OK",        "evidence": "Dr. Rossi [timbro]"},
    {"id": "08",  "status": "OK",        "evidence": "ATC: N02BG10"},
    {"id": "05A", "status": "VIOLATED",  "evidence": "no special-needs statement found"},
    {"id": "10A", "status": "UNCERTAIN", "evidence": "posology not clearly stated"}
  ],
  "summary": {
    "violated":  ["05A"],
    "ok":        ["01", "02", "08"],
    "uncertain": ["10A"]
  }
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(doc: Document, rules: List[Rule]) -> Tuple[str, str]:
    """
    Build the (system_message, user_message) pair for a single document.

    Parameters
    ----------
    doc   : Document dataclass from src.common.schema
    rules : list of Rule dataclasses

    Returns
    -------
    (system_str, user_str) both as plain strings
    """
    # Build the rules block
    rules_block = "\n".join(
        f"  - Rule {r.id}: {r.description}"
        for r in rules
    )

    user_msg = f"""\
{_EXAMPLE}

---

### YOUR TASK

Compliance Rules to evaluate:
{rules_block}

Document ID: {doc.doc_id}
Document Text (OCR-extracted):
\"\"\"
{doc.text.strip()}
\"\"\"

Evaluate every rule listed above and return ONLY the JSON object described. \
Replace "DOC_000" with "{doc.doc_id}".
"""

    return _SYSTEM_PROMPT, user_msg
