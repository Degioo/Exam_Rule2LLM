from collections import defaultdict
from typing import Dict, List, Tuple

from ..common.schema import RuleStatus, Prediction


def to_binary(status: str) -> int:
    """Binary label: 1 if VIOLATED else 0 (OK or UNCERTAIN)."""
    return 1 if status == RuleStatus.VIOLATED else 0


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(preds: List[Prediction], gold: Dict[str, Dict[str, List[str]]]) -> Dict:
    """
    Returns:
      - per_rule: {rule_id: {tp, fp, fn, precision, recall, f1, coverage}}
      - overall: macro averages
    Coverage = fraction of predictions not UNCERTAIN for that rule.
    """
    by_rule = defaultdict(list)
    for pr in preds:
        by_rule[pr.rule_id].append(pr)

    per_rule = {}
    macro_p = macro_r = macro_f1 = 0.0
    n_rules = 0

    for rule_id, rule_preds in by_rule.items():
        tp = fp = fn = 0
        total = 0
        non_uncertain = 0

        for pr in rule_preds:
            total += 1
            if pr.status != RuleStatus.UNCERTAIN:
                non_uncertain += 1

            # Get gold labels for this doc
            if pr.doc_id not in gold:
                # If doc not in gold, skip or count as ...? 
                # Ideally gold should cover all docs.
                continue
                
            g = gold[pr.doc_id]
            gold_violated = set(g.get("violated", []))
            gold_is_viol = 1 if rule_id in gold_violated else 0
            pred_is_viol = to_binary(pr.status)

            if pred_is_viol == 1 and gold_is_viol == 1:
                tp += 1
            elif pred_is_viol == 1 and gold_is_viol == 0:
                fp += 1
            elif pred_is_viol == 0 and gold_is_viol == 1:
                fn += 1

        p, r, f1 = precision_recall_f1(tp, fp, fn)
        coverage = non_uncertain / total if total else 0.0

        per_rule[rule_id] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
            "coverage": coverage
        }

        macro_p += p
        macro_r += r
        macro_f1 += f1
        n_rules += 1

    overall = {
        "macro_precision": macro_p / n_rules if n_rules else 0.0,
        "macro_recall": macro_r / n_rules if n_rules else 0.0,
        "macro_f1": macro_f1 / n_rules if n_rules else 0.0,
    }

    return {"per_rule": per_rule, "overall": overall}
