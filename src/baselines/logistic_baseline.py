from typing import Dict, List
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..common.schema import Document, Rule, Prediction, RuleStatus


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_labels(docs: List[Document], gold: Dict, rule_id: str):
    y = []
    for d in docs:
        if d.doc_id not in gold:
             # Fallback if doc is not in gold, although this should be handled upstream
             y.append(0)
             continue
        violated = set(gold[d.doc_id].get("violated", []))
        y.append(1 if rule_id in violated else 0)
    return y


def logistic_predict(
    docs: List[Document],
    rules: List[Rule],
    gold: Dict,
    model_name: str = "TFIDF+LogReg"
) -> List[Prediction]:
    """
    Train one binary classifier per rule on the available docs (demo setting).
    With very small data, this is illustrative rather than performance-oriented.
    """
    preds: List[Prediction] = []

    texts = [normalize_text(d.text) for d in docs]
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform(texts)

    for r in rules:
        y = build_labels(docs, gold, r.id)

        # Handle simplified case where all labels are same (e.g. all 0 or all 1)
        if len(set(y)) < 2:
            # Cannot train logistic regression with 1 class
            # Default to majority
            majority_val = y[0]
            for d in docs:
                status = RuleStatus.VIOLATED if majority_val == 1 else RuleStatus.OK
                preds.append(Prediction(
                    doc_id=d.doc_id,
                    model=model_name,
                    rule_id=r.id,
                    status=status,
                    evidence=f"OneClassOnly({majority_val})"
                ))
            continue

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        proba = clf.predict_proba(X)[:, 1]  # P(violated)

        for d, p in zip(docs, proba):
            status = RuleStatus.VIOLATED if p >= 0.5 else RuleStatus.OK
            preds.append(Prediction(
                doc_id=d.doc_id,
                model=model_name,
                rule_id=r.id,
                status=status,
                evidence=f"P(violated)={p:.3f}"
            ))

    return preds
