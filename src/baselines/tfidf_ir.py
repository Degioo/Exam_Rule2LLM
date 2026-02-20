from typing import List
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..common.schema import Document, Rule, Prediction, RuleStatus


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tfidf_ir_predict(
    docs: List[Document],
    rules: List[Rule],
    model_name: str = "TFIDF-IR",
    threshold: float = 0.10
) -> List[Prediction]:
    """
    For each (doc, rule) compute cosine similarity between doc text and rule description.
    If similarity < threshold => VIOLATED else OK.
    Note: This baseline cannot produce UNCERTAIN (by design).
    """
    preds: List[Prediction] = []

    doc_texts = [normalize_text(d.text) for d in docs]
    # Fit on docs + rules descriptions to share vocabulary
    corpus = doc_texts + [normalize_text(r.description) for r in rules]

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform(corpus)

    X_docs = X[:len(docs)]
    X_rules = X[len(docs):]

    sim = cosine_similarity(X_docs, X_rules)  # shape: (n_docs, n_rules)

    for i, d in enumerate(docs):
        for j, r in enumerate(rules):
            score = float(sim[i, j])
            status = RuleStatus.OK if score >= threshold else RuleStatus.VIOLATED
            preds.append(Prediction(
                doc_id=d.doc_id,
                model=model_name,
                rule_id=r.id,
                status=status,
                evidence=f"cosine_tfidf={score:.3f} (thr={threshold})"
            ))

    return preds
