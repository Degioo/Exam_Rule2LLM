from typing import List
import re
import os
import numpy as np
from gensim.models import KeyedVectors

from ..common.schema import Document, Prediction, RuleStatus


def tokenize(s: str) -> List[str]:
    s = s.lower()
    # keep letters/numbers/basic accents; remove punctuation-like noise
    s = re.sub(r"[^a-zàèéìòù0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def fasttext_predict_05A(
    docs: List[Document],
    model_path: str,
    threshold: float = 0.55,
    model_name: str = "fastText-sim-05A"
) -> List[Prediction]:
    """
    Baseline semantica per la regola 05A usando Gensim per efficienza.
    Supporta file .bin (FastText nativo) e .vec (word2vec testo).
    """
    print(f"Loading vectors from {model_path}...")
    ext = os.path.splitext(model_path)[1].lower()
    try:
        if ext == ".bin":
            # FastText binary format (.bin) — use fasttext library
            import fasttext
            _ft_model = fasttext.load_model(model_path)
            words = _ft_model.get_words()
            vectors = np.array([_ft_model.get_word_vector(w) for w in words], dtype=np.float32)
            ft = KeyedVectors(vector_size=vectors.shape[1])
            ft.add_vectors(words, vectors)
            del _ft_model, vectors  # free memory
        else:
            # Word2Vec text format (.vec) — use Gensim directly
            ft = KeyedVectors.load_word2vec_format(model_path, binary=False, limit=300000)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    # keyword "concettuali" (alcune sono radici/forme comuni)
    keywords = [
        "refrattario", "resistente", "responsivo", "risposta",
        "mancata", "non", "industriale", "farmaco"
    ]
    
    # Filter keywords present in the loaded vocab
    key_vecs = []
    for k in keywords:
        if k in ft:
            key_vecs.append(ft[k])
        # fallback for OOV keywords?? nothing for now
    
    if not key_vecs:
        print("Warning: no keywords found in model vocabulary!")
        return []

    preds: List[Prediction] = []
    for d in docs:
        toks = tokenize(d.text)

        best = 0.0
        best_tok = None
        for t in toks:
            if t in ft:
                tv = ft[t]
                # compute max sim against any keyword
                score = max(cosine(tv, kv) for kv in key_vecs)
                if score > best:
                    best = score
                    best_tok = t

        status = RuleStatus.OK if best >= threshold else RuleStatus.VIOLATED
        preds.append(Prediction(
            doc_id=d.doc_id,
            model=model_name,
            rule_id="05A",
            status=status,
            evidence=f"best_sim={best:.3f}; token={best_tok}; thr={threshold}"
        ))

    return preds
