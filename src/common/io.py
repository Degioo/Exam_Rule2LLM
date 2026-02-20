import json
from pathlib import Path
from typing import Dict, List

from .schema import Document, Rule


def load_docs(docs_dir: str) -> List[Document]:
    docs = []
    # docs_dir might be a string or Path, make sure it's Path
    dpath = Path(docs_dir)
    if not dpath.exists():
        print(f"Warning: {dpath} does not exist.")
        return []
        
    for p in sorted(dpath.glob("*.txt")):
        doc_id = p.stem
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(doc_id=doc_id, text=text))
    return docs


def load_rules(rules_path: str) -> List[Rule]:
    rpath = Path(rules_path)
    if not rpath.exists():
         print(f"Warning: {rpath} does not exist.")
         return []
         
    data = json.loads(rpath.read_text(encoding="utf-8"))
    rules = [Rule(id=r["id"], description=r["description"]) for r in data]
    return rules


def load_gold(gold_path: str) -> Dict[str, Dict[str, List[str]]]:
    gpath = Path(gold_path)
    if not gpath.exists():
        print(f"Warning: {gpath} does not exist.")
        return {}
        
    return json.loads(gpath.read_text(encoding="utf-8"))
