from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class RuleStatus(str, Enum):
    OK = "OK"
    VIOLATED = "VIOLATED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class Rule:
    id: str
    description: str


@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class Prediction:
    doc_id: str
    model: str
    rule_id: str
    status: RuleStatus
    evidence: str = ""
