from typing import List

from .common import ForgeModel


class EvaluationCase(ForgeModel):
    id: str
    question: str
    case_type: str
    expected_source_documents: List[str] = []
    expected_key_facts: List[str] = []
    evaluation_type: str = "grounded_answer"
    notes: str = ""


class CaseResult(ForgeModel):
    case_id: str
    question: str
    answer: str
    passed: bool
    scores: dict[str, float] = {}
    citations: List[str] = []
    failure_category: str | None = None
    failure_evidence: str = ""

