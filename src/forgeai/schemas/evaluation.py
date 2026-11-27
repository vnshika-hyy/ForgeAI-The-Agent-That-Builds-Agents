from typing import List

from pydantic import Field, model_validator

from .common import ForgeModel


class EvaluationCase(ForgeModel):
    id: str
    question: str
    case_type: str
    expected_behavior: str = "answer"
    expected_source_documents: List[str] = []
    expected_key_facts: List[str] = []
    supporting_chunk_ids: List[str] = []
    required_tool: str | None = None
    evaluation_type: str = "grounded_answer"
    notes: str = ""


class EvaluationSet(ForgeModel):
    cases: List[EvaluationCase]
    warnings: List[str] = []


class CaseResult(ForgeModel):
    case_id: str
    question: str
    answer: str
    passed: bool
    case_type: str = ""
    expected_behavior: str = "answer"
    actual_refused: bool = False
    grounded: bool = False
    correct: bool = False
    appropriate_refusal: bool = False
    fact_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_sources: List[str] = []
    actual_sources: List[str] = []
    retrieved_sources: List[str] = []
    retrieved_chunk_ids: List[str] = []
    tools_expected: List[str] = []
    tools_used: List[str] = []
    deterministic_checks: dict[str, object] = {}
    matched_facts: List[str] = []
    missing_facts: List[str] = []
    judge_reason: str = ""
    error: str | None = None
    scores: dict[str, float] = {}
    citations: List[str] = []
    failure_category: str | None = None
    failure_classification: dict[str, object] = {}
    failure_evidence: str = ""


class EvaluationRunResult(ForgeModel):
    results: List[CaseResult]
    total: int = 0
    passed: int = 0
    failed: int = 0
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    groundedness_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    average_fact_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def calculate_summary(self) -> "EvaluationRunResult":
        self.total = len(self.results)
        self.passed = sum(result.passed for result in self.results)
        self.failed = self.total - self.passed
        self.score = self.passed / self.total if self.total else 0.0
        self.groundedness_rate = (
            sum(result.grounded for result in self.results) / self.total
            if self.total
            else 0.0
        )
        refusal_results = [
            result
            for result in self.results
            if result.expected_behavior == "refuse"
        ]
        self.refusal_accuracy = (
            sum(result.appropriate_refusal for result in refusal_results)
            / len(refusal_results)
            if refusal_results
            else 0.0
        )
        fact_results = [
            result
            for result in self.results
            if result.matched_facts or result.missing_facts
        ]
        self.average_fact_coverage = (
            sum(result.fact_coverage for result in fact_results) / len(fact_results)
            if fact_results
            else 0.0
        )
        return self

