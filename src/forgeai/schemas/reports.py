from typing import List, Literal

from pydantic import model_validator

from .common import ForgeModel
from .evaluation import CaseResult


class IterationScorecard(ForgeModel):
    iteration: int
    score_before: float
    score_after: float
    failure_cases: List[str] = []
    failure_categories: dict[str, int] = {}
    changes: List[str] = []
    reasons: List[str] = []
    intended_cases: List[str] = []
    diagnosis: str = ""
    stop_reason: str = ""


class QualityReport(ForgeModel):
    total_tests: int
    passed: int
    failed: int
    score: float
    failure_categories: dict[str, int] = {}
    unresolved_failures: List[str] = []
    likely_causes: List[str] = []
    known_limitations: List[str] = []


class EvidenceReference(ForgeModel):
    source: str
    chunk_id: str | None = None
    page: int | None = None
    excerpt: str | None = None


class RequirementGroundability(ForgeModel):
    requirement: str
    status: Literal["supported", "partially_supported", "unsupported"]
    reason: str
    evidence: list[EvidenceReference] = []


class GroundabilityReport(ForgeModel):
    findings: list[RequirementGroundability] = []
    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0
    warnings: list[str] = []

    @model_validator(mode="after")
    def calculate_counts(self) -> "GroundabilityReport":
        self.supported_count = sum(
            finding.status == "supported" for finding in self.findings
        )
        self.partially_supported_count = sum(
            finding.status == "partially_supported" for finding in self.findings
        )
        self.unsupported_count = sum(
            finding.status == "unsupported" for finding in self.findings
        )
        return self

