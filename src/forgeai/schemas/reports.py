from typing import List

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


class QualityReport(ForgeModel):
    total_tests: int
    passed: int
    failed: int
    score: float
    failure_categories: dict[str, int] = {}
    unresolved_failures: List[str] = []
    likely_causes: List[str] = []
    known_limitations: List[str] = []

