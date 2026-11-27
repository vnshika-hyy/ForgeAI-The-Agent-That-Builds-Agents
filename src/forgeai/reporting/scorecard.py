from ..schemas.evaluation import CaseResult, EvaluationRunResult


def build_scorecard(results: list[CaseResult]) -> EvaluationRunResult:
    """Compute aggregate evaluation metrics from actual case results."""
    return EvaluationRunResult(results=results)
