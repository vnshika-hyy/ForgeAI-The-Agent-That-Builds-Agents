import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..llm.client import LLMClient
from ..schemas.common import ForgeModel
from ..schemas.evaluation import CaseResult, EvaluationCase


FailureCategory = Literal[
    "retrieval_miss",
    "prompt_defect",
    "hallucination",
    "wrong_refusal",
]


class FailureClassification(ForgeModel):
    case_id: str
    category: FailureCategory
    reason: str
    supporting_signals: list[str] = Field(default_factory=list)


class FailureClassifier:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client
        self._prompt_path = (
            Path(__file__).resolve().parents[3]
            / "prompts"
            / "failure_classifier.txt"
        )

    @staticmethod
    def _deterministic(
        case: EvaluationCase,
        result: CaseResult,
    ) -> FailureClassification | None:
        checks = result.deterministic_checks
        if result.expected_behavior in {"answer", "answer_cautiously", "use_tool"} and result.actual_refused:
            return FailureClassification(
                case_id=case.id,
                category="wrong_refusal",
                reason="The case expected an answer, but the agent refused.",
                supporting_signals=["expected_behavior=answer", "actual_refused=true"],
            )

        invalid_sources = (
            checks.get("source_ids_valid") is False
            or "invalid source reference" in result.failure_evidence.casefold()
            or "unknown chunk ids" in result.failure_evidence.casefold()
        )
        unsupported_claims = checks.get("unsupported_claims") or (
            "unsupported claims:" in result.failure_evidence.casefold()
        )
        if invalid_sources or unsupported_claims:
            signals = []
            if invalid_sources:
                signals.append("invalid_or_fabricated_source_reference")
            if unsupported_claims:
                signals.append("judge_detected_unsupported_claims")
            return FailureClassification(
                case_id=case.id,
                category="hallucination",
                reason="The response used invalid grounding or contained unsupported claims.",
                supporting_signals=signals,
            )

        expected_chunks = set(case.supporting_chunk_ids)
        expected_sources = set(case.expected_source_documents)
        retrieved_chunks = set(result.retrieved_chunk_ids)
        retrieved_sources = set(result.retrieved_sources)
        has_expected_evidence = bool(expected_chunks or expected_sources)
        necessary_evidence_missing = (
            bool(expected_chunks) and not expected_chunks.intersection(retrieved_chunks)
        ) or (
            not expected_chunks
            and bool(expected_sources)
            and not expected_sources.intersection(retrieved_sources)
        )
        missed_facts = bool(result.missing_facts or checks.get("unmatched_expected_facts"))
        if (
            has_expected_evidence
            and necessary_evidence_missing
            and missed_facts
        ):
            return FailureClassification(
                case_id=case.id,
                category="retrieval_miss",
                reason="Expected supporting evidence existed but was not surfaced by runtime retrieval.",
                supporting_signals=[
                    "expected_evidence_not_retrieved",
                    "expected_facts_missing",
                ],
            )

        if (
            has_expected_evidence
            and not necessary_evidence_missing
            and (missed_facts or not result.passed)
        ):
            return FailureClassification(
                case_id=case.id,
                category="prompt_defect",
                reason="Relevant evidence was retrieved, but the agent did not follow the expected behavior.",
                supporting_signals=[
                    "expected_evidence_retrieved",
                    "response_failed_case",
                ],
            )
        return None

    def classify(
        self,
        case: EvaluationCase,
        result: CaseResult,
    ) -> FailureClassification:
        if result.passed:
            raise ValueError("FailureClassifier only classifies failed cases.")
        deterministic = self._deterministic(case, result)
        if deterministic is not None:
            return deterministic
        if self.llm_client is None:
            return FailureClassification(
                case_id=case.id,
                category="prompt_defect",
                reason="Failure signals were insufficient for a more specific deterministic category.",
                supporting_signals=["deterministic_signals_inconclusive"],
            )

        prompt = (
            self._prompt_path.read_text(encoding="utf-8")
            + "\n\nCASE:\n"
            + case.model_dump_json(indent=2)
            + "\n\nFAILED RESULT:\n"
            + result.model_dump_json(indent=2)
        )
        verdict = self.llm_client.generate_json(prompt, FailureClassification)
        return verdict.model_copy(update={"case_id": case.id})

    def classify_all(
        self,
        cases: list[EvaluationCase],
        results: list[CaseResult],
    ) -> list[FailureClassification]:
        cases_by_id = {case.id: case for case in cases}
        classifications: list[FailureClassification] = []
        for result in results:
            if result.passed:
                continue
            case = cases_by_id.get(result.case_id)
            if case is None:
                classifications.append(
                    FailureClassification(
                        case_id=result.case_id,
                        category="prompt_defect",
                        reason="Failed result has no matching evaluation case.",
                        supporting_signals=["case_definition_missing"],
                    )
                )
                continue
            classification = self.classify(case, result)
            result.failure_category = classification.category
            result.failure_classification = classification.model_dump(mode="json")
            classifications.append(classification)
        return classifications


def classifications_json(
    classifications: list[FailureClassification],
) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in classifications],
        ensure_ascii=False,
        indent=2,
    )
