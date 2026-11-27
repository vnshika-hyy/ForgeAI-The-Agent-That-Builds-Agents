import re
from typing import Mapping

from pydantic import Field

from ..runtime.agent_runner import AgentResponse
from ..schemas.common import ForgeModel
from ..schemas.evaluation import EvaluationCase


class DeterministicCheckResult(ForgeModel):
    refusal_correct: bool | None = None
    wrong_refusal: bool = False
    source_ids_valid: bool = True
    expected_source_hit: bool | None = None
    source_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    required_tool_used: bool | None = None
    matched_facts: list[str] = Field(default_factory=list)
    unmatched_expected_facts: list[str] = Field(default_factory=list)
    exact_fact_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    source_grounded: bool = False
    hard_failures: list[str] = Field(default_factory=list)


def _normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def check_case_deterministically(
    case: EvaluationCase,
    response: AgentResponse,
    known_chunk_sources: Mapping[str, str],
) -> DeterministicCheckResult:
    checks = DeterministicCheckResult()
    expected_behavior = case.expected_behavior
    if expected_behavior == "refuse":
        checks.refusal_correct = response.refused
        if not response.refused:
            checks.hard_failures.append("Expected a refusal, but the agent answered.")
    elif expected_behavior in {"answer", "answer_cautiously", "use_tool"}:
        checks.wrong_refusal = response.refused
        if response.refused:
            checks.hard_failures.append(
                "The agent refused a case that expected an answer."
            )

    invalid_ids = [
        chunk_id
        for chunk_id in response.source_chunk_ids
        if chunk_id not in known_chunk_sources
    ]
    checks.source_ids_valid = not invalid_ids
    if invalid_ids:
        checks.hard_failures.append(
            "The response cited unknown chunk IDs: " + ", ".join(invalid_ids)
        )

    actual_sources = {
        known_chunk_sources[chunk_id]
        for chunk_id in response.source_chunk_ids
        if chunk_id in known_chunk_sources
    }
    expected_chunks = set(case.supporting_chunk_ids)
    expected_sources = set(case.expected_source_documents)
    if not expected_sources:
        expected_sources = {
            known_chunk_sources[chunk_id]
            for chunk_id in expected_chunks
            if chunk_id in known_chunk_sources
        }

    if expected_chunks or expected_sources:
        if expected_sources:
            denominator = len(expected_sources)
            overlap = len(actual_sources & expected_sources)
        else:
            denominator = len(expected_chunks)
            overlap = len(set(response.source_chunk_ids) & expected_chunks)
        checks.source_recall = overlap / denominator if denominator else 0.0
        checks.expected_source_hit = overlap > 0
        if not checks.expected_source_hit and not response.refused:
            checks.hard_failures.append(
                "The response did not cite any expected supporting source."
            )
        if (
            case.case_type == "cross_document"
            and checks.source_recall < 1.0
            and not response.refused
        ):
            checks.hard_failures.append(
                "The response did not cite all expected cross-document sources."
            )
    else:
        checks.expected_source_hit = None
        checks.source_recall = 1.0

    required_tool = case.required_tool if case.case_type == "tool_required" else None
    if case.case_type == "tool_required":
        checks.required_tool_used = bool(
            required_tool and required_tool in response.tools_used
        )
        if not required_tool:
            checks.hard_failures.append(
                "The tool-required case does not specify a required tool."
            )
        elif not checks.required_tool_used:
            checks.hard_failures.append(
                f"Required tool '{required_tool}' was not used."
            )

    expected_facts = case.expected_key_facts
    if expected_facts:
        normalized_answer = _normalize(response.answer)
        checks.matched_facts = [
            fact
            for fact in expected_facts
            if _normalize(fact) and _normalize(fact) in normalized_answer
        ]
        matched_normalized = {_normalize(fact) for fact in checks.matched_facts}
        checks.unmatched_expected_facts = [
            fact
            for fact in expected_facts
            if _normalize(fact) not in matched_normalized
        ]
        checks.exact_fact_coverage = len(checks.matched_facts) / len(expected_facts)

    has_valid_citation = bool(response.source_chunk_ids) and checks.source_ids_valid
    has_expected_source_hit = (
        checks.expected_source_hit is not False
        and checks.source_ids_valid
    )
    tool_grounding = bool(
        required_tool and checks.required_tool_used and checks.source_ids_valid
    )
    if response.refused and expected_behavior == "refuse":
        checks.source_grounded = checks.source_ids_valid
    elif expected_behavior == "clarify":
        checks.source_grounded = checks.source_ids_valid
    else:
        checks.source_grounded = (
            (has_valid_citation and has_expected_source_hit) or tool_grounding
        )
    if (
        case.case_type == "cross_document"
        and checks.source_recall < 1.0
        and not response.refused
    ):
        checks.source_grounded = False

    if (
        expected_behavior not in {"refuse", "clarify"}
        and not response.refused
        and not checks.source_grounded
    ):
        checks.hard_failures.append(
            "The response has no valid supporting source or successful required tool."
        )
    return checks
