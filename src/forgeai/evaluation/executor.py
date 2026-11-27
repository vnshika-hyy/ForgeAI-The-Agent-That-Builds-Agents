from ..ingestion.chunker import Chunk
from ..runtime.agent_runner import AgentResponse, AgentRunner
from ..schemas.evaluation import (
    CaseResult,
    EvaluationCase,
    EvaluationRunResult,
    EvaluationSet,
)
from .deterministic_checks import (
    DeterministicCheckResult,
    check_case_deterministically,
)
from .judge import Judge, JudgeResult


class EvalExecutor:
    def __init__(self, agent_runner: AgentRunner, judge: Judge) -> None:
        self.agent_runner = agent_runner
        self.judge = judge

    def _known_chunks(self) -> dict[str, Chunk]:
        retriever = getattr(self.agent_runner, "retriever", None)
        chunks = getattr(retriever, "chunks", [])
        return {
            chunk.chunk_id or f"{chunk.source}::chunk_{chunk.chunk_index}": chunk
            for chunk in chunks
        }

    def _failure_result(
        self,
        case: EvaluationCase,
        error: str,
        response: AgentResponse | None = None,
        checks: DeterministicCheckResult | None = None,
    ) -> CaseResult:
        response = response or AgentResponse(
            answer="",
            refused=False,
        )
        known_chunks = self._known_chunks()
        actual_sources = sorted(
            {
                known_chunks[chunk_id].source
                for chunk_id in response.source_chunk_ids
                if chunk_id in known_chunks
            }
        )
        return CaseResult(
            case_id=case.id,
            question=case.question,
            answer=response.answer,
            passed=False,
            case_type=case.case_type,
            expected_behavior=case.expected_behavior,
            actual_refused=response.refused,
            grounded=False,
            correct=False,
            appropriate_refusal=False,
            fact_coverage=checks.exact_fact_coverage if checks else 0.0,
            expected_sources=case.expected_source_documents,
            actual_sources=actual_sources,
            retrieved_sources=sorted(
                {
                    known_chunks[chunk_id].source
                    for chunk_id in response.retrieved_chunk_ids
                    if chunk_id in known_chunks
                }
            ),
            retrieved_chunk_ids=response.retrieved_chunk_ids,
            tools_expected=[case.required_tool] if case.required_tool else [],
            tools_used=response.tools_used,
            deterministic_checks=(
                checks.model_dump(mode="json") if checks else {}
            ),
            matched_facts=checks.matched_facts if checks else [],
            missing_facts=(
                checks.unmatched_expected_facts if checks else case.expected_key_facts
            ),
            judge_reason="",
            error=error,
            scores={},
            citations=response.source_chunk_ids,
            failure_evidence=error,
        )

    @staticmethod
    def _should_judge(
        case: EvaluationCase,
        response: AgentResponse,
        checks: DeterministicCheckResult,
    ) -> bool:
        if case.expected_behavior == "refuse" and response.refused:
            return False
        if response.refused:
            return False
        if not checks.source_ids_valid:
            return False
        if (
            case.expected_behavior not in {"clarify", "refuse"}
            and not checks.source_grounded
        ):
            return False
        return True

    @staticmethod
    def _combine(
        case: EvaluationCase,
        response: AgentResponse,
        checks: DeterministicCheckResult,
        judge_result: JudgeResult | None,
        known_chunks: dict[str, object],
    ) -> CaseResult:
        expected_behavior = case.expected_behavior
        actual_sources = sorted(
            {
                known_chunks[chunk_id].source
                for chunk_id in response.source_chunk_ids
                if chunk_id in known_chunks
            }
        )

        if judge_result is None:
            fact_coverage = checks.exact_fact_coverage
            matched_facts = checks.matched_facts
            missing_facts = checks.unmatched_expected_facts
            judge_grounded = True
            judge_correct = True
            if expected_behavior == "refuse":
                expected_behavior_met = bool(checks.refusal_correct)
            elif expected_behavior in {"clarify", "answer_cautiously"}:
                expected_behavior_met = False
            else:
                expected_behavior_met = not checks.wrong_refusal
            judge_reason = ""
        else:
            fact_coverage = judge_result.fact_coverage
            matched_facts = judge_result.matched_facts
            missing_facts = judge_result.missing_expected_facts
            judge_grounded = judge_result.grounded
            judge_correct = judge_result.correct
            expected_behavior_met = judge_result.expected_behavior_met
            judge_reason = judge_result.reason

        appropriate_refusal = (
            bool(checks.refusal_correct)
            if expected_behavior == "refuse"
            else (
                judge_result.appropriate_refusal
                if judge_result is not None
                else False
            )
        )
        grounded = checks.source_grounded and judge_grounded

        if expected_behavior == "refuse":
            correct = bool(checks.refusal_correct)
            behavior_met = correct
        elif expected_behavior == "clarify":
            behavior_met = expected_behavior_met
            correct = behavior_met and judge_correct
        elif expected_behavior in {"answer", "answer_cautiously", "use_tool"}:
            behavior_met = not response.refused and expected_behavior_met
            correct = (
                behavior_met
                and judge_correct
                and fact_coverage == 1.0
                and checks.required_tool_used is not False
            )
        else:
            behavior_met = expected_behavior_met
            correct = behavior_met and judge_correct

        unsupported_claims = bool(
            judge_result and judge_result.unsupported_claims
        )
        if unsupported_claims:
            correct = False

        passed = (
            not checks.hard_failures
            and correct
            and behavior_met
            and (grounded or expected_behavior in {"refuse", "clarify"})
        )
        failure_reasons = list(checks.hard_failures)
        if not behavior_met and not failure_reasons:
            failure_reasons.append("Expected behavior was not followed.")
        if not grounded and expected_behavior not in {"refuse", "clarify"}:
            failure_reasons.append("Response was not grounded in validated evidence.")
        if missing_facts:
            failure_reasons.append(
                "Missing expected facts: " + "; ".join(missing_facts)
            )
        if judge_result and judge_result.unsupported_claims:
            failure_reasons.append(
                "Unsupported claims: " + "; ".join(judge_result.unsupported_claims)
            )
        if judge_reason and not passed:
            failure_reasons.append(judge_reason)

        expected_sources = case.expected_source_documents or sorted(
            {
                known_chunks[chunk_id].source
                for chunk_id in case.supporting_chunk_ids
                if chunk_id in known_chunks
            }
        )
        tools_expected = (
            [case.required_tool]
            if case.case_type == "tool_required" and case.required_tool
            else []
        )
        return CaseResult(
            case_id=case.id,
            question=case.question,
            answer=response.answer,
            passed=passed,
            case_type=case.case_type,
            expected_behavior=expected_behavior,
            actual_refused=response.refused,
            grounded=grounded,
            correct=correct,
            appropriate_refusal=appropriate_refusal,
            fact_coverage=fact_coverage,
            expected_sources=expected_sources,
            actual_sources=actual_sources,
            retrieved_sources=sorted(
                {
                    known_chunks[chunk_id].source
                    for chunk_id in response.retrieved_chunk_ids
                    if chunk_id in known_chunks
                }
            ),
            retrieved_chunk_ids=response.retrieved_chunk_ids,
            tools_expected=tools_expected,
            tools_used=response.tools_used,
            deterministic_checks=checks.model_dump(mode="json"),
            matched_facts=matched_facts,
            missing_facts=missing_facts,
            judge_reason=judge_reason,
            scores={
                "source_recall": checks.source_recall,
                "fact_coverage": fact_coverage,
                "grounded": float(grounded),
                "correct": float(correct),
            },
            citations=response.source_chunk_ids,
            failure_evidence="; ".join(failure_reasons),
        )

    def run_case(self, case: EvaluationCase) -> CaseResult:
        try:
            response = self.agent_runner.answer(case.question)
        except Exception as exc:
            return self._failure_result(
                case,
                f"AgentRunner error: {type(exc).__name__}: {exc}",
            )

        known_chunks = self._known_chunks()
        known_chunk_sources = {
            chunk_id: chunk.source
            for chunk_id, chunk in known_chunks.items()
        }
        checks = check_case_deterministically(
            case,
            response,
            known_chunk_sources,
        )
        judge_result: JudgeResult | None = None
        if self._should_judge(case, response, checks):
            validated_evidence = [
                {
                    "chunk_id": chunk_id,
                    "source": known_chunks[chunk_id].source,
                    "page": known_chunks[chunk_id].page,
                    "text": known_chunks[chunk_id].text,
                }
                for chunk_id in response.source_chunk_ids
                if chunk_id in known_chunks
            ]
            try:
                judge_result = self.judge.judge(
                    case,
                    response,
                    deterministic_checks=checks,
                    validated_evidence=validated_evidence,
                )
            except Exception as exc:
                return self._failure_result(
                    case,
                    f"Judge error: {type(exc).__name__}: {exc}",
                    response,
                    checks,
                )
        return self._combine(
            case,
            response,
            checks,
            judge_result,
            known_chunks,
        )

    def run(self, evaluation_set: EvaluationSet) -> list[CaseResult]:
        results: list[CaseResult] = []
        for case in evaluation_set.cases:
            try:
                results.append(self.run_case(case))
            except Exception as exc:
                results.append(
                    self._failure_result(
                        case,
                        f"Evaluation error: {type(exc).__name__}: {exc}",
                    )
                )
        return results

    @staticmethod
    def summarize(results: list[CaseResult]) -> EvaluationRunResult:
        return EvaluationRunResult(results=results)
