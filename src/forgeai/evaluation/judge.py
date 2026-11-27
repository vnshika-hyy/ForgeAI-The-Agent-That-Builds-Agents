import json
import re
from pathlib import Path

from pydantic import Field

from ..llm.client import LLMClient
from ..runtime.agent_runner import AgentResponse
from ..schemas.common import ForgeModel
from ..schemas.evaluation import EvaluationCase
from .deterministic_checks import DeterministicCheckResult


class FactAssessment(ForgeModel):
    fact: str
    covered: bool
    reason: str = ""


class JudgeVerdict(ForgeModel):
    fact_assessments: list[FactAssessment] = Field(default_factory=list)
    grounded: bool
    correct: bool
    appropriate_refusal: bool
    expected_behavior_met: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str


class JudgeResult(ForgeModel):
    grounded: bool
    correct: bool
    fact_coverage: float = Field(ge=0.0, le=1.0)
    appropriate_refusal: bool
    expected_behavior_met: bool
    matched_facts: list[str] = Field(default_factory=list)
    missing_expected_facts: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str


class Judge:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self._prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "judge.txt"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^\w]+", " ", text.casefold()).strip()

    def judge(
        self,
        case: EvaluationCase,
        response: AgentResponse,
        *,
        deterministic_checks: DeterministicCheckResult | None = None,
        validated_evidence: list[dict[str, object]] | None = None,
    ) -> JudgeResult:
        checks = deterministic_checks or DeterministicCheckResult()
        evidence = validated_evidence or []
        input_data = {
            "question": case.question,
            "expected_behavior": case.expected_behavior,
            "expected_facts": [
                {
                    "fact": fact,
                    "deterministically_matched": fact in checks.matched_facts,
                }
                for fact in case.expected_key_facts
            ],
            "expected_sources": case.expected_source_documents,
            "expected_chunk_ids": case.supporting_chunk_ids,
            "actual_response": response.answer,
            "actual_refused": response.refused,
            "actual_validated_sources": response.source_chunk_ids,
            "tools_used": response.tools_used,
            "deterministic_check_results": checks.model_dump(mode="json"),
            "validated_evidence": evidence,
        }
        prompt = (
            self._prompt_path.read_text(encoding="utf-8")
            + "\n\nEVALUATION INPUT:\n"
            + json.dumps(input_data, ensure_ascii=False)
        )
        verdict = self.llm_client.generate_json(prompt, JudgeVerdict)

        semantic_assessments: dict[str, bool] = {}
        expected_normalized = {
            self._normalize(fact): fact for fact in case.expected_key_facts
        }
        for assessment in verdict.fact_assessments:
            normalized = self._normalize(assessment.fact)
            if normalized in expected_normalized and normalized not in semantic_assessments:
                semantic_assessments[normalized] = assessment.covered

        matched_facts = list(checks.matched_facts)
        missing_facts: list[str] = []
        for fact in case.expected_key_facts:
            normalized = self._normalize(fact)
            if fact in checks.matched_facts:
                continue
            if semantic_assessments.get(normalized, False):
                matched_facts.append(fact)
            else:
                missing_facts.append(fact)
        fact_coverage = (
            len(matched_facts) / len(case.expected_key_facts)
            if case.expected_key_facts
            else 1.0
        )
        return JudgeResult(
            grounded=verdict.grounded,
            correct=verdict.correct and fact_coverage == 1.0,
            fact_coverage=max(0.0, min(1.0, fact_coverage)),
            appropriate_refusal=verdict.appropriate_refusal,
            expected_behavior_met=verdict.expected_behavior_met,
            matched_facts=matched_facts,
            missing_expected_facts=missing_facts,
            unsupported_claims=verdict.unsupported_claims,
            reason=verdict.reason.strip(),
        )
