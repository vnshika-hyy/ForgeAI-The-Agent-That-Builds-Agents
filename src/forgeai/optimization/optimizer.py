from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from ..evaluation.failure_classifier import (
    FailureClassification,
    FailureClassifier,
)
from ..schemas.agent_definition import AgentDefinition, RetrievalDefinition
from ..schemas.evaluation import CaseResult, EvaluationSet, EvaluationRunResult
from ..schemas.reports import GroundabilityReport, IterationScorecard
from .critic import Critic


@dataclass
class OptimizationResult:
    agent_definition: AgentDefinition
    initial_results: list[CaseResult]
    final_results: list[CaseResult]
    classifications: list[FailureClassification] = field(default_factory=list)
    classification_history: list[list[FailureClassification]] = field(default_factory=list)
    scorecards: list[IterationScorecard] = field(default_factory=list)
    agent_versions: list[AgentDefinition] = field(default_factory=list)
    results_history: list[list[CaseResult]] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def initial_score(self) -> float:
        return EvaluationRunResult(results=self.initial_results).score

    @property
    def final_score(self) -> float:
        return EvaluationRunResult(results=self.final_results).score


class Optimizer:
    MAX_ITERATIONS = 3

    def __init__(
        self,
        failure_classifier: FailureClassifier,
        critic: Critic | None = None,
    ) -> None:
        self.failure_classifier = failure_classifier
        self.critic = critic or Critic()

    @staticmethod
    def _boundary_block(groundability: GroundabilityReport) -> str:
        boundaries = [
            f"- {finding.requirement} [{finding.status}]: {finding.reason}"
            for finding in groundability.findings
            if finding.status in {"unsupported", "partially_supported"}
        ]
        if not boundaries:
            return ""
        return (
            "Authoritative GroundabilityGuard boundaries (must not be weakened):\n"
            + "\n".join(boundaries)
        )

    @staticmethod
    def _append_once(prompt: str, instruction: str) -> str:
        if instruction in prompt:
            return prompt
        return f"{prompt.rstrip()}\n\n{instruction}"

    def _apply_category_change(
        self,
        agent: AgentDefinition,
        category: str,
        classifications: list[FailureClassification],
        groundability: GroundabilityReport,
    ) -> tuple[AgentDefinition, str, str, list[str]] | None:
        affected = [
            classification
            for classification in classifications
            if classification.category == category
        ]
        if not affected:
            return None
        case_ids = [item.case_id for item in affected]
        diagnosis = self.critic.diagnose(affected).get(category, "")
        updated = agent.model_copy(deep=True)

        if category == "retrieval_miss":
            retrieval = agent.retrieval
            if retrieval.top_k < 20:
                updated.retrieval = retrieval.model_copy(
                    update={"top_k": min(20, retrieval.top_k + 2)}
                )
                change = f"retrieval.top_k: {retrieval.top_k} -> {updated.retrieval.top_k}"
                reason = "Retrieve more candidate chunks for cases that missed expected evidence."
            elif retrieval.similarity_threshold > 0:
                updated.retrieval = retrieval.model_copy(
                    update={"similarity_threshold": 0.0}
                )
                change = (
                    "retrieval.similarity_threshold: "
                    f"{retrieval.similarity_threshold} -> 0.0"
                )
                reason = "Allow lower-scoring relevant candidates through the existing threshold."
            elif retrieval.chunk_size > 200:
                new_size = max(200, retrieval.chunk_size - 100)
                updated.retrieval = RetrievalDefinition(
                    chunk_size=new_size,
                    chunk_overlap=min(retrieval.chunk_overlap, new_size - 1),
                    top_k=retrieval.top_k,
                    similarity_threshold=retrieval.similarity_threshold,
                )
                change = f"retrieval.chunk_size: {retrieval.chunk_size} -> {new_size}"
                reason = "Use smaller, more focused retrieval chunks for missed facts."
            else:
                return None
        elif category == "hallucination":
            instruction = (
                "Grounding repair: state factual claims only when supported by "
                "retrieved supplied evidence; cite the supporting source. If the "
                "evidence is absent or insufficient, explicitly refuse rather than "
                "guessing."
            )
            updated.system_prompt = self._append_once(
                updated.system_prompt,
                instruction,
            )
            change = "Appended evidence-only and unsupported-answer refusal instruction."
            reason = "Reduce unsupported factual claims without changing retrieval, tools, or scope."
        elif category == "prompt_defect":
            cases = ", ".join(case_ids)
            instruction = (
                f"Behavior repair for evaluation cases {cases}: follow the supplied "
                "scope, use retrieved evidence directly, include required facts "
                "concisely, and follow the configured expected response behavior."
            )
            updated.system_prompt = self._append_once(
                updated.system_prompt,
                instruction,
            )
            change = "Appended a case-targeted behavior and evidence-use instruction."
            reason = diagnosis
        elif category == "wrong_refusal":
            instruction = (
                f"Refusal repair for evaluation cases {', '.join(case_ids)}: when a "
                "request is within scope and supported by retrieved evidence, answer "
                "it rather than refusing. Continue to refuse unsupported requests."
            )
            updated.system_prompt = self._append_once(
                updated.system_prompt,
                instruction,
            )
            change = "Appended an answer-supported-requests instruction."
            reason = diagnosis
        else:
            return None

        boundary_block = self._boundary_block(groundability)
        if boundary_block and boundary_block not in updated.system_prompt:
            updated.system_prompt = self._append_once(
                updated.system_prompt,
                boundary_block,
            )
            change += " Preserved authoritative groundability boundaries."
        return updated, change, reason or diagnosis, case_ids

    def _apply_changes(
        self,
        agent: AgentDefinition,
        classifications: list[FailureClassification],
        groundability: GroundabilityReport,
    ) -> tuple[AgentDefinition, list[str], list[str], list[str]]:
        updated = agent
        changes: list[str] = []
        reasons: list[str] = []
        targets: list[str] = []
        for category in (
            "retrieval_miss",
            "prompt_defect",
            "hallucination",
            "wrong_refusal",
        ):
            change = self._apply_category_change(
                updated,
                category,
                classifications,
                groundability,
            )
            if change is None:
                continue
            updated, description, reason, case_ids = change
            changes.append(f"{category}: {description}")
            reasons.append(f"{category}: {reason}")
            targets.extend(case_ids)
        return updated, changes, reasons, list(dict.fromkeys(targets))

    def optimize(
        self,
        agent_definition: AgentDefinition,
        evaluation_set: EvaluationSet,
        evaluate: Callable[[AgentDefinition, EvaluationSet], list[CaseResult]],
        groundability: GroundabilityReport,
    ) -> OptimizationResult:
        current_agent = agent_definition.model_copy(deep=True)
        boundary_block = self._boundary_block(groundability)
        if boundary_block and boundary_block not in current_agent.system_prompt:
            current_agent.system_prompt = self._append_once(
                current_agent.system_prompt,
                boundary_block,
            )
        initial_results = evaluate(current_agent, evaluation_set)
        current_results = initial_results
        best_agent = current_agent
        best_results = current_results
        classifications: list[FailureClassification] = []
        classification_history: list[list[FailureClassification]] = []
        scorecards: list[IterationScorecard] = []
        versions = [current_agent.model_copy(deep=True)]
        results_history = [initial_results]
        stop_reason = "maximum iterations reached"

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            before_score = EvaluationRunResult(results=current_results).score
            if all(result.passed for result in current_results):
                stop_reason = "all evaluation cases passed"
                break

            classifications = self.failure_classifier.classify_all(
                evaluation_set.cases,
                current_results,
            )
            classification_history.append(list(classifications))
            diagnosis_by_category = self.critic.diagnose(classifications)
            candidate, changes, reasons, targeted_cases = self._apply_changes(
                current_agent,
                classifications,
                groundability,
            )
            if not changes:
                stop_reason = "no safe targeted changes are available"
                break

            candidate_results = evaluate(candidate, evaluation_set)
            after_score = EvaluationRunResult(results=candidate_results).score
            category_counts = dict(Counter(item.category for item in classifications))
            scorecards.append(
                IterationScorecard(
                    iteration=iteration,
                    score_before=before_score,
                    score_after=after_score,
                    failure_cases=[item.case_id for item in classifications],
                    failure_categories=category_counts,
                    changes=changes,
                    reasons=reasons,
                    intended_cases=targeted_cases,
                    diagnosis="; ".join(diagnosis_by_category.values()),
                    stop_reason=(
                        "no meaningful improvement"
                        if after_score <= before_score
                        else ""
                    ),
                )
            )
            versions.append(candidate.model_copy(deep=True))
            results_history.append(candidate_results)
            current_agent = candidate
            current_results = candidate_results

            if after_score > EvaluationRunResult(results=best_results).score:
                best_agent = candidate
                best_results = candidate_results
            if all(result.passed for result in candidate_results):
                stop_reason = "all evaluation cases passed"
                break
            if after_score <= before_score:
                stop_reason = "iteration produced no meaningful improvement"
                break
            if iteration == self.MAX_ITERATIONS:
                stop_reason = "maximum of three optimization iterations reached"

        if not scorecards and not any(result.passed for result in initial_results):
            classifications = self.failure_classifier.classify_all(
                evaluation_set.cases,
                initial_results,
            )
            if not classification_history:
                classification_history.append(list(classifications))
        final_classifications = (
            self.failure_classifier.classify_all(
                evaluation_set.cases,
                best_results,
            )
            if any(not result.passed for result in best_results)
            else []
        )
        return OptimizationResult(
            agent_definition=best_agent,
            initial_results=initial_results,
            final_results=best_results,
            classifications=final_classifications,
            classification_history=classification_history,
            scorecards=scorecards,
            agent_versions=versions,
            results_history=results_history,
            stop_reason=stop_reason,
        )
