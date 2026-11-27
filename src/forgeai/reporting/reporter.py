from pathlib import Path
from typing import Any

from pydantic import Field

from ..evaluation.failure_classifier import FailureClassification
from ..optimization.optimizer import OptimizationResult
from ..schemas.agent_definition import AgentDefinition
from ..schemas.common import ForgeModel
from ..schemas.evaluation import CaseResult, EvaluationRunResult
from ..schemas.reports import GroundabilityReport, IterationScorecard
from .scorecard import build_scorecard


class FinalReport(ForgeModel):
    initial_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    total_cases: int
    passed_cases: int
    failed_cases: int
    groundedness_rate: float = Field(ge=0.0, le=1.0)
    refusal_accuracy: float = Field(ge=0.0, le=1.0)
    average_fact_coverage: float = Field(ge=0.0, le=1.0)
    iterations: list[IterationScorecard] = Field(default_factory=list)
    remaining_failures: list[CaseResult] = Field(default_factory=list)
    failure_categories: dict[str, int] = Field(default_factory=dict)
    classifications: list[FailureClassification] = Field(default_factory=list)
    likely_causes: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    retrieval_configuration: dict[str, Any] = Field(default_factory=dict)
    groundability_warnings: list[str] = Field(default_factory=list)
    unresolved_unsupported_capabilities: list[str] = Field(default_factory=list)
    stop_reason: str = ""


class Reporter:
    DEFAULT_LIMITATIONS = [
        "Retrieval uses lexical matching and may miss semantically related evidence.",
        "LLM Judge assessments can vary by local model and model version.",
        "Expected-fact semantic validation is not a formal entailment proof.",
        "PDF page provenance may be approximate when source extraction does not preserve page-level chunks.",
        "Weather tool results depend on availability of the external Open-Meteo service.",
    ]

    def build(
        self,
        optimization: OptimizationResult,
        groundability: GroundabilityReport,
        *,
        known_limitations: list[str] | None = None,
    ) -> FinalReport:
        initial = build_scorecard(optimization.initial_results)
        final = build_scorecard(optimization.final_results)
        remaining = [result for result in optimization.final_results if not result.passed]
        categories: dict[str, int] = {}
        for classification in optimization.classifications:
            categories[classification.category] = (
                categories.get(classification.category, 0) + 1
            )
        return FinalReport(
            initial_score=initial.score,
            final_score=final.score,
            total_cases=final.total,
            passed_cases=final.passed,
            failed_cases=final.failed,
            groundedness_rate=final.groundedness_rate,
            refusal_accuracy=final.refusal_accuracy,
            average_fact_coverage=final.average_fact_coverage,
            iterations=optimization.scorecards,
            remaining_failures=remaining,
            failure_categories=categories,
            classifications=optimization.classifications,
            likely_causes=[
                f"{item.case_id} ({item.category}): {item.reason}"
                for item in optimization.classifications
            ],
            known_limitations=list(
                dict.fromkeys([*(known_limitations or []), *self.DEFAULT_LIMITATIONS])
            ),
            selected_tools=optimization.agent_definition.tools,
            retrieval_configuration=optimization.agent_definition.retrieval.model_dump(),
            groundability_warnings=groundability.warnings,
            unresolved_unsupported_capabilities=[
                f"{finding.requirement}: {finding.reason}"
                for finding in groundability.findings
                if finding.status in {"unsupported", "partially_supported"}
            ],
            stop_reason=optimization.stop_reason,
        )

    @staticmethod
    def to_markdown(report: FinalReport) -> str:
        lines = [
            "# ForgeAI Final Report",
            "",
            "## Evaluation summary",
            "",
            f"- Initial score: {report.initial_score:.1%}",
            f"- Final score: {report.final_score:.1%}",
            f"- Cases: {report.total_cases} ({report.passed_cases} passed, {report.failed_cases} failed)",
            f"- Groundedness rate: {report.groundedness_rate:.1%}",
            f"- Refusal accuracy: {report.refusal_accuracy:.1%}",
            f"- Average fact coverage: {report.average_fact_coverage:.1%}",
            "",
            f"Stop reason: {report.stop_reason or 'not recorded'}",
            "",
            "## Optimization iterations",
            "",
        ]
        if not report.iterations:
            lines.append("No optimization iterations were needed or applied.")
        for iteration in report.iterations:
            lines.extend(
                [
                    f"### Iteration {iteration.iteration}",
                    f"- Score: {iteration.score_before:.1%} → {iteration.score_after:.1%}",
                    f"- Failed cases: {', '.join(iteration.failure_cases) or 'none'}",
                    f"- Diagnoses: {iteration.diagnosis or 'not recorded'}",
                ]
            )
            lines.extend(f"- Change: {change}" for change in iteration.changes)
            lines.extend(f"- Reason: {reason}" for reason in iteration.reasons)
            if iteration.stop_reason:
                lines.append(f"- Stop: {iteration.stop_reason}")
            lines.append("")

        lines.extend(["## Remaining failures", ""])
        if report.remaining_failures:
            for result in report.remaining_failures:
                lines.append(
                    f"- **{result.case_id}** ({result.failure_category or 'unclassified'}): "
                    f"{result.failure_evidence or result.error or result.judge_reason or 'No reason recorded.'}"
                )
            if report.likely_causes:
                lines.extend(
                    ["", "Likely causes:", *(f"- {cause}" for cause in report.likely_causes)]
                )
        else:
            lines.append("No failed cases remain.")

        lines.extend(
            [
                "",
                "## Groundability warnings",
                "",
                *(f"- {warning}" for warning in report.groundability_warnings),
                "",
                "## Unresolved unsupported or partially supported capabilities",
                "",
                *(
                    f"- {capability}"
                    for capability in report.unresolved_unsupported_capabilities
                ),
                "",
                "## Configuration",
                "",
                f"- Selected tools: {', '.join(report.selected_tools) or 'none'}",
                f"- Retrieval: `{report.retrieval_configuration}`",
                "",
                "## Known limitations",
                "",
                *(f"- {limitation}" for limitation in report.known_limitations),
                "",
            ]
        )
        return "\n".join(lines)

    def write(self, report: FinalReport, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "final_report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (output_dir / "final_report.md").write_text(
            self.to_markdown(report),
            encoding="utf-8",
        )
