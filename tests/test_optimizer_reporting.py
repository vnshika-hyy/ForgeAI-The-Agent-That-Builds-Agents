from pathlib import Path

from forgeai.evaluation.failure_classifier import (
    FailureClassification,
    FailureClassifier,
)
from forgeai.optimization.optimizer import OptimizationResult, Optimizer
from forgeai.reporting.reporter import Reporter
from forgeai.schemas.agent_definition import AgentDefinition, RetrievalDefinition
from forgeai.schemas.evaluation import CaseResult, EvaluationCase, EvaluationSet
from forgeai.schemas.reports import (
    GroundabilityReport,
    IterationScorecard,
    RequirementGroundability,
)


def make_agent() -> AgentDefinition:
    return AgentDefinition(
        agent_name="Test agent",
        purpose="Test purpose",
        system_prompt="Original prompt body.",
        retrieval=RetrievalDefinition(
            chunk_size=900,
            chunk_overlap=150,
            top_k=1,
        ),
        tools=[],
        refusal_policy="Refuse unsupported requests.",
        citation_policy="Cite sources.",
        output_contract="Be concise.",
    )


def case_result(case: EvaluationCase, passed: bool, **updates) -> CaseResult:
    values = {
        "case_id": case.id,
        "question": case.question,
        "answer": "Test answer",
        "passed": passed,
        "case_type": case.case_type,
        "expected_behavior": case.expected_behavior,
        "failure_evidence": "Expected evidence was not retrieved.",
        "missing_facts": ["Expected fact"],
    }
    values.update(updates)
    return CaseResult(**values)


def test_optimizer_preserves_groundability_boundaries_and_targets_hallucination():
    case = EvaluationCase(
        id="eval_1",
        question="A question",
        case_type="answerable",
        expected_behavior="answer",
    )
    report = GroundabilityReport(findings=[
        RequirementGroundability(
            requirement="Personal account balance",
            status="unsupported",
            reason="No personal account records are provided.",
        )
    ])

    class FixedClassifier:
        def classify_all(self, cases, results):
            return [
                FailureClassification(
                    case_id="eval_1",
                    category="hallucination",
                    reason="Unsupported claims.",
                    supporting_signals=["unsupported_claims"],
                )
            ]

    def evaluate(agent, evaluation_set):
        fixed = "No personal account records are provided."
        passed = "Grounding repair:" in agent.system_prompt
        assert fixed in agent.system_prompt
        return [case_result(case, passed)]

    optimized = Optimizer(FixedClassifier()).optimize(
        make_agent(),
        EvaluationSet(cases=[case]),
        evaluate,
        report,
    )

    assert optimized.agent_definition.system_prompt.startswith("Original prompt body.")
    assert "Grounding repair:" in optimized.agent_definition.system_prompt
    assert "No personal account records are provided." in optimized.agent_definition.system_prompt
    assert optimized.scorecards[0].changes
    assert optimized.scorecards[0].intended_cases == ["eval_1"]


def test_optimizer_applies_targeted_prompt_and_wrong_refusal_repairs():
    cases = [
        EvaluationCase(
            id="prompt_case",
            question="Evidence-backed question",
            case_type="answerable",
            expected_behavior="answer",
        ),
        EvaluationCase(
            id="refusal_case",
            question="Supported request",
            case_type="answerable",
            expected_behavior="answer",
        ),
    ]

    class FixedClassifier:
        def classify_all(self, cases, results):
            return [
                FailureClassification(
                    case_id="prompt_case",
                    category="prompt_defect",
                    reason="Prompt needs evidence-use instruction.",
                ),
                FailureClassification(
                    case_id="refusal_case",
                    category="wrong_refusal",
                    reason="The agent refused supported requests.",
                ),
            ]

    def evaluate(agent, evaluation_set):
        fixed = (
            "Behavior repair for evaluation cases prompt_case"
            in agent.system_prompt
            and "Refusal repair for evaluation cases refusal_case"
            in agent.system_prompt
        )
        return [case_result(case, fixed) for case in evaluation_set.cases]

    result = Optimizer(FixedClassifier()).optimize(
        make_agent(),
        EvaluationSet(cases=cases),
        evaluate,
        GroundabilityReport(findings=[]),
    )

    assert "Behavior repair for evaluation cases prompt_case" in result.agent_definition.system_prompt
    assert "Refusal repair for evaluation cases refusal_case" in result.agent_definition.system_prompt
    assert result.scorecards[0].failure_categories == {
        "prompt_defect": 1,
        "wrong_refusal": 1,
    }


def test_optimizer_stops_at_maximum_three_iterations():
    cases = [
        EvaluationCase(
            id=f"eval_{index}",
            question=f"Question {index}",
            case_type="answerable",
        )
        for index in range(4)
    ]
    evaluation_set = EvaluationSet(cases=cases)

    class FixedClassifier:
        def classify_all(self, cases, results):
            return [
                FailureClassification(
                    case_id=case.id,
                    category="retrieval_miss",
                    reason="Expected evidence missed.",
                    supporting_signals=["expected_evidence_not_retrieved"],
                )
                for case in cases
            ]

    def evaluate(agent, eval_set):
        passed_count = (agent.retrieval.top_k - 1) // 2
        return [
            case_result(case, index < passed_count)
            for index, case in enumerate(eval_set.cases)
        ]

    optimized = Optimizer(FixedClassifier()).optimize(
        make_agent(),
        evaluation_set,
        evaluate,
        GroundabilityReport(findings=[]),
    )

    assert len(optimized.scorecards) == 3
    assert optimized.stop_reason == "maximum of three optimization iterations reached"
    assert len(optimized.agent_versions) == 4
    assert len(optimized.results_history) == 4


def test_optimizer_stops_on_plateau_and_no_safe_change():
    case = EvaluationCase(
        id="eval_1",
        question="Question",
        case_type="answerable",
    )
    evaluation_set = EvaluationSet(cases=[case])
    agent = make_agent().model_copy(
        update={
            "retrieval": RetrievalDefinition(
                chunk_size=200,
                chunk_overlap=0,
                top_k=20,
                similarity_threshold=0.0,
            )
        }
    )

    class FixedClassifier:
        def classify_all(self, cases, results):
            return [
                FailureClassification(
                    case_id="eval_1",
                    category="retrieval_miss",
                    reason="Still missing.",
                )
            ]

    result = case_result(case, False)
    optimized = Optimizer(FixedClassifier()).optimize(
        agent,
        evaluation_set,
        lambda candidate, cases: [result],
        GroundabilityReport(findings=[]),
    )

    assert optimized.stop_reason == "no safe targeted changes are available"
    assert optimized.scorecards == []


def test_reporter_reports_real_scores_and_writes_json_markdown(tmp_path: Path):
    passed = CaseResult(
        case_id="eval_1",
        question="Q1",
        answer="A1",
        passed=True,
        grounded=True,
        matched_facts=["Fact"],
        fact_coverage=1.0,
    )
    failed = CaseResult(
        case_id="eval_2",
        question="Q2",
        answer="A2",
        passed=False,
        grounded=False,
        failure_category="hallucination",
        failure_evidence="Unsupported claim.",
    )
    optimization = OptimizationResult(
        agent_definition=make_agent(),
        initial_results=[passed, failed],
        final_results=[passed, failed],
        scorecards=[
            IterationScorecard(
                iteration=1,
                score_before=0.5,
                score_after=0.5,
                failure_cases=["eval_2"],
                failure_categories={"hallucination": 1},
                changes=["Added evidence-only instructions."],
                reasons=["Reduce unsupported claims."],
                intended_cases=["eval_2"],
                stop_reason="no meaningful improvement",
            )
        ],
        stop_reason="iteration produced no meaningful improvement",
    )
    groundability = GroundabilityReport(findings=[
        RequirementGroundability(
            requirement="Sensitive records",
            status="unsupported",
            reason="No supporting documents.",
        )
    ])

    reporter = Reporter()
    report = reporter.build(optimization, groundability)
    reporter.write(report, tmp_path)

    assert report.initial_score == 0.5
    assert report.final_score == 0.5
    assert report.failed_cases == 1
    assert report.remaining_failures[0].case_id == "eval_2"
    assert (tmp_path / "final_report.json").is_file()
    assert "50.0%" in (tmp_path / "final_report.md").read_text()
    assert report.model_dump_json()
