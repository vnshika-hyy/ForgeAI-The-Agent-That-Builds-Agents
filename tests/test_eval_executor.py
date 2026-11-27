from types import SimpleNamespace

from forgeai.evaluation.executor import EvalExecutor
from forgeai.evaluation.judge import JudgeResult
from forgeai.ingestion.chunker import Chunk
from forgeai.runtime.agent_runner import AgentResponse
from forgeai.schemas.evaluation import (
    CaseResult,
    EvaluationCase,
    EvaluationRunResult,
    EvaluationSet,
)


class FakeAgentRunner:
    def __init__(self, responses: dict[str, AgentResponse | Exception]):
        self.responses = responses
        self.calls: list[str] = []
        self.retriever = SimpleNamespace(
            chunks=[
                Chunk(
                    text="The Alpha team receives 7 training days each year.",
                    source="policy.txt",
                    document_type="text",
                    chunk_index=0,
                    chunk_id="policy.txt::chunk_0",
                    metadata={},
                )
            ]
        )

    def answer(self, question: str) -> AgentResponse:
        self.calls.append(question)
        value = self.responses[question]
        if isinstance(value, Exception):
            raise value
        return value


class FakeJudge:
    def __init__(self, result: JudgeResult):
        self.result = result
        self.calls = 0

    def judge(self, case, response, **kwargs):
        self.calls += 1
        return self.result


def judge_result(
    *,
    fact_coverage: float = 1.0,
    correct: bool = True,
    grounded: bool = True,
    behavior_met: bool = True,
) -> JudgeResult:
    return JudgeResult(
        grounded=grounded,
        correct=correct,
        fact_coverage=fact_coverage,
        appropriate_refusal=False,
        expected_behavior_met=behavior_met,
        reason="Test judge result.",
    )


def make_case(
    question: str,
    *,
    expected_behavior: str = "answer",
    case_type: str = "answerable",
    facts: list[str] | None = None,
    sources: list[str] | None = None,
    chunk_ids: list[str] | None = None,
    required_tool: str | None = None,
) -> EvaluationCase:
    return EvaluationCase(
        id=question,
        question=question,
        case_type=case_type,
        expected_behavior=expected_behavior,
        expected_key_facts=facts or [],
        expected_source_documents=sources or [],
        supporting_chunk_ids=chunk_ids or [],
        required_tool=required_tool,
    )


def test_correct_expected_refusal_passes_deterministically():
    case = make_case(
        "request outside scope",
        expected_behavior="refuse",
        case_type="out_of_scope",
    )
    runner = FakeAgentRunner({
        case.question: AgentResponse(answer="I can't help with that.", refused=True)
    })
    judge = FakeJudge(judge_result())

    result = EvalExecutor(runner, judge).run_case(case)

    assert result.passed
    assert result.appropriate_refusal
    assert judge.calls == 0


def test_wrong_refusal_fails_regardless_of_optimistic_judge():
    case = make_case(
        "answerable request",
        expected_behavior="answer",
        facts=["The Alpha team receives 7 training days each year."],
        sources=["policy.txt"],
    )
    runner = FakeAgentRunner({
        case.question: AgentResponse(answer="I cannot answer.", refused=True)
    })
    judge = FakeJudge(judge_result())

    result = EvalExecutor(runner, judge).run_case(case)

    assert not result.passed
    assert not result.correct
    assert judge.calls == 0


def test_semantic_fact_paraphrase_receives_judge_credit():
    fact = "The Alpha team receives seven training days annually."
    case = make_case(
        "How much training does Alpha receive?",
        facts=[fact],
        sources=["policy.txt"],
        chunk_ids=["policy.txt::chunk_0"],
    )
    runner = FakeAgentRunner({
        case.question: AgentResponse(
            answer="Alpha gets 7 training days every year.",
            source_chunk_ids=["policy.txt::chunk_0"],
        )
    })
    judge = FakeJudge(judge_result())

    result = EvalExecutor(runner, judge).run_case(case)

    assert result.passed
    assert result.fact_coverage == 1.0
    assert result.grounded
    assert judge.calls == 1


def test_missing_fact_lowers_coverage():
    case = make_case(
        "Question with two required facts",
        facts=["Fact one is true.", "Fact two is true."],
        sources=["policy.txt"],
        chunk_ids=["policy.txt::chunk_0"],
    )
    runner = FakeAgentRunner({
        case.question: AgentResponse(
            answer="Fact one is true.",
            source_chunk_ids=["policy.txt::chunk_0"],
        )
    })
    judge = FakeJudge(judge_result(fact_coverage=0.5, correct=False))

    result = EvalExecutor(runner, judge).run_case(case)

    assert not result.passed
    assert result.fact_coverage == 0.5


def test_plausible_uncited_answer_cannot_be_grounded_by_judge():
    case = make_case(
        "How many training days?",
        facts=["The Alpha team receives 7 training days each year."],
        sources=["policy.txt"],
        chunk_ids=["policy.txt::chunk_0"],
    )
    runner = FakeAgentRunner({
        case.question: AgentResponse(
            answer="The Alpha team receives 7 training days each year."
        )
    })
    judge = FakeJudge(judge_result(grounded=True))

    result = EvalExecutor(runner, judge).run_case(case)

    assert not result.passed
    assert not result.grounded
    assert judge.calls == 0


def test_tool_required_case_passes_only_when_expected_tool_was_used():
    case = make_case(
        "Calculate the total",
        expected_behavior="use_tool",
        case_type="tool_required",
        required_tool="calculator",
    )
    response = AgentResponse(
        answer="The result is 42.",
        tools_used=["calculator"],
    )
    runner = FakeAgentRunner({case.question: response})
    judge = FakeJudge(judge_result())

    result = EvalExecutor(runner, judge).run_case(case)

    assert result.passed
    assert result.tools_used == ["calculator"]
    assert result.deterministic_checks["required_tool_used"] is True


def test_judge_cannot_override_missing_required_tool():
    case = make_case(
        "Calculate the total",
        expected_behavior="use_tool",
        case_type="tool_required",
        facts=["The Alpha team receives 7 training days each year."],
        sources=["policy.txt"],
        chunk_ids=["policy.txt::chunk_0"],
        required_tool="calculator",
    )
    runner = FakeAgentRunner({
        case.question: AgentResponse(
            answer="The Alpha team receives 7 training days each year.",
            source_chunk_ids=["policy.txt::chunk_0"],
        )
    })
    judge = FakeJudge(judge_result())

    result = EvalExecutor(runner, judge).run_case(case)

    assert not result.passed
    assert result.deterministic_checks["required_tool_used"] is False
    assert judge.calls == 1


def test_concise_complete_answer_outscores_verbose_incomplete_answer():
    facts = ["Fact one is true.", "Fact two is true."]
    concise_case = make_case(
        "Concise",
        facts=facts,
        sources=["policy.txt"],
        chunk_ids=["policy.txt::chunk_0"],
    )
    verbose_case = make_case(
        "Verbose",
        facts=facts,
        sources=["policy.txt"],
        chunk_ids=["policy.txt::chunk_0"],
    )
    runner = FakeAgentRunner({
        "Concise": AgentResponse(
            answer="Fact one is true. Fact two is true.",
            source_chunk_ids=["policy.txt::chunk_0"],
        ),
        "Verbose": AgentResponse(
            answer=("Fact one is true. " + "Irrelevant detail. " * 20),
            source_chunk_ids=["policy.txt::chunk_0"],
        ),
    })

    class ResponseJudge:
        def judge(self, case, response, **kwargs):
            if case.question == "Concise":
                return judge_result()
            return judge_result(fact_coverage=0.5, correct=False)

    executor = EvalExecutor(runner, ResponseJudge())
    concise = executor.run_case(concise_case)
    verbose = executor.run_case(verbose_case)

    assert concise.passed
    assert not verbose.passed
    assert verbose.fact_coverage == 0.5


def test_executor_isolates_agent_errors_and_continues():
    first = make_case("raises", expected_behavior="refuse", case_type="out_of_scope")
    second = make_case("continues", expected_behavior="refuse", case_type="out_of_scope")
    runner = FakeAgentRunner({
        "raises": RuntimeError("agent failed"),
        "continues": AgentResponse(answer="No supported data.", refused=True),
    })
    executor = EvalExecutor(runner, FakeJudge(judge_result()))

    results = executor.run(EvaluationSet(cases=[first, second]))

    assert not results[0].passed
    assert "agent failed" in results[0].error
    assert results[1].passed
    assert runner.calls == ["raises", "continues"]


def test_executor_isolates_judge_errors_and_continues():
    first = make_case(
        "judge raises",
        facts=["The Alpha team receives 7 training days each year."],
        sources=["policy.txt"],
        chunk_ids=["policy.txt::chunk_0"],
    )
    second = make_case(
        "still running",
        expected_behavior="refuse",
        case_type="out_of_scope",
    )
    runner = FakeAgentRunner({
        first.question: AgentResponse(
            answer="The Alpha team receives 7 training days each year.",
            source_chunk_ids=["policy.txt::chunk_0"],
        ),
        second.question: AgentResponse(answer="I can't answer that.", refused=True),
    })

    class FirstJudgeCallRaises:
        def __init__(self):
            self.calls = 0

        def judge(self, case, response, **kwargs):
            self.calls += 1
            raise RuntimeError("judge unavailable")

    judge = FirstJudgeCallRaises()
    results = EvalExecutor(runner, judge).run(
        EvaluationSet(cases=[first, second])
    )

    assert not results[0].passed
    assert "Judge error" in results[0].error
    assert results[1].passed
    assert runner.calls == ["judge raises", "still running"]


def test_evaluation_summary_computes_8_of_10_score():
    results = [
        CaseResult(
            case_id=str(index),
            question="Question",
            answer="Answer",
            passed=index < 8,
        )
        for index in range(10)
    ]

    summary = EvaluationRunResult(results=results)

    assert summary.total == 10
    assert summary.passed == 8
    assert summary.failed == 2
    assert summary.score == 0.8
    assert summary.model_dump_json()
