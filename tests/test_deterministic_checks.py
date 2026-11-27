from forgeai.evaluation.deterministic_checks import check_case_deterministically
from forgeai.runtime.agent_runner import AgentResponse
from forgeai.schemas.evaluation import EvaluationCase


def make_case(**overrides) -> EvaluationCase:
    values = {
        "id": "eval_001",
        "question": "Question?",
        "case_type": "answerable",
        "expected_behavior": "answer",
        "expected_key_facts": [],
    }
    values.update(overrides)
    return EvaluationCase(**values)


def test_expected_refusal_is_checked_deterministically():
    checks = check_case_deterministically(
        make_case(
            case_type="out_of_scope",
            expected_behavior="refuse",
        ),
        AgentResponse(answer="I can't answer that.", refused=True),
        {},
    )

    assert checks.refusal_correct is True
    assert not checks.hard_failures


def test_wrong_refusal_is_a_deterministic_failure():
    checks = check_case_deterministically(
        make_case(expected_behavior="answer"),
        AgentResponse(answer="I can't answer that.", refused=True),
        {},
    )

    assert checks.wrong_refusal
    assert any("refused" in failure for failure in checks.hard_failures)


def test_normalized_exact_fact_match_is_detected():
    case = make_case(
        expected_key_facts=["Employees receive 24 days of annual leave."],
    )
    response = AgentResponse(
        answer="Employees receive 24 days of annual leave!",
        source_chunk_ids=["policy.txt::chunk_0"],
    )

    checks = check_case_deterministically(
        case,
        response,
        {"policy.txt::chunk_0": "policy.txt"},
    )

    assert checks.matched_facts == case.expected_key_facts
    assert checks.exact_fact_coverage == 1.0
    assert checks.source_grounded


def test_invalid_and_missing_expected_sources_cannot_be_grounded():
    case = make_case(
        expected_source_documents=["policy.txt"],
        expected_key_facts=["A supported fact."],
    )
    response = AgentResponse(
        answer="A supported fact.",
        source_chunk_ids=["fake.pdf::chunk_9"],
    )

    checks = check_case_deterministically(
        case,
        response,
        {"policy.txt::chunk_0": "policy.txt"},
    )

    assert not checks.source_ids_valid
    assert checks.expected_source_hit is False
    assert not checks.source_grounded
    assert checks.hard_failures


def test_cross_document_case_requires_source_from_both_documents():
    case = make_case(
        case_type="cross_document",
        expected_source_documents=["policy_a.txt", "policy_b.txt"],
    )
    response = AgentResponse(
        answer="Combined answer.",
        source_chunk_ids=["policy_a.txt::chunk_0"],
    )

    checks = check_case_deterministically(
        case,
        response,
        {
            "policy_a.txt::chunk_0": "policy_a.txt",
            "policy_b.txt::chunk_0": "policy_b.txt",
        },
    )

    assert checks.source_recall == 0.5
    assert not checks.source_grounded
    assert any("all expected cross-document" in item for item in checks.hard_failures)


def test_tool_required_case_without_a_named_tool_fails():
    checks = check_case_deterministically(
        make_case(case_type="tool_required", expected_behavior="use_tool"),
        AgentResponse(answer="A result."),
        {},
    )

    assert checks.required_tool_used is False
    assert any("does not specify" in item for item in checks.hard_failures)
