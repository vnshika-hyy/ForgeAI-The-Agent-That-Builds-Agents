from forgeai.evaluation.failure_classifier import FailureClassifier
from forgeai.schemas.evaluation import CaseResult, EvaluationCase


def make_case(**overrides):
    values = {
        "id": "eval_001",
        "question": "Question?",
        "case_type": "answerable",
        "expected_behavior": "answer",
    }
    values.update(overrides)
    return EvaluationCase(**values)


def make_result(**overrides):
    values = {
        "case_id": "eval_001",
        "question": "Question?",
        "answer": "Answer",
        "passed": False,
    }
    values.update(overrides)
    return CaseResult(**values)


def test_classifies_wrong_refusal_deterministically():
    result = FailureClassifier().classify(
        make_case(expected_behavior="answer"),
        make_result(actual_refused=True),
    )

    assert result.category == "wrong_refusal"
    assert result.case_id == "eval_001"


def test_classifies_retrieval_miss_from_missing_expected_chunks():
    result = FailureClassifier().classify(
        make_case(
            supporting_chunk_ids=["policy.txt::chunk_4"],
            expected_key_facts=["A fact from the policy."],
        ),
        make_result(
            missing_facts=["A fact from the policy."],
            retrieved_chunk_ids=["policy.txt::chunk_0"],
        ),
    )

    assert result.category == "retrieval_miss"
    assert "expected_evidence_not_retrieved" in result.supporting_signals


def test_classifies_hallucination_from_unsupported_claim_signal():
    result = FailureClassifier().classify(
        make_case(),
        make_result(failure_evidence="Unsupported claims: invented data."),
    )

    assert result.category == "hallucination"


def test_classifies_prompt_defect_when_expected_evidence_was_retrieved():
    result = FailureClassifier().classify(
        make_case(
            supporting_chunk_ids=["policy.txt::chunk_0"],
            expected_key_facts=["A fact from the policy."],
        ),
        make_result(
            missing_facts=["A fact from the policy."],
            retrieved_chunk_ids=["policy.txt::chunk_0"],
        ),
    )

    assert result.category == "prompt_defect"
