from forgeai.evaluation.deterministic_checks import check_case_deterministically
from forgeai.evaluation.judge import Judge
from forgeai.runtime.agent_runner import AgentResponse
from forgeai.schemas.evaluation import EvaluationCase


class FakeLLMClient:
    def __init__(self, verdict: dict):
        self.verdict = verdict
        self.prompt = ""

    def generate_json(self, prompt: str, schema: type):
        self.prompt = prompt
        return schema.model_validate(self.verdict)


def make_case(facts: list[str]) -> EvaluationCase:
    return EvaluationCase(
        id="eval_001",
        question="What is the annual leave entitlement?",
        case_type="answerable",
        expected_behavior="answer",
        expected_key_facts=facts,
        expected_source_documents=["policy.txt"],
        supporting_chunk_ids=["policy.txt::chunk_0"],
    )


def test_judge_accepts_semantic_paraphrase_for_expected_fact():
    fact = "Employees receive 24 annual leave days."
    case = make_case([fact])
    response = AgentResponse(
        answer="Staff members are entitled to twenty-four days of annual leave.",
        source_chunk_ids=["policy.txt::chunk_0"],
    )
    checks = check_case_deterministically(
        case,
        response,
        {"policy.txt::chunk_0": "policy.txt"},
    )
    client = FakeLLMClient({
        "fact_assessments": [
            {"fact": fact, "covered": True, "reason": "Same entitlement, paraphrased."}
        ],
        "grounded": True,
        "correct": True,
        "appropriate_refusal": False,
        "expected_behavior_met": True,
        "unsupported_claims": [],
        "reason": "The response states the expected entitlement.",
    })

    result = Judge(client).judge(
        case,
        response,
        deterministic_checks=checks,
        validated_evidence=[
            {
                "chunk_id": "policy.txt::chunk_0",
                "source": "policy.txt",
                "text": "Employees receive 24 annual leave days.",
            }
        ],
    )

    assert result.fact_coverage == 1.0
    assert result.matched_facts == [fact]
    assert result.correct


def test_missing_expected_fact_reduces_fact_coverage():
    facts = [
        "Employees receive 24 annual leave days.",
        "The annual leave year starts in January.",
    ]
    case = make_case(facts)
    response = AgentResponse(
        answer="Employees receive 24 annual leave days.",
        source_chunk_ids=["policy.txt::chunk_0"],
    )
    checks = check_case_deterministically(
        case,
        response,
        {"policy.txt::chunk_0": "policy.txt"},
    )
    client = FakeLLMClient({
        "fact_assessments": [
            {
                "fact": facts[1],
                "covered": False,
                "reason": "The answer does not mention the start of the leave year.",
            }
        ],
        "grounded": True,
        "correct": False,
        "appropriate_refusal": False,
        "expected_behavior_met": True,
        "unsupported_claims": [],
        "reason": "One expected fact is missing.",
    })

    result = Judge(client).judge(
        case,
        response,
        deterministic_checks=checks,
        validated_evidence=[],
    )

    assert result.fact_coverage == 0.5
    assert result.missing_expected_facts == [facts[1]]
    assert not result.correct


def test_judge_receives_neutral_evidence_focused_prompt():
    case = make_case(["A complete concise fact."])
    response = AgentResponse(answer="A complete concise fact.")
    client = FakeLLMClient({
        "fact_assessments": [],
        "grounded": False,
        "correct": False,
        "appropriate_refusal": False,
        "expected_behavior_met": False,
        "unsupported_claims": [],
        "reason": "Evidence is missing.",
    })

    Judge(client).judge(case, response)

    assert "Longer answers are NOT inherently better" in client.prompt
    assert "QUESTION" in client.prompt
    assert "VALIDATED EVIDENCE" in client.prompt
