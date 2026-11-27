from forgeai.evaluation.eval_generator import EvalGenerator
from forgeai.ingestion.loader import Document
from forgeai.schemas.agent_definition import AgentDefinition, RetrievalDefinition
from forgeai.schemas.reports import (
    GroundabilityReport,
    RequirementGroundability,
)
from forgeai.schemas.requirements import Requirements


class FakeLLMClient:
    def __init__(self, respond):
        self.respond = respond
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, schema: type):
        self.prompts.append(prompt)
        return schema.model_validate({"cases": self.respond(prompt)})


def make_requirements() -> Requirements:
    return Requirements(
        agent_name="Generic Assistant",
        purpose="Answer questions from supplied information.",
        scope="Supplied documents",
        target_audience="Users",
        desired_tone="Clear",
        must_answer_topics=["Information in the supplied documents"],
        must_refuse_topics=["Unsupported personal information"],
    )


def make_agent(tools: list[str] | None = None) -> AgentDefinition:
    return AgentDefinition(
        agent_name="Generic Assistant",
        purpose="Answer questions from supplied information.",
        system_prompt="Use evidence.",
        retrieval=RetrievalDefinition(
            chunk_size=200,
            chunk_overlap=0,
            top_k=5,
        ),
        tools=tools or [],
        refusal_policy="Refuse unsupported requests.",
        citation_policy="Cite source chunks.",
        output_contract="Be clear.",
    )


def make_documents() -> list[Document]:
    return [
        Document(
            source="policy_a.txt",
            document_type="text",
            text="The Alpha team receives 7 training days each year.",
        ),
        Document(
            source="policy_b.txt",
            document_type="text",
            text="The Beta team receives 9 training days each year.",
        ),
    ]


def empty_groundability() -> GroundabilityReport:
    return GroundabilityReport(findings=[])


def candidate(
    question: str,
    case_type: str,
    behavior: str,
    *,
    facts: list[str] | None = None,
    chunk_ids: list[str] | None = None,
    required_tool: str | None = None,
    notes: str = "",
) -> dict:
    return {
        "question": question,
        "case_type": case_type,
        "expected_behavior": behavior,
        "expected_key_facts": facts or [],
        "supporting_chunk_ids": chunk_ids or [],
        "required_tool": required_tool,
        "notes": notes,
    }


def empty_response(_prompt: str) -> list[dict]:
    return []


def test_answerable_case_preserves_real_chunk_evidence():
    def respond(prompt: str) -> list[dict]:
        if "Generation category: answerable" in prompt:
            return [
                candidate(
                    "How many training days does the Alpha team receive?",
                    "answerable",
                    "answer",
                    facts=["The Alpha team receives 7 training days each year."],
                    chunk_ids=["policy_a.txt::chunk_0"],
                )
            ]
        return []

    result = EvalGenerator(FakeLLMClient(respond)).generate(
        make_requirements(),
        empty_groundability(),
        make_agent(),
        make_documents(),
    )

    answerable = next(case for case in result.cases if case.case_type == "answerable")
    assert answerable.expected_source_documents == ["policy_a.txt"]
    assert answerable.supporting_chunk_ids == ["policy_a.txt::chunk_0"]
    assert answerable.expected_key_facts == [
        "The Alpha team receives 7 training days each year."
    ]


def test_fabricated_answerable_evidence_is_rejected():
    def respond(prompt: str) -> list[dict]:
        if "Generation category: answerable" in prompt:
            return [
                candidate(
                    "What is in the secret policy?",
                    "answerable",
                    "answer",
                    facts=["The secret policy grants 50 days."],
                    chunk_ids=["fake.pdf::chunk_99"],
                )
            ]
        return []

    result = EvalGenerator(FakeLLMClient(respond)).generate(
        make_requirements(),
        empty_groundability(),
        make_agent(),
        make_documents(),
    )

    assert not any(case.case_type == "answerable" for case in result.cases)
    assert result.warnings


def test_out_of_scope_case_is_forced_to_refuse():
    def respond(prompt: str) -> list[dict]:
        if "Generation category: out_of_scope" in prompt:
            return [
                candidate(
                    "What personal records are not included in the documents?",
                    "out_of_scope",
                    "answer",
                    notes="The documents contain no personal records.",
                )
            ]
        return []

    result = EvalGenerator(FakeLLMClient(respond)).generate(
        make_requirements(),
        empty_groundability(),
        make_agent(),
        [],
    )

    case = next(case for case in result.cases if case.case_type == "out_of_scope")
    assert case.expected_behavior == "refuse"
    assert case.expected_key_facts == []
    assert case.notes


def test_question_deduplication_normalizes_case_and_punctuation():
    def respond(prompt: str) -> list[dict]:
        if "Generation category: answerable" in prompt:
            return [
                candidate(
                    "  How MANY training days? ",
                    "answerable",
                    "answer",
                    facts=["The Alpha team receives 7 training days each year."],
                    chunk_ids=["policy_a.txt::chunk_0"],
                ),
                candidate(
                    "how many training days",
                    "answerable",
                    "answer",
                    facts=["The Alpha team receives 7 training days each year."],
                    chunk_ids=["policy_a.txt::chunk_0"],
                ),
            ]
        return []

    result = EvalGenerator(FakeLLMClient(respond)).generate(
        make_requirements(),
        empty_groundability(),
        make_agent(),
        make_documents(),
    )

    matches = [
        case
        for case in result.cases
        if EvalGenerator._normalize_question(case.question)
        == "how many training days"
    ]
    assert len(matches) == 1


def test_cross_document_requires_two_valid_sources():
    def respond(prompt: str) -> list[dict]:
        if "Generation category: cross_document" in prompt:
            return [
                candidate(
                    "Compare the training days for Alpha and Beta.",
                    "cross_document",
                    "answer",
                    facts=[
                        "The Alpha team receives 7 training days each year.",
                        "The Beta team receives 9 training days each year.",
                    ],
                    chunk_ids=[
                        "policy_a.txt::chunk_0",
                        "policy_b.txt::chunk_0",
                    ],
                ),
                candidate(
                    "Compare two facts from one policy.",
                    "cross_document",
                    "answer",
                    facts=["The Alpha team receives 7 training days each year."],
                    chunk_ids=[
                        "policy_a.txt::chunk_0",
                        "policy_a.txt::chunk_1",
                    ],
                ),
            ]
        return []

    result = EvalGenerator(FakeLLMClient(respond)).generate(
        make_requirements(),
        empty_groundability(),
        make_agent(),
        make_documents(),
    )

    cross_cases = [case for case in result.cases if case.case_type == "cross_document"]
    assert len(cross_cases) == 1
    assert set(cross_cases[0].expected_source_documents) == {
        "policy_a.txt",
        "policy_b.txt",
    }


def test_tool_cases_are_only_requested_for_selected_tools():
    def respond(prompt: str) -> list[dict]:
        if "Generation category: tool_required" in prompt:
            selected = prompt.split("Selected tool: ", 1)[1].splitlines()[0]
            return [
                candidate(
                    "Calculate a total using the supplied values.",
                    "tool_required",
                    "use_tool",
                    required_tool=selected,
                )
            ]
        return []

    no_tools_client = FakeLLMClient(respond)
    EvalGenerator(no_tools_client).generate(
        make_requirements(),
        empty_groundability(),
        make_agent(),
        make_documents(),
    )
    assert not any("Generation category: tool_required" in p for p in no_tools_client.prompts)

    calculator_client = FakeLLMClient(respond)
    result = EvalGenerator(calculator_client).generate(
        make_requirements(),
        empty_groundability(),
        make_agent(["calculator"]),
        make_documents(),
    )
    tool_cases = [case for case in result.cases if case.case_type == "tool_required"]
    assert tool_cases
    assert {case.required_tool for case in tool_cases} == {"calculator"}


def test_minimum_set_assembly_and_json_serialization():
    answerable_index = 0

    def respond(prompt: str) -> list[dict]:
        nonlocal answerable_index
        category = prompt.split("Generation category: ", 1)[1].splitlines()[0]
        if category == "answerable":
            source = "policy_a.txt"
            fact = "The Alpha team receives 7 training days each year."
            chunk_id = "policy_a.txt::chunk_0"
            candidates = [
                candidate(
                    f"How many Alpha training days are noted, question {index}?",
                    category,
                    "answer",
                    facts=[fact],
                    chunk_ids=[chunk_id],
                )
                for index in range(answerable_index, answerable_index + 5)
            ]
            answerable_index += 5
            return candidates
        if category == "out_of_scope":
            return [
                candidate(
                    f"Request unsupported personal detail {index}?",
                    category,
                    "refuse",
                    notes="No personal records are supplied.",
                )
                for index in range(4)
            ]
        if category == "ambiguous":
            return [
                candidate(
                    f"Clarify this vague training request {index}?",
                    category,
                    "clarify",
                )
                for index in range(3)
            ]
        if category == "cross_document":
            return [
                candidate(
                    f"Compare both teams' training days, version {index}?",
                    category,
                    "answer",
                    facts=[
                        "The Alpha team receives 7 training days each year.",
                        "The Beta team receives 9 training days each year.",
                    ],
                    chunk_ids=[
                        "policy_a.txt::chunk_0",
                        "policy_b.txt::chunk_0",
                    ],
                )
                for index in range(3)
            ]
        return []

    result = EvalGenerator(FakeLLMClient(respond)).generate(
        make_requirements(),
        empty_groundability(),
        make_agent(),
        make_documents(),
    )

    assert len(result.cases) >= 20
    assert result.cases[0].id == "eval_001"
    assert result.model_dump_json()


def test_insufficient_evidence_returns_shortage_warning_without_fabrication():
    result = EvalGenerator(FakeLLMClient(empty_response)).generate(
        make_requirements(),
        GroundabilityReport(
            findings=[
                RequirementGroundability(
                    requirement="No source-backed facts",
                    status="unsupported",
                    reason="There is no document evidence.",
                )
            ]
        ),
        make_agent(),
        [],
    )

    assert len(result.cases) < 20
    assert any("insufficient" in warning for warning in result.warnings)
    assert any("Cross-document generation was not possible" in warning for warning in result.warnings)
