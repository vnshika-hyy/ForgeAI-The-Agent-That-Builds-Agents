from forgeai.ingestion.loader import Document
from forgeai.runtime.agent_runner import AgentRunner
from forgeai.schemas.agent_definition import AgentDefinition, RetrievalDefinition
from forgeai.tools.registry import ToolRegistry


class FakeLLMClient:
    def __init__(self, responses: list[dict]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, schema: type):
        self.prompts.append(prompt)
        return schema.model_validate(next(self.responses))


def make_agent(tools: list[str] | None = None) -> AgentDefinition:
    return AgentDefinition(
        agent_name="Generic Assistant",
        purpose="Answer questions using supplied evidence.",
        system_prompt="Use the supplied sources.",
        retrieval=RetrievalDefinition(
            chunk_size=900,
            chunk_overlap=0,
            top_k=3,
        ),
        tools=tools or [],
        refusal_policy="Refuse unsupported claims.",
        citation_policy="Cite source chunks.",
        output_contract="Answer clearly.",
    )


def make_document() -> Document:
    return Document(
        source="policy.txt",
        document_type="text",
        text="Employees receive 24 days of annual leave each year.",
    )


def test_runner_retrieves_evidence_and_validates_citations():
    llm = FakeLLMClient([
        {
            "action": "answer",
            "answer": "Employees receive 24 days of annual leave.",
            "source_chunk_ids": ["policy.txt::chunk_0"],
        }
    ])
    runner = AgentRunner(make_agent(), [make_document()], llm, ToolRegistry())

    response = runner.answer("How many annual leave days are provided?")

    assert response.answer.startswith("Employees receive 24")
    assert response.source_chunk_ids == ["policy.txt::chunk_0"]
    assert not response.refused
    assert "annual leave" in llm.prompts[0]


def test_runner_refuses_fabricated_source_ids():
    llm = FakeLLMClient([
        {
            "action": "answer",
            "answer": "Unsupported claim.",
            "source_chunk_ids": ["secret.pdf::chunk_88"],
        }
    ])
    runner = AgentRunner(make_agent(), [make_document()], llm, ToolRegistry())

    response = runner.answer("What is the leave entitlement?")

    assert response.refused
    assert response.source_chunk_ids == []
    assert "invalid source reference" in response.answer


def test_runner_dispatches_only_selected_tools():
    llm = FakeLLMClient([
        {
            "action": "tool_call",
            "tool_name": "calculator",
            "arguments": {"expression": "12 * 8.5"},
        },
        {
            "action": "answer",
            "answer": "The result is 102.",
            "source_chunk_ids": [],
        },
    ])
    runner = AgentRunner(
        make_agent(["calculator"]),
        [],
        llm,
        ToolRegistry(),
    )

    response = runner.answer("What is 12 times 8.5?")

    assert not response.refused
    assert response.answer == "The result is 102."
    assert response.tools_used == ["calculator"]
    assert "102.0" in llm.prompts[1]


def test_runner_stops_repeated_tool_calls():
    llm = FakeLLMClient([
        {
            "action": "tool_call",
            "tool_name": "current_date",
            "arguments": {},
        }
        for _ in range(4)
    ])
    runner = AgentRunner(
        make_agent(["current_date"]),
        [],
        llm,
        ToolRegistry(),
    )

    response = runner.answer("What is today's date?")

    assert response.refused
    assert len(llm.prompts) == AgentRunner.MAX_TOOL_CALLS + 1
    assert "tool-call limit" in response.answer


def test_runner_refuses_uncited_answer_without_tool_use():
    llm = FakeLLMClient([
        {
            "action": "answer",
            "answer": "The answer is 24 days.",
            "source_chunk_ids": [],
        }
    ])
    runner = AgentRunner(make_agent(), [make_document()], llm, ToolRegistry())

    response = runner.answer("How many days?")

    assert response.refused
    assert response.source_chunk_ids == []
