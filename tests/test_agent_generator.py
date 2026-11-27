from forgeai.generation.agent_generator import (
    AgentGenerationResponse,
    AgentGenerator,
)
from forgeai.schemas.agent_definition import AgentDefinition
from forgeai.schemas.requirements import Requirements
from forgeai.schemas.reports import (
    GroundabilityReport,
    RequirementGroundability,
)
from forgeai.tools.registry import ToolRegistry


class FakeLLMClient:
    def __init__(self, response: dict):
        self.response = response

    def generate_json(self, prompt: str, schema: type):
        return schema.model_validate(self.response)


def make_requirements(
    purpose: str = "Answer HR policy questions",
) -> Requirements:
    return Requirements(
        agent_name="Policy Assistant",
        purpose=purpose,
        scope="Supplied policy documents",
        target_audience="Employees",
        desired_tone="Professional",
        must_answer_topics=["HR policies"],
    )


def empty_groundability() -> GroundabilityReport:
    return GroundabilityReport(findings=[], warnings=[])


def make_response(**overrides):
    response = {
        "system_prompt": "Answer clearly and ground factual claims in evidence.",
        "retrieval": {
            "chunk_size": 900,
            "chunk_overlap": 150,
            "top_k": 5,
            "similarity_threshold": 0.0,
        },
        "tools": [],
        "refusal_policy": "Refuse unsupported claims.",
        "citation_policy": "Reference the source used.",
        "output_contract": "Provide a clear answer.",
    }
    response.update(overrides)
    return response


def test_hr_agent_selects_no_unnecessary_tools():
    generator = AgentGenerator(
        FakeLLMClient(make_response()),
        ToolRegistry(),
    )

    agent = generator.generate(
        make_requirements(),
        empty_groundability(),
        [],
    )

    assert agent.tools == []


def test_unknown_tools_are_rejected():
    generator = AgentGenerator(
        FakeLLMClient(make_response(tools=[
            {"name": "calculator", "reason": "Calculate totals"},
            {"name": "web_search", "reason": "Search the web"},
            {"name": "magic_database", "reason": "Invented tool"},
        ])),
        ToolRegistry(),
    )

    agent = generator.generate(make_requirements(), empty_groundability(), [])

    assert agent.tools == ["calculator"]
    assert "calculator" in agent.tool_selection_reasons


def test_duplicate_tools_are_removed():
    generator = AgentGenerator(
        FakeLLMClient(make_response(tools=[
            {"name": "calculator", "reason": "Calculate totals"},
            {"name": "calculator", "reason": "Duplicate"},
            {"name": "current_date", "reason": "Check a date"},
        ])),
        ToolRegistry(),
    )

    agent = generator.generate(make_requirements(), empty_groundability(), [])

    assert agent.tools == ["calculator", "current_date"]


def test_groundability_partial_boundary_is_appended():
    report = GroundabilityReport(findings=[
        RequirementGroundability(
            requirement="Tell employees their current leave balance",
            status="partially_supported",
            reason="No employee-specific balance data is supplied.",
        )
    ])
    generator = AgentGenerator(
        FakeLLMClient(make_response(
            system_prompt="Answer HR policy questions.",
        )),
        ToolRegistry(),
    )

    agent = generator.generate(make_requirements(), report, [])

    assert "Knowledge boundaries:" in agent.system_prompt
    assert "current leave balance" in agent.system_prompt
    assert "No employee-specific balance data" in agent.system_prompt


def test_unsupported_capability_becomes_refusal_boundary():
    report = GroundabilityReport(findings=[
        RequirementGroundability(
            requirement="Provide employee salary information",
            status="unsupported",
            reason="No salary information exists in supplied sources.",
        )
    ])
    generator = AgentGenerator(
        FakeLLMClient(make_response()),
        ToolRegistry(),
    )

    agent = generator.generate(make_requirements(), report, [])

    assert "employee salary information" in agent.system_prompt
    assert "Refuse or qualify" in agent.refusal_policy
    assert "No salary information" in agent.system_prompt


def test_invalid_retrieval_values_are_clamped():
    generator = AgentGenerator(
        FakeLLMClient(make_response(retrieval={
            "chunk_size": -20,
            "chunk_overlap": 90000,
            "top_k": 9999,
            "similarity_threshold": -5.0,
        })),
        ToolRegistry(),
    )

    agent = generator.generate(make_requirements(), empty_groundability(), [])

    assert agent.retrieval.chunk_size == 200
    assert agent.retrieval.chunk_overlap == 199
    assert agent.retrieval.top_k == 20
    assert agent.retrieval.similarity_threshold == 0.0


def test_valid_configuration_serializes():
    generator = AgentGenerator(
        FakeLLMClient(make_response()),
        ToolRegistry(),
    )

    agent = generator.generate(make_requirements(), empty_groundability(), [])

    assert isinstance(agent, AgentDefinition)
    assert agent.model_dump_json()
