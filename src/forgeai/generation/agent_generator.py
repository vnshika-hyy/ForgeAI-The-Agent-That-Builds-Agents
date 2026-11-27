from pathlib import Path

from pydantic import BaseModel

from ..ingestion.loader import Document
from ..llm.client import LLMClient
from ..schemas.agent_definition import AgentDefinition, RetrievalDefinition
from ..schemas.requirements import Requirements
from ..schemas.reports import GroundabilityReport
from ..tools.registry import ToolRegistry


class ToolSelection(BaseModel):
    name: str
    reason: str = ""


class RetrievalResponse(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    similarity_threshold: float = 0.0


class AgentGenerationResponse(BaseModel):
    system_prompt: str
    retrieval: RetrievalResponse = RetrievalResponse()
    tools: list[ToolSelection] = []
    refusal_policy: str
    citation_policy: str
    output_contract: str


class AgentGenerator:
    """Creates a declarative agent configuration from requirements."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self._prompts_dir = Path(__file__).resolve().parents[3] / "prompts"

    def _sanitize_retrieval(
        self,
        retrieval: RetrievalResponse,
    ) -> RetrievalDefinition:
        chunk_size = max(200, min(4000, retrieval.chunk_size))
        chunk_overlap = max(0, min(retrieval.chunk_overlap, chunk_size - 1))
        top_k = max(1, min(20, retrieval.top_k))
        similarity_threshold = max(
            0.0,
            min(1.0, retrieval.similarity_threshold),
        )
        return RetrievalDefinition(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def _sanitize_tools(
        self,
        selections: list[ToolSelection],
    ) -> tuple[list[str], dict[str, str]]:
        tools: list[str] = []
        reasons: dict[str, str] = {}
        for selection in selections:
            name = selection.name.strip()
            if not name or name in tools:
                continue
            if self.tool_registry.get(name) is None:
                continue
            tools.append(name)
            reasons[name] = selection.reason.strip() or (
                f"The requirements indicate a need for the {name} tool."
            )
        return tools, reasons

    def _groundability_boundaries(
        self,
        groundability: GroundabilityReport,
    ) -> str:
        limitations = [
            f"- {finding.requirement} [{finding.status}]: {finding.reason}"
            for finding in groundability.findings
            if finding.status != "supported"
        ]
        if not limitations:
            return ""
        return (
            "\n\nKnowledge boundaries:\n"
            "The following limitations are authoritative. Do not use them to "
            "claim capabilities that the supplied evidence does not support:\n"
            + "\n".join(limitations)
        )

    def generate(
        self,
        requirements: Requirements,
        groundability: GroundabilityReport,
        documents: list[Document],
    ) -> AgentDefinition:
        document_names = ", ".join(
            document.source for document in documents
        ) or "No documents supplied"
        prompt_template = (
            self._prompts_dir / "agent_generator.txt"
        ).read_text(encoding="utf-8")
        prompt = (
            prompt_template
            + "\n\nREQUIREMENTS:\n"
            + requirements.to_pretty_json()
            + "\n\nGROUNDABILITY REPORT:\n"
            + groundability.model_dump_json(indent=2)
            + "\n\nAVAILABLE DOCUMENTS:\n"
            + document_names
            + "\n\nAVAILABLE TOOLS:\n"
            + self.tool_registry.descriptions()
        )
        response = self.llm_client.generate_json(
            prompt,
            AgentGenerationResponse,
        )

        tools, tool_reasons = self._sanitize_tools(response.tools)
        retrieval = self._sanitize_retrieval(response.retrieval)
        system_prompt = (
            response.system_prompt.strip()
            + self._groundability_boundaries(groundability)
        )

        refusal_policy = response.refusal_policy.strip()
        if groundability.unsupported_count or groundability.partially_supported_count:
            refusal_policy = (
                refusal_policy
                + "\nRefuse or qualify unsupported and partially supported claims "
                "according to the Knowledge boundaries section."
            ).strip()

        return AgentDefinition(
            agent_name=requirements.agent_name,
            purpose=requirements.purpose,
            system_prompt=system_prompt,
            retrieval=retrieval,
            tools=tools,
            tool_selection_reasons=tool_reasons,
            refusal_policy=refusal_policy,
            citation_policy=response.citation_policy.strip(),
            output_contract=response.output_contract.strip(),
            metadata={
                "groundability_supported_count": groundability.supported_count,
                "groundability_partially_supported_count": (
                    groundability.partially_supported_count
                ),
                "groundability_unsupported_count": (
                    groundability.unsupported_count
                ),
            },
        )
