import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..ingestion.chunker import chunk_documents
from ..ingestion.loader import Document
from ..ingestion.vector_store import LocalTextStore
from ..llm.client import LLMClient
from ..schemas.agent_definition import AgentDefinition
from ..schemas.common import ForgeModel
from ..tools.registry import ToolRegistry


class AgentTurn(ForgeModel):
    action: Literal["answer", "tool_call"]
    answer: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)
    refused: bool = False
    tool_name: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)


class AgentResponse(ForgeModel):
    action: Literal["answer"] = "answer"
    answer: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    refused: bool = False


class AgentRunner:
    MAX_TOOL_CALLS = 3

    def __init__(
        self,
        agent_definition: AgentDefinition,
        documents: list[Document],
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
    ) -> None:
        self.agent_definition = agent_definition
        self.documents = documents
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        retrieval = agent_definition.retrieval
        chunks = chunk_documents(
            documents,
            chunk_size=retrieval.chunk_size,
            chunk_overlap=retrieval.chunk_overlap,
        )
        self.retriever = LocalTextStore(chunks)
        self.allowed_tools = [
            name
            for name in dict.fromkeys(agent_definition.tools)
            if tool_registry.get(name) is not None
        ]
        self.csv_files = self._csv_allowlist(documents)
        self._runtime_prompt = (
            Path(__file__).resolve().parents[3]
            / "prompts"
            / "agent_runtime.txt"
        ).read_text(encoding="utf-8")

    @staticmethod
    def _csv_allowlist(documents: list[Document]) -> dict[str, str]:
        candidates: dict[str, str] = {}
        duplicates: set[str] = set()
        for document in documents:
            if document.document_type != "csv" or not document.source.lower().endswith(".csv"):
                continue
            file_path = document.metadata.get("file_path")
            if not file_path:
                continue
            path = Path(file_path).resolve()
            if not path.is_file():
                continue
            if document.source in candidates:
                duplicates.add(document.source)
            else:
                candidates[document.source] = str(path)
        for duplicate in duplicates:
            candidates.pop(duplicate, None)
        return candidates

    def _prompt(
        self,
        question: str,
        evidence: list[dict[str, object]],
        tool_history: list[dict[str, object]],
    ) -> str:
        tools_text = "\n".join(
            f"- {name}: {self.tool_registry.get(name).description}"
            for name in self.allowed_tools
            if self.tool_registry.get(name) is not None
        ) or "No tools are enabled."
        return (
            f"{self.agent_definition.system_prompt}\n\n"
            f"Refusal policy:\n{self.agent_definition.refusal_policy}\n\n"
            f"Citation policy:\n{self.agent_definition.citation_policy}\n\n"
            f"Output contract:\n{self.agent_definition.output_contract}\n\n"
            f"{self._runtime_prompt}\n\n"
            f"ENABLED TOOLS:\n{tools_text}\n\n"
            f"RETRIEVED EVIDENCE:\n"
            f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
            f"TOOL RESULTS SO FAR:\n"
            f"{json.dumps(tool_history, ensure_ascii=False)}\n\n"
            f"USER QUESTION:\n{question}"
        )

    def _refusal(
        self,
        message: str,
        tools_used: list[str] | None = None,
        retrieved_chunk_ids: list[str] | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            answer=message,
            source_chunk_ids=[],
            retrieved_chunk_ids=retrieved_chunk_ids or [],
            tools_used=tools_used or [],
            refused=True,
        )

    def answer(self, question: str) -> AgentResponse:
        question = question.strip()
        if not question:
            return self._refusal("Please provide a question.")

        retrieval = self.agent_definition.retrieval
        retrieved = [
            result
            for result in self.retriever.search(question, top_k=retrieval.top_k)
            if result.score >= retrieval.similarity_threshold
        ]
        evidence = [
            {
                "chunk_id": result.chunk.chunk_id
                or f"{result.chunk.source}::chunk_{result.chunk.chunk_index}",
                "source": result.chunk.source,
                "page": result.chunk.page,
                "text": result.chunk.text,
            }
            for result in retrieved
        ]
        valid_source_ids = {
            str(item["chunk_id"])
            for item in evidence
        }
        retrieved_chunk_ids = [
            str(item["chunk_id"])
            for item in evidence
        ]
        tool_history: list[dict[str, object]] = []
        tools_used: list[str] = []

        for _ in range(self.MAX_TOOL_CALLS + 1):
            turn = self.llm_client.generate_json(
                self._prompt(question, evidence, tool_history),
                AgentTurn,
            )
            if turn.action == "tool_call":
                if len(tool_history) >= self.MAX_TOOL_CALLS:
                    return self._refusal(
                        "I could not complete the request within the permitted tool-call limit.",
                        tools_used,
                        retrieved_chunk_ids,
                    )
                tool_name = turn.tool_name or ""
                result = self.tool_registry.execute_tool(
                    tool_name,
                    turn.arguments,
                    self.allowed_tools,
                    csv_files=self.csv_files,
                )
                if (
                    tool_name in self.allowed_tools
                    and self.tool_registry.get(tool_name) is not None
                ):
                    tools_used.append(tool_name)
                tool_history.append(
                    {
                        "tool": tool_name,
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                    }
                )
                continue

            if turn.refused:
                return AgentResponse(
                    answer=turn.answer.strip()
                    or "The request cannot be answered within the agent's boundaries.",
                    source_chunk_ids=[],
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    tools_used=tools_used,
                    refused=True,
                )
            answer = turn.answer.strip()
            if not answer:
                return self._refusal(
                    "The agent did not produce an answer supported by its available context.",
                    tools_used,
                    retrieved_chunk_ids,
                )

            submitted_ids = turn.source_chunk_ids
            if any(chunk_id not in valid_source_ids for chunk_id in submitted_ids):
                return self._refusal(
                    "The response included an invalid source reference and was withheld.",
                    tools_used,
                    retrieved_chunk_ids,
                )
            source_ids = list(dict.fromkeys(submitted_ids))
            successful_tool_used = any(
                result["success"] for result in tool_history
            )
            if not source_ids and not successful_tool_used:
                return self._refusal(
                    "The supplied documents do not provide evidence to answer this question.",
                    tools_used,
                    retrieved_chunk_ids,
                )
            return AgentResponse(
                answer=answer,
                source_chunk_ids=source_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                tools_used=tools_used,
                refused=False,
            )

        return self._refusal(
            "I could not complete the request within the permitted tool-call limit.",
            tools_used,
            retrieved_chunk_ids,
        )
