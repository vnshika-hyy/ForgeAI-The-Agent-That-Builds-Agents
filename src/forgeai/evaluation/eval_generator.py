import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..ingestion.chunker import Chunk, chunk_documents
from ..ingestion.loader import Document
from ..llm.client import LLMClient
from ..schemas.agent_definition import AgentDefinition
from ..schemas.common import ForgeModel
from ..schemas.evaluation import EvaluationCase, EvaluationSet
from ..schemas.reports import GroundabilityReport
from ..schemas.requirements import Requirements


CaseType = Literal[
    "answerable",
    "out_of_scope",
    "ambiguous",
    "cross_document",
    "tool_required",
]
ExpectedBehavior = Literal[
    "answer",
    "refuse",
    "clarify",
    "answer_cautiously",
    "use_tool",
]


class GeneratedEvalCandidate(ForgeModel):
    question: str
    case_type: CaseType
    expected_behavior: ExpectedBehavior
    expected_key_facts: list[str] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    required_tool: str | None = None
    notes: str = ""


class GeneratedEvalBatch(ForgeModel):
    cases: list[GeneratedEvalCandidate] = Field(default_factory=list)


class EvalGenerator:
    MINIMUM_CASES = 20
    EVIDENCE_BATCH_SIZE = 5
    ANSWERABLE_TARGET = 10
    MAX_ANSWERABLE_BATCHES = 6
    MAX_FILL_BATCHES = 6

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self._prompts_dir = Path(__file__).resolve().parents[3] / "prompts"

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"[^\w]+", " ", question.casefold()).strip()

    @staticmethod
    def _normalize_fact(text: str) -> str:
        return re.sub(r"[^\w]+", " ", text.casefold()).strip()

    @staticmethod
    def _balanced_chunks(chunks: list[Chunk]) -> list[Chunk]:
        by_source: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, []).append(chunk)
        balanced: list[Chunk] = []
        sources = list(by_source)
        index = 0
        while True:
            added = False
            for source in sources:
                if index < len(by_source[source]):
                    balanced.append(by_source[source][index])
                    added = True
            if not added:
                break
            index += 1
        return balanced

    @staticmethod
    def _chunks_by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
        return {
            chunk.chunk_id or f"{chunk.source}::chunk_{chunk.chunk_index}": chunk
            for chunk in chunks
        }

    def _base_context(
        self,
        requirements: Requirements,
        groundability: GroundabilityReport,
        agent_definition: AgentDefinition,
        documents: list[Document],
    ) -> str:
        context = {
            "requirements": requirements.model_dump(mode="json"),
            "groundability_report": groundability.model_dump(mode="json"),
            "agent_definition": agent_definition.model_dump(mode="json"),
            "documents": [
                {
                    "source": document.source,
                    "document_type": document.document_type,
                    "metadata": document.metadata,
                }
                for document in documents
            ],
        }
        return json.dumps(context, ensure_ascii=False)

    @staticmethod
    def _evidence_payload(chunks: list[Chunk]) -> list[dict[str, str | int | None]]:
        return [
            {
                "chunk_id": chunk.chunk_id
                or f"{chunk.source}::chunk_{chunk.chunk_index}",
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text[:1200],
            }
            for chunk in chunks
        ]

    def _generate_batch(
        self,
        *,
        case_type: CaseType,
        count: int,
        prompt_name: str,
        context: str,
        evidence: list[Chunk],
        tool_name: str | None = None,
    ) -> list[GeneratedEvalCandidate]:
        prompt_template = (self._prompts_dir / prompt_name).read_text(
            encoding="utf-8"
        )
        prompt = (
            f"Generation category: {case_type}\n"
            f"Requested case count: {count}\n"
            f"Selected tool: {tool_name or 'none'}\n\n"
            + prompt_template
            + "\n\nCONFIGURATION CONTEXT:\n"
            + context
            + "\n\nSUPPLIED EVIDENCE CHUNKS:\n"
            + json.dumps(self._evidence_payload(evidence), ensure_ascii=False)
        )
        batch = self.llm_client.generate_json(prompt, GeneratedEvalBatch)
        return batch.cases[:count]

    def _validate_candidate(
        self,
        candidate: GeneratedEvalCandidate,
        requested_type: CaseType,
        supplied_ids: set[str],
        chunks_by_id: dict[str, Chunk],
        selected_tool: str | None = None,
    ) -> EvaluationCase | None:
        question = candidate.question.strip()
        if not question or candidate.case_type != requested_type:
            return None

        cited_ids = list(
            dict.fromkeys(
                chunk_id
                for chunk_id in candidate.supporting_chunk_ids
                if chunk_id in supplied_ids and chunk_id in chunks_by_id
            )
        )
        cited_chunks = [chunks_by_id[chunk_id] for chunk_id in cited_ids]
        grounded_facts: list[str] = []
        fact_sources: set[str] = set()

        if requested_type in {"answerable", "cross_document"}:
            for fact in candidate.expected_key_facts:
                normalized_fact = self._normalize_fact(fact)
                if not normalized_fact:
                    continue
                matching_chunks = [
                    chunk
                    for chunk in cited_chunks
                    if normalized_fact in self._normalize_fact(chunk.text)
                ]
                if matching_chunks:
                    grounded_facts.append(fact.strip())
                    fact_sources.update(chunk.source for chunk in matching_chunks)
            if not grounded_facts:
                return None
            cited_chunks = [
                chunk
                for chunk in cited_chunks
                if any(
                    self._normalize_fact(fact) in self._normalize_fact(chunk.text)
                    for fact in grounded_facts
                )
            ]
            cited_ids = [
                chunk.chunk_id or f"{chunk.source}::chunk_{chunk.chunk_index}"
                for chunk in cited_chunks
            ]

        if requested_type == "cross_document" and len(fact_sources) < 2:
            return None
        if requested_type == "tool_required":
            if (
                not selected_tool
                or candidate.required_tool != selected_tool
                or candidate.required_tool is None
            ):
                return None

        if requested_type == "answerable":
            behavior = "answer"
        elif requested_type == "cross_document":
            behavior = "answer"
        elif requested_type == "out_of_scope":
            behavior = "refuse"
        elif requested_type == "tool_required":
            behavior = "use_tool"
        else:
            behavior = candidate.expected_behavior

        expected_sources = sorted({chunk.source for chunk in cited_chunks})
        notes = candidate.notes.strip()
        if requested_type == "out_of_scope" and not notes:
            notes = (
                "The request exceeds the configured scope or asks for information "
                "not supported by supplied sources."
            )
        return EvaluationCase(
            id="pending",
            question=question,
            case_type=requested_type,
            expected_behavior=behavior,
            expected_source_documents=expected_sources,
            expected_key_facts=grounded_facts,
            supporting_chunk_ids=cited_ids,
            required_tool=selected_tool if requested_type == "tool_required" else None,
            evaluation_type="grounded_answer",
            notes=notes,
        )

    def _add_batch(
        self,
        cases: list[EvaluationCase],
        question_keys: set[str],
        candidates: list[GeneratedEvalCandidate],
        requested_type: CaseType,
        supplied_chunks: list[Chunk],
        chunks_by_id: dict[str, Chunk],
        selected_tool: str | None = None,
    ) -> None:
        supplied_ids = {
            chunk.chunk_id or f"{chunk.source}::chunk_{chunk.chunk_index}"
            for chunk in supplied_chunks
        }
        for candidate in candidates:
            case = self._validate_candidate(
                candidate,
                requested_type,
                supplied_ids,
                chunks_by_id,
                selected_tool,
            )
            if case is None:
                continue
            key = self._normalize_question(case.question)
            if not key or key in question_keys:
                continue
            question_keys.add(key)
            cases.append(case)

    def generate(
        self,
        requirements: Requirements,
        groundability: GroundabilityReport,
        agent_definition: AgentDefinition,
        documents: list[Document],
    ) -> EvaluationSet:
        chunks = chunk_documents(
            documents,
            chunk_size=agent_definition.retrieval.chunk_size,
            chunk_overlap=agent_definition.retrieval.chunk_overlap,
        )
        chunks = self._balanced_chunks(chunks)
        chunks_by_id = self._chunks_by_id(chunks)
        context = self._base_context(
            requirements,
            groundability,
            agent_definition,
            documents,
        )
        cases: list[EvaluationCase] = []
        question_keys: set[str] = set()
        warnings: list[str] = []

        evidence_groups = [
            chunks[index : index + self.EVIDENCE_BATCH_SIZE]
            for index in range(0, len(chunks), self.EVIDENCE_BATCH_SIZE)
        ]

        if evidence_groups:
            answerable_batches = 0
            group_index = 0
            while (
                sum(case.case_type == "answerable" for case in cases)
                < self.ANSWERABLE_TARGET
                and answerable_batches < self.MAX_ANSWERABLE_BATCHES
            ):
                evidence = evidence_groups[group_index % len(evidence_groups)]
                candidates = self._generate_batch(
                    case_type="answerable",
                    count=5,
                    prompt_name="eval_answerable.txt",
                    context=context,
                    evidence=evidence,
                )
                self._add_batch(
                    cases,
                    question_keys,
                    candidates,
                    "answerable",
                    evidence,
                    chunks_by_id,
                )
                answerable_batches += 1
                group_index += 1

        boundary_evidence = chunks[: self.EVIDENCE_BATCH_SIZE]
        candidates = self._generate_batch(
            case_type="out_of_scope",
            count=4,
            prompt_name="eval_boundary.txt",
            context=context,
            evidence=boundary_evidence,
        )
        self._add_batch(
            cases,
            question_keys,
            candidates,
            "out_of_scope",
            boundary_evidence,
            chunks_by_id,
        )

        candidates = self._generate_batch(
            case_type="ambiguous",
            count=3,
            prompt_name="eval_ambiguous.txt",
            context=context,
            evidence=boundary_evidence,
        )
        self._add_batch(
            cases,
            question_keys,
            candidates,
            "ambiguous",
            boundary_evidence,
            chunks_by_id,
        )

        source_count = len({chunk.source for chunk in chunks})
        if source_count >= 2:
            cross_evidence = chunks[: min(len(chunks), self.EVIDENCE_BATCH_SIZE * 2)]
            candidates = self._generate_batch(
                case_type="cross_document",
                count=3,
                prompt_name="eval_cross_document.txt",
                context=context,
                evidence=cross_evidence,
            )
            before_count = len(cases)
            self._add_batch(
                cases,
                question_keys,
                candidates,
                "cross_document",
                cross_evidence,
                chunks_by_id,
            )
            if len(cases) == before_count:
                warnings.append(
                    "No valid cross-document cases could be grounded in evidence "
                    "from two distinct sources."
                )
        else:
            warnings.append(
                "Cross-document generation was not possible because fewer than "
                "two distinct document sources were supplied."
            )

        for tool_name in dict.fromkeys(agent_definition.tools):
            candidates = self._generate_batch(
                case_type="tool_required",
                count=2,
                prompt_name="eval_tool.txt",
                context=context,
                evidence=boundary_evidence,
                tool_name=tool_name,
            )
            self._add_batch(
                cases,
                question_keys,
                candidates,
                "tool_required",
                boundary_evidence,
                chunks_by_id,
                selected_tool=tool_name,
            )

        fill_batches = 0
        while (
            len(cases) < self.MINIMUM_CASES
            and evidence_groups
            and fill_batches < self.MAX_FILL_BATCHES
        ):
            evidence = evidence_groups[fill_batches % len(evidence_groups)]
            candidates = self._generate_batch(
                case_type="answerable",
                count=5,
                prompt_name="eval_answerable.txt",
                context=context,
                evidence=evidence,
            )
            self._add_batch(
                cases,
                question_keys,
                candidates,
                "answerable",
                evidence,
                chunks_by_id,
            )
            fill_batches += 1

        if len(cases) < self.MINIMUM_CASES:
            warnings.append(
                f"Generated {len(cases)} valid distinct cases; at least "
                f"{self.MINIMUM_CASES} are recommended. The supplied corpus or "
                "grounded candidate output was insufficient to safely fill the set."
            )

        final_cases = [
            case.model_copy(update={"id": f"eval_{index:03d}"})
            for index, case in enumerate(cases, start=1)
        ]
        return EvaluationSet(cases=final_cases, warnings=warnings)
