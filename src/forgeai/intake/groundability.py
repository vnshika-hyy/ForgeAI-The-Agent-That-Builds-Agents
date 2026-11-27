from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel

from ..ingestion.chunker import Chunk
from ..ingestion.loader import Document
from ..ingestion.vector_store import LocalTextStore
from ..llm.client import LLMClient
from ..schemas.requirements import Requirements
from ..schemas.reports import (
    EvidenceReference,
    GroundabilityReport,
    RequirementGroundability,
)


class GroundabilityAssessment(BaseModel):
    status: Literal["supported", "partially_supported", "unsupported"]
    reason: str
    supporting_chunk_ids: list[str] = []


class GroundabilityGuard:
    """Judges whether requirements are supported by retrieved source evidence."""

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: LocalTextStore,
    ) -> None:
        self.llm_client = llm_client
        self.retriever = retriever
        self._prompts_dir = Path(__file__).resolve().parents[3] / "prompts"

    @staticmethod
    def _claims(requirements: Requirements) -> list[str]:
        claims: list[str] = []
        seen: set[str] = set()
        for claim in [
            *requirements.must_answer_topics,
            *requirements.required_capabilities,
        ]:
            normalized = claim.strip()
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            claims.append(normalized)
        return claims

    @staticmethod
    def _evidence(
        chunks: Iterable[Chunk],
        accepted_ids: Iterable[str],
    ) -> list[EvidenceReference]:
        accepted = set(accepted_ids)
        evidence: list[EvidenceReference] = []
        for chunk in chunks:
            chunk_id = chunk.chunk_id or (
                f"{chunk.source}::chunk_{chunk.chunk_index}"
            )
            if chunk_id not in accepted:
                continue
            evidence.append(
                EvidenceReference(
                    source=chunk.source,
                    chunk_id=chunk_id,
                    page=chunk.page,
                    excerpt=chunk.text,
                )
            )
        return evidence

    def assess(
        self,
        requirements: Requirements,
        documents: list[Document],
    ) -> GroundabilityReport:
        findings: list[RequirementGroundability] = []

        for claim in self._claims(requirements):
            retrieved = self.retriever.search(claim, top_k=5)
            if not retrieved:
                findings.append(
                    RequirementGroundability(
                        requirement=claim,
                        status="unsupported",
                        reason="No adequate relevant document evidence was retrieved.",
                        evidence=[],
                    )
                )
                continue

            evidence_text = "\n\n".join(
                f"CHUNK_ID: {result.chunk.chunk_id or f'{result.chunk.source}::chunk_{result.chunk.chunk_index}'}\n"
                f"SOURCE: {result.chunk.source}\n"
                f"CONTENT:\n{result.chunk.text}"
                for result in retrieved
            )
            prompt_template = (
                self._prompts_dir / "groundability.txt"
            ).read_text(encoding="utf-8")
            prompt = (
                prompt_template
                + "\n\nREQUIREMENT:\n"
                + claim
                + "\n\nSUPPLIED EVIDENCE:\n"
                + evidence_text
            )
            assessment = self.llm_client.generate_json(
                prompt,
                GroundabilityAssessment,
            )

            supplied_ids = {
                result.chunk.chunk_id
                or f"{result.chunk.source}::chunk_{result.chunk.chunk_index}"
                for result in retrieved
            }
            accepted_ids = {
                chunk_id
                for chunk_id in assessment.supporting_chunk_ids
                if chunk_id in supplied_ids
            }
            evidence_chunks = [
                result.chunk
                for result in retrieved
                if (
                    result.chunk.chunk_id
                    or f"{result.chunk.source}::chunk_{result.chunk.chunk_index}"
                )
                in accepted_ids
            ]
            evidence = self._evidence(evidence_chunks, accepted_ids)

            status = assessment.status
            reason = assessment.reason.strip() or (
                "The supplied evidence was not sufficient to support this requirement."
            )
            if not evidence:
                status = "unsupported"
                if assessment.status != "unsupported":
                    reason = (
                        "No valid cited evidence was provided for this requirement."
                        + (f" {reason}" if assessment.reason.strip() else "")
                    )

            findings.append(
                RequirementGroundability(
                    requirement=claim,
                    status=status,
                    reason=reason,
                    evidence=evidence,
                )
            )

        warnings = [
            f"{finding.requirement}: {finding.reason}"
            for finding in findings
            if finding.status != "supported"
        ]
        return GroundabilityReport(findings=findings, warnings=warnings)
