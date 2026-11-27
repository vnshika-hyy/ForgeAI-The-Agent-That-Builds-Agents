from pathlib import Path
from typing import Mapping

from pydantic import BaseModel

from ..ingestion.loader import Document
from ..llm.client import LLMClient
from ..schemas.requirements import Requirements


class ClarificationQuestions(BaseModel):
    questions: list[str]


class IntakeInterviewer:
    """Identifies missing requirements from a brief and document context."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self._prompts_dir = Path(__file__).resolve().parents[3] / "prompts"

    def _read_prompt(self, filename: str) -> str:
        return (self._prompts_dir / filename).read_text(encoding="utf-8")

    @staticmethod
    def _document_context(
        documents: list[Document],
        *,
        max_total_chars: int = 4000,
        max_per_document_chars: int = 800,
    ) -> str:
        if not documents:
            return "No documents were supplied."

        sections: list[str] = []
        remaining = max_total_chars
        for document in documents:
            if remaining <= 0:
                break

            metadata = ", ".join(
                f"{key}={value}" for key, value in document.metadata.items()
            )
            representative = " ".join(document.text.split())
            representative = representative[:max_per_document_chars].strip()
            if len(" ".join(document.text.split())) > max_per_document_chars:
                representative += "..."

            section = (
                f"Document: {document.source}\n"
                f"Type: {document.document_type}\n"
                f"Metadata: {metadata or 'none'}\n"
                f"Representative content: {representative or '[empty]'}"
            )
            sections.append(section[:remaining])
            remaining -= len(section[:remaining])

        if len(sections) < len(documents):
            sections.append("[Additional documents omitted from this bounded summary]")
        return "\n\n".join(sections)

    @staticmethod
    def _clean_questions(questions: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for question in questions:
            normalized = question.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
            if len(cleaned) == 5:
                break
        return cleaned

    def generate_questions(
        self,
        brief: str,
        documents: list[Document],
    ) -> list[str]:
        document_context = self._document_context(documents)
        prompt_template = self._read_prompt("intake_questions.txt")
        prompt = (
            prompt_template
            + "\n\nUSER BRIEF:\n"
            + brief.strip()
            + "\n\nINSPECTED DOCUMENT CONTEXT:\n"
            + document_context
        )
        result = self.llm_client.generate_json(prompt, ClarificationQuestions)
        return self._clean_questions(result.questions)

    def build_requirements(
        self,
        brief: str,
        documents: list[Document],
        answers: Mapping[str, str],
    ) -> Requirements:
        document_context = self._document_context(documents)
        answers_text = "\n".join(
            f"{question.strip()}: {answer.strip()}"
            for question, answer in answers.items()
            if question.strip() and answer.strip()
        )
        if not answers_text:
            answers_text = "No clarification answers were provided."

        prompt_template = self._read_prompt("intake_requirements.txt")
        prompt = (
            prompt_template
            + "\n\nORIGINAL USER BRIEF:\n"
            + brief.strip()
            + "\n\nINSPECTED DOCUMENT CONTEXT:\n"
            + document_context
            + "\n\nCLARIFICATION QUESTIONS AND ANSWERS:\n"
            + answers_text
        )
        return self.llm_client.generate_json(prompt, Requirements)
