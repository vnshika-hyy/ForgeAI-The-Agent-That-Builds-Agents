from pathlib import Path

from forgeai.ingestion.loader import Document
from forgeai.intake.interviewer import ClarificationQuestions, IntakeInterviewer
from forgeai.schemas.requirements import Requirements


class FakeLLMClient:
    def __init__(self, question_data: dict | None = None, requirements: dict | None = None):
        self.question_data = question_data
        self.requirements = requirements

    def generate_json(self, prompt: str, schema: type):
        if schema is ClarificationQuestions:
            return ClarificationQuestions.model_validate(self.question_data)
        return Requirements.model_validate(self.requirements)


def test_generate_questions_enforces_maximum_of_five():
    fake = FakeLLMClient(
        question_data={
            "questions": [
                "Question 1?",
                "Question 2?",
                "Question 3?",
                "Question 4?",
                "Question 5?",
                "Question 6?",
                "Question 7?",
            ]
        }
    )
    interviewer = IntakeInterviewer(fake)

    questions = interviewer.generate_questions("A general brief", [])

    assert len(questions) == 5


def test_generate_questions_cleans_duplicates_blanks_and_whitespace():
    fake = FakeLLMClient(
        question_data={
            "questions": [
                "  Who is the audience?  ",
                "",
                "who is the audience?",
                "   ",
                "What tone should it use?",
            ]
        }
    )
    interviewer = IntakeInterviewer(fake)

    questions = interviewer.generate_questions("A general brief", [])

    assert questions == [
        "Who is the audience?",
        "What tone should it use?",
    ]


def test_build_requirements_returns_structured_requirements():
    requirements = {
        "agent_name": "HR Policy Assistant",
        "purpose": "Answer HR policy questions for shore staff.",
        "scope": "HR policies applicable to shore staff",
        "target_audience": "Shore staff",
        "desired_tone": "Professional and concise",
        "must_answer_topics": ["Annual leave", "HR policies"],
        "must_refuse_topics": [],
        "required_capabilities": ["Answer policy questions"],
        "document_sources": ["hr-policy.md"],
    }
    fake = FakeLLMClient(requirements=requirements)
    interviewer = IntakeInterviewer(fake)
    document = Document(
        source="hr-policy.md",
        document_type="text",
        text="Shore staff receive annual leave according to company policy.",
        metadata={"file_path": "/docs/hr-policy.md"},
    )

    result = interviewer.build_requirements(
        "I need an assistant that answers HR policy questions for shore staff.",
        [document],
        {"Audience": "Shore staff", "Tone": "Professional and concise"},
    )

    assert isinstance(result, Requirements)
    assert result.target_audience == "Shore staff"
    assert result.desired_tone == "Professional and concise"
    assert "HR" in result.purpose
    assert "HR policies" in result.must_answer_topics


def test_generate_questions_accepts_no_questions():
    fake = FakeLLMClient(question_data={"questions": []})
    interviewer = IntakeInterviewer(fake)

    questions = interviewer.generate_questions(
        "I need a concise assistant for the policies in the supplied documents.",
        [],
    )

    assert questions == []
