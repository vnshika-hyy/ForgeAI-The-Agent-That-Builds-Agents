from forgeai.ingestion.chunker import chunk_document
from forgeai.ingestion.loader import Document
from forgeai.ingestion.vector_store import LocalTextStore
from forgeai.intake.groundability import GroundabilityGuard
from forgeai.schemas.requirements import Requirements
from forgeai.schemas.reports import GroundabilityReport, RequirementGroundability


class FakeLLMClient:
    def __init__(self, assessments: list[dict]):
        self.assessments = iter(assessments)

    def generate_json(self, prompt: str, schema: type):
        assessment = next(self.assessments)
        return schema.model_validate(assessment)


def make_requirements(
    *,
    must_answer_topics: list[str] | None = None,
    required_capabilities: list[str] | None = None,
) -> Requirements:
    return Requirements(
        agent_name="Test Assistant",
        purpose="Test agent",
        scope="Test scope",
        target_audience="Test users",
        desired_tone="Professional",
        must_answer_topics=must_answer_topics or [],
        required_capabilities=required_capabilities or [],
    )


def make_store(source: str, text: str) -> LocalTextStore:
    document = Document(
        source=source,
        document_type="text",
        text=text,
        metadata={"file_path": f"/docs/{source}"},
    )
    return LocalTextStore(chunk_document(document, chunk_size=900, chunk_overlap=0))


def test_supported_requirement_preserves_real_evidence():
    store = make_store(
        "leave_policy.txt",
        "Full-time shore employees receive 24 days of annual leave per year.",
    )
    fake = FakeLLMClient([
        {
            "status": "supported",
            "reason": "The policy directly states the annual entitlement.",
            "supporting_chunk_ids": ["leave_policy.txt::chunk_0"],
        }
    ])
    guard = GroundabilityGuard(fake, store)

    report = guard.assess(
        make_requirements(must_answer_topics=["Annual leave entitlement"]),
        [],
    )

    assert report.supported_count == 1
    assert report.findings[0].status == "supported"
    assert report.findings[0].evidence[0].source == "leave_policy.txt"
    assert report.findings[0].evidence[0].chunk_id == "leave_policy.txt::chunk_0"


def test_unrelated_requirement_is_unsupported():
    store = make_store(
        "leave_policy.txt",
        "Employees receive 24 annual leave days per year.",
    )
    guard = GroundabilityGuard(FakeLLMClient([]), store)

    report = guard.assess(
        make_requirements(must_answer_topics=["Employee salary bands"]),
        [],
    )

    assert report.unsupported_count == 1
    assert report.findings[0].status == "unsupported"
    assert report.findings[0].evidence == []


def test_partial_support_preserves_limitation():
    store = make_store(
        "leave_policy.txt",
        "Employees receive 24 annual leave days per year.",
    )
    fake = FakeLLMClient([
        {
            "status": "partially_supported",
            "reason": "The policy covers entitlement but not each employee's remaining balance.",
            "supporting_chunk_ids": ["leave_policy.txt::chunk_0"],
        }
    ])
    guard = GroundabilityGuard(fake, store)

    report = guard.assess(
        make_requirements(
            must_answer_topics=[
                "Tell each employee their current remaining leave balance"
            ]
        ),
        [],
    )

    finding = report.findings[0]
    assert finding.status == "partially_supported"
    assert "remaining balance" in finding.reason
    assert finding.requirement.startswith("Tell each employee")


def test_fabricated_citation_is_not_included():
    store = make_store(
        "leave_policy.txt",
        "Employees receive 24 annual leave days per year.",
    )
    fake = FakeLLMClient([
        {
            "status": "supported",
            "reason": "The model claims support.",
            "supporting_chunk_ids": ["secret_database.pdf::chunk_99"],
        }
    ])
    guard = GroundabilityGuard(fake, store)

    report = guard.assess(
        make_requirements(must_answer_topics=["Annual leave"]),
        [],
    )

    assert report.findings[0].status == "unsupported"
    assert report.findings[0].evidence == []
    assert "No valid cited evidence" in report.findings[0].reason


def test_counts_are_calculated_from_findings():
    findings = [
        RequirementGroundability(
            requirement="A",
            status="supported",
            reason="Supported",
        ),
        RequirementGroundability(
            requirement="B",
            status="supported",
            reason="Supported",
        ),
        RequirementGroundability(
            requirement="C",
            status="partially_supported",
            reason="Partial",
        ),
        RequirementGroundability(
            requirement="D",
            status="unsupported",
            reason="Unsupported",
        ),
        RequirementGroundability(
            requirement="E",
            status="unsupported",
            reason="Unsupported",
        ),
    ]

    report = GroundabilityReport(
        findings=findings,
        supported_count=99,
        partially_supported_count=99,
        unsupported_count=99,
    )

    assert report.supported_count == 2
    assert report.partially_supported_count == 1
    assert report.unsupported_count == 2
