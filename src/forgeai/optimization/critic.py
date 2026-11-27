from ..evaluation.failure_classifier import FailureClassification


class Critic:
    """Produces targeted diagnoses from classified failures."""

    def diagnose(
        self,
        classifications: list[FailureClassification],
    ) -> dict[str, str]:
        diagnoses = {
            "retrieval_miss": (
                "Relevant expected evidence was not surfaced; retrieval coverage "
                "should be increased without changing agent scope."
            ),
            "prompt_defect": (
                "Evidence was available, but the system prompt did not reliably "
                "guide the response behavior."
            ),
            "hallucination": (
                "The response exceeded its validated evidence; strengthen grounding "
                "and refusal constraints."
            ),
            "wrong_refusal": (
                "The agent refused an answerable request; clarify that supported "
                "requests should be answered while preserving unsupported boundaries."
            ),
        }
        return {
            category: diagnoses[category]
            for category in dict.fromkeys(item.category for item in classifications)
        }
