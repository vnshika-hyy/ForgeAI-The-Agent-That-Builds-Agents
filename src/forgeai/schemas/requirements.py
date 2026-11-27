from typing import List

from .common import ForgeModel


class Requirements(ForgeModel):
    agent_name: str
    purpose: str
    scope: str
    target_audience: str
    desired_tone: str
    must_answer_topics: List[str] = []
    must_refuse_topics: List[str] = []
    required_capabilities: List[str] = []
    document_sources: List[str] = []
    tool_requirements: List[str] = []
    assumptions: List[str] = []
    identified_grounding_gaps: List[str] = []

