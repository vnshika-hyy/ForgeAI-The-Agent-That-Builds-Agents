from typing import Dict, List

from .common import ForgeModel


class RetrievalDefinition(ForgeModel):
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    similarity_threshold: float = 0.0


class AgentDefinition(ForgeModel):
    agent_name: str
    version: str = "1"
    purpose: str
    system_prompt: str
    retrieval: RetrievalDefinition = RetrievalDefinition()
    tools: List[str] = []
    tool_selection_reasons: Dict[str, str] = {}
    refusal_policy: str
    citation_policy: str
    output_contract: str
    metadata: Dict[str, object] = {}

