from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class ForgeAIConfig(BaseModel):
    provider: str = Field(default="ollama")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str = Field(default="qwen3:8b")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)

    @classmethod
    def from_env(cls) -> "ForgeAIConfig":
        return cls(
            provider=os.getenv("FORGEAI_LLM_PROVIDER", "ollama"),
            ollama_base_url=os.getenv(
                "FORGEAI_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            ollama_model=os.getenv("FORGEAI_OLLAMA_MODEL", "qwen3:8b"),
            temperature=float(os.getenv("FORGEAI_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("FORGEAI_MAX_TOKENS", "2048")),
        )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

