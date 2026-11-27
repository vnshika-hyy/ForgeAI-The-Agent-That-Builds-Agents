import json
import logging
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import ForgeAIConfig

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, config: ForgeAIConfig | None = None) -> None:
        self.config = config or ForgeAIConfig.from_env()

    def generate(self, prompt: str, system: str = "") -> str:
        if self.config.provider != "ollama":
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        try:
            response = httpx.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except httpx.HTTPError as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError(
                "Unable to reach Ollama. Start Ollama and confirm FORGEAI_OLLAMA_MODEL."
            ) from exc

    def generate_json(self, prompt: str, schema: type[T], retries: int = 2) -> T:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            text = self.generate(
                prompt
                + "\n\nReturn exactly one valid JSON object matching this schema. "
                "Do not wrap it in Markdown."
            )
            try:
                parsed = json.loads(self._extract_json(text))
                return schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning("Structured JSON attempt %d failed: %s", attempt + 1, exc)
        raise ValueError(f"LLM returned invalid structured JSON: {last_error}")

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return text

