from typing import Any

from pydantic import BaseModel, ConfigDict


class ForgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_pretty_json(self) -> str:
        return self.model_dump_json(indent=2, ensure_ascii=False)

