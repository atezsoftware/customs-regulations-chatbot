"""A bounded source correction changes content, never legal version identity."""

from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)


class ChunkCorrection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1, max_length=500)
    expected_updated_at: AwareDatetime
    text: str | None = None
    heading_path: list[str] | None = None
    chunk_metadata: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def require_content_change(self) -> Self:
        fields = {"text", "heading_path", "chunk_metadata"} & self.model_fields_set
        if not fields or any(getattr(self, field) is None for field in fields):
            raise ValueError("Supply at least one non-null content field")
        if self.text is not None and not self.text.strip():
            raise ValueError("Chunk text cannot be emptied")
        return self


class ChunkCorrectionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updates: list[ChunkCorrection] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def distinct_targets(self) -> Self:
        if len({update.chunk_id for update in self.updates}) != len(self.updates):
            raise ValueError("Each canonical chunk may appear only once")
        return self
