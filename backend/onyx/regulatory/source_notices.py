"""Typed catalogue notices, distinct from canonical text and source files."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.regulatory.legistation_metadata import LegistationMetadata

SOURCE_NOTICE_ID_PREFIX = "notice:"


class SourceNotice(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: str
    document_set_id: int
    identity_key: str
    instrument_type: str
    instrument_number: str | None = None
    instrument_year: int | None = None
    instrument_date: str | None = None
    catalog_title: str
    aliases: list[str] = Field(default_factory=list)
    legistation_metadata: LegistationMetadata
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    content_state: Literal["metadata_only", "materialized"] = "metadata_only"
    user_file_id: str | None = None
    revision: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_content_identity(self) -> "SourceNotice":
        UUID(self.id)
        if self.user_file_id is not None:
            UUID(self.user_file_id)
        if (self.content_state == "materialized") != (self.user_file_id is not None):
            raise ValueError("Materialized notices require an actual source file.")
        return self


def source_notice_id(value: str) -> str:
    """Expose a notice target without masquerading as a readable source UUID."""
    return SOURCE_NOTICE_ID_PREFIX + str(UUID(value))


def parse_source_notice_id(value: str) -> str | None:
    if not value.startswith(SOURCE_NOTICE_ID_PREFIX):
        return None
    try:
        return str(UUID(value.removeprefix(SOURCE_NOTICE_ID_PREFIX)))
    except ValueError:
        return None
