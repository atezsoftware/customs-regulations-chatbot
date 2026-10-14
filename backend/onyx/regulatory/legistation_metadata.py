"""Optional file-level descriptions kept outside canonical regulatory evidence."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, ValidationError


class LegistationMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")

    document_type: str | None = None
    subject: str | None = None
    official_title: str | None = None
    official_gazette_date: str | None = None
    validity_source: str | None = None
    validity_status: str | None = None
    repeal_note: str | None = None


def parse_legistation_metadata(value: object) -> LegistationMetadata | None:
    """Ignore absent or malformed optional annotations without blocking source reads."""
    if not isinstance(value, Mapping):
        return None
    try:
        metadata = LegistationMetadata.model_validate(dict(value))
    except ValidationError:
        return None
    if not any(item for item in metadata.model_dump().values()):
        return None
    return metadata
