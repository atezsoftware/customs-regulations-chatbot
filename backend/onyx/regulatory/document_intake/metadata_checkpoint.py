"""Immutable identities required to resume acknowledged metadata requests."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MetadataLeaseLost(RuntimeError):
    """An expired metadata owner must leave its durable recovery intent untouched."""


def metadata_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def metadata_recipe_fingerprint(
    components: Mapping[str, str], constants: Mapping[str, object]
) -> str:
    return metadata_digest(
        {"version": 1, "components": components, "constants": constants}
    )


class MetadataCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    recipe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_binding: dict[str, Any]
    source_snapshot: dict[str, Any]
    request_hashes: dict[str, list[str]]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def build(
        cls,
        *,
        recipe_sha256: str,
        provider_binding: dict[str, Any],
        source_snapshot: dict[str, Any],
        request_hashes: dict[str, list[str]],
    ) -> "MetadataCheckpoint":
        payload = {
            "version": 1,
            "recipe_sha256": recipe_sha256,
            "provider_binding": provider_binding,
            "source_snapshot": source_snapshot,
            "request_hashes": request_hashes,
        }
        return cls(**payload, checkpoint_sha256=metadata_digest(payload))

    def validate_checksum(self) -> None:
        if self.checkpoint_sha256 != metadata_digest(
            self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        ):
            raise ValueError("The frozen metadata checkpoint changed")
