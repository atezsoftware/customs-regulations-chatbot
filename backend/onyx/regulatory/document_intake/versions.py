"""Server-recorded preparation targets; older queued requests keep their semantics."""

from collections.abc import Mapping
from typing import Any, Literal

from onyx.regulatory.document_intake.models import (
    IntakeChunkingVersion,
    IntakePreparationPolicy,
)

IntakeFormatterVersion = Literal[2, 3, 4]
INTAKE_FORMATTER_VERSION: IntakeFormatterVersion = 4


def reprepare_target_versions(
    entry: Mapping[str, Any],
) -> tuple[IntakeChunkingVersion, IntakeFormatterVersion]:
    keys = {"target_chunking_version", "target_formatter_version"}
    if not keys.intersection(entry):
        return "2", 3
    if not keys.issubset(entry):
        raise ValueError("The frozen reprepare target versions are incomplete")
    chunking, formatter = (
        entry["target_chunking_version"],
        entry["target_formatter_version"],
    )
    if chunking in {"4", "5"} and type(formatter) is int and formatter == 4:
        policy = IntakePreparationPolicy.model_validate(
            entry.get("target_preparation_policy")
        )
        if policy.chunking_version != chunking:
            raise ValueError("The frozen policy and chunking version differ")
        return policy.chunking_version, 4
    if (
        chunking == "3"
        and type(formatter) is int
        and formatter == 4
        and entry.get("target_preparation_policy") is None
    ):
        return "3", 4
    raise ValueError("The frozen reprepare target versions are not supported")
