"""Server-recorded preparation targets; older queued requests keep their semantics."""

from collections.abc import Mapping
from typing import Any, Literal

from onyx.regulatory.document_intake.models import IntakeChunkingVersion

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
    if chunking == "3" and type(formatter) is int and formatter == 4:
        return "3", 4
    raise ValueError("The frozen reprepare target versions are not supported")
