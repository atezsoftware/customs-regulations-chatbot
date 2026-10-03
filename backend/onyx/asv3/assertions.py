"""Bind each cited answer block to its own original-source assessment."""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from onyx.asv3.citation_numbers import extract_citation_numbers


class AssertionWitness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation: Annotated[int, Field(strict=True, ge=1)]
    source_quote: str = Field(min_length=1, max_length=800)


class AssertionVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    status: Literal["supported", "unsupported", "uncertain"]
    witnesses: list[AssertionWitness] = Field(default_factory=list, max_length=32)
    missing_conditions: list[str] = Field(default_factory=list, max_length=16)
    explanation: str = Field(
        min_length=1,
        max_length=4000,
        description="One short sentence about this block's decisive support or defect. Put actionable gaps in missing_conditions; avoid repeating the answer or other assessments.",
    )


class AssertionUnit(TypedDict):
    unit_id: str
    text: str
    evidence_numbers: list[int]


def assertion_inventory(answer: str) -> list[AssertionUnit]:
    """Split presentation blocks, without guessing their legal truth or atomicity."""
    blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", answer):
        lines: list[str] = []
        for line in paragraph.splitlines():
            if re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line) and lines:
                blocks.append("\n".join(lines).strip())
                lines = []
            lines.append(line)
        if lines:
            blocks.append("\n".join(lines).strip())
    units: list[AssertionUnit] = []
    for block in blocks:
        numbers = extract_citation_numbers(block)
        if not numbers:
            continue
        digest = hashlib.sha256(block.encode()).hexdigest()[:12]
        units.append(
            {
                "unit_id": f"au{len(units)}-{digest}",
                "text": block,
                "evidence_numbers": list(numbers),
            }
        )
    return units
