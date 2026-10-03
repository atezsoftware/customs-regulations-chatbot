"""Bind every answer block to an explicit source, scenario or presentation basis."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from onyx.asv3.citation_numbers import extract_citation_numbers


class AssertionWitness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation: Annotated[int, Field(strict=True, ge=1)]
    source_quote: str = Field(
        min_length=1,
        max_length=800,
        description="A short contiguous verbatim passage copied from this citation's supplied original. Do not paraphrase, concatenate separated clauses or insert ellipses unless they occur literally in the original.",
    )


class AssertionVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    status: Literal["supported", "unsupported", "uncertain"]
    basis: Literal["original", "scenario", "presentation", "evidence_gap"] = "original"
    scenario_quotes: list[Annotated[str, Field(min_length=1, max_length=800)]] = Field(
        default_factory=list,
        max_length=6,
        description="Contiguous literal user-supplied facts supporting a facts-only block or arithmetic. Legal consequences require original sources instead.",
    )
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
    presentation_only: bool


def presentation_block(text: str) -> bool:
    """Recognize structure, not whether a heading's content is legally true."""
    return all(
        re.fullmatch(r"\s*(?:#{1,6}\s+.+|(?:[-*_]\s*){3,}|\*\*[^*]+:\*\*)\s*", line)
        for line in text.splitlines()
    )


def assertion_support_defect(
    unit: AssertionUnit,
    check: AssertionVerification,
    originals: Mapping[int, str],
    scenario: str,
    *,
    allow_explicit_gaps: bool = False,
) -> str | None:
    # Literal normalization is shared with the quotation contract.
    from onyx.asv3.quotations import normalized

    if check.basis == "evidence_gap":
        if (
            allow_explicit_gaps
            and check.status == "uncertain"
            and check.missing_conditions
            and not unit["evidence_numbers"]
            and not check.witnesses
        ):
            return None
        return "An evidence-gap notice must disclose an unresolved issue, without asserting its legal answer."
    if check.status != "supported" or check.missing_conditions:
        return "This block still has an unsupported outcome or missing condition."
    if check.basis != "original":
        if unit["evidence_numbers"] or check.witnesses:
            return "A cited block requires an original-source assessment."
        if check.basis == "presentation":
            if unit["presentation_only"] and not check.scenario_quotes:
                return None
            return "Presentation-only prose must be a heading or label without a substantive claim; remove unnecessary introductory prose instead of researching it."
        if check.scenario_quotes and all(
            normalized(quote) in normalized(scenario) for quote in check.scenario_quotes
        ):
            return None
        return "A facts-only block needs literal user-supplied facts; legal applications require inline originals."
    if not unit["evidence_numbers"]:
        return "This legal assertion has no inline original-source citation."
    if check.scenario_quotes or {w.citation for w in check.witnesses} != set(
        unit["evidence_numbers"]
    ):
        return "Each inline citation must have this block's own original witness."
    if any(
        w.citation not in originals
        or not w.source_quote.strip()
        or normalized(w.source_quote) not in normalized(originals[w.citation])
        for w in check.witnesses
    ):
        return "A source witness is not a contiguous literal passage in its original."
    return None


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
        if not block:
            continue
        numbers = extract_citation_numbers(block)
        digest = hashlib.sha256(block.encode()).hexdigest()[:12]
        units.append(
            {
                "unit_id": f"au{len(units)}-{digest}",
                "text": block,
                "evidence_numbers": list(numbers),
                "presentation_only": presentation_block(block),
            }
        )
    return units
