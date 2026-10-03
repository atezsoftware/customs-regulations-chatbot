"""Bind every answer block to an explicit source, scenario or presentation basis."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.witnesses import original_witness_text


class AssertionWitness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation: Annotated[int, Field(strict=True, ge=1)]
    source_quote: str = Field(
        default="",
        max_length=800,
        description="Legacy literal witness when no witness catalogue is supplied. Leave empty when selecting witness_id; never paraphrase or insert ellipses.",
    )
    witness_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Select a supplied witness_spans identifier from this citation's original. Do not calculate offsets, invent IDs or recopy its text.",
    )

    @model_validator(mode="after")
    def exactly_one_witness(self) -> AssertionWitness:
        if bool(self.source_quote.strip()) == bool(self.witness_id):
            raise ValueError("Select exactly one original witness ID or literal quote")
        return self


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
        default="",
        max_length=4000,
        description="Omit for a supported block. For a defect, give one short actionable sentence without repeating the answer or other assessments.",
    )


def assertion_witness_valid(
    witness: AssertionWitness, originals: Mapping[int, str]
) -> bool:
    from onyx.asv3.quotations import normalized

    text = originals.get(witness.citation)
    if text is None:
        return False
    if witness.witness_id:
        return (
            not witness.source_quote.strip()
            and original_witness_text(witness.citation, text, witness.witness_id)
            is not None
        )
    return bool(witness.source_quote.strip()) and normalized(
        witness.source_quote
    ) in normalized(text)


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
    if any(not assertion_witness_valid(w, originals) for w in check.witnesses):
        return "A source witness does not select a catalogued passage or contiguous literal quote in its exact original."
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
    bound_blocks: list[str] = []
    for block in blocks:
        if (
            bound_blocks
            and bound_blocks[-1].endswith(":")
            and not presentation_block(bound_blocks[-1])
            and not extract_citation_numbers(bound_blocks[-1])
            and re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", block)
        ):
            # An introductory clause and its first list item form one cited assertion.
            bound_blocks[-1] += "\n\n" + block
        else:
            bound_blocks.append(block)
    units: list[AssertionUnit] = []
    for block in bound_blocks:
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
