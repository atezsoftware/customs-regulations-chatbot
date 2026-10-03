"""Patch identified publication defects without regenerating unrelated answer detail."""

import json

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.assertions import assertion_inventory, assertion_witness_valid
from onyx.asv3.authority import unresolved_authority_gap
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    MaterialSourceOmission,
    ResearchModel,
    StructuredOutputError,
)
from onyx.asv3.models import RunStopped, ToolOutcome
from onyx.prompts.asv3.research import ANSWER_REPAIR_PROMPT
from onyx.tracing.flows import LLMFlow


class AnswerUnitReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    replacement: str = Field(min_length=1, max_length=20000)


class AnswerUnitInsertion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    after_unit_id: str
    omission_ids: list[str] = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=20000)


class AnswerRepairPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    replacements: list[AnswerUnitReplacement] = Field(
        default_factory=list, max_length=64
    )
    insertions: list[AnswerUnitInsertion] = Field(default_factory=list, max_length=64)


def repair_publication_candidate(
    model: ResearchModel,
    ledger: EvidenceLedger,
    *,
    answer: str,
    scenario: str,
    gap: ToolOutcome,
    evidence: str,
    research_state: dict[str, JsonValue],
) -> str | None:
    """Return None when no exact defect can be localized; patches still require full review."""
    units = assertion_inventory(answer)
    declared = gap.data.get("assertion_gaps", [])
    negative_ids = (
        {
            str(row["unit_id"])
            for row in declared
            if isinstance(row, dict) and isinstance(row.get("unit_id"), str)
        }
        if isinstance(declared, list)
        else set()
    )
    target_ids = {
        unit["unit_id"]
        for unit in units
        if unit["unit_id"] in negative_ids
        or unresolved_authority_gap(unit["text"], ledger) is not None
    }
    omitted = gap.data.get("omitted_material_source_details", [])
    try:
        omissions = (
            [MaterialSourceOmission.model_validate(row) for row in omitted]
            if isinstance(omitted, list)
            else []
        )
    except ValueError:
        return None
    for omission in omissions:
        original = ledger.get(omission.witness.citation)
        if original is None or not assertion_witness_valid(
            omission.witness, {omission.witness.citation: original.text}
        ):
            return None
    required_omissions = {f"om{index}": row for index, row in enumerate(omissions)}
    if not target_ids and not required_omissions:
        return None
    allowed = set(ledger.citation_mapping())
    unit_ids = {unit["unit_id"] for unit in units}

    def validate(text: str) -> None:
        patch = AnswerRepairPatch.model_validate_json(text)
        identities = [row.unit_id for row in patch.replacements]
        if set(identities) != target_ids or len(identities) != len(target_ids):
            raise ValueError(
                "Replace each exact rejected unit once; other units are immutable"
            )
        assigned = [
            identifier for row in patch.insertions for identifier in row.omission_ids
        ]
        if set(assigned) != set(required_omissions) or len(assigned) != len(
            required_omissions
        ):
            raise ValueError("Add each witnessed missing requirement exactly once")
        for insertion in patch.insertions:
            if insertion.after_unit_id not in unit_ids:
                raise ValueError("Insertions need an exact existing answer unit")
            numbers = set(extract_citation_numbers(insertion.text))
            if any(
                required_omissions[key].witness.citation not in numbers
                for key in insertion.omission_ids
            ):
                raise ValueError(
                    "Each added detail needs its own required inline original"
                )
        for replacement in [
            *[row.replacement for row in patch.replacements],
            *[row.text for row in patch.insertions],
        ]:
            if not replacement.strip():
                raise ValueError(
                    "Preserve supported detail or disclose the exact unresolved outcome"
                )
            if set(extract_citation_numbers(replacement)) - allowed:
                raise ValueError("A replacement cannot cite an unrecorded original")
            defect = unresolved_authority_gap(replacement, ledger)
            if defect is not None:
                raise ValueError(
                    "The replacement retains an unverified governing attribution: "
                    + json.dumps(defect, ensure_ascii=False)
                )

    try:
        text = model.invoke_text(
            ANSWER_REPAIR_PROMPT,
            json.dumps(
                {
                    "language": model.context.language,
                    "scenario": scenario,
                    "answer_units": units,
                    "target_unit_ids": sorted(target_ids),
                    "required_omissions": {
                        key: row.model_dump(mode="json")
                        for key, row in required_omissions.items()
                    },
                    "publication_gap": gap.model_dump(mode="json"),
                    "research_state": research_state,
                    "evidence": evidence,
                },
                ensure_ascii=False,
            ),
            LLMFlow.ASV3_ANSWER_REPAIR,
            max_tokens=9000,
            response_model_override=AnswerRepairPatch,
            response_validator=validate,
        )
        patch = AnswerRepairPatch.model_validate_json(text)
        replacements = {row.unit_id: row.replacement for row in patch.replacements}
        insertions: dict[str, list[str]] = {}
        for row in patch.insertions:
            insertions.setdefault(row.after_unit_id, []).append(row.text)
    except (StructuredOutputError, RunStopped):
        model.context.check_active()
        return answer
    # Replace in source order, preserving exact formatting and all untouched blocks.
    parts: list[str] = []
    offset = 0
    for unit in units:
        start = answer.find(unit["text"], offset)
        if start < 0:
            return answer
        end = start + len(unit["text"])
        parts.extend(
            [answer[offset:start], replacements.get(unit["unit_id"], unit["text"])]
        )
        for insertion in insertions.get(unit["unit_id"], []):
            parts.append("\n\n" + insertion)
        offset = end
    parts.append(answer[offset:])
    return "".join(parts)
