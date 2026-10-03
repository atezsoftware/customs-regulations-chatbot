"""Patch identified publication defects without regenerating unrelated answer detail."""

import json

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import unresolved_authority_gap
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel, StructuredOutputError
from onyx.asv3.models import RunStopped, ToolOutcome
from onyx.prompts.asv3.research import ANSWER_REPAIR_PROMPT
from onyx.tracing.flows import LLMFlow


class AnswerUnitReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    replacement: str = Field(min_length=1, max_length=20000)


class AnswerRepairPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    replacements: list[AnswerUnitReplacement] = Field(min_length=1, max_length=64)


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
    if not target_ids:
        return None
    allowed = set(ledger.citation_mapping())

    def validate(text: str) -> None:
        patch = AnswerRepairPatch.model_validate_json(text)
        identities = [row.unit_id for row in patch.replacements]
        if set(identities) != target_ids or len(identities) != len(target_ids):
            raise ValueError(
                "Replace each exact rejected unit once; other units are immutable"
            )
        for row in patch.replacements:
            if not row.replacement.strip():
                raise ValueError(
                    "Preserve supported detail or disclose the exact unresolved outcome"
                )
            if set(extract_citation_numbers(row.replacement)) - allowed:
                raise ValueError("A replacement cannot cite an unrecorded original")
            defect = unresolved_authority_gap(row.replacement, ledger)
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
        replacements = {
            row.unit_id: row.replacement
            for row in AnswerRepairPatch.model_validate_json(text).replacements
        }
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
        offset = end
    parts.append(answer[offset:])
    return "".join(parts)
