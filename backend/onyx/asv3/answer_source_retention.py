"""Retain originals already selected for supported outcome resolutions."""

from __future__ import annotations

import hashlib
import json

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.outcome_map import OutcomeCondition, OutcomeMap, OutcomeResolution
from onyx.asv3.parallel_execution import parallel_execution_enabled


def selected_outcome_source_gap(
    answer: str,
    call_id: str | None,
    context: RunContext,
    ledger: EvidenceLedger,
    outcomes: OutcomeMap,
) -> ToolOutcome | None:
    """Check retained citation coverage; applicability remains model-assessed."""
    if not parallel_execution_enabled(context):
        return None
    scope_hash = hashlib.sha256(
        json.dumps(context.scope, sort_keys=True).encode()
    ).hexdigest()
    if (outcomes.run_id, outcomes.scope_hash) != (context.run_id, scope_hash):
        return ToolOutcome(
            status=OutcomeStatus.DENIED,
            summary="Retained outcome sources belong to another run or scope.",
            data={"retained_outcome_scope_mismatch": True},
        )
    bound = context.services.get("outcome_map")
    if isinstance(bound, OutcomeMap) and bound is not outcomes:
        return ToolOutcome(
            status=OutcomeStatus.DENIED,
            summary="Use this request's actual retained outcome sources.",
            data={"retained_outcome_request_mismatch": True},
        )
    snapshot = outcomes.export()
    resolution_rows, condition_rows = snapshot["resolutions"], snapshot["conditions"]
    assert isinstance(resolution_rows, list) and isinstance(condition_rows, list)
    conditions: dict[str, tuple[OutcomeCondition, dict[str, JsonValue]]] = {}
    for row in condition_rows:
        assert isinstance(row, dict) and isinstance(row["source_hashes"], dict)
        condition = OutcomeCondition.model_validate(row["condition"])
        conditions[condition.condition_id] = condition, row["source_hashes"]

    cited = set(extract_citation_numbers(answer))
    delivered = ledger.completely_delivered(call_id or "")
    missing_outcomes: list[str] = []
    missing_conditions: list[str] = []
    missing_citations: set[int] = set()
    undelivered: set[int] = set()
    invalid_witnesses: list[JsonValue] = []
    retained_sources: list[JsonValue] = []
    for row in resolution_rows:
        resolution = OutcomeResolution.model_validate(row)
        if resolution.status not in {"supported", "conditional"}:
            continue
        required = set(resolution.evidence_numbers)
        witnesses: list[JsonValue] = []
        invalid = False
        for identity in resolution.condition_ids:
            condition, hashes = conditions[identity]
            condition_gap = False
            for witness in condition.witnesses:
                required.add(witness.citation)
                expected_hash = hashes[str(witness.citation)]
                item = ledger.get(witness.citation)
                bound: dict[str, JsonValue] = {
                    "condition_id": identity,
                    **witness.model_dump(mode="json"),
                    "text_hash": expected_hash,
                }
                witnesses.append(bound)
                if (
                    item is None
                    or item.text_hash != expected_hash
                    or not witness.start_char < witness.end_char <= len(item.text)
                ):
                    invalid_witnesses.append(bound)
                    invalid = condition_gap = True
                if witness.citation not in cited or witness.citation not in delivered:
                    condition_gap = True
            if condition_gap and identity not in missing_conditions:
                missing_conditions.append(identity)
        absent = required - cited
        unread = required - delivered
        if absent or unread or invalid:
            missing_outcomes.append(resolution.outcome_id)
            missing_citations.update(absent)
            undelivered.update(unread)
            retained_sources.append(
                {
                    "outcome_id": resolution.outcome_id,
                    "status": resolution.status,
                    "condition_ids": list(resolution.condition_ids),
                    "required_evidence_numbers": sorted(required),
                    "missing_inline_citations": sorted(absent),
                    "undelivered_citations": sorted(unread),
                    "witnesses": witnesses,
                }
            )
    if not missing_outcomes:
        return None
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Retain the originals already selected for supported outcomes beside their actual answer claims; correct only the identified omissions.",
        data={
            "missing_selected_outcome_sources": True,
            "outcome_ids": missing_outcomes,
            "condition_ids": missing_conditions,
            "missing_inline_citations": sorted(missing_citations),
            "undelivered_citations": sorted(undelivered),
            "invalid_source_witnesses": invalid_witnesses,
            "retained_outcome_sources": retained_sources,
        },
    )
