"""A source-writer-bound correction of contradictory terminal resolution metadata."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, Decision, HarnessView, RunContext
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.retained_answer import resolve_retained_answer

_DETAIL = "Supported outcome needs originals and no open gap"
_TERMINALS = frozenset(
    {
        "submit_answer",
        "submit_partial_answer",
        "submit_retained_answer",
        "submit_retained_partial_answer",
    }
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _validation_gap(
    call: CapabilityCall, context: RunContext, registry: CapabilityRegistry
) -> str | None:
    outcomes = context.services.get("outcome_map")
    ledger = context.services.get("evidence")
    if not isinstance(outcomes, OutcomeMap) or not isinstance(ledger, EvidenceLedger):
        raise ValueError("Terminal metadata needs its outcome and original records")
    isolated = copy.copy(context)
    isolated.services = dict(context.services)
    validation = copy.copy(outcomes)
    validation.restore(outcomes.export(), ledger)
    isolated.services["outcome_map"] = validation
    # Followup validation belongs to actual terminal dispatch. It has its own
    # registry cache and is outside this resolution-only correction.
    checked = call.model_copy(
        update={
            "arguments": {
                key: value
                for key, value in call.arguments.items()
                if key != "_required_followups"
            }
        }
    )
    gap = registry.outcome_metadata_gap(checked, isolated)
    if gap is None:
        return None
    if gap.data.get("invalid_outcome_metadata") is not True:
        raise ValueError("Terminal metadata has a different publication defect")
    detail = gap.data.get("detail")
    return detail if isinstance(detail, str) else "Invalid terminal metadata"


@dataclass(frozen=True)
class CroppedTerminalMetadataRepair:
    decision: Decision
    canonical_call: CapabilityCall
    arguments_json: str
    resolutions_json: str
    mutable_indexes: tuple[int, ...]
    writer_call_id: str
    fence_json: str
    source_rows_json: str
    payload_json: str

    def apply(
        self,
        patch: dict[str, JsonValue],
        context: RunContext,
        registry: CapabilityRegistry,
    ) -> Decision:
        if _fence(context) != self.fence_json:
            raise ValueError("Terminal metadata correction changed execution ownership")
        if context.services.get("last_model_call_id") != self.writer_call_id:
            raise ValueError(
                "Terminal metadata correction lost its actual source writer"
            )
        ledger = context.services.get("evidence")
        assert isinstance(ledger, EvidenceLedger)
        rows = cast(list[dict[str, JsonValue]], json.loads(self.source_rows_json))
        numbers = {cast(int, row["citation"]) for row in rows}
        if not numbers.issubset(ledger.completely_delivered(self.writer_call_id)):
            raise ValueError(
                "Terminal metadata sources were not delivered to its writer"
            )
        if _source_rows(ledger, numbers) != self.source_rows_json:
            raise ValueError("Terminal metadata original identity or text changed")
        replacement = patch.get("resolutions")
        if set(patch) != {"resolutions"} or not isinstance(replacement, list):
            raise ValueError("Terminal metadata patch may contain only resolutions")
        prior = cast(list[dict[str, JsonValue]], json.loads(self.resolutions_json))
        if len(replacement) != len(prior):
            raise ValueError("Terminal metadata patch changed resolution identities")
        for index, (old, new) in enumerate(zip(prior, replacement, strict=True)):
            if not isinstance(new, dict):
                raise ValueError("Terminal metadata resolution must remain an object")
            if index not in self.mutable_indexes:
                if _json(old) != _json(new):
                    raise ValueError(
                        "Terminal metadata patch changed a valid resolution"
                    )
                continue
            if set(new) - {"status", "gap"} != set(old) - {"status", "gap"} or any(
                new.get(key) != value
                for key, value in old.items()
                if key not in {"status", "gap"}
            ):
                raise ValueError(
                    "Terminal metadata patch changed originals or conditions"
                )
            status, gap = new.get("status"), new.get("gap", "")
            if (
                not isinstance(status, str)
                or status not in {old.get("status"), "unresolved"}
                or not isinstance(gap, str)
                or gap not in {"", old.get("gap")}
            ):
                raise ValueError("Terminal metadata patch invented a status or gap")
        original = cast(dict[str, JsonValue], json.loads(self.arguments_json))
        coverage = original["_coverage"]
        assert isinstance(coverage, dict)
        coverage["resolutions"] = replacement
        canonical = self.canonical_call.model_copy(
            update={
                "arguments": {
                    **self.canonical_call.arguments,
                    "_coverage": coverage,
                }
            }
        )
        if (detail := _validation_gap(canonical, context, registry)) is not None:
            raise ValueError("Terminal metadata correction remains invalid: " + detail)
        corrected = self.decision.calls[0].model_copy(update={"arguments": original})
        # Keep the actual source-bearing native message; this patch is separately
        # traced and is neither a new answer nor a delivery of original texts.
        return self.decision.model_copy(update={"calls": [corrected]})


def _fence(context: RunContext) -> str:
    return _json(
        [context.run_id, context.scope, context.depth, context.services.get("task_id")]
    )


def _source_rows(ledger: EvidenceLedger, numbers: set[int]) -> str:
    rows: list[dict[str, JsonValue]] = []
    for number in sorted(numbers):
        item = ledger.get(number)
        if (
            item is None
            or hashlib.sha256(item.text.encode()).hexdigest() != item.text_hash
        ):
            raise ValueError("Terminal metadata requires immutable original text")
        rows.append(
            {
                "citation": number,
                "source_id": item.source_id,
                "chunk_id": item.chunk_id,
                "text_hash": item.text_hash,
                "metadata": item.metadata,
            }
        )
    return _json(rows)


def prepare_cropped_terminal_metadata_repair(
    decision: Decision,
    view: HarnessView,
    context: RunContext,
    writer_call_id: str | None,
) -> CroppedTerminalMetadataRepair | None:
    """Inspect on a private outcome snapshot; healthy dispatch stays untouched."""
    if (
        context.services.get("asv3_exp_cropped") is not True
        or context.services.get("asv3_exp_cheap") is True
        or len(decision.calls) != 1
        or decision.calls[0].name not in _TERMINALS
        or decision.calls[0].argument_error is not None
    ):
        return None
    registry = context.services.get("registry")
    outcomes = context.services.get("outcome_map")
    ledger = context.services.get("evidence")
    if not (
        isinstance(registry, CapabilityRegistry)
        and isinstance(outcomes, OutcomeMap)
        and isinstance(ledger, EvidenceLedger)
    ):
        return None
    canonical = resolve_retained_answer(
        decision, context, view.draft_to_repair, request=view.request
    ).calls[0]
    if (
        canonical.argument_error is not None
        or _validation_gap(canonical, context, registry) != _DETAIL
    ):
        return None
    call = decision.calls[0]
    coverage = call.arguments.get("_coverage")
    resolutions = coverage.get("resolutions") if isinstance(coverage, dict) else None
    if not isinstance(resolutions, list) or not all(
        isinstance(row, dict) for row in resolutions
    ):
        return None
    mutable = tuple(
        index
        for index, row in enumerate(resolutions)
        if isinstance(row, dict)
        and row.get("status") in {"supported", "conditional"}
        and isinstance(gap := row.get("gap"), str)
        and gap.strip()
    )
    if not mutable:
        return None
    answer = canonical.arguments.get("answer")
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or not writer_call_id
        or context.services.get("last_model_call_id") != writer_call_id
    ):
        raise ValueError(
            "Terminal metadata correction needs its unchanged source writer"
        )
    numbers = set(extract_citation_numbers(answer))
    for row in resolutions:
        assert isinstance(row, dict)
        numbers.update(cast(list[int], row.get("evidence_numbers", [])))
    outcome_ids = [
        cast(str, row["outcome_id"]) for row in resolutions if isinstance(row, dict)
    ]
    conditions = outcomes.view(outcome_ids=outcome_ids).get("conditions", [])
    declared = coverage.get("conditions", []) if isinstance(coverage, dict) else []
    for collection in (conditions, declared):
        if isinstance(collection, list):
            for condition in collection:
                witnesses = (
                    condition.get("witnesses") if isinstance(condition, dict) else None
                )
                if isinstance(witnesses, list):
                    for witness in witnesses:
                        citation = (
                            witness.get("citation")
                            if isinstance(witness, dict)
                            else None
                        )
                        if type(citation) is int:
                            numbers.add(citation)
    if not numbers.issubset(ledger.completely_delivered(writer_call_id)):
        raise ValueError(
            "Terminal metadata correction cannot borrow another writer's sources"
        )
    source_rows = _source_rows(ledger, numbers)
    return CroppedTerminalMetadataRepair(
        decision=decision.model_copy(deep=True),
        canonical_call=canonical.model_copy(deep=True),
        arguments_json=_json(call.arguments),
        resolutions_json=_json(resolutions),
        mutable_indexes=mutable,
        writer_call_id=writer_call_id,
        fence_json=_fence(context),
        source_rows_json=source_rows,
        payload_json=_json(
            {
                "request": view.request,
                "call_id": "metadata:0",
                "validation_error": _DETAIL,
                "resolutions": resolutions,
                "mutable_resolution_indexes": list(mutable),
                "existing_outcome_conditions": outcomes.view(
                    delivered_citations=ledger.completely_delivered(writer_call_id)
                ),
                "unchanged_unapproved_answer": answer,
                "unchanged_answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "preserved_gap_observations": [
                    cast(dict[str, JsonValue], resolutions[index])["gap"]
                    for index in mutable
                ],
                "patch_keys": ["resolutions"],
            }
        ),
    )
