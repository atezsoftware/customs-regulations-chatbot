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
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.retained_answer import resolve_retained_answer

_DETAIL = "Supported outcome needs originals and no open gap"
_CONDITION_DETAILS = frozenset(
    {
        "Supported outcome must retain its source conditions",
        "Resolution condition belongs to another outcome",
    }
)
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
    condition_bindings_json: str | None = None

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
        if self.condition_bindings_json is not None:
            if (
                _json(replacement) != self.condition_bindings_json
                or _json(_condition_bindings(self.canonical_call, context))
                != self.condition_bindings_json
            ):
                raise ValueError(
                    "Terminal condition bindings changed recorded relationships"
                )
        for index, (old, new) in enumerate(zip(prior, replacement, strict=True)):
            if self.condition_bindings_json is not None:
                continue
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


def _condition_bindings(
    call: CapabilityCall,
    context: RunContext,
) -> list[dict[str, JsonValue]]:
    """Rebind only host-validated conditions and their existing original witnesses."""
    outcomes, ledger = (
        context.services.get("outcome_map"),
        context.services.get("evidence"),
    )
    if not isinstance(outcomes, OutcomeMap) or not isinstance(ledger, EvidenceLedger):
        raise ValueError(
            "Condition bindings need their owned outcome and source records"
        )
    coverage = call.arguments.get("_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Condition bindings need recorded coverage")
    payload = dict(coverage)
    if "_outcomes" in call.arguments:
        if "outcomes" in payload:
            raise ValueError("Use only _outcomes to declare outcomes")
        payload["outcomes"] = call.arguments["_outcomes"]
    update = OutcomeUpdate.model_validate(payload)
    staged = copy.copy(outcomes)
    staged.restore(outcomes.export(), ledger)
    staged.update(update.model_copy(update={"resolutions": []}), ledger)
    state = staged.view()
    conditions = state["conditions"]
    declared_outcomes = state["outcomes"]
    assert isinstance(conditions, list) and isinstance(declared_outcomes, list)
    known = {row["condition_id"] for row in conditions if isinstance(row, dict)}
    known_outcomes = {
        row["outcome_id"] for row in declared_outcomes if isinstance(row, dict)
    }
    raw = coverage.get("resolutions")
    if not isinstance(raw, list):
        raise ValueError("Condition bindings need recorded resolutions")
    result: list[dict[str, JsonValue]] = []
    for row in raw:
        if not isinstance(row, dict) or row.get("outcome_id") not in known_outcomes:
            raise ValueError("Condition binding refers to an unknown outcome")
        current_ids = row.get("condition_ids", [])
        current_numbers = row.get("evidence_numbers", [])
        if not isinstance(current_ids, list) or not isinstance(current_numbers, list):
            raise ValueError("Condition binding fields must remain lists")
        if any(
            not isinstance(identity, str) or identity not in known
            for identity in current_ids
        ):
            raise ValueError("Condition binding refers to an unknown source condition")
        applicable = [
            condition
            for condition in conditions
            if isinstance(condition, dict)
            and isinstance(condition.get("outcome_ids"), list)
            and row["outcome_id"] in condition["outcome_ids"]
        ]
        identities = [condition["condition_id"] for condition in applicable]
        retained = [identity for identity in current_ids if identity in identities]
        corrected = copy.deepcopy(row)
        corrected["condition_ids"] = retained
        if row.get("status") in {"supported", "conditional"}:
            corrected["condition_ids"] = retained + [
                identity for identity in identities if identity not in retained
            ]
            citations = [
                witness["citation"]
                for condition in applicable
                for witness in cast(list[dict[str, JsonValue]], condition["witnesses"])
            ]
            corrected["evidence_numbers"] = current_numbers + sorted(
                {cast(int, number) for number in citations}
                - set(cast(list[int], current_numbers))
            )
        result.append(corrected)
    return result


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
    if canonical.argument_error is not None:
        return None
    detail = _validation_gap(canonical, context, registry)
    if detail not in {
        _DETAIL,
        *_CONDITION_DETAILS,
    }:
        return None
    call = decision.calls[0]
    coverage = call.arguments.get("_coverage")
    resolutions = coverage.get("resolutions") if isinstance(coverage, dict) else None
    if not isinstance(resolutions, list) or not all(
        isinstance(row, dict) for row in resolutions
    ):
        return None
    bindings = (
        _condition_bindings(canonical, context)
        if detail in _CONDITION_DETAILS
        else None
    )
    mutable = (
        tuple(
            index
            for index, (old, new) in enumerate(zip(resolutions, bindings, strict=True))
            if old != new
        )
        if bindings is not None
        else tuple(
            index
            for index, row in enumerate(resolutions)
            if isinstance(row, dict)
            and row.get("status") in {"supported", "conditional"}
            and isinstance(gap := row.get("gap"), str)
            and gap.strip()
        )
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
                "validation_error": detail,
                "resolutions": resolutions,
                "mutable_resolution_indexes": list(mutable),
                "existing_outcome_conditions": outcomes.view(
                    delivered_citations=ledger.completely_delivered(writer_call_id)
                ),
                "unchanged_unapproved_answer": answer,
                "unchanged_answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "preserved_gap_observations": [
                    cast(dict[str, JsonValue], resolutions[index]).get("gap", "")
                    for index in mutable
                ],
                "patch_keys": ["resolutions"],
            }
        ),
        condition_bindings_json=_json(bindings) if bindings is not None else None,
    )
