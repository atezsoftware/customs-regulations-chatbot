"""Captured 5464 bindings are corrected without regenerating its retained answer."""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_repair_state import CroppedRepairState
from onyx.asv3.cropped_terminal_metadata import prepare_cropped_terminal_metadata_repair
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.retained_answer import resolve_retained_answer
from onyx.asv3.terminal_fact_references import normalize_terminal_fact_references
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_native_model_adapter import model

_FIXTURE = (
    Path(__file__).with_name("fixtures") / "cropped_terminal_conditions_5464.json.gz"
)


class _CapturedLedger(EvidenceLedger):
    """Read-only sparse projection keeps the captured global citation numbers."""

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__()
        self.originals = {
            row["citation"]: EvidenceItem.model_validate(row["item"])
            for row in raw["records"]
        }
        self.deliveries = raw["deliveries"]
        self.unread: set[int] = set()
        for item in self.originals.values():
            assert hashlib.sha256(item.text.encode()).hexdigest() == item.text_hash
        for delivery in self.deliveries:
            for row in delivery["records"]:
                item = self.originals[row["citation"]]
                assert item.identity == (
                    row["source_id"],
                    row["chunk_id"],
                    row["text_hash"],
                )
                if row["complete"]:
                    assert row["start_char"] == 0 and row["end_char"] == len(item.text)
                    assert (
                        row["passage_hash"]
                        == hashlib.sha256(item.text.encode()).hexdigest()
                    )

    def get(self, number: int) -> EvidenceItem | None:
        return self.originals.get(number)

    def completely_delivered(self, call_id: str) -> set[int]:
        return {
            row["citation"]
            for delivery in self.deliveries
            if delivery["call_id"] == call_id
            for row in delivery["records"]
            if row["complete"]
        } - self.unread


def _captured(
    which: str,
) -> tuple[
    ResearchModel,
    HarnessView,
    Decision,
    _CapturedLedger,
    OutcomeMap,
    CapabilityRegistry,
    MagicMock,
    CroppedRepairState,
]:
    data = json.loads(gzip.decompress(_FIXTURE.read_bytes()))
    owner = data["context"]
    context = RunContext(
        run_id=owner["run_id"], scope=owner["scope"], depth=owner["depth"]
    )
    ledger = _CapturedLedger(data["evidence_export_sparse"])
    outcomes = OutcomeMap(
        data["questions"],
        context,
        factual_context=data["factual_context"],
        reuse_retained_conditions=True,
    )
    outcomes.restore(data["outcome_map_before_dispatch"], ledger)
    guard = MagicMock(
        return_value=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="An independent publication guard still applies.",
        )
    )
    registry = CapabilityRegistry()
    registry.register(
        ToolSpec(
            name="submit_answer",
            description="Guarded publication",
            parameters={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "basis": {"type": "string", "enum": ["originals"]},
                },
                "required": ["answer", "basis"],
                "additionalProperties": False,
            },
            handler=guard,
        )
    )
    source = data[
        "first_terminal_call" if which == "first" else "latest_retained_repair_call"
    ]
    context.services.update(
        asv3_exp_cropped=True,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
        evidence=ledger,
        outcome_map=outcomes,
        registry=registry,
        last_model_call_id=source["writer_call_id"],
    )
    selected = model(limit=1_047_552)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    adapter.last_call_id = source["writer_call_id"]
    view = HarnessView(
        request=data["request"],
        questions=data["questions"],
        facts=[],
        receipts=[],
        evidence=[],
        original_evidence=[],
        tools=registry.definitions(context),
        draft_to_repair=data["first_terminal_call"]["arguments"]["answer"],
    )
    decision = Decision(
        calls=[
            CapabilityCall(
                name=source["tool_name"],
                arguments=normalize_terminal_fact_references(
                    source["tool_name"], source["arguments"], context
                ),
            )
        ]
    )
    canonical = resolve_retained_answer(
        decision, context, view.draft_to_repair, request=view.request
    )
    assert canonical.calls[0].argument_error is None
    expected_hash = (
        data["first_terminal_answer_sha256"]
        if which == "first"
        else data["checkpoint_last_draft_sha256"]
    )
    assert (
        hashlib.sha256(str(canonical.calls[0].arguments["answer"]).encode()).hexdigest()
        == expected_hash
    )
    repair_state = CroppedRepairState(context, view.request)
    context.services["cropped_repair_state"] = repair_state
    return adapter, view, decision, ledger, outcomes, registry, guard, repair_state


@pytest.mark.parametrize(
    "which,missing", [("first", {2, 3, 52}), ("retained", {2, 3, 8, 52, 180, 203})]
)
def test_captured_binding_repair_retains_body_conditions_and_actual_writer(
    which: str,
    missing: set[int],
) -> None:
    adapter, view, decision, ledger, outcomes, registry, guard, repair_state = (
        _captured(which)
    )
    before = outcomes.export()
    sources = {n: item.model_dump(mode="json") for n, item in ledger.originals.items()}
    contract = prepare_cropped_terminal_metadata_repair(
        decision, view, adapter.context, adapter.last_call_id
    )
    assert contract is not None and contract.condition_bindings_json is not None
    original = contract.canonical_call.arguments
    original_coverage = original["_coverage"]
    assert isinstance(original_coverage, dict)
    old_rows = original_coverage["resolutions"]
    rows = json.loads(contract.condition_bindings_json)
    assert (
        set(rows[0]["evidence_numbers"]) - set(old_rows[0]["evidence_numbers"])
        == missing
    )
    corrected = adapter._repair_cropped_terminal_metadata(decision, view, adapter.llm)
    expanded = resolve_retained_answer(
        corrected, adapter.context, view.draft_to_repair, request=view.request
    ).calls[0]
    after_coverage = expanded.arguments["_coverage"]
    assert isinstance(after_coverage, dict)
    assert after_coverage["conditions"] == original_coverage["conditions"]
    assert {k: v for k, v in expanded.arguments.items() if k != "_coverage"} == {
        k: v for k, v in original.items() if k != "_coverage"
    }
    for old, new in zip(old_rows, after_coverage["resolutions"], strict=True):
        assert {
            k: v
            for k, v in old.items()
            if k not in {"condition_ids", "evidence_numbers"}
        } == {
            k: v
            for k, v in new.items()
            if k not in {"condition_ids", "evidence_numbers"}
        }
        assert set(old["evidence_numbers"]) <= set(new["evidence_numbers"])
    assert outcomes.export() == before
    assert repair_state.export()["records"] == []
    assert adapter.context.services["last_model_call_id"] == adapter.last_call_id
    assert isinstance(adapter.llm, MagicMock)
    adapter.llm.invoke.assert_not_called()
    result = registry.dispatch(expanded, adapter.context)
    assert result.status == OutcomeStatus.PARTIAL
    guard.assert_called_once()
    assert {
        n: item.model_dump(mode="json") for n, item in ledger.originals.items()
    } == sources


@pytest.mark.parametrize(
    "tamper",
    [
        "binding",
        "evidence",
        "status",
        "gap",
        "body",
        "scope",
        "writer",
        "source",
        "unread",
    ],
)
def test_captured_condition_correction_cannot_change_semantics_or_borrow_sources(
    tamper: str,
) -> None:
    adapter, view, decision, ledger, _outcomes, registry, _guard, _state = _captured(
        "first"
    )
    contract = prepare_cropped_terminal_metadata_repair(
        decision, view, adapter.context, adapter.last_call_id
    )
    assert contract is not None and contract.condition_bindings_json is not None
    patch: dict[str, JsonValue] = {
        "resolutions": json.loads(contract.condition_bindings_json)
    }
    rows = patch["resolutions"]
    if tamper == "binding":
        rows[0]["condition_ids"] = []
    elif tamper == "evidence":
        rows[0]["evidence_numbers"].append(999)
    elif tamper == "status":
        rows[0]["status"] = "unresolved"
    elif tamper == "gap":
        rows[0]["gap"] = "Invented limitation"
    elif tamper == "body":
        patch["answer"] = "Rewritten legal answer"
    elif tamper == "scope":
        adapter.context.scope["asv3_document_set_id"] = 99
    elif tamper == "writer":
        adapter.context.services["last_model_call_id"] = "foreign-writer"
    elif tamper == "source":
        ledger.originals[2].text += " Altered."
    else:
        ledger.unread.add(2)
    with pytest.raises(ValueError):
        contract.apply(patch, adapter.context, registry)


def test_unknown_source_condition_fails_before_any_source_acquisition() -> None:
    adapter, view, decision, _ledger, _outcomes, _registry, _guard, state = _captured(
        "first"
    )
    decision = decision.model_copy(deep=True)
    coverage = decision.calls[0].arguments["_coverage"]
    assert isinstance(coverage, dict)
    rows = coverage["resolutions"]
    assert isinstance(rows, list) and isinstance(rows[2], dict)
    condition_ids = rows[2]["condition_ids"]
    assert isinstance(condition_ids, list)
    condition_ids.append("unknown_condition")
    with pytest.raises(RunStopped, match="unknown source condition"):
        adapter._repair_cropped_terminal_metadata(decision, view, adapter.llm)
    assert state.export()["records"] == []
    assert isinstance(adapter.llm, MagicMock)
    adapter.llm.invoke.assert_not_called()
