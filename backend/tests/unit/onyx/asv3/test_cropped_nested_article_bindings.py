"""Only executed and committed composition subcalls can establish article identity."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.cropped_article_bindings import CroppedArticleBindings
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.sandbox import build_sandbox_specs
from tests.unit.onyx.asv3.test_cropped_focused_paragraph_repair import _CAPTURE, _setup


def _call(*, parallel: bool = False) -> CapabilityCall:
    steps = [
        {
            "id": "context",
            "tool": "read_chunk_context",
            "arguments": {
                "source_id": _CAPTURE["rows"][2]["source_id"],
                "chunk_id": _CAPTURE["rows"][2]["chunk_id"],
            },
        }
    ]
    if parallel:
        steps.append(
            {
                "id": "range",
                "tool": "read_source_range",
                "arguments": {
                    "source_id": _CAPTURE["rows"][0]["source_id"],
                    "start": 30,
                    "limit": 3,
                },
            }
        )
    return CapabilityCall(
        name="compose_tool_calls", arguments={"steps": steps, "max_parallel": 2}
    )


def _install_composer(harness: Harness) -> None:
    # Reuse the actual composition handler and schema; its broker is unused by compose.
    spec = next(
        item
        for item in build_sandbox_specs(cast(CorpusBroker, MagicMock()))
        if item.name == "compose_tool_calls"
    )
    harness.registry.register(spec)


def test_actual_parallel_compose_retains_child_proofs_before_model_stripping() -> None:
    harness, repair, target = _setup()
    _install_composer(harness)
    before = harness.evidence.export()
    receipts = harness.dispatch_repair_reads([_call(parallel=True)])
    assert receipts[0].outcome.status == OutcomeStatus.FOUND
    assert repair._governing_gap(target, harness, [3]) is None
    state = harness.context.services["cropped_article_bindings"]
    saved = state.export()
    assert len(saved["reads"]) == 2
    assert {read["method"] for read in saved["reads"]} == {
        "read_source_range",
        "read_chunk_context",
    }
    assert all(
        [m["position"] for m in read["read"]["members"]] == [30, 31, 32]
        for read in saved["reads"]
    )
    assert len({read["call_id"] for read in saved["reads"]}) == 2
    assert harness.evidence.export() == before
    assert harness.evidence.completely_delivered("not-a-model-call") == set()
    assert (
        "cropped_ordered_source_read"
        not in harness._model_tool_result(receipts[0]).content
    )
    restored = CroppedArticleBindings(harness.context, harness.request)
    restored.restore(saved, harness.context, harness.evidence)
    assert sorted(restored.identity_bindings(harness.context, harness.evidence)) == [
        2,
        3,
    ]


def test_dispatched_child_is_not_bound_until_actual_parent_commit() -> None:
    harness, repair, target = _setup()
    _install_composer(harness)
    call = _call()
    before = harness.evidence.export()
    outcome = harness.registry.dispatch(call, harness.context)
    state = harness.context.services["cropped_article_bindings"]
    assert state.export()["reads"] == []
    assert harness.evidence.export() == before
    assert repair._governing_gap(target, harness, [3]) is not None
    harness._commit_receipt(ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0))
    assert repair._governing_gap(target, harness, [3]) is None


def test_forged_aggregate_dictionary_cannot_create_an_executed_read() -> None:
    harness, repair, target = _setup()
    _install_composer(harness)
    call = _call()
    outcome = harness.registry.dispatch(call, harness.context)
    state = harness.context.services["cropped_article_bindings"]
    # A checkpoint never persists uncommitted captures. Recreate that trusted state.
    restored = CroppedArticleBindings(harness.context, harness.request)
    restored.restore(state.export(), harness.context, harness.evidence)
    harness.context.services["cropped_article_bindings"] = restored
    assert "cropped_ordered_source_read" in json.dumps(outcome.data)
    harness._commit_receipt(ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0))
    assert restored.export()["reads"] == []
    assert repair._governing_gap(target, harness, [3]) is not None


@pytest.mark.parametrize("discard", ["cancelled", "foreign_owner", "foreign_scope"])
def test_uncommitted_or_foreign_parent_cannot_promote_child_proof(discard: str) -> None:
    harness, repair, target = _setup()
    _install_composer(harness)
    call = _call()
    outcome = harness.registry.dispatch(call, harness.context)
    state = harness.context.services["cropped_article_bindings"]
    receipt = ToolReceipt(
        call=call, outcome=outcome, evidence_ids=[1, 2, 3], elapsed_seconds=0
    )
    if discard == "cancelled":
        receipt.outcome.status = OutcomeStatus.CANCELLED
        assert state.record_receipt(receipt, harness.context, harness.evidence) == 0
    elif discard == "foreign_owner":
        child = harness.context.child()
        child.services["task_id"] = "different-owner"
        assert state.record_receipt(receipt, child, harness.evidence) == 0
    else:
        child = harness.context.child()
        child.scope = {"different": "corpus"}
        with pytest.raises(ValueError, match="Foreign"):
            state.record_receipt(receipt, child, harness.evidence)
    assert state.export()["reads"] == []
    assert repair._governing_gap(target, harness, [3]) is not None


def test_late_ledger_body_tamper_rejects_actual_captured_child() -> None:
    harness, _repair, _target = _setup()
    _install_composer(harness)
    call = _call()
    outcome = harness.registry.dispatch(call, harness.context)
    state = harness.context.services["cropped_article_bindings"]
    harness.evidence._items[3].text = harness.evidence._items[3].text.replace(
        "fiil", "olay", 1
    )
    with pytest.raises(ValueError, match="canonical original"):
        harness._commit_receipt(
            ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0)
        )
    assert state.export()["reads"] == []


def test_actual_gapped_compose_does_not_establish_requested_paragraph() -> None:
    harness, repair, target = _setup(holey=True)
    _install_composer(harness)
    harness.dispatch_repair_reads([_call()])
    state = harness.context.services["cropped_article_bindings"]
    assert len(state.export()["reads"]) == 1
    assert state.identity_bindings(harness.context, harness.evidence) == {}
    assert repair._governing_gap(target, harness, [2]) is not None


def test_actual_composer_rejects_nested_compose_before_any_source_execution() -> None:
    harness, _repair, _target = _setup()
    _install_composer(harness)
    call = CapabilityCall(
        name="compose_tool_calls",
        arguments={
            "steps": [
                {
                    "id": "inner",
                    "tool": "compose_tool_calls",
                    "arguments": _call().arguments,
                }
            ]
        },
    )
    outcome = harness.registry.dispatch(call, harness.context)
    assert outcome.status == OutcomeStatus.INVALID
    state = harness.context.services["cropped_article_bindings"]
    assert state.export()["reads"] == [] and state._pending == {}


def test_cancelled_dispatch_discards_uncommitted_child_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, _repair, _target = _setup()
    _install_composer(harness)
    spec = harness.registry.get("compose_tool_calls")
    assert spec is not None
    original = spec.handler

    def cancel_after_children(
        arguments: dict[str, JsonValue], context: RunContext
    ) -> ToolOutcome:
        result = original(arguments, context)
        context.cancel()
        context.check_active()
        return result

    monkeypatch.setitem(
        harness.registry._specs,
        "compose_tool_calls",
        spec.model_copy(update={"handler": cancel_after_children}),
    )
    outcome = harness.registry.dispatch(_call(), harness.context)
    assert outcome.status == OutcomeStatus.CANCELLED
    state = harness.context.services["cropped_article_bindings"]
    assert state.export()["reads"] == [] and state._pending == {}


@pytest.mark.parametrize("variant", ["default", "cheap"])
def test_other_workflows_do_not_capture_composed_article_bindings(variant: str) -> None:
    harness, _repair, _target = _setup()
    _install_composer(harness)
    state = harness.context.services["cropped_article_bindings"]
    harness.context.services["asv3_exp_cropped"] = False
    harness.context.services["asv3_exp_cheap"] = variant == "cheap"
    harness.dispatch_repair_reads([_call()])
    assert state.export()["reads"] == []
