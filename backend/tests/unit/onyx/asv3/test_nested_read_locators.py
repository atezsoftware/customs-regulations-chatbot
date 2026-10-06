"""Compound read navigation remains owned and never substitutes for delivery."""

import copy
import json
from typing import cast
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, RunContext, ToolReceipt
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.sandbox import compose
from onyx.asv3.working_memory import WorkingMemory
from onyx.db.asv3_corpus import CorpusChunk
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker

SOURCE = "3491f232-1f96-43e5-a130-ceb576e76bfa"


def acquired(
    *, tool: str = "read_provision", arguments: dict[str, JsonValue] | None = None
) -> tuple[RunContext, ToolReceipt]:
    rows = [
        CorpusChunk(
            "first",
            UUID(SOURCE),
            "MADDE 71 — The exact operative condition.",
            12,
            0,
            ("Instrument", "MADDE 71"),
            {},
            None,
            None,
            "active",
        ),
        CorpusChunk(
            "continuation",
            UUID(SOURCE),
            "The actual exception and later step.",
            13,
            1,
            ("Instrument", "MADDE 71"),
            {},
            None,
            None,
            "active",
        ),
    ]
    context = RunContext(
        run_id="owned-run",
        scope={"tenant": "A", "pc_set": 15},
        services={
            "research_profile": "experimental",
            "experimental_parallel": True,
            "task_id": "owned-child",
        },
    )
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    registry = CapabilityRegistry(build_corpus_specs(MemoryBroker(rows)))
    context.services["registry"] = registry
    call = CapabilityCall(
        name="compose_tool_calls",
        call_id="actual-outer-call",
        arguments={
            "steps": [
                {
                    "id": "read",
                    "tool": tool,
                    "arguments": arguments
                    or {
                        "source_id": SOURCE,
                        "article": "71",
                        "paragraph": "2",
                        "start": 12,
                    },
                }
            ]
        },
    )
    outcome = compose(call.arguments, context)
    return context, ToolReceipt(
        call=call,
        outcome=outcome,
        elapsed_seconds=0,
        evidence_ids=ledger.add(outcome.evidence, context),
    )


def cursors(memory: WorkingMemory) -> list[dict[str, JsonValue]]:
    return [
        row
        for row in cast(list[dict[str, JsonValue]], memory.export()["locators"])
        if isinstance(details := row.get("context"), dict)
        and details.get("kind") == "compound_read_navigation"
    ]


def test_actual_compose_retains_partial_subunit_and_continuation_without_new_proof() -> (
    None
):
    context, receipt = acquired()
    ledger = cast(EvidenceLedger, context.services["evidence"])
    originals = ledger.serialize_records(receipt.evidence_ids)
    memory = WorkingMemory(context.scope)
    memory.observe(receipt, context=context)
    row = cursors(memory)[0]
    assert row["source_id"] == SOURCE and row["article"] == "71"
    assert row["paragraph"] == "2" and row["subunit_verified"] is False
    assert row["scan_truncated"] is False and row["evidence_truncated"] is False
    assert row["next_position"] == 14 and row["evidence_next_position"] is None
    assert row["status"] == "partial" and row["absence_proven"] is False
    assert "article_closure_complete" not in row
    assert row["tool"] == "read_provision" and row["receipt_id"] == receipt.call.call_id
    details = cast(dict[str, JsonValue], row["context"])
    steps = cast(list[dict[str, JsonValue]], receipt.call.arguments["steps"])
    assert details["arguments"] == steps[0]["arguments"]
    assert "not current model delivery" in str(details["notice"])
    assert ledger.serialize_records(receipt.evidence_ids) == originals
    assert ledger.completely_delivered(receipt.call.call_id) == set()


def test_actual_sibling_page_preserves_next_offset_and_false_article_closure() -> None:
    context, receipt = acquired(
        tool="read_chunk_context",
        arguments={
            "source_id": SOURCE,
            "chunk_id": "first",
            "offset": 0,
            "limit": 1,
        },
    )
    memory = WorkingMemory(context.scope)
    memory.observe(receipt, context=context)
    row = cursors(memory)[0]
    assert row["chunk_id"] == "first" and row["has_more"] is True
    assert row["next_offset"] == 1 and row["article_closure_complete"] is False
    assert row["status"] == "partial"


def test_dependent_source_reference_only_exports_whitelisted_actual_read_arguments() -> (
    None
):
    context, receipt = acquired()
    steps = cast(list[dict[str, JsonValue]], receipt.call.arguments["steps"])
    args = cast(dict[str, JsonValue], steps[0]["arguments"])
    args["source_id"] = {"$ref": "identity.data.sources.0.source_id"}
    args["private_payload"] = "PRIVATE"
    steps.insert(
        0,
        {"id": "identity", "tool": "resolve_source", "arguments": {"query": "PRIVATE"}},
    )
    results = cast(dict[str, JsonValue], receipt.outcome.data["steps"])
    results["identity"] = {
        "status": "found",
        "data": {
            "sources": [{"source_id": SOURCE}],
            "private_payload": "PRIVATE",
        },
    }
    memory = WorkingMemory(context.scope)
    memory.observe(receipt, context=context)
    assert cursors(memory)[0]["source_id"] == SOURCE
    assert "PRIVATE" not in json.dumps(cursors(memory))


def test_exact_range_arguments_remain_distinct_and_restore_retains_owner() -> None:
    context, receipt = acquired(
        tool="read_source_range",
        arguments={
            "source_id": SOURCE,
            "start": 12,
            "limit": 1,
        },
    )
    memory = WorkingMemory(context.scope)
    memory.observe(receipt, context=context)
    other = receipt.model_copy(deep=True)
    steps = cast(list[dict[str, JsonValue]], other.call.arguments["steps"])
    cast(dict[str, JsonValue], steps[0]["arguments"])["limit"] = 2
    other.outcome = compose(other.call.arguments, context)
    other.evidence_ids = cast(EvidenceLedger, context.services["evidence"]).add(
        other.outcome.evidence, context
    )
    memory.observe(other, context=context)
    assert len(cursors(memory)) == 2
    restored = WorkingMemory(context.scope)
    restored.restore(memory.export())
    assert restored.export() == memory.export()
    # Cleared receipt evidence during checkpoint restoration is still navigation only.
    cleared = receipt.model_copy(deep=True)
    cleared.outcome.evidence = []
    restored.observe(cleared, context=context)
    assert len(cursors(restored)) == 2
    context.services["task_id"] = "foreign-child"
    with pytest.raises(ValueError, match="owner changed"):
        restored.observe(receipt, context=context)


@pytest.mark.parametrize(
    "flags,depth",
    [
        ({}, 0),
        ({"research_profile": "experimental", "experimental_parallel": False}, 0),
        ({"research_profile": "experimental", "experimental_parallel": "true"}, 0),
        (
            {
                "research_profile": "experimental",
                "experimental_parallel": False,
                "serial_session_diagnostics": True,
                "lean_native_mode": True,
            },
            1,
        ),
    ],
)
def test_protected_profiles_keep_exact_legacy_export(
    flags: dict[str, JsonValue], depth: int
) -> None:
    context, receipt = acquired()
    context.services = {
        "evidence": context.services["evidence"],
        "task_id": "owned-child",
        **flags,
    }
    context.depth = depth
    legacy, scoped = WorkingMemory(context.scope), WorkingMemory(context.scope)
    legacy.observe(receipt)
    scoped.observe(receipt, context=context)
    assert scoped.export() == legacy.export() and not cursors(scoped)


def test_typed_hosted_serial_session_can_retain_navigation_with_parallel_false() -> (
    None
):
    context, receipt = acquired()
    context.services.update(
        experimental_parallel=False,
        serial_session_diagnostics=True,
        lean_native_mode=True,
    )
    memory = WorkingMemory(context.scope)
    memory.observe(receipt, context=context)
    assert cursors(memory) and context.services["experimental_parallel"] is False


def test_actual_exact_chunk_read_retains_its_canonical_anchor() -> None:
    context, receipt = acquired(
        tool="read_chunk", arguments={"source_id": SOURCE, "chunk_id": "first"}
    )
    memory = WorkingMemory(context.scope)
    memory.observe(receipt, context=context)
    assert cursors(memory)[0]["chunk_id"] == "first"
    assert cursors(memory)[0]["status"] == "found"


@pytest.mark.parametrize(
    "alteration",
    [
        "foreign_source",
        "wrong_article",
        "not_found",
        "forged_closure",
        "forged_absence",
        "negative_position",
        "string_position",
        "string_boolean",
        "missing_ids",
        "extra_step",
        "duplicate_step",
        "unknown_tool",
        "private_argument",
    ],
)
def test_unbound_or_malformed_nested_results_never_create_compound_locator(
    alteration: str,
) -> None:
    context, receipt = acquired()
    steps = cast(list[dict[str, JsonValue]], receipt.call.arguments["steps"])
    args = cast(dict[str, JsonValue], steps[0]["arguments"])
    results = cast(dict[str, JsonValue], receipt.outcome.data["steps"])
    result = cast(dict[str, JsonValue], results["read"])
    data = cast(dict[str, JsonValue], result["data"])
    if alteration == "foreign_source":
        args["source_id"] = "da0b4eb9-4b8b-4177-93f5-592e04e537b6"
    elif alteration == "wrong_article":
        data["article"] = "72"
    elif alteration == "not_found":
        result["status"] = "not_found"
    elif alteration == "forged_closure":
        data["article_closure_complete"] = True
    elif alteration == "forged_absence":
        data["absence_proven"] = True
    elif alteration == "negative_position":
        data["next_position"] = -1
    elif alteration == "string_position":
        data["next_position"] = "14"
    elif alteration == "string_boolean":
        data["scan_truncated"] = "false"
    elif alteration == "missing_ids":
        receipt.evidence_ids = []
    elif alteration == "extra_step":
        results["undeclared"] = copy.deepcopy(result)
    elif alteration == "duplicate_step":
        steps.append(copy.deepcopy(steps[0]))
    elif alteration == "unknown_tool":
        steps[0]["tool"] = "unknown"
    else:
        args["paragraph"] = {"private_payload": "PRIVATE"}
    memory = WorkingMemory(context.scope)
    memory.observe(receipt, context=context)
    assert not cursors(memory)


def test_captured_scope_change_fails_closed() -> None:
    context, receipt = acquired()
    memory = WorkingMemory(context.scope)
    context.scope["tenant"] = "B"
    with pytest.raises(ValueError, match="scope changed"):
        memory.observe(receipt, context=context)
