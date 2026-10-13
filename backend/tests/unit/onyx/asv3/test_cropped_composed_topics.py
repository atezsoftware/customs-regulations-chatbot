"""Only executed and committed static compositions can partition topic originals."""

import copy
import json
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_task_packets import (
    CroppedTaskPacketStore,
    packet_for_view,
    restore_cropped_topic_context,
)
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchUpdate, build_research_specs
from onyx.asv3.sandbox import compose
from tests.unit.onyx.asv3.test_cropped_task_packets import (
    REQUEST,
    finding,
    fixture,
    need,
    topics,
    view,
)
from tests.unit.onyx.asv3.test_cropped_task_packets import (
    receipt as source_receipt,
)


def setup(
    *,
    shared: bool = False,
    unbound: bool = False,
    pending: str | None = None,
    mixed_pending: bool = False,
    pending_shared: bool = False,
) -> tuple[Any, ...]:
    context, ledger, state, store = fixture(3 if mixed_pending else 2)
    topics(state, ledger)

    def read(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        number = int(str(arguments["source_id"]).split("-")[-1])
        selected_pending = (
            pending if not mixed_pending or arguments.get("pending") is True else None
        )
        item = ledger.get(number)
        assert item is not None
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL
            if selected_pending == "partial"
            else OutcomeStatus.FOUND,
            summary="Actual original",
            evidence=[item.model_copy(deep=True)],
            data={
                "source_id": item.source_id,
                "has_more": False,
                "continuation_note": f"Complete topic {number} conditions and exceptions",
                **(
                    {selected_pending: True}
                    if selected_pending and selected_pending != "partial"
                    else {}
                ),
                **(
                    {"next_position": 11, "has_more": True}
                    if selected_pending == "partial"
                    else {}
                ),
            },
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_provision",
                description="Read actual fixture original",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "pending": {"type": "boolean"},
                    },
                    "required": ["source_id"],
                    "additionalProperties": False,
                },
                handler=read,
                parallel_safe=True,
            ),
            ToolSpec(
                name="compose_tool_calls",
                description="Execute actual composition",
                parameters={
                    "type": "object",
                    "properties": {"steps": {"type": "array"}},
                    "required": ["steps"],
                    "additionalProperties": False,
                },
                handler=compose,
                parallel_safe=True,
            ),
        ]
    )
    context.services["registry"] = registry
    call = CapabilityCall(
        name="compose_tool_calls",
        arguments={
            "steps": [
                {
                    "id": "a",
                    "tool": "read_provision",
                    "arguments": {"source_id": "source-1", "_need_id": "A"},
                },
                {
                    "id": "b",
                    "tool": "read_provision",
                    "arguments": {
                        "source_id": "source-1" if shared else "source-2",
                        **({} if unbound else {"_need_id": "B"}),
                    },
                },
            ]
        },
    )
    if mixed_pending:
        cast(list[JsonValue], call.arguments["steps"]).append(
            {
                "id": "c",
                "tool": "read_provision",
                "arguments": {
                    "source_id": "source-1" if pending_shared else "source-3",
                    "_need_id": "B",
                    "pending": True,
                },
            }
        )
    harness = Harness(
        request=REQUEST,
        context=context,
        registry=registry,
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    return context, ledger, state, store, registry, call, harness


def execute(values: tuple[Any, ...], *, commit: bool = True) -> ToolReceipt:
    context, _, _, _, registry, call, harness = values
    outcome = registry.dispatch(call, context)
    assert outcome.status in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}
    receipt = ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0)
    if commit:
        harness._commit_receipt(receipt)
    return receipt


def test_actual_dispatch_and_parent_commit_partition_and_restore_exact_journal() -> (
    None
):
    values = setup()
    context, ledger, state, store, _, _, harness = values
    receipt = execute(values)
    raw = receipt.model_dump(mode="json")
    saved = store.export()
    assert len(saved["receipts"]) == 1
    record = saved["receipts"][0]
    assert record["raw"] == raw
    assert record["need_ids"] == ["A", "B"]
    assert [step["citations"] for step in record["composed_steps"]] == [[1], [2]]
    projected = packet_for_view(
        view(ledger, [receipt]), [receipt], state, context, ledger
    )
    assert projected is not None
    local, manifest = projected
    assert local.required_evidence_numbers == [1]
    assert [r["citation"] for r in local.original_evidence] == [1]
    step_data = local.receipts[0].outcome.data
    assert set(cast(dict[str, JsonValue], step_data["steps"])) == {"a"}
    projection = cast(
        dict[str, JsonValue], step_data["cropped_composed_topic_projection"]
    )
    assert projection["parked_step_ids"] == ["b"]
    assert ledger.get(2).text not in json.dumps(local.model_dump(mode="json"))
    assert store.export()["receipts"][0]["raw"] == raw
    restored = restore_cropped_topic_context(local, context)
    assert restored.receipts[0].model_dump(mode="json") == raw
    assert restored.required_evidence_numbers == [1, 2]
    resumed = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    resumed.restore(saved, ledger)
    assert resumed.export() == saved
    assert harness.receipts[0].model_dump(mode="json") == raw
    assert manifest["legal_approval"] is False


def test_pending_dispatch_is_not_checkpointed_or_accepted_without_commit() -> None:
    values = setup()
    context, ledger, state, store, _, _, _ = values
    receipt = execute(values, commit=False)
    assert store.export()["receipts"] == []
    receipt.evidence_ids = [1, 2]
    store.record_receipts("root", [receipt], state, ledger, context=context)
    assert store.export()["receipts"][0]["composed_steps"] == []
    assert (
        packet_for_view(view(ledger, [receipt]), [receipt], state, context, ledger)
        is None
    )


@pytest.mark.parametrize(
    "defect",
    [
        "no_dispatch",
        "changed_parent",
        "changed_text",
        "foreign_owner",
        "foreign_depth",
        "failure",
        "reference",
        "duplicate",
        "dependency",
        "parent_outcome",
    ],
)
def test_uncertain_parent_or_child_proof_never_partitions(defect: str) -> None:
    values = setup()
    context, ledger, state, store, registry, call, harness = values
    if defect == "reference":
        call.arguments["steps"][1]["arguments"]["source_id"] = {
            "$ref": "a.data.source_id"
        }
        call.arguments["steps"][1]["depends_on"] = ["a"]
    if defect == "duplicate":
        call.arguments["steps"][1]["arguments"] = copy.deepcopy(
            call.arguments["steps"][0]["arguments"]
        )
    if defect == "dependency":
        call.arguments["steps"][1]["depends_on"] = ["a"]
    if defect == "parent_outcome":
        call.arguments["_outcomes"] = [{"outcome_id": "o1"}]
    outcome = (
        registry.dispatch(call, context)
        if defect != "no_dispatch"
        else ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Unproven",
            evidence=[ledger.get(1), ledger.get(2)],
            data={"steps": {}},
        )
    )
    receipt = ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0)
    if defect == "changed_parent":
        receipt.outcome.data["changed"] = True
    elif defect == "changed_text":
        receipt.outcome.evidence[0].metadata["publication_snapshot"] = "changed"
        receipt.outcome.evidence[0].text += "changed"
    elif defect == "foreign_owner":
        context.services["task_id"] = "foreign"
    elif defect == "foreign_depth":
        context.depth = 1
    elif defect == "failure":
        receipt.outcome.status = OutcomeStatus.ERROR
    try:
        harness._commit_receipt(receipt)
        store.record_receipts(
            str(context.services.get("task_id") or "root"),
            [receipt],
            state,
            ledger,
            context=context,
        )
    except ValueError:
        pass
    assert all(not row["composed_steps"] for row in store.export()["receipts"])


@pytest.mark.parametrize("shared,unbound", [(True, False), (False, True), (True, True)])
def test_shared_or_unbound_step_keeps_its_original_protected(
    shared: bool, unbound: bool
) -> None:
    values = setup(shared=shared, unbound=unbound)
    context, ledger, state, store, *_ = values
    receipt = execute(values)
    projected = packet_for_view(
        view(ledger, [receipt]), [receipt], state, context, ledger
    )
    assert projected is None
    assert len(store.export()["receipts"]) == 1


def test_all_active_topics_still_receive_every_original_without_savings_claim() -> None:
    values = setup()
    context, ledger, state, _, *_ = values
    receipt = execute(values)
    state.update(ResearchUpdate(active_need_ids=["A", "B"]), ledger)
    assert (
        packet_for_view(view(ledger, [receipt]), [receipt], state, context, ledger)
        is None
    )


@pytest.mark.parametrize(
    "defect",
    [
        "citation",
        "parent_output",
        "need",
        "parent_definition",
        "source_hash",
        "source_swap",
        "captured_original",
        "captured_call",
        "source_swap_rehashed",
    ],
)
def test_checkpoint_binding_mutations_fail_closed(defect: str) -> None:
    from onyx.asv3.cropped_task_packets import _digest

    values = setup()
    context, ledger, _, store, *_ = values
    execute(values)
    saved = copy.deepcopy(store.export())
    row = saved["receipts"][0]
    if defect == "citation":
        row["composed_steps"][0]["citations"] = [2]
    elif defect == "parent_output":
        row["raw"]["outcome"]["data"]["steps"]["a"]["summary"] = "Changed"
    elif defect == "need":
        row["composed_steps"][0]["need_ids"] = ["B"]
    elif defect == "parent_definition":
        row["raw"]["call"]["arguments"]["steps"][0]["arguments"]["_need_id"] = "B"
    elif defect in {"source_swap", "source_swap_rehashed"}:
        a, b = row["composed_steps"]
        a["citations"], b["citations"] = b["citations"], a["citations"]
        a["sources"], b["sources"] = b["sources"], a["sources"]
        if defect == "source_swap_rehashed":
            from onyx.asv3.cropped_task_packets import _capture_hash, _ComposedStep

            for step in (a, b):
                step["capture_hash"] = _capture_hash(
                    _ComposedStep.model_validate(step), row["owner"], row["call_id"]
                )
    elif defect == "captured_original":
        row["composed_steps"][0]["captured_originals"] = row["composed_steps"][1][
            "captured_originals"
        ]
    elif defect == "captured_call":
        row["composed_steps"][0]["captured_call"]["arguments"]["source_id"] = "source-2"
    else:
        row["composed_steps"][0]["sources"][0]["text_hash"] = "0" * 64
    saved["integrity"] = _digest({k: v for k, v in saved.items() if k != "integrity"})
    resumed = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    with pytest.raises(ValueError):
        resumed.restore(cast(dict[str, JsonValue], saved), ledger)


def test_old_checkpoint_has_no_invented_child_proof() -> None:
    from onyx.asv3.cropped_task_packets import _digest

    values = setup()
    context, ledger, _, store, *_ = values
    execute(values)
    saved = copy.deepcopy(store.export())
    row = saved["receipts"][0]
    row.pop("composed_steps")
    row["need_ids"] = []
    saved["integrity"] = _digest({k: v for k, v in saved.items() if k != "integrity"})
    resumed = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    resumed.restore(cast(dict[str, JsonValue], saved), ledger)
    restored_rows = cast(list[dict[str, JsonValue]], resumed.export()["receipts"])
    assert restored_rows[0]["composed_steps"] == []


@pytest.mark.parametrize(
    "pending",
    [
        "partial",
        "has_more",
        "not_read",
        "scan_truncated",
        "evidence_truncated",
        "truncated",
    ],
)
def test_pending_child_keeps_complete_parent_and_continuation_visible(
    pending: str,
) -> None:
    values = setup(pending=pending)
    context, ledger, state, store, registry, call, harness = values
    receipt = ToolReceipt(
        call=call, outcome=registry.dispatch(call, context), elapsed_seconds=0
    )
    harness._commit_receipt(receipt)
    store.record_receipts("root", [receipt], state, ledger, context=context)
    assert all(
        not row["partitionable"]
        for row in store.export()["receipts"][0]["composed_steps"]
    )
    assert (
        packet_for_view(view(ledger, [receipt]), [receipt], state, context, ledger)
        is None
    )


@pytest.mark.parametrize(
    "pending",
    ["partial", "has_more", "not_read", "scan_truncated", "evidence_truncated"],
)
def test_open_pending_child_passes_through_without_activating_archived_complete_sibling(
    pending: str,
) -> None:
    values = setup(pending=pending, mixed_pending=True)
    context, ledger, state, store, *_ = values
    state.update(ResearchUpdate(needs=[need("B")]), ledger)
    receipt = execute(values)
    raw = receipt.model_dump(mode="json")
    result = packet_for_view(view(ledger, [receipt]), [receipt], state, context, ledger)
    assert result is not None
    local, manifest = result
    assert manifest["active_need_ids"] == ["A"]
    assert local.required_evidence_numbers == [1, 3]
    assert manifest["pending_passthrough_original_ids"] == [3]
    local_steps = cast(dict[str, Any], local.receipts[0].outcome.data["steps"])
    assert set(local_steps) == {"a", "c"}
    assert local_steps["c"] == raw["outcome"]["data"]["steps"]["c"]
    index = cast(list[dict[str, Any]], manifest["detached_need_index"])[0]
    assert index["delivered_original_ids"] == [3]
    assert index["unread_original_ids"] == [2]
    assert index["pending_composed_operations"][0]["step_id"] == "c"
    assert (
        index["pending_composed_operations"][0]["continuation"]
        == store.export()["receipts"][0]["composed_steps"][2]["continuation"]
    )
    assert ledger.get(2).text not in json.dumps(local.model_dump(mode="json"))
    restored = restore_cropped_topic_context(local, context)
    assert restored.receipts[0].model_dump(mode="json") == raw
    assert restored.required_evidence_numbers == [1, 2, 3]
    saved = store.export()
    resumed = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    resumed.restore(saved, ledger)
    assert resumed.export() == saved


@pytest.mark.parametrize(
    "reason", ["shared_active", "required", "dependency", "candidate"]
)
def test_pending_passthrough_preserves_existing_dependency_and_candidate_protection(
    reason: str,
) -> None:
    values = setup(
        pending="partial", mixed_pending=True, pending_shared=reason == "shared_active"
    )
    context, ledger, state, _, *_ = values
    if reason != "candidate":
        state.update(ResearchUpdate(needs=[need("B")]), ledger)
    if reason == "dependency":
        state.update(ResearchUpdate(needs=[need("A", depends=["B"])]), ledger)
    receipt = execute(values)
    original = view(ledger, [receipt])
    if reason == "required":
        original.required_evidence_numbers = [3]
    assert packet_for_view(original, [receipt], state, context, ledger) is None


@pytest.mark.parametrize(
    "tool,data",
    [
        (
            "read_provision",
            {
                "next_position": 71,
                "scan_truncated": False,
                "evidence_truncated": False,
                "evidence_next_position": None,
            },
        ),
        ("read_source_range", {"next_position": 12, "has_more": False}),
        (
            "read_chunk_context",
            {"next_offset": 8, "has_more": False, "article_closure_complete": False},
        ),
        ("resolve_source", {"next_offset": 20, "has_more": False}),
    ],
)
def test_completed_corpus_bookmark_is_not_a_pending_continuation(
    tool: str, data: dict[str, JsonValue]
) -> None:
    from onyx.asv3.cropped_composed_topics import topic_step_partitionable

    assert topic_step_partitionable("found", data, tool)
    assert not topic_step_partitionable("partial", data, tool)
    assert not topic_step_partitionable("found", {**data, "has_more": True}, tool)


@pytest.mark.parametrize(
    "pending",
    [
        None,
        "partial",
        "scan_truncated",
        "evidence_truncated",
        "has_more",
        "next_cursor",
        "unknown",
    ],
)
def test_follow_reference_terminal_bookmark_keeps_real_pending_work(
    pending: str | None,
) -> None:
    from onyx.asv3.cropped_composed_topics import topic_step_partitionable

    data: dict[str, JsonValue] = {
        "next_position": 1252,
        "scan_truncated": False,
        "evidence_truncated": False,
        "evidence_next_position": None,
    }
    if pending in {"scan_truncated", "evidence_truncated", "has_more"}:
        data[pending] = True
    elif pending == "next_cursor":
        data["next_cursor"] = "continuation"
    elif pending == "unknown":
        data.pop("scan_truncated")
    assert topic_step_partitionable(
        "partial" if pending == "partial" else "found", data, "follow_reference"
    ) is (pending is None)


def setup_retained_reference() -> tuple[tuple[Any, ...], ToolReceipt]:
    values = setup()
    context, ledger, state, store, registry, call, harness = values
    ledger.add(
        [
            EvidenceItem(
                source_id="source-3",
                chunk_id="chunk-3",
                text="Old B witness, already read.",
            )
        ],
        context,
    )
    old = source_receipt(3, "B", ledger)
    harness._commit_receipt(old)
    store.record_receipts("root", [old], state, ledger, context=context)
    for spec in build_research_specs(state, ledger):
        registry.register(spec)
    call.arguments["steps"].insert(
        0,
        {
            "id": "retain",
            "tool": "update_research",
            "arguments": {
                "findings": [finding("B", 3, ledger).model_dump(mode="json")]
            },
        },
    )
    return values, old


@pytest.mark.parametrize("explicit_binding", [False, True])
def test_actual_retained_witness_is_not_new_delivery_or_whole_topic_dependency(
    explicit_binding: bool,
) -> None:
    values, old = setup_retained_reference()
    context, ledger, state, store, *_ = values
    if explicit_binding:
        values[5].arguments["steps"][0]["arguments"]["_need_id"] = "B"
    current = execute(values)
    raw = current.model_dump(mode="json")
    assert set(current.evidence_ids) == {1, 2}
    saved = store.export()
    parent = saved["receipts"][-1]
    reference_step = parent["composed_steps"][0]
    assert reference_step["citations"] == []
    assert reference_step["captured_originals"] == []
    assert reference_step["references"][0]["source"]["citation"] == 3
    assert reference_step["references"][0]["need_id"] == "B"
    projected = packet_for_view(
        view(ledger, [old, current]), [old, current], state, context, ledger
    )
    assert projected is not None
    local, manifest = projected
    assert manifest["active_need_ids"] == ["A"]
    assert manifest["reference_passthrough_original_ids"] == [3]
    assert local.required_evidence_numbers == [1, 3]
    assert local.receipts[0].evidence_ids == [1]
    assert (
        cast(dict[str, JsonValue], local.receipts[0].outcome.data["steps"])["retain"]
        == raw["outcome"]["data"]["steps"]["retain"]
    )
    restored = restore_cropped_topic_context(local, context)
    assert [r.model_dump(mode="json") for r in restored.receipts] == [
        old.model_dump(mode="json"),
        raw,
    ]
    assert restored.required_evidence_numbers == [1, 2, 3]
    resumed = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    resumed.restore(saved, ledger)
    assert resumed.export() == saved


@pytest.mark.parametrize(
    "reason", ["required", "shared_active", "dependency", "all_active"]
)
def test_retained_reference_keeps_existing_material_dependency_closure(
    reason: str,
) -> None:
    values, old = setup_retained_reference()
    context, ledger, state, _, _, call, _ = values
    if reason == "shared_active":
        call.arguments["steps"][1]["arguments"]["source_id"] = "source-3"
    if reason == "dependency":
        state.update(ResearchUpdate(needs=[need("A", depends=["B"])]), ledger)
    if reason == "all_active":
        state.update(ResearchUpdate(active_need_ids=["A", "B"]), ledger)
    current = execute(values)
    original = view(ledger, [old, current])
    if reason == "required":
        original.required_evidence_numbers = [3]
    assert packet_for_view(original, [old, current], state, context, ledger) is None


@pytest.mark.parametrize("mutation", ["need", "source", "range"])
def test_retained_reference_checkpoint_rechecks_actual_result_and_source(
    mutation: str,
) -> None:
    from onyx.asv3.cropped_task_packets import _capture_hash, _ComposedStep, _digest

    values, _ = setup_retained_reference()
    context, ledger, _, store, *_ = values
    execute(values)
    saved = copy.deepcopy(store.export())
    parent = saved["receipts"][-1]
    step = parent["composed_steps"][0]
    reference = step["references"][0]
    if mutation == "need":
        reference["need_id"] = "A"
    elif mutation == "source":
        reference["source"] = parent["composed_steps"][1]["sources"][0]
    else:
        reference["end_char"] = 1
    step["capture_hash"] = _capture_hash(
        _ComposedStep.model_validate(step), parent["owner"], parent["call_id"]
    )
    saved["integrity"] = _digest({k: v for k, v in saved.items() if k != "integrity"})
    with pytest.raises(ValueError):
        CroppedTaskPacketStore(context, REQUEST, policy=store.policy).restore(
            saved, ledger
        )


def test_restored_conservative_pending_classification_is_not_promoted() -> None:
    from onyx.asv3.cropped_task_packets import _capture_hash, _ComposedStep, _digest

    values = setup()
    context, ledger, _, store, *_ = values
    execute(values)
    saved = store.export()
    row = saved["receipts"][0]
    step = row["composed_steps"][0]
    step["partitionable"] = False
    step["capture_hash"] = _capture_hash(
        _ComposedStep.model_validate(step), row["owner"], row["call_id"]
    )
    saved["integrity"] = _digest({k: v for k, v in saved.items() if k != "integrity"})
    restored = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    restored.restore(saved, ledger)
    restored_rows = cast(list[dict[str, Any]], restored.export()["receipts"])
    assert restored_rows[0]["composed_steps"][0]["partitionable"] is False


@pytest.mark.parametrize("candidate", [False, True])
def test_actual_source_free_invalid_sibling_remains_visible_without_discarding_valid_proof(
    candidate: bool,
) -> None:
    values = setup()
    context, ledger, state, store, registry, call, _ = values
    if not candidate:
        state.update(ResearchUpdate(needs=[need("B")]), ledger)
    rejected = ToolOutcome(
        status=OutcomeStatus.INVALID,
        summary="The article is not an explicit reference.",
        data={"has_more": True, "next_position": 17},
    )
    registry.register(
        ToolSpec(
            name="follow_reference",
            description="Actual rejection",
            parameters={"type": "object", "additionalProperties": False},
            handler=lambda _a, _c: rejected,
            parallel_safe=True,
        )
    )
    call.arguments["steps"].append(
        {"id": "invalid", "tool": "follow_reference", "arguments": {"_need_id": "B"}}
    )
    current = execute(values)
    raw = current.model_dump(mode="json")
    saved = store.export()
    assert len(saved["receipts"][0]["composed_steps"]) == 3
    assert saved["receipts"][0]["composed_steps"][2]["partitionable"] is False
    projected = packet_for_view(
        view(ledger, [current]), [current], state, context, ledger
    )
    if candidate:
        assert projected is None
    else:
        assert projected is not None
        local, manifest = projected
        assert local.required_evidence_numbers == [1]
        local_steps = cast(
            dict[str, JsonValue], local.receipts[0].outcome.data["steps"]
        )
        assert set(local_steps) == {"a", "invalid"}
        assert local_steps["invalid"] == raw["outcome"]["data"]["steps"]["invalid"]
        detached = cast(list[dict[str, Any]], manifest["detached_need_index"])
        operation = detached[0]["pending_composed_operations"][0]
        assert operation["status"] == "invalid"
        assert operation["continuation"] == rejected.data
        assert (
            restore_cropped_topic_context(local, context)
            .receipts[0]
            .model_dump(mode="json")
            == raw
        )
    resumed = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    resumed.restore(saved, ledger)
    assert resumed.export() == saved


@pytest.mark.parametrize(
    "defect",
    [
        "source",
        "read",
        "reference",
        "unknown_data",
        "error",
        "cancelled",
        "denied",
        "unavailable",
    ],
)
def test_sourceful_or_untrusted_failure_still_protects_entire_parent(
    defect: str,
) -> None:
    values = setup()
    context, ledger, state, store, registry, call, _ = values
    rejected = ToolOutcome(
        status=OutcomeStatus.INVALID, summary="Failure remains visible"
    )
    if defect == "source":
        rejected.evidence = [ledger.get(1)]
    elif defect == "read":
        rejected.original_reads = source_receipt(1, "A", ledger).outcome.original_reads
    elif defect == "reference":
        rejected.data = {"citation": 1}
    elif defect == "unknown_data":
        rejected.data = {"payload": {"locator": "source-1"}}
    else:
        rejected.status = OutcomeStatus(defect)
    registry.register(
        ToolSpec(
            name="follow_reference",
            description="Actual failure",
            parameters={"type": "object", "additionalProperties": False},
            handler=lambda _a, _c: rejected,
            parallel_safe=True,
        )
    )
    call.arguments["steps"].append(
        {"id": "invalid", "tool": "follow_reference", "arguments": {"_need_id": "B"}}
    )
    current = execute(values)
    store.record_receipts("root", [current], state, ledger, context=context)
    assert store.export()["receipts"][0]["composed_steps"] == []
