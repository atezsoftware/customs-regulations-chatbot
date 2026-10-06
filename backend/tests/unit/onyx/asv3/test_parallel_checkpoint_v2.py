"""Lossless shared child validation under the unchanged physical storage capacity."""

import copy
import gc
import hashlib
import json
import tracemalloc
from contextlib import closing
from typing import Callable, cast
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import Decision, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.asv3.parallel_checkpoint import (
    compact_parallel_checkpoint,
    parallel_checkpoint_digest,
    restore_parallel_checkpoint,
    share_parallel_snapshot,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workers import WorkerPool
from onyx.db.asv3_runs import decode_asv3_checkpoint, encode_asv3_checkpoint
from tests.unit.onyx.asv3.test_parallel_answers import original


def digest(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def large_child() -> tuple[RunContext, dict[str, JsonValue]]:
    context = RunContext(
        run_id="run",
        scope={"document_sets": [15]},
        services={"research_profile": "experimental", "experimental_parallel": True},
    )
    ledger = EvidenceLedger()
    for number in range(60):
        item = original(f"Original {number}, İĞŞçöü — 東京. " * 5)
        item.chunk_id = str(uuid4())
        assert item.search_doc is not None
        item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
        ledger.add([item], context)
    records: list[dict[str, JsonValue]] = []
    for number in range(1, 61):
        item = ledger.get(number)
        assert item is not None
        records.append({"citation": number, "text": item.text})
    for number in range(430):
        call = f"actual-call-{number}"
        ledger.record_delivery(call, "asv3_researcher", records)
        ledger.pin_delivery(call)
    answer = "Exact useful answer with all source detail [1].\n" * 100
    store = ParallelAnswerReceipts(context, "The full question", user_id="owner")
    assignment: dict[str, JsonValue] = {
        "question_id": "q0",
        "question": "The independent issue",
        "outcome_ids": ["outcome0"],
    }

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        harness = Harness(
            request=task,
            context=child,
            evidence=ledger,
            registry=CapabilityRegistry(),
            decide=lambda _: Decision(answer=answer),
        )
        harness.last_draft = answer
        child.services["last_model_call_id"] = "actual-call-429"
        store.seal(
            child,
            assignment=assignment,
            answer=answer,
            status=OutcomeStatus.FOUND,
            model_call_id="actual-call-429",
            ledger=ledger,
            validate_body=lambda: None,
        )
        callback = cast(
            Callable[[dict[str, JsonValue]], None],
            child.services["record_child_checkpoint"],
        )
        callback(harness.snapshot())
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=answer)

    with closing(WorkerPool(context, runner)) as workers:
        task = workers.spawn(
            "The independent issue",
            independent_question=True,
            assignment_id="q0",
            outcome_ids=["outcome0"],
        )
        workers.wait_until_all([task])
        exported = workers.export()
    snapshot: dict[str, JsonValue] = {
        "version": 1,
        "sequence": 1,
        "run_id": "run",
        "request": "The full question",
        "research_profile": "experimental",
        "parallel_research": True,
        "scope": context.scope,
        "evidence": ledger.export(),
        "workers": exported,
        "last_draft": answer,
        "parallel_answers": store.export(),
    }
    return context, snapshot


def test_logical_child_above_8mb_roundtrips_through_real_worker_without_amplification() -> (
    None
):
    context, snapshot = large_child()
    raw_tasks = cast(
        list[dict[str, JsonValue]],
        cast(dict[str, JsonValue], snapshot["workers"])["tasks"],
    )
    original_wrapper = cast(dict[str, JsonValue], raw_tasks[0]["child_checkpoint"])
    original_child = cast(dict[str, JsonValue], original_wrapper["snapshot"])
    assert len(json.dumps(original_child, ensure_ascii=False).encode()) > 8_000_000
    compact = compact_parallel_checkpoint(snapshot)
    assert len(json.dumps(compact, ensure_ascii=False).encode()) < 8_000_000
    restored = decode_asv3_checkpoint(encode_asv3_checkpoint(snapshot))
    assert restored == snapshot
    with closing(
        WorkerPool(
            context, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as workers:
        gc.collect()
        tracemalloc.start()
        workers.restore(cast(dict[str, JsonValue], restored["workers"]))
        child = workers.checkpoint(cast(str, raw_tasks[0]["task_id"]))
        exported = workers.export()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < 12_000_000
        assert child == original_child
        assert exported == snapshot["workers"]
        assert child is not None
        deliveries = cast(
            list[dict[str, JsonValue]],
            cast(dict[str, JsonValue], child["evidence"])["deliveries"],
        )
        assert deliveries[0]["records"] is deliveries[1]["records"]
        assert original_wrapper["integrity"] == digest(
            {
                key: value
                for key, value in original_wrapper.items()
                if key != "integrity"
            }
        )
        harness = Harness(
            request="The independent issue",
            context=context,
            registry=CapabilityRegistry(),
            decide=lambda _: Decision(answer="Done"),
        )
        harness.restore(child)
        assert harness.last_draft == original_child["last_draft"]
        assert harness.evidence.completely_delivered("actual-call-429") == set(
            range(1, 61)
        )
        store = ParallelAnswerReceipts(context, "The full question", user_id="owner")
        store.restore(
            cast(dict[str, JsonValue], restored["parallel_answers"]),
            context,
            "The full question",
            harness.evidence,
        )
        sealed = cast(
            list[dict[str, JsonValue]],
            cast(dict[str, JsonValue], snapshot["parallel_answers"])["receipts"],
        )[0]
        store.verify(
            context,
            receipt_id=cast(str, sealed["receipt_id"]),
            task_id=cast(str, raw_tasks[0]["task_id"]),
            assignment={
                "question_id": "q0",
                "question": "The independent issue",
                "outcome_ids": ["outcome0"],
            },
            answer=cast(str, snapshot["last_draft"]),
            status=OutcomeStatus.FOUND,
            ledger=harness.evidence,
            validate_body=lambda: None,
        )


def test_plain_json_worker_restore_also_preserves_exported_readset_aliases() -> None:
    context, snapshot = large_child()
    plain = json.loads(json.dumps(snapshot["workers"], ensure_ascii=False))
    with closing(
        WorkerPool(
            context, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as workers:
        workers.restore(plain)
        exported = workers.export()
        tasks = cast(list[dict[str, JsonValue]], exported["tasks"])
        child = cast(
            dict[str, JsonValue],
            cast(dict[str, JsonValue], tasks[0]["child_checkpoint"])["snapshot"],
        )
        deliveries = cast(
            list[dict[str, JsonValue]],
            cast(dict[str, JsonValue], child["evidence"])["deliveries"],
        )
        assert deliveries[0]["records"] is deliveries[1]["records"]
        assert exported == plain
        plain["tasks"][0]["child_checkpoint"]["outcome_ids"].append("foreign-outcome")
        assert workers.export() == snapshot["workers"]
        assert workers.checkpoint(cast(str, tasks[0]["task_id"])) is not None


def test_old_storage_v1_keeps_exact_values_and_existing_amplification_guard() -> None:
    context = RunContext(run_id="run")
    ledger = EvidenceLedger()
    item = original("Exact original.")
    ledger.add([item], context)
    ledger.record_delivery(
        "actual", "asv3_researcher", [{"citation": 1, "text": item.text}]
    )
    snapshot: dict[str, JsonValue] = {
        "research_profile": "experimental",
        "parallel_research": True,
        "evidence": ledger.export(),
    }
    evidence = cast(dict[str, JsonValue], snapshot["evidence"])
    record = cast(list[dict[str, JsonValue]], evidence["records"])[0]
    delivery = cast(list[dict[str, JsonValue]], evidence["deliveries"])[0]
    row = cast(list[dict[str, JsonValue]], delivery["records"])[0]
    compact = {
        **snapshot,
        "evidence": {
            **evidence,
            "records": [digest(record)],
            "deliveries": [{**delivery, "records": [digest(row)]}],
        },
        "parallel_checkpoint_storage": {
            "version": 1,
            "records": {digest(record): record},
            "delivery_rows": {digest(row): row},
        },
    }
    assert restore_parallel_checkpoint(compact) == snapshot
    deliveries = cast(
        list[dict[str, JsonValue]],
        cast(dict[str, JsonValue], compact["evidence"])["deliveries"],
    )
    deliveries[0]["records"] = cast(list[JsonValue], deliveries[0]["records"]) * 30000
    with pytest.raises(ValueError, match="storage capacity"):
        restore_parallel_checkpoint(compact)


def test_existing_shared_value_cannot_bypass_a_smaller_requested_capacity() -> None:
    snapshot: dict[str, JsonValue] = {
        "evidence": {"version": 1, "records": []},
        "native": "x" * 1000,
    }
    shared = share_parallel_snapshot(snapshot)
    with pytest.raises(ValueError, match="storage capacity"):
        share_parallel_snapshot(shared, max_unit_bytes=100)


@pytest.mark.parametrize("bad", [(1, 2), {"set"}, object(), {1: "not a JSON key"}])
def test_whole_snapshot_json_is_validated_before_immutable_attachment(
    bad: object,
) -> None:
    snapshot = cast(
        dict[str, JsonValue], {"evidence": {"version": 1, "records": []}, "native": bad}
    )
    with pytest.raises(ValueError):
        share_parallel_snapshot(snapshot)


def test_snapshot_cycles_and_unknown_nested_reference_types_fail_closed() -> None:
    snapshot: dict[str, JsonValue] = {"evidence": {"version": 1, "records": []}}
    snapshot["self"] = snapshot
    with pytest.raises(ValueError):
        share_parallel_snapshot(snapshot)


def test_nested_workers_are_pooled_and_exactly_restored() -> None:
    ledger: dict[str, JsonValue] = {"version": 1, "records": [], "deliveries": []}
    grandchild: dict[str, JsonValue] = {
        "evidence": ledger,
        "native": "exact grandchild",
    }
    child: dict[str, JsonValue] = {
        "evidence": ledger,
        "workers": {"tasks": [{"child_checkpoint": {"snapshot": grandchild}}]},
    }
    snapshot: dict[str, JsonValue] = {
        "research_profile": "experimental",
        "parallel_research": True,
        "evidence": ledger,
        "workers": {"tasks": [{"child_checkpoint": {"snapshot": child}}]},
    }
    assert (
        restore_parallel_checkpoint(compact_parallel_checkpoint(snapshot)) == snapshot
    )


@pytest.mark.parametrize(
    "value",
    [
        {"ü": [None, True, 3, -0.0, 1.5, "東京\n  "]},
        {"a": ["same", "same"], "z": {"c": 5}},
        {"nan": float("nan"), "infinity": float("inf")},
    ],
)
def test_streamed_digest_is_byte_identical_to_old_json_sha(
    value: dict[str, JsonValue],
) -> None:
    assert parallel_checkpoint_digest(value) == digest(value)


def test_v2_preserves_legitimate_duplicate_order_and_rejects_wrong_typed_edges() -> (
    None
):
    context = RunContext(run_id="run")
    ledger = EvidenceLedger()
    item = original("Full original")
    ledger.add([item], context)
    ledger.record_delivery(
        "actual", "asv3_researcher", [{"citation": 1, "text": item.text}] * 2
    )
    snapshot: dict[str, JsonValue] = {
        "research_profile": "experimental",
        "parallel_research": True,
        "evidence": ledger.export(),
    }
    compact = compact_parallel_checkpoint(snapshot)
    assert restore_parallel_checkpoint(compact) == snapshot
    changed = copy.deepcopy(compact)
    storage = cast(dict[str, JsonValue], changed["parallel_checkpoint_storage"])
    readsets = cast(dict[str, JsonValue], storage["readsets"])
    old = next(iter(readsets))
    readsets[old] = [next(iter(cast(dict[str, JsonValue], storage["records"])))]
    with pytest.raises(ValueError):
        restore_parallel_checkpoint(changed)
