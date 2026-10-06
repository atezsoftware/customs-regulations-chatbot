"""Exercise durable parallel snapshots through real child checkpoint bindings."""

import base64
import copy
import hashlib
import json
import zlib
from contextlib import closing
from typing import Callable, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import Decision, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.asv3.parallel_checkpoint import (
    compact_parallel_checkpoint,
    restore_parallel_checkpoint,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workers import WorkerPool
from onyx.db.asv3_runs import decode_asv3_checkpoint, encode_asv3_checkpoint
from tests.unit.onyx.asv3.test_parallel_answers import original

STORAGE = "parallel_checkpoint_storage"
ANSWER = "A complete source-supported result, İĞŞçöü — 東京 [1].\n" * 450


def wrapped_snapshot(*, large: bool = False) -> tuple[RunContext, dict[str, JsonValue]]:
    root = RunContext(run_id="run", scope={"document_sets": [15]})
    ledger = EvidenceLedger()
    item = original("Canonical original. " * (15000 if large else 2))
    if large:
        item.metadata["additional_context"] = "Immutable provenance. " * 110000
    ledger.add([item], root)
    ledger.record_delivery(
        "actual-accepted-call", "asv3_researcher", [{"citation": 1, "text": item.text}]
    )
    ledger.pin_delivery("actual-accepted-call")

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        harness = Harness(
            request=task,
            context=child,
            registry=CapabilityRegistry(),
            decide=lambda _: Decision(answer=ANSWER),
            evidence=ledger,
        )
        harness.last_draft = ANSWER
        snapshot = harness.snapshot()
        snapshot["turns"] = [
            {"assistant": {"content": task + " exact native body"}, "results": []}
        ]
        callback = cast(
            Callable[[dict[str, JsonValue]], None],
            child.services["record_child_checkpoint"],
        )
        callback(snapshot)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=ANSWER)

    with closing(WorkerPool(root, runner)) as pool:
        for number in range(3):
            task = pool.spawn(
                f"Assigned issue {number}",
                independent_question=True,
                assignment_id=f"q{number}",
                outcome_ids=[f"outcome{number}"],
            )
            pool.wait_until_all([task])
        workers = pool.export()
    snapshot: dict[str, JsonValue] = {
        "version": 1,
        "run_id": root.run_id,
        "sequence": 20,
        "research_profile": "experimental",
        "parallel_research": True,
        "request": "The complete request",
        "scope": root.scope,
        "evidence": ledger.export(),
        "workers": workers,
        "last_draft": ANSWER,
    }
    return root, snapshot


def test_actual_independent_snapshots_above_aggregate_capacity_roundtrip_losslessly() -> (
    None
):
    context, snapshot = wrapped_snapshot(large=True)
    original_json = json.dumps(snapshot, ensure_ascii=False)
    assert len(original_json.encode()) > 8_000_000
    compact = compact_parallel_checkpoint(snapshot)
    assert len(json.dumps(compact, ensure_ascii=False).encode()) < 8_000_000
    assert json.dumps(snapshot, ensure_ascii=False) == original_json
    encoded = encode_asv3_checkpoint(snapshot)
    restored = decode_asv3_checkpoint(encoded)
    assert restored == snapshot
    assert json.dumps(restored, ensure_ascii=False) == original_json
    root_records = cast(dict[str, JsonValue], restored["evidence"])["records"]
    tasks = cast(
        list[dict[str, JsonValue]],
        cast(dict[str, JsonValue], restored["workers"])["tasks"],
    )
    child = cast(dict[str, JsonValue], tasks[0]["child_checkpoint"])
    child_snapshot = cast(dict[str, JsonValue], child["snapshot"])
    assert (
        cast(list[JsonValue], root_records)[0]
        is cast(
            list[JsonValue],
            cast(dict[str, JsonValue], child_snapshot["evidence"])["records"],
        )[0]
    )
    ledger = EvidenceLedger()
    ledger.restore(cast(dict[str, JsonValue], restored["evidence"]), context)
    assert ledger.get(1) is not None
    assert ledger.completely_delivered("actual-accepted-call") == {1}
    with closing(
        WorkerPool(
            context, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as workers:
        workers.restore(cast(dict[str, JsonValue], restored["workers"]))
        assert workers.checkpoint(cast(str, tasks[0]["task_id"])) == child_snapshot
        assert all(
            task.outcome is not None and task.outcome.summary == ANSWER
            for task in workers.results()
        )
    assert restored["last_draft"] == ANSWER


def test_historical_metadata_ranges_and_pins_do_not_merge_with_new_root_values() -> (
    None
):
    _, snapshot = wrapped_snapshot()
    before = copy.deepcopy(snapshot)
    ledger = cast(dict[str, JsonValue], snapshot["evidence"])
    records = cast(list[dict[str, JsonValue]], ledger["records"])
    item = cast(dict[str, JsonValue], records[0]["item"])
    cast(dict[str, JsonValue], item["metadata"])["publication_revision"] = 5
    item["question_ids"] = ["new-root-question"]
    text = cast(str, item["text"])
    deliveries = cast(list[dict[str, JsonValue]], ledger["deliveries"])
    partial = {
        **cast(list[dict[str, JsonValue]], deliveries[0]["records"])[0],
        "start_char": 2,
        "end_char": 8,
        "complete": False,
        "passage_hash": hashlib.sha256(text[2:8].encode()).hexdigest(),
    }
    deliveries.append(
        {"call_id": "new-call", "flow": "asv3_coordinator", "records": [partial]}
    )
    cast(list[JsonValue], ledger["pinned_delivery_calls"]).append("new-call")
    restored = restore_parallel_checkpoint(compact_parallel_checkpoint(snapshot))
    assert restored == snapshot
    assert restored["workers"] == before["workers"]
    assert restored["evidence"] != before["evidence"]


def test_accepted_body_originals_and_actual_delivery_survive_codec() -> None:
    from tests.unit.onyx.asv3.test_parallel_answers import fixture, seal, verify

    root, child, ledger, receipts = fixture()
    receipt_id = seal(receipts, child, ledger, answer=ANSWER)
    snapshot: dict[str, JsonValue] = {
        "version": 1,
        "run_id": root.run_id,
        "sequence": 2,
        "research_profile": "experimental",
        "parallel_research": True,
        "evidence": ledger.export(),
        "parallel_answers": receipts.export(),
        "last_draft": ANSWER,
    }
    restored = decode_asv3_checkpoint(encode_asv3_checkpoint(snapshot))
    restored_ledger = EvidenceLedger()
    restored_ledger.restore(cast(dict[str, JsonValue], restored["evidence"]), root)
    restored_receipts = ParallelAnswerReceipts(
        root,
        "Explain permission and its conditions for the supplied transaction.",
        user_id="owner",
    )
    restored_receipts.restore(
        cast(dict[str, JsonValue], restored["parallel_answers"]),
        root,
        "Explain permission and its conditions for the supplied transaction.",
        restored_ledger,
    )
    verify(restored_receipts, root, restored_ledger, receipt_id, answer=ANSWER)
    assert restored == snapshot


@pytest.mark.parametrize(
    "change",
    ["missing", "value", "unknown_ref", "duplicate", "format", "profile", "nested_row"],
)
def test_compact_checkpoint_rejects_modified_pool_and_refs(change: str) -> None:
    _, snapshot = wrapped_snapshot()
    compact = compact_parallel_checkpoint(snapshot)
    storage = cast(dict[str, JsonValue], compact[STORAGE])
    pool = cast(dict[str, JsonValue], storage["records"])
    identifier = next(iter(pool))
    ledger = cast(dict[str, JsonValue], compact["evidence"])
    if change == "missing":
        del pool[identifier]
    elif change == "value":
        cast(dict[str, JsonValue], pool[identifier])["citation"] = 2
    elif change == "unknown_ref":
        cast(list[JsonValue], ledger["records"])[0] = "0" * 64
    elif change == "duplicate":
        cast(list[JsonValue], ledger["records"]).append(identifier)
    elif change == "format":
        storage["version"] = 2
    elif change == "profile":
        compact["parallel_research"] = False
    else:
        rows = cast(dict[str, JsonValue], storage["delivery_rows"])
        old = next(iter(rows))
        row = cast(dict[str, JsonValue], rows.pop(old))
        row["complete"] = {"reference": old}
        key = hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        rows[key] = row
    with pytest.raises(ValueError):
        restore_parallel_checkpoint(compact)


@pytest.mark.parametrize("target", ["root", "child"])
def test_reference_amplification_fails_capacity_check_before_model_validation(
    target: str,
) -> None:
    _, snapshot = wrapped_snapshot()
    compact = compact_parallel_checkpoint(snapshot)
    if target == "child":
        tasks = cast(
            list[dict[str, JsonValue]],
            cast(dict[str, JsonValue], compact["workers"])["tasks"],
        )
        wrapper = cast(dict[str, JsonValue], tasks[0]["child_checkpoint"])
        compact_unit = cast(dict[str, JsonValue], wrapper["snapshot"])
    else:
        compact_unit = compact
    ledger = cast(dict[str, JsonValue], compact_unit["evidence"])
    deliveries = cast(list[dict[str, JsonValue]], ledger["deliveries"])
    reference = cast(list[JsonValue], deliveries[0]["records"])[0]
    deliveries[0]["records"] = [reference] * 30000
    assert len(json.dumps(compact).encode()) < 8_000_000
    with pytest.raises(ValueError, match="storage capacity"):
        restore_parallel_checkpoint(compact)


@pytest.mark.parametrize(
    "profile,parallel", [("normal", False), ("deep", False), ("experimental", False)]
)
def test_other_profiles_and_legacy_checkpoint_keep_identical_encoding(
    profile: str, parallel: bool
) -> None:
    _, snapshot = wrapped_snapshot()
    snapshot.update(research_profile=profile, parallel_research=parallel)
    assert compact_parallel_checkpoint(snapshot) is snapshot
    assert restore_parallel_checkpoint(snapshot) is snapshot
    encoded = encode_asv3_checkpoint(snapshot)
    expected = json.dumps(
        {
            "version": 1,
            "encoding": "zlib-base64",
            "data": base64.b64encode(
                zlib.compress(
                    json.dumps(snapshot, ensure_ascii=False).encode(), level=3
                )
            ).decode("ascii"),
        }
    )
    assert encoded == expected
    assert decode_asv3_checkpoint(encoded) == snapshot


def test_shared_pool_values_are_immutable_and_ordinary_evidence_models_can_change() -> (
    None
):
    root, snapshot = wrapped_snapshot()
    restored = restore_parallel_checkpoint(compact_parallel_checkpoint(snapshot))
    record = cast(
        list[dict[str, JsonValue]],
        cast(dict[str, JsonValue], restored["evidence"])["records"],
    )[0]
    metadata = cast(
        dict[str, JsonValue], cast(dict[str, JsonValue], record["item"])["metadata"]
    )
    with pytest.raises(TypeError, match="immutable"):
        metadata["publication_revision"] = 99
    ledger = EvidenceLedger()
    ledger.restore(cast(dict[str, JsonValue], restored["evidence"]), root)
    item = ledger.get(1)
    assert item is not None
    item.metadata["publication_revision"] = 99
    item.question_ids.append("new-question")
    assert metadata["publication_revision"] == 4


def test_valid_new_pool_hash_cannot_replace_child_historical_integrity() -> None:
    context, snapshot = wrapped_snapshot()
    compact = compact_parallel_checkpoint(snapshot)
    storage = cast(dict[str, JsonValue], compact[STORAGE])
    pool = cast(dict[str, JsonValue], storage["records"])
    previous = next(iter(pool))
    record = cast(dict[str, JsonValue], pool.pop(previous))
    item = cast(dict[str, JsonValue], record["item"])
    cast(dict[str, JsonValue], item["metadata"])["publication_revision"] = 5
    changed = hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    pool[changed] = record
    tasks = cast(
        list[dict[str, JsonValue]],
        cast(dict[str, JsonValue], compact["workers"])["tasks"],
    )
    units = [
        compact,
        *(
            cast(
                dict[str, JsonValue],
                cast(dict[str, JsonValue], task["child_checkpoint"])["snapshot"],
            )
            for task in tasks
        ),
    ]
    for unit in units:
        cast(dict[str, JsonValue], unit["evidence"])["records"] = [changed]
    restored = restore_parallel_checkpoint(compact)
    with closing(
        WorkerPool(
            context, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as workers:
        with pytest.raises(ValueError, match="integrity changed"):
            workers.restore(cast(dict[str, JsonValue], restored["workers"]))
