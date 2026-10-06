"""Reuse only fully validated immutable graphs, with unchanged task fences."""

import copy
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Callable, cast
from unittest.mock import patch

import pytest
from pydantic import JsonValue

from onyx.asv3 import parallel_checkpoint as codec
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.workers import WorkerPool
from tests.unit.onyx.asv3.test_parallel_answers import original


def snapshot() -> dict[str, JsonValue]:
    context = RunContext(run_id="run")
    ledger = EvidenceLedger()
    item = original("Exact operative source, qualification and exception.")
    ledger.add([item], context)
    ledger.record_delivery(
        "actual", "asv3_coordinator", [{"citation": 1, "text": item.text}]
    )
    return {
        "run_id": "run",
        "request": "Question",
        "evidence": ledger.export(),
        "native": {"messages": ["Exact immutable body."]},
    }


def test_validated_reuse_avoids_pool_serialization_and_keeps_aliases() -> None:
    source = snapshot()
    shared = codec.share_parallel_snapshot(source)
    with (
        patch.object(codec, "_compact_v2", wraps=codec._compact_v2) as compact,
        patch.object(codec, "_restore_v2", wraps=codec._restore_v2) as restore,
    ):
        assert codec.share_parallel_snapshot(shared) is shared
        assert codec.share_parallel_snapshot(copy.deepcopy(shared)) is shared
        compact.assert_not_called()
        restore.assert_not_called()
        assert codec.share_parallel_snapshot(dict(shared)) == source
        assert compact.call_count == restore.call_count == 1
    assert shared == source
    source["native"] = {"messages": ["Foreign new body."]}
    assert shared != source


@pytest.mark.parametrize("target", ["root", "nested-dict", "nested-list", "row"])
def test_base_class_mutation_cannot_bypass_validation(target: str) -> None:
    shared = codec.share_parallel_snapshot(snapshot())
    native = cast(dict[str, JsonValue], shared["native"])
    if target == "root":
        dict.__setitem__(shared, "request", "Another question")
    elif target == "nested-dict":
        dict.__setitem__(native, "messages", [])
    elif target == "nested-list":
        list.append(cast(list[JsonValue], native["messages"]), "Unvalidated text")
    else:
        evidence = cast(dict[str, JsonValue], shared["evidence"])
        delivery = cast(list[dict[str, JsonValue]], evidence["deliveries"])[0]
        row = cast(list[dict[str, JsonValue]], delivery["records"])[0]
        dict.__setitem__(row, "text_hash", "0" * 64)
    with pytest.raises(ValueError, match="checkpoint changed"):
        codec.share_parallel_snapshot(shared)


def test_forged_wrapper_and_capacity_metadata_never_grant_trust() -> None:
    shared = codec.share_parallel_snapshot(snapshot())
    assert isinstance(shared, codec._SharedSnapshot)
    forged = codec._SharedSnapshot(shared)
    forged.compact_bytes = 1
    with patch.object(codec, "_restore_v2", wraps=codec._restore_v2) as restore:
        assert codec.share_parallel_snapshot(forged) == shared
        restore.assert_called_once()
    shared.compact_bytes = 1
    with pytest.raises(ValueError, match="capacity changed"):
        codec.share_parallel_snapshot(shared)


def test_validated_capacity_and_certificate_lifetime_are_exact() -> None:
    shared = codec.share_parallel_snapshot(snapshot())
    assert isinstance(shared, codec._SharedSnapshot)
    capacity = shared.compact_bytes
    assert codec.share_parallel_snapshot(shared, max_unit_bytes=capacity) is shared
    with pytest.raises(ValueError, match="storage capacity"):
        codec.share_parallel_snapshot(shared, max_unit_bytes=capacity - 1)
    identity = id(shared)
    assert identity in codec._validated_snapshots
    del shared
    gc.collect()
    assert identity not in codec._validated_snapshots


def test_concurrent_worker_saves_keep_bindings_and_validate_once() -> None:
    context = RunContext(
        run_id="run",
        services={"research_profile": "experimental", "experimental_parallel": True},
    )
    ready = threading.Barrier(3)
    compact_calls: list[str] = []
    lock = threading.Lock()
    original_compact = codec._compact_v2

    def compact(
        value: dict[str, JsonValue], maximum: int, *, fast_capacity: bool = False
    ) -> dict[str, JsonValue]:
        with lock:
            compact_calls.append(cast(str, value["request"]))
        return original_compact(value, maximum, fast_capacity=fast_capacity)

    def run(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        value = snapshot()
        value["request"] = task
        ready.wait(timeout=3)
        callback = cast(
            Callable[[dict[str, JsonValue]], None],
            child.services["record_child_checkpoint"],
        )
        callback(value)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=task)

    with patch.object(codec, "_compact_v2", side_effect=compact):
        with closing(WorkerPool(context, run)) as workers:
            tasks = [
                workers.spawn(
                    f"Question {index}",
                    independent_question=True,
                    assignment_id=f"q{index}",
                    outcome_ids=[f"outcome{index}"],
                )
                for index in range(3)
            ]
            workers.wait_until_all(tasks)
            with ThreadPoolExecutor(max_workers=3) as reads:
                saved = list(reads.map(workers.checkpoint, tasks))
            assert all(row is not None for row in saved)
            assert [row["request"] for row in saved if row] == [
                f"Question {index}" for index in range(3)
            ]
            assert len(compact_calls) == 3
            exported = workers.export()
            raw_tasks = cast(list[dict[str, JsonValue]], exported["tasks"])
            wrapper = cast(dict[str, JsonValue], raw_tasks[0]["child_checkpoint"])
            wrapper["outcome_ids"] = ["foreign-outcome"]
            with closing(
                WorkerPool(
                    context,
                    lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done"),
                )
            ) as restored:
                with pytest.raises(ValueError, match="integrity changed"):
                    restored.restore(exported)
