"""Compare scoped compact-size accounting with the original JSON encoder."""

import copy
import json
import random
from contextlib import closing
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import JsonValue

from onyx.asv3 import parallel_checkpoint as codec
from onyx.asv3 import workers as worker_module
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.workers import WorkerPool
from tests.unit.onyx.asv3.test_parallel_checkpoint_reuse import snapshot


def samples() -> list[dict[str, JsonValue]]:
    shared: JsonValue = {"ç": ['İĞŞöç — 東京\n\t"\\', None, True, 1, -0.0]}
    values: list[dict[str, JsonValue]] = [
        {},
        {"empty": [[], {}, ""]},
        {'"\\\n\t東京': {"": "Escaped keys"}},
        {"alias": [shared, shared], "again": shared},
        {"float": [float("inf"), float("-inf"), float("nan"), 1e-100, 1e100]},
    ]
    rng = random.Random(19)

    def value(depth: int) -> JsonValue:
        if depth == 0 or rng.randrange(3) == 0:
            return rng.choice([None, True, False, 0, 12.5, 'ü\n"\\東京'])
        if rng.randrange(2):
            return [value(depth - 1) for _ in range(rng.randrange(5))]
        return {f"key-{index}-ç": value(depth - 1) for index in range(rng.randrange(5))}

    values.extend({"value": value(6)} for _ in range(100))
    return values


@pytest.mark.parametrize("value", samples())
def test_exact_bytes_and_capacity_boundaries(value: dict[str, JsonValue]) -> None:
    expected = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    assert codec._capacity(value, expected) == expected
    assert codec._capacity_walk(value, expected) == expected
    assert codec._capacity_walk(value, expected + 1) == expected
    with pytest.raises(ValueError, match="storage capacity"):
        codec._capacity_walk(value, expected - 1)


@pytest.mark.parametrize("maximum", [0, -1, True, 2.0])
def test_invalid_capacity_matches_legacy(maximum: int) -> None:
    for function in (codec._capacity, codec._capacity_walk):
        with pytest.raises(ValueError, match="Invalid checkpoint storage capacity"):
            function({}, maximum)


def test_cycles_invalid_values_and_mutated_aliases_remain_checked() -> None:
    cycle: dict[str, JsonValue] = {}
    cycle["self"] = cycle
    for function in (codec._capacity, codec._capacity_walk):
        with pytest.raises(ValueError, match="Circular reference"):
            function(cycle, 100_000)
        with pytest.raises(ValueError, match="Invalid checkpoint JSON value"):
            function(cast(dict[str, JsonValue], {"foreign": object()}), 100_000)
    shared: dict[str, JsonValue] = {"value": "short"}
    value: dict[str, JsonValue] = {"copies": [shared] * 20}
    old_size = codec._capacity_walk(value, 100_000)
    dict.__setitem__(shared, "value", "ü" * 1000)
    assert codec._capacity_walk(value, 100_000) == codec._capacity(value, 100_000)
    with pytest.raises(ValueError, match="storage capacity"):
        codec._capacity_walk(value, old_size)


def test_nonstring_keys_keep_encoder_handling() -> None:
    value = cast(dict[str, JsonValue], {1: "one", False: [1, None], None: "null"})
    assert codec._capacity_walk(value, 1000) == codec._capacity(value, 1000)


def test_full_traversal_remains_exact_when_bounded_memo_is_full() -> None:
    shared: dict[str, JsonValue] = {"body": "Literal İğşç — 東京"}
    value: dict[str, JsonValue] = {
        "unique": [{str(index): [index]} for index in range(30)],
        "aliases": [shared] * 30,
    }
    with patch.object(codec, "_CAPACITY_MEMO_ITEMS", 2):
        assert codec._capacity_walk(value, 100_000) == codec._capacity(value, 100_000)


def test_fast_share_preserves_full_values_and_raw_codec_path() -> None:
    source = snapshot()
    legacy = codec.share_parallel_snapshot(source)
    with patch.object(codec, "_capacity_walk", wraps=codec._capacity_walk) as walk:
        fast = codec.share_parallel_snapshot(source, fast_capacity=True)
        assert walk.call_count == 2
        assert fast == legacy == source
        assert isinstance(fast, codec._SharedSnapshot)
        assert isinstance(legacy, codec._SharedSnapshot)
        assert fast.compact_bytes == legacy.compact_bytes
        before = walk.call_count
        scoped = {
            **source,
            "research_profile": "experimental",
            "parallel_research": True,
        }
        compact = codec.compact_parallel_checkpoint(scoped)
        assert codec.restore_parallel_checkpoint(compact) == scoped
        codec.share_parallel_snapshot(copy.deepcopy(source))
        assert walk.call_count == before
    expected = codec._capacity(codec._compact_v2(source, 100_000), 100_000)
    assert (
        codec.share_parallel_snapshot(
            source, max_unit_bytes=expected, fast_capacity=True
        )
        == source
    )
    with pytest.raises(ValueError, match="storage capacity"):
        codec.share_parallel_snapshot(
            source, max_unit_bytes=expected - 1, fast_capacity=True
        )


def test_fast_share_retains_base_method_mutation_refusal() -> None:
    shared = codec.share_parallel_snapshot(snapshot(), fast_capacity=True)
    native = cast(dict[str, JsonValue], shared["native"])
    list.append(cast(list[JsonValue], native["messages"]), "Unvalidated source")
    with pytest.raises(ValueError, match="checkpoint changed"):
        codec.share_parallel_snapshot(shared, fast_capacity=True)


@pytest.mark.parametrize("profile", ["experimental", "standard"])
def test_nonparallel_worker_keeps_original_capacity_path(profile: str) -> None:
    context = RunContext(
        run_id="run",
        services={"research_profile": profile, "experimental_parallel": False},
    )
    with closing(
        WorkerPool(
            context, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as pool:
        task_id = pool.spawn("Question", assignment_id="assignment")
        pool.wait_until_all([task_id])
        with patch.object(codec, "_capacity_walk", wraps=codec._capacity_walk) as walk:
            pool.record_checkpoint(task_id, snapshot())
            assert pool.checkpoint(task_id) == snapshot()
            walk.assert_not_called()


def test_generated_wrapper_binding_and_untrusted_hash_verification() -> None:
    context = RunContext(
        run_id="run",
        services={"research_profile": "experimental", "experimental_parallel": True},
    )
    source = snapshot()
    with closing(
        WorkerPool(
            context, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as pool:
        task_id = pool.spawn(
            "Question",
            independent_question=True,
            assignment_id="assignment",
            outcome_ids=["outcome"],
        )
        pool.wait_until_all([task_id])
        pool.record_checkpoint_control(
            task_id, {key: part for key, part in source.items() if key != "evidence"}
        )
        capture = pool.capture_checkpoint_state()
        with patch.object(
            worker_module,
            "parallel_checkpoint_digest",
            wraps=codec.parallel_checkpoint_digest,
        ) as digest:
            materialized = pool.materialize_checkpoint_state(
                capture, cast(dict[str, JsonValue], source["evidence"])
            )
            assert digest.call_count == 2  # Control verification and generated wrapper.
        rows = cast(list[dict[str, JsonValue]], materialized["tasks"])
        wrapped = cast(dict[str, JsonValue], rows[0]["child_checkpoint"])
        assert wrapped["snapshot"] == source
        assert wrapped["integrity"] == codec.parallel_checkpoint_digest(
            {key: part for key, part in wrapped.items() if key != "integrity"}
        )
        pool._validated_checkpoint(pool._tasks[task_id], wrapped)
        tampered = {**wrapped, "integrity": "0" * 64}
        with pytest.raises(ValueError, match="integrity changed"):
            pool._validated_checkpoint(pool._tasks[task_id], tampered)
        scoped = {
            **source,
            "research_profile": "experimental",
            "parallel_research": True,
            "workers": materialized,
        }
        assert (
            codec.restore_parallel_checkpoint(codec.compact_parallel_checkpoint(scoped))
            == scoped
        )


def test_generated_wrapper_still_revalidates_issued_graph() -> None:
    context = RunContext(
        run_id="run",
        services={"research_profile": "experimental", "experimental_parallel": True},
    )
    source = snapshot()
    with closing(
        WorkerPool(
            context, lambda *_: ToolOutcome(status=OutcomeStatus.FOUND, summary="Done")
        )
    ) as pool:
        task_id = pool.spawn(
            "Question",
            independent_question=True,
            assignment_id="assignment",
            outcome_ids=["outcome"],
        )
        pool.wait_until_all([task_id])
        pool.record_checkpoint_control(
            task_id, {key: part for key, part in source.items() if key != "evidence"}
        )
        capture = pool.capture_checkpoint_state()

        def tamper(value: dict[str, JsonValue]) -> str:
            result = codec.parallel_checkpoint_digest(value)
            if "snapshot" in value:
                dict.__setitem__(
                    cast(dict[str, JsonValue], value["snapshot"]),
                    "request",
                    "Foreign request",
                )
            return result

        with patch.object(
            worker_module, "parallel_checkpoint_digest", side_effect=tamper
        ):
            with pytest.raises(ValueError, match="checkpoint changed"):
                pool.materialize_checkpoint_state(
                    capture, cast(dict[str, JsonValue], source["evidence"])
                )
