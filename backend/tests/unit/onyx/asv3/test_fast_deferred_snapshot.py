"""Deferred root capture preserves actual Cropped checkpoint and resume boundaries."""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3 import cropped_cost_budget, llm_adapter, models, progress, runtime
from onyx.asv3.harness import Harness
from onyx.asv3.models import RunContext, RunStopped
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_FAST_CROPPED_VARIANT,
)
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.asv3_runs import decode_asv3_checkpoint, encode_asv3_checkpoint
from onyx.llm.model_response import ModelResponse
from onyx.regulatory.source_notices import SourceNotice
from onyx.tracing.framework.setup import get_trace_provider
from tests.unit.onyx.asv3 import test_runtime
from tests.unit.onyx.asv3.test_cropped_focused_repair_runtime import (
    priced_scripted_provider as priced_scripted_provider,
)
from tests.unit.onyx.asv3.test_runtime import response, setup_run

pytestmark = pytest.mark.usefixtures(
    "empty_source_inventory", "priced_scripted_provider"
)


@dataclass
class _RunResult:
    checkpoints: list[dict[str, JsonValue]]
    roots: list[Harness]
    full_snapshots: list[Harness]


def _uuid_sequence(start: int) -> Callable[[], UUID]:
    values = count(start)
    return lambda: UUID(int=next(values))


def _run(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    *,
    cancel_after_read: bool = False,
    previous: dict[str, JsonValue] | None = None,
) -> _RunResult:
    for index, module in enumerate(
        (models, llm_adapter, progress, cropped_cost_budget, test_runtime), start=1
    ):
        monkeypatch.setattr(module, "uuid4", _uuid_sequence(index * 1_000))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.0)
    span_ids = count(100 if previous is not None else 1)
    monkeypatch.setattr(
        get_trace_provider(), "gen_span_id", lambda: f"span_{next(span_ids):024d}"
    )
    kwargs, broker, llm, _, _ = setup_run(monkeypatch)
    if previous is not None:
        for index, module in enumerate(
            (llm_adapter, progress, cropped_cost_budget, test_runtime), start=1
        ):
            monkeypatch.setattr(module, "uuid4", _uuid_sequence(10_000 + index * 1_000))
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
    source = str(broker.sources[0].id)

    def notices(
        _queries: list[str] | None,
        _identifiers: list[str] | None,
        context: RunContext,
    ) -> list[SourceNotice]:
        context.check_active()
        return []

    def page(
        source_id: str, context: RunContext, **_options: object
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        context.check_active()
        assert source_id == source
        if cancel_after_read:
            broker.cancelled.set()
        return broker.sources[0], [broker.chunks[source]], False

    monkeypatch.setattr(broker, "source_notices", notices, raising=False)
    monkeypatch.setattr(broker, "page", page)
    saved: list[dict[str, JsonValue]] = []

    def persist(**arguments: Any) -> None:
        snapshot = copy.deepcopy(arguments["snapshot"])
        assert decode_asv3_checkpoint(encode_asv3_checkpoint(snapshot)) == snapshot
        saved.append(snapshot)

    monkeypatch.setattr(runtime, "save_asv3_checkpoint", persist)
    if previous is not None:
        monkeypatch.setattr(
            runtime, "load_asv3_checkpoint", lambda **_options: copy.deepcopy(previous)
        )
        kwargs["resume_message_id"] = kwargs["assistant_message_id"]

    roots: list[Harness] = []
    full_snapshots: list[Harness] = []
    original_snapshot = Harness.snapshot

    def snapshot(active: Harness) -> dict[str, JsonValue]:
        full_snapshots.append(active)
        return original_snapshot(active)

    def create(**arguments: Any) -> Harness:
        active = Harness(**arguments)
        roots.append(active)
        return active

    monkeypatch.setattr(Harness, "snapshot", snapshot)
    monkeypatch.setattr(runtime, "Harness", create)
    invocation = 0

    def invoke(**_arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1 and previous is None:
            return response(
                calls=[("read_source_range", {"source_id": source, "_language": "tr"})]
            )
        assert invocation == (1 if previous is not None else 2)
        return response(
            calls=[
                (
                    "submit_answer",
                    {"answer": "Tamir şartları [1].", "basis": "originals"},
                )
            ]
        )

    llm.invoke.side_effect = invoke
    if cancel_after_read:
        with pytest.raises(RunStopped, match="Research cancelled"):
            runtime.run_asv3_loop(**kwargs)
    else:
        runtime.run_asv3_loop(**kwargs)
    assert len(roots) == 1
    return _RunResult(saved, roots, full_snapshots)


def _without_variant(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _without_variant(part)
            for key, part in value.items()
            if key not in {"asv3_workflow_variant", "asv3_workflow_policy"}
        }
    if isinstance(value, list):
        return [_without_variant(part) for part in value]
    return value


def _assert_equal(actual: JsonValue, expected: JsonValue, path: str = "") -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _assert_equal(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_equal(left, right, f"{path}/{index}")
    else:
        assert actual == expected, path


def test_fast_deferred_capture_preserves_durable_payload_and_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as baseline_patch:
        baseline = _run(baseline_patch, ASV3_EXP_CROPPED_VARIANT)
    with monkeypatch.context() as fast_patch:
        fast = _run(fast_patch, ASV3_FAST_CROPPED_VARIANT)

    assert baseline.full_snapshots
    assert not fast.full_snapshots
    assert len(fast.checkpoints) == len(baseline.checkpoints) > 2
    _assert_equal(
        _without_variant(fast.checkpoints), _without_variant(baseline.checkpoints)
    )
    last = fast.checkpoints[-1]
    assert last["evidence"] and last["receipts"] and last["turns"]
    assert last["asv3_workflow_variant"] == ASV3_FAST_CROPPED_VARIANT
    assert [row["sequence"] for row in fast.checkpoints] == list(
        range(1, len(fast.checkpoints) + 1)
    )
    with monkeypatch.context() as resume_patch:
        resumed = _run(resume_patch, ASV3_FAST_CROPPED_VARIANT, previous=last)
    with monkeypatch.context() as baseline_resume_patch:
        baseline_resumed = _run(
            baseline_resume_patch,
            ASV3_EXP_CROPPED_VARIANT,
            previous=baseline.checkpoints[-1],
        )
    assert not resumed.full_snapshots
    assert baseline_resumed.full_snapshots
    _assert_equal(
        _without_variant(resumed.checkpoints),
        _without_variant(baseline_resumed.checkpoints),
    )
    assert resumed.checkpoints[-1]["run_id"] == last["run_id"]
    evidence = resumed.checkpoints[-1]["evidence"]
    prior_evidence = last["evidence"]
    assert isinstance(evidence, dict) and isinstance(prior_evidence, dict)
    assert evidence["records"] == prior_evidence["records"]
    resumed_sequence = resumed.checkpoints[0]["sequence"]
    prior_sequence = last["sequence"]
    assert isinstance(resumed_sequence, int) and not isinstance(resumed_sequence, bool)
    assert isinstance(prior_sequence, int) and not isinstance(prior_sequence, bool)
    assert resumed_sequence > prior_sequence


def test_fast_deferred_capture_keeps_pending_checkpoint_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as baseline_patch:
        baseline = _run(
            baseline_patch, ASV3_EXP_CROPPED_VARIANT, cancel_after_read=True
        )
    with monkeypatch.context() as fast_patch:
        fast = _run(fast_patch, ASV3_FAST_CROPPED_VARIANT, cancel_after_read=True)
    assert baseline.full_snapshots and not fast.full_snapshots
    assert fast.checkpoints
    _assert_equal(
        _without_variant(fast.checkpoints), _without_variant(baseline.checkpoints)
    )
    assert fast.checkpoints[-1]["pending_call_count"] == 1
    assert not fast.checkpoints[-1]["receipts"]
    evidence = fast.checkpoints[-1]["evidence"]
    assert isinstance(evidence, dict) and not evidence["records"]
