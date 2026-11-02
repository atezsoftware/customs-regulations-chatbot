"""Host finish preserves one reserved governing read without reopening discovery."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import cropped_focused_repair
from onyx.asv3.cropped_cost_budget import CroppedCostBudget, CroppedRepairBudgetReached
from onyx.asv3.cropped_repair_state import CroppedRepairTarget
from onyx.asv3.cropped_research_limit import CroppedResearchLimit, CroppedResearchPolicy
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from tests.unit.onyx.asv3.test_cropped_retained_article_repair import _Checker, _setup


def _close_discovery(
    context: RunContext, request: str, *, critical_reads: int = 16
) -> CroppedResearchLimit:
    controller = CroppedResearchLimit(
        context,
        request,
        policy=CroppedResearchPolicy(
            enforce_observed_limits=False, critical_reads=critical_reads
        ),
    )
    context.services["cropped_research_controller"] = controller
    controller.force_finish(
        "Research allocation reached; finish with retained evidence"
    )
    return controller


def test_host_finish_allows_reserved_known_article_then_keeps_discovery_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operative = "(6) Ödeme süresinde ödenen cezaya indirim uygulanır."
    harness, repair, target, _broker = _setup(
        "5326", "17", "MADDE 17 — (1) Para cezasının türü belirlenir.", operative
    )
    controller = _close_discovery(harness.context, harness.request)
    model = cast(ResearchModel, MagicMock())
    checker = _Checker(harness.evidence, operative)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)
    assert repair.state.reserve(target)

    numbers, observations = repair._acquire(target, harness, model)
    repair._check(target, harness, model, numbers, observations)

    assert repair.state.records_for([target])[0]["status"] == "found"
    assert len(harness.receipts) == 1
    assert controller.progress()["critical_reads_reserved"] == 1
    actual_read = harness.receipts[0].call
    assert not controller.admit_tool(
        actual_read.name, actual_read.arguments, call_id=actual_read.call_id
    ).allowed
    assert not controller.admit_tool(
        "search_corpus", {"query": "a broad extra search"}, call_id=actual_read.call_id
    ).allowed
    with pytest.raises(ValueError, match="exact canonical read"):
        with controller.focused_governing_read(actual_read, repair.state, target):
            pytest.fail("Completed repair reopened a source read")
    assert not repair.state.reserve(target)


def test_scoped_permit_binds_active_governing_target_and_exact_call() -> None:
    harness, repair, _target, _broker = _setup(
        "5326", "17", "MADDE 17 — (1) Para cezasının türü belirlenir.", "(6) İndirim."
    )
    controller = _close_discovery(harness.context, harness.request)
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6001", article="30"
    )
    assert repair.state.reserve(target)
    actual_read = CapabilityCall(
        name="read_named_provision",
        arguments={"source_name": "6001 sayılı Kanun", "article": "30"},
    )
    with controller.focused_governing_read(actual_read, repair.state, target):
        assert not controller.admit_tool(
            "search_corpus",
            {"query": "a broad extra search"},
            call_id=actual_read.call_id,
        ).allowed
        assert not controller.admit_tool(
            actual_read.name, actual_read.arguments, call_id="another-call"
        ).allowed
        assert controller.admit_tool(
            actual_read.name, actual_read.arguments, call_id=actual_read.call_id
        ).allowed
        repeated = controller.admit_tool(
            actual_read.name, actual_read.arguments, call_id=actual_read.call_id
        )
        assert not repeated.allowed
        assert (
            repeated.reason == "critical_read_already_reserved_reuse_recorded_evidence"
        )
    nongoverning = target.model_copy(update={"check_kind": "related_source"})
    assert repair.state.reserve(nongoverning)
    with pytest.raises(ValueError, match="exact canonical read"):
        with controller.focused_governing_read(actual_read, repair.state, nongoverning):
            pytest.fail("A non-governing target acquired a privileged read")


def test_unknown_statute_is_actually_attempted_and_critical_allowance_is_bounded() -> (
    None
):
    harness, repair, _target, _broker = _setup(
        "5326", "17", "MADDE 17 — (1) Para cezasının türü belirlenir.", "(6) İndirim."
    )
    controller = _close_discovery(harness.context, harness.request, critical_reads=1)
    looked_up: list[str] = []

    def lookup(arguments: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        context.check_active()
        looked_up.append(str(arguments["source_name"]))
        return ToolOutcome(
            status=OutcomeStatus.NOT_FOUND,
            summary="No authorized source identity was resolved.",
            data={"sources": [], "has_more": False, "absence_proven": False},
        )

    harness.registry.register(
        ToolSpec(
            name="read_named_provision",
            description="Exact named original read",
            parameters={"type": "object"},
            handler=lookup,
        )
    )
    model = cast(ResearchModel, MagicMock())
    first = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6001", article="30"
    )
    assert repair.state.reserve(first)
    numbers, observations = repair._acquire(first, harness, model)
    repair._check(first, harness, model, numbers, observations)
    assert looked_up == ["6001 sayılı Kanun"]
    assert repair.state.records_for([first])[0]["status"] == "not_found"
    assert not repair.state.reserve(first)

    second = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    assert repair.state.reserve(second)
    numbers, observations = repair._acquire(second, harness, model)
    repair._check(second, harness, model, numbers, observations)
    record = repair.state.records_for([second])[0]
    assert record["status"] == "incomplete"
    detail = json.loads(str(record["detail"]))
    assert detail["observations"][0]["result"]["not_read"] is True
    assert detail["absence_proven"] is False
    assert looked_up == ["6001 sayılı Kanun"]
    assert controller.progress()["critical_reads_reserved"] == 1


def test_unreserved_target_cannot_reopen_host_finish() -> None:
    harness, repair, target, _broker = _setup(
        "5326", "17", "MADDE 17 — (1) Para cezasının türü belirlenir.", "(6) İndirim."
    )
    controller = _close_discovery(harness.context, harness.request)
    _numbers, observations = repair._acquire(
        target, harness, cast(ResearchModel, MagicMock()), read_incomplete_original=True
    )
    assert observations[0]["status"] == "partial"
    result = observations[0]["result"]
    assert isinstance(result, dict) and result["not_read"] is True
    assert controller.progress()["critical_reads_reserved"] == 0


@pytest.mark.parametrize("stop", ["cancelled", "repair_budget"])
def test_reserved_read_preserves_actual_cancellation_and_repair_budget(
    stop: str,
) -> None:
    harness, repair, target, _broker = _setup(
        "5326", "17", "MADDE 17 — (1) Para cezasının türü belirlenir.", "(6) İndirim."
    )
    controller = _close_discovery(harness.context, harness.request)
    assert repair.state.reserve(target)
    if stop == "cancelled":
        harness.context.cancel()
        expected = RunStopped
    else:
        budget = CroppedCostBudget(harness.context, harness.request)
        budget.record_external_charge(
            "exhausted-repair",
            known_cost_usd=budget.repair_usd,
            uncertain_upper_usd=0,
            context=harness.context,
        )
        harness.context.services["cropped_cost_budget"] = budget
        expected = CroppedRepairBudgetReached
    with pytest.raises(expected):
        repair._acquire(
            target,
            harness,
            cast(ResearchModel, MagicMock()),
            read_incomplete_original=True,
        )
    assert harness.receipts == []
    assert controller.progress()["critical_reads_reserved"] == 0
