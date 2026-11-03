"""Unassigned Cropped workers read sources without mutating shared outcomes."""

from typing import cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import JsonValue

from onyx.asv3.cropped_repair_model import fresh_model
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.outcome_map import OutcomeUpdate
from onyx.asv3.workers import WorkerPool
from tests.unit.onyx.asv3.test_native_model_adapter import model
from tests.unit.onyx.asv3.test_native_outcome_metadata import (
    harness,
    native_parameters,
    read_call,
    requested,
    research_context,
    source_registry,
)


@pytest.mark.parametrize("assigned", [None, []])
def test_unassigned_spawned_worker_and_focused_model_expose_only_authorized_actions(
    assigned: list[str] | None,
) -> None:
    context, outcomes, _ledger = research_context()
    context.services.update(research_profile="experimental", asv3_exp_cropped=True)
    before = outcomes.export()
    observed: list[RunContext] = []

    def runner(_task: str, child: RunContext, _updates: object) -> ToolOutcome:
        observed.append(child)
        adapter = fresh_model(ResearchModel(model(), child, lean_native_mode=True))
        for owner in (child, adapter.context):
            parameters = native_parameters(owner)
            properties = parameters["properties"]
            assert isinstance(properties, dict)
            assert "_outcomes" not in properties and "_coverage" not in properties
            validator = Draft202012Validator(parameters)
            assert validator.is_valid(read_call().arguments)
            assert not validator.is_valid(read_call({"_coverage": {}}).arguments)
        assert "Where useful and exposed" in adapter._research_instruction()
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Worker complete")

    pool = WorkerPool(context, runner)
    try:
        task_id = pool.spawn("Read the original", outcome_ids=assigned)
        pool.wait_until_all([task_id])
        assert len(observed) == 1 and observed[0].depth == 1
        assert outcomes.export() == before
    finally:
        pool.close()


@pytest.mark.parametrize(
    "metadata",
    [
        {"_coverage": {}},
        {"_outcomes": []},
        {"_outcomes": [], "_coverage": {"conditions": [], "resolutions": []}},
    ],
)
@pytest.mark.parametrize("through_harness", [False, True])
def test_empty_cropped_child_metadata_does_not_block_actual_source_read(
    metadata: dict[str, JsonValue], through_harness: bool
) -> None:
    parent, outcomes, ledger = research_context()
    parent.services.update(research_profile="experimental", asv3_exp_cropped=True)
    child = parent.child()
    observed: list[dict[str, JsonValue]] = []

    def read(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        observed.append(arguments)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original read")

    registry = source_registry(read)
    before = outcomes.export()
    call = read_call(metadata)
    outcome = (
        harness(child, registry, ledger)._dispatch([call])[0].outcome
        if through_harness
        else registry.dispatch(call, child)
    )
    assert outcome.status is OutcomeStatus.FOUND
    assert observed == [{"source_id": "original", "article": "1"}]
    assert outcomes.export() == before
    assert call.arguments == read_call(metadata).arguments


@pytest.mark.parametrize(
    "metadata",
    [
        {"_outcomes": [requested()]},
        {
            "_coverage": {
                "resolutions": [{"outcome_id": "first", "status": "unresolved"}]
            }
        },
        {"_coverage": {"conditions": ["malformed"]}},
        {"_coverage": {"unexpected": []}},
    ],
)
def test_unassigned_cropped_child_still_rejects_real_or_malformed_metadata(
    metadata: dict[str, JsonValue],
) -> None:
    parent, outcomes, ledger = research_context()
    parent.services.update(research_profile="experimental", asv3_exp_cropped=True)
    child = parent.child()
    observed: list[dict[str, JsonValue]] = []
    registry = source_registry(
        lambda arguments, _context: (
            observed.append(arguments)
            or ToolOutcome(status=OutcomeStatus.FOUND, summary="Original read")
        )
    )
    before = outcomes.export()
    outcome = (
        harness(child, registry, ledger)._dispatch([read_call(metadata)])[0].outcome
    )
    assert outcome.status is OutcomeStatus.INVALID
    assert outcome.data["invalid_outcome_metadata"] is True
    assert observed == [] and outcomes.export() == before


def test_assigned_cropped_child_keeps_only_its_existing_subset_authority() -> None:
    parent, outcomes, ledger = research_context()
    parent.services.update(research_profile="experimental", asv3_exp_cropped=True)
    outcomes.update(OutcomeUpdate.model_validate({"outcomes": [requested()]}), ledger)
    child = parent.child()
    child.services["task_outcome_ids"] = ["first"]
    properties = native_parameters(child)["properties"]
    assert isinstance(properties, dict)
    assert "_outcomes" in properties and "_coverage" in properties
    registry = source_registry(
        lambda _arguments, _context: ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Original read"
        )
    )
    assert (
        registry.dispatch(read_call({"_outcomes": [requested()]}), child).status
        is OutcomeStatus.FOUND
    )
    before = outcomes.export()
    denied = registry.dispatch(read_call({"_outcomes": [requested("other")]}), child)
    assert denied.status is OutcomeStatus.INVALID
    assert denied.data["detail"] == "Researcher may update only assigned outcomes"
    assert outcomes.export() == before


@pytest.mark.parametrize("kind", ["root", "normal", "deep", "cheap"])
def test_root_and_other_workflows_keep_existing_outcome_contract(kind: str) -> None:
    context, outcomes, _ledger = research_context()
    context.services.update(research_profile="experimental", asv3_exp_cropped=True)
    if kind != "root":
        context = context.child()
        if kind == "cheap":
            context.services["asv3_exp_cheap"] = True
        else:
            context.services.pop("asv3_exp_cropped")
            context.services["research_profile"] = kind
    properties = cast(dict[str, JsonValue], native_parameters(context)["properties"])
    assert "_outcomes" in properties and "_coverage" in properties
    registry = source_registry(
        lambda _arguments, _context: ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Original read"
        )
    )
    before = outcomes.export()
    outcome = registry.dispatch(read_call({"_coverage": {}}), context)
    assert outcome.status is (
        OutcomeStatus.FOUND if kind == "root" else OutcomeStatus.INVALID
    )
    assert outcomes.export() == before
