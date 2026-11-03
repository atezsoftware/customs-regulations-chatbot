"""A found governing original gets one bounded, assertion-preserving citation correction."""

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.cropped_citation_repair import citation_repair_plan
from onyx.asv3.cropped_cost_budget import CroppedCostBudget, CroppedRepairBudgetReached
from onyx.asv3.cropped_focused_repair import CroppedFocusedRepair
from onyx.asv3.cropped_handoff_transport import expand_cropped_handoff_transport
from onyx.asv3.cropped_repair_state import CroppedRepairState, extract_repair_targets
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.llm.model_response import ModelResponse
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cropped_repair_model import payload
from tests.unit.onyx.asv3.test_native_model_adapter import model, native_response

_CAPTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "cropped_citation_6391.json").read_text()
)
_OPTIONS = {
    "strict_reference_boundaries": True,
    "syntactic_reference_binding": True,
    "strict_numeric_paragraphs": True,
}


def _setup() -> tuple[Harness, CroppedFocusedRepair, ResearchModel, Any, str]:
    context = RunContext(
        services={
            "asv3_exp_cropped": True,
            "asv3_workflow_variant": "asv3_tuned",
            "research_profile": "normal",
        }
    )
    ledger = EvidenceLedger()
    ledger.add(
        [EvidenceItem.model_validate(row["item"]) for row in _CAPTURE["records"]],
        context,
    )
    context.services["evidence"] = ledger
    # Global citation numbers are run-local; original source/chunk/text hashes stay exact.
    failed = _CAPTURE["failed_unit"].replace("[6][96]", "[1][2]")
    draft = failed + "\n\n" + _CAPTURE["preserved_condition"]
    corrected = failed + "[3]" + "\n\n" + _CAPTURE["preserved_condition"]
    partial: dict[str, str] = {}

    def guard(answer: str) -> ToolOutcome | None:
        gap = canonical_named_authority_gap(answer, ledger, **_OPTIONS)
        return (
            ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Attach the obtained governing original.",
                data=gap,
            )
            if gap
            else None
        )

    def submit(arguments: dict[str, Any], _context: RunContext) -> ToolOutcome:
        answer = arguments["answer"]
        gap = guard(answer)
        if gap is not None:
            return gap
        partial["answer"] = answer
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL, summary="Supported partial answer"
        )

    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    adapter._native_output_capacity = 8192
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_partial_answer",
                description="Publish only a guarded partial answer.",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                handler=submit,
            )
        ]
    )
    harness = Harness(
        request="Explain the cited decision and its limits.",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=lambda _view: Decision(),
        draft_guard=guard,
        partial_submission=lambda: partial.get("answer"),
    )
    harness.last_draft = draft
    harness.publication_gap = guard(draft)
    assert harness.publication_gap is not None
    targets = extract_repair_targets(harness.publication_gap.data, draft, ledger)
    assert (
        len(targets) == 1
        and targets[0].instrument_number == "4458"
        and targets[0].article == "211"
    )
    state = CroppedRepairState(context, harness.request)
    assert state.reserve(targets[0])
    state.finish(
        targets[0].key,
        "found",
        [3, 4, 5],
        "Captured governing original and its material conditions obtained.",
    )
    assert state.reserve_integration(targets, "coordinator")
    repair = CroppedFocusedRepair(state)
    harness.repair_decision = lambda active: repair.decide(active, adapter)
    return harness, repair, adapter, selected, corrected


def _response(harness: Harness, replacement: str) -> Callable[..., ModelResponse]:
    assert harness.last_draft is not None

    def invoke(**arguments: Any) -> ModelResponse:
        packet = payload(arguments["prompt"])
        retained = packet["draft_to_repair"]
        assert isinstance(retained, dict)
        units = retained["units"]
        assert isinstance(units, list) and isinstance(units[0], dict)
        return native_response(
            "submit_retained_partial_answer",
            json.dumps(
                {
                    "retained_answer_edits": [
                        {
                            "unit_id": units[0]["unit_id"],
                            "replacement": replacement + "\n\n",
                        }
                    ]
                }
            ),
        )

    return invoke


def test_captured_6391_found_original_is_cited_without_changing_case_or_conditions() -> (
    None
):
    harness, repair, adapter, selected, corrected = _setup()
    selected.invoke.side_effect = _response(harness, corrected.split("\n\n")[0])

    result = harness.run()

    assert result.status == OutcomeStatus.PARTIAL
    assert isinstance(result.answer, str)
    assert result.answer == corrected
    assert _CAPTURE["preserved_condition"] in result.answer
    assert "özgün metni elde edilemedi" not in result.answer
    assert selected.invoke.call_count == 1
    packet = expand_cropped_handoff_transport(
        payload(selected.invoke.call_args.kwargs["prompt"])
    )
    gap = packet["publication_gap"]
    assert isinstance(gap, dict)
    correction = gap["citation_only_correction"]
    assert isinstance(correction, dict)
    requirements, texts = (
        correction["unit_original_requirements"],
        correction["unit_texts"],
    )
    assert isinstance(requirements, dict) and list(requirements.values()) == [
        [[3, 4, 5]]
    ]
    assert isinstance(texts, dict) and list(texts.values()) == [
        _CAPTURE["failed_unit"].replace("[6][96]", "[1][2]")
    ]
    assert [receipt.call.name for receipt in result.receipts] == [
        "submit_partial_answer"
    ]
    assert adapter.last_call_id is not None
    assert harness.evidence.completely_delivered(adapter.last_call_id) == {
        1,
        2,
        3,
        4,
        5,
        6,
    }
    assert (
        harness.evidence.delivery_flow(adapter.last_call_id)
        == LLMFlow.ASV3_ANSWER_REPAIR.value
    )
    assert (
        harness.draft_guard is not None and harness.draft_guard(result.answer) is None
    )
    integrations = repair.state.export()["integrations"]
    assert isinstance(integrations, dict) and len(integrations) == 2


@pytest.mark.parametrize(
    "change",
    [
        "wording",
        "condition",
        "unrelated_citation",
        "remove_old_citation",
        "no_correction",
    ],
)
def test_citation_correction_rejects_substantive_or_unrelated_changes_once(
    change: str,
) -> None:
    harness, repair, adapter, selected, corrected = _setup()
    assert harness.last_draft is not None and harness.publication_gap is not None
    original = harness.last_draft
    replacement = corrected.split("\n\n")[0]
    if change == "wording":
        replacement = replacement.replace("reddine", "kabulüne")
    elif change == "condition":
        replacement = replacement.replace(
            "; doğrudan ve süresinde ek tahakkuka karşı açılmış bir dava değildir.", "."
        )
    elif change == "unrelated_citation":
        replacement = replacement.replace("[3]", "[6]")
    elif change == "remove_old_citation":
        replacement = replacement.replace("[1]", "")
    else:
        replacement = original.split("\n\n")[0]
    selected.invoke.side_effect = _response(harness, replacement)
    with pytest.raises(RunStopped, match="Citation correction"):
        repair.decide(harness, adapter)
    assert harness.last_draft == original
    assert selected.invoke.call_count == 1
    with pytest.raises(RunStopped, match="already received"):
        repair.decide(harness, adapter)
    assert selected.invoke.call_count == 1
    assert harness.receipts == []


def test_citation_pass_reservation_survives_checkpoint_restore_without_a_retry() -> (
    None
):
    harness, repair, adapter, selected, _corrected = _setup()
    assert harness.last_draft is not None
    selected.invoke.side_effect = _response(
        harness, harness.last_draft.split("\n\n")[0]
    )
    with pytest.raises(RunStopped, match="Citation correction"):
        repair.decide(harness, adapter)
    restored = CroppedRepairState(harness.context, harness.request)
    restored.restore(copy.deepcopy(repair.state.export()))
    with pytest.raises(RunStopped, match="already received"):
        CroppedFocusedRepair(restored).decide(harness, adapter)
    assert selected.invoke.call_count == 1


@pytest.mark.parametrize("stop", ["budget", "cancelled"])
def test_citation_correction_preserves_actual_budget_and_cancellation(
    stop: str,
) -> None:
    harness, repair, adapter, selected, _corrected = _setup()
    if stop == "budget":
        budget = CroppedCostBudget(harness.context, harness.request)
        budget.record_external_charge(
            "repair-allocation",
            known_cost_usd=budget.repair_usd,
            uncertain_upper_usd=0,
            context=harness.context,
        )
        harness.context.services["cropped_cost_budget"] = budget
        expected = CroppedRepairBudgetReached
    else:
        harness.context.cancel()
        expected = RunStopped
    with pytest.raises(expected):
        repair.decide(harness, adapter)
    selected.invoke.assert_not_called()


@pytest.mark.parametrize("defect", ["not_found", "unknown_citations", "stale_unit"])
def test_citation_plan_cannot_treat_an_unresolved_or_unrelated_gap_as_found(
    defect: str,
) -> None:
    harness, repair, _adapter, _selected, _corrected = _setup()
    assert harness.last_draft is not None and harness.publication_gap is not None
    targets = extract_repair_targets(
        harness.publication_gap.data, harness.last_draft, harness.evidence
    )
    records = repair.state.records_for(targets)
    gap = copy.deepcopy(harness.publication_gap.data)
    if defect == "not_found":
        records[0]["status"] = "not_found"
    elif defect == "unknown_citations":
        gap["unknown_citations"] = [999]
    else:
        targets[0] = targets[0].model_copy(update={"unit_ids": ["stale"]})
    assert citation_repair_plan(harness.last_draft, gap, targets, records) is None
