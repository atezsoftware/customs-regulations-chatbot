"""Multiple exhausted sources in one assertion retain precise, guarded limitations."""

from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.cropped_focused_repair import (
    CroppedFocusedRepair,
    source_gap_sentence,
    unresolved_governing_projection,
)
from onyx.asv3.cropped_repair_state import CroppedRepairState, CroppedRepairTarget
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import Decision, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.registry import CapabilityRegistry

_SUPPORTED = "Yönetmelikte sınır kapısında tahsil usulü düzenlenir [1]."
_JOINT = (
    "6001 sayılı Kanun m.30 ve 5326 sayılı Kanun m.17 uyarınca çıkışa izin verilir."
)
_REPEATED = "6001 sayılı Kanun m.30 uyarınca ayrıca indirim uygulanır."
_DRAFT = "\n\n".join((_SUPPORTED, _JOINT, _REPEATED))


def _setup() -> tuple[Harness, CroppedFocusedRepair, list[CroppedRepairTarget]]:
    context = RunContext(services={"asv3_exp_cropped": True})
    ledger = EvidenceLedger()
    harness = Harness(
        request="HGS borcu ve ödeme seçenekleri?",
        context=context,
        registry=CapabilityRegistry(),
        evidence=ledger,
        decide=lambda _view: Decision(),
    )
    units = assertion_inventory(_DRAFT)
    targets = [
        CroppedRepairTarget(
            check_kind="governing_original",
            instrument_number="6001",
            article="30",
            unit_ids=[units[1]["unit_id"], units[2]["unit_id"]],
        ),
        CroppedRepairTarget(
            check_kind="governing_original",
            instrument_number="5326",
            article="17",
            unit_ids=[units[1]["unit_id"]],
        ),
    ]
    state = CroppedRepairState(context, harness.request)
    for target, status in zip(targets, ("not_found", "incomplete"), strict=True):
        state.reserve(target)
        state.finish(target.key, status, [], "Focused attempt exhausted.")
    state.reserve_integration(targets, "coordinator")
    harness.last_draft = _DRAFT
    return harness, CroppedFocusedRepair(state), targets


def _gap(targets: list[CroppedRepairTarget]) -> ToolOutcome:
    entries: list[JsonValue] = [
        {
            "unit_id": unit,
            "instrument_number": target.instrument_number,
            "article": target.article,
        }
        for target in targets
        for unit in target.unit_ids
    ]
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Read the governing originals.",
        data={"named_authority_gaps": entries},
    )


def test_joint_assertion_discloses_each_exhausted_source_once() -> None:
    harness, repair, targets = _setup()
    expected = "\n\n".join(
        (
            _SUPPORTED,
            source_gap_sentence(targets[0], "not_found"),
            source_gap_sentence(targets[1], "incomplete"),
        )
    )
    assert (
        unresolved_governing_projection(
            _DRAFT, targets, repair.state.records_for(targets)
        )
        == expected
    )
    guard = MagicMock(
        side_effect=lambda answer: None if answer == expected else _gap(targets)
    )
    harness.draft_guard = guard
    decision = repair._exhausted_decision(harness)
    assert decision is not None
    assert decision.calls[0].name == "submit_partial_answer"
    assert decision.calls[0].arguments == {"answer": expected}
    assert [call.args[0] for call in guard.call_args_list] == [_DRAFT, expected]
    assert all(not repair.state.reserve(target) for target in targets)


@pytest.mark.parametrize("change", ["found", "unfinished", "unrelated_defect"])
def test_joint_limitations_cannot_remove_verified_or_unfinished_sources_or_skip_guard(
    change: str,
) -> None:
    harness, repair, targets = _setup()
    if change in {"found", "unfinished"}:
        records = repair.state.records_for(targets)
        records[1]["status"] = change if change == "found" else "in_progress"
        assert unresolved_governing_projection(_DRAFT, targets, records) is None
    else:
        harness.draft_guard = MagicMock(
            side_effect=[
                _gap(targets),
                ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="Another assertion remains unsupported.",
                    data={"unknown_citations": [999]},
                ),
            ]
        )
        assert repair._exhausted_decision(harness) is None
        assert harness.draft_guard.call_count == 2
