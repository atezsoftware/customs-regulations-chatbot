"""A captured full-law source gap can end as a guarded supported partial answer."""

import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.cropped_focused_repair import (
    CroppedFocusedRepair,
    authority_answer_with_cropped_disclosures,
    authority_gap_with_cropped_disclosures,
    source_gap_sentence,
    unresolved_governing_projection,
)
from onyx.asv3.cropped_repair_state import (
    CroppedRepairState,
    CroppedRepairTarget,
    RepairStatus,
    extract_repair_targets,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import Decision, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.registry import CapabilityRegistry

_FIXTURE = Path(__file__).with_name("fixtures") / "cropped_full_instrument_5434.json"


def _check(text: str) -> dict[str, JsonValue] | None:
    return native_named_authority_gap(
        text,
        EvidenceLedger(),
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )


def _captured() -> tuple[Harness, CroppedFocusedRepair, CroppedRepairTarget, str]:
    raw = json.loads(_FIXTURE.read_text())
    context = RunContext(services={"asv3_exp_cropped": True})
    ledger = EvidenceLedger()
    draft = raw["supported_unit"] + "\n\n" + raw["disclosure_unit"]
    original_gap = _check(draft)
    assert original_gap is not None
    target = extract_repair_targets(original_gap, draft, ledger)[0]
    assert target.article is None
    assert target.instrument_number == raw["gap_entry"]["instrument_number"]
    state = CroppedRepairState(context, "Captured 5434 question")
    state.reserve(target)
    state.finish(
        target.key,
        cast(RepairStatus, raw["source_attempt_status"]),
        [],
        raw["source_attempt_detail"],
    )
    state.reserve_integration([target], "coordinator")

    def guard(text: str) -> ToolOutcome | None:
        gap = authority_gap_with_cropped_disclosures(text, state, _check)
        return (
            ToolOutcome(
                status=OutcomeStatus.PARTIAL, summary="Source identity", data=gap
            )
            if gap is not None
            else None
        )

    harness = Harness(
        request=state.request,
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _view: Decision(),
        evidence=ledger,
        draft_guard=guard,
    )
    harness.last_draft = draft
    harness.publication_gap = guard(draft)
    return harness, CroppedFocusedRepair(state), target, raw["supported_unit"]


def test_captured_5434_exhausted_full_law_attempt_keeps_supported_partial_body() -> (
    None
):
    harness, repair, target, supported = _captured()
    model = MagicMock()
    assert repair.state.reserve_integration([target], "coordinator") is False
    assert repair.state.reserve(target) is False
    decision = repair.decide(harness, model)
    assert decision.calls[0].name == "submit_partial_answer"
    assert decision.calls[0].arguments["answer"] == (
        supported + "\n\n5393 sayılı Kanunun özgün metni elde edilemedi."
    )
    assert harness.draft_guard is not None
    assert harness.draft_guard(str(decision.calls[0].arguments["answer"])) is None
    assert model.mock_calls == []


@pytest.mark.parametrize("status", ["not_found", "incomplete", "failed"])
def test_full_instrument_host_notices_require_the_completed_own_attempt(
    status: RepairStatus,
) -> None:
    state = CroppedRepairState(RunContext(), "Question")
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="5393"
    )
    notice = source_gap_sentence(target, status)
    assert "None" not in notice and "maddesinin" not in notice
    assert authority_answer_with_cropped_disclosures(notice, state) == notice
    state.reserve(target)
    assert authority_answer_with_cropped_disclosures(notice, state) == notice
    state.finish(target.key, status, [], "Original was not independently verified.")
    assert authority_answer_with_cropped_disclosures(notice, state) == ""
    if status != "not_found":
        assert "özgün metni elde edilemedi" not in notice
    unsupported = "5393 sayılı Kanun taşıta izin verir."
    for text in (notice + " " + unsupported, notice + "\n\n" + unsupported):
        assert (
            _check(authority_answer_with_cropped_disclosures(text, state)) is not None
        )


def test_full_instrument_projection_cannot_approve_an_independent_new_statute() -> None:
    harness, repair, _target, _supported = _captured()
    assert harness.last_draft is not None
    harness.last_draft += "\n\n6455 sayılı Kanun m.11 uyarınca ceza uygulanmaz."
    assert repair._exhausted_decision(harness) is None


def test_full_instrument_projection_cannot_remove_a_found_or_unmapped_attempt() -> None:
    harness, repair, target, _supported = _captured()
    assert harness.last_draft is not None
    records = repair.state.records_for([target])
    assert unresolved_governing_projection(harness.last_draft, [target], records)
    records[0]["status"] = "found"
    records[0]["evidence_numbers"] = [1]
    assert (
        unresolved_governing_projection(harness.last_draft, [target], records) is None
    )
    assert unresolved_governing_projection(harness.last_draft, [target], []) is None
