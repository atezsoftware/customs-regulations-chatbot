"""Disclosure filtering preserves the identities of the exact owned answer units."""

from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.cropped_article_bindings import cropped_named_authority_gap
from onyx.asv3.cropped_focused_repair import (
    CroppedFocusedRepair,
    authority_gap_with_cropped_disclosures,
    source_gap_sentence,
)
from onyx.asv3.cropped_repair_state import (
    CroppedRepairState,
    CroppedRepairTarget,
    extract_repair_targets,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.registry import CapabilityRegistry
from tests.unit.onyx.asv3.test_cropped_authority_5315 import OPTIONS, captured


def _record(state: CroppedRepairState, target: CroppedRepairTarget) -> None:
    state.reserve(target)
    state.finish(target.key, "not_found", [], "A completed focused original attempt.")
    state.reserve_integration([target], "coordinator")


def _setup() -> tuple[CroppedRepairState, CroppedRepairTarget]:
    state = CroppedRepairState(
        RunContext(services={"asv3_exp_cropped": True}), "Question"
    )
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="10"
    )
    _record(state, target)
    return state, target


def _check(text: str) -> dict[str, JsonValue] | None:
    return native_named_authority_gap(
        text,
        EvidenceLedger(),
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )


def test_unknown_article_after_owned_notice_maps_to_raw_unit_and_stays_blocked() -> (
    None
):
    state, prior = _setup()
    draft = (
        "Başvuruda belge sunulur [1].\n\n"
        + source_gap_sentence(prior, "not_found")
        + "\n\n6455 sayılı Kanun m.11 uyarınca ceza uygulanmaz."
    )
    gap = authority_gap_with_cropped_disclosures(draft, state, _check)
    assert gap is not None
    targets = extract_repair_targets(gap, draft, EvidenceLedger())
    assert len(targets) == 1
    assert targets[0].unit_ids == [assertion_inventory(draft)[2]["unit_id"]]
    assert targets[0].article == "11"
    assert state.records_for(targets) == []


def test_duplicate_identical_units_map_by_exact_kept_order() -> None:
    state, prior = _setup()
    claim = "6455 sayılı Kanun m.11 uyarınca ceza uygulanmaz."
    draft = claim + "\n\n" + source_gap_sentence(prior, "not_found") + "\n\n" + claim
    raw = assertion_inventory(draft)
    gap = authority_gap_with_cropped_disclosures(draft, state, _check)
    assert gap is not None
    targets = extract_repair_targets(gap, draft, EvidenceLedger())
    assert targets[0].unit_ids == [raw[0]["unit_id"], raw[2]["unit_id"]]


@pytest.mark.parametrize("kind", ["unit_id", "unit_ids", "answer_unit_ids"])
def test_nested_diagnostic_unit_fields_map_without_mutating_checker_result(
    kind: str,
) -> None:
    state, prior = _setup()
    draft = source_gap_sentence(prior, "not_found") + "\n\nActual claim."
    projected = assertion_inventory("Actual claim.")[0]["unit_id"]
    raw_id = assertion_inventory(draft)[1]["unit_id"]
    value: JsonValue = projected if kind == "unit_id" else [projected]
    diagnostic: dict[str, JsonValue] = {"nested": [{kind: value}]}
    result = authority_gap_with_cropped_disclosures(
        draft, state, lambda _text: diagnostic
    )
    assert result == {"nested": [{kind: raw_id if kind == "unit_id" else [raw_id]}]}
    assert diagnostic == {"nested": [{kind: value}]}


def test_ambiguous_repartition_uses_full_raw_checker_instead_of_guessing() -> None:
    state, prior = _setup()
    draft = (
        "Aşağıdaki koşul:\n\n"
        + source_gap_sentence(prior, "not_found")
        + "\n\n- 6455 sayılı Kanun m.11 uyarınca ceza uygulanmaz."
    )
    checker = MagicMock(side_effect=_check)
    result = authority_gap_with_cropped_disclosures(draft, state, checker)
    assert result is not None
    assert checker.call_count == 1
    assert checker.call_args.args == (draft,)


def test_stale_diagnostic_id_falls_back_to_full_raw_check_and_remains_rejected() -> (
    None
):
    state, prior = _setup()
    draft = (
        source_gap_sentence(prior, "not_found")
        + "\n\n6455 sayılı Kanun m.11 uyarınca ceza uygulanmaz."
    )
    stale = {
        "named_authority_gaps": [
            {"unit_id": "au0-stale", "instrument_number": "6455", "article": "11"}
        ]
    }
    checker = MagicMock(side_effect=[stale, _check(draft)])
    result = authority_gap_with_cropped_disclosures(draft, state, checker)
    assert result is not None
    assert checker.call_count == 2
    assert checker.call_args.args == (draft,)
    targets = extract_repair_targets(result, draft, EvidenceLedger())
    assert targets[0].unit_ids == [assertion_inventory(draft)[1]["unit_id"]]


def test_true_exhausted_unknown_after_prior_notice_projects_only_actual_claim() -> None:
    context = RunContext(services={"asv3_exp_cropped": True})
    state = CroppedRepairState(context, "Question")
    prior = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="10"
    )
    _record(state, prior)
    supported = "Başvuruda belge sunulur [1]."
    draft = (
        supported
        + "\n\n"
        + source_gap_sentence(prior, "not_found")
        + "\n\n6455 sayılı Kanun m.11 uyarınca ceza uygulanmaz."
    )
    gap = authority_gap_with_cropped_disclosures(draft, state, _check)
    assert gap is not None
    target = extract_repair_targets(gap, draft, EvidenceLedger())[0]
    _record(state, target)

    def guard(text: str) -> ToolOutcome | None:
        current = authority_gap_with_cropped_disclosures(text, state, _check)
        return (
            ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Actual governing gap",
                data=current,
            )
            if current
            else None
        )

    harness = Harness(
        request="Question",
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _view: Decision(),
        draft_guard=guard,
    )
    harness.last_draft = draft
    decision = CroppedFocusedRepair(state)._exhausted_decision(harness)
    assert decision is not None
    assert decision.calls[0].name == "submit_partial_answer"
    assert decision.calls[0].arguments[
        "answer"
    ] == supported + "\n\n" + source_gap_sentence(
        prior, "not_found"
    ) + "\n\n" + source_gap_sentence(target, "not_found")


def test_actual_5315_disclaimer_and_owned_notice_keep_complete_partial_body() -> None:
    context, ledger, explanation = captured()
    state = CroppedRepairState(context, "Captured 5315 question")
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="4925", article="13"
    )
    _record(state, target)
    draft = source_gap_sentence(target, "not_found") + "\n\n" + explanation

    def guard(text: str) -> ToolOutcome | None:
        current = authority_gap_with_cropped_disclosures(
            text,
            state,
            lambda body: cropped_named_authority_gap(body, ledger, context, **OPTIONS),
        )
        return (
            ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Actual governing gap",
                data=current,
            )
            if current
            else None
        )

    harness = Harness(
        request=state.request,
        context=context,
        registry=CapabilityRegistry(),
        evidence=ledger,
        decide=lambda _view: Decision(),
        draft_guard=guard,
    )
    harness.last_draft = draft
    harness.receipts = [
        ToolReceipt(
            call=CapabilityCall(
                name="submit_partial_answer", arguments={"answer": draft}
            ),
            outcome=ToolOutcome(status=OutcomeStatus.PARTIAL, summary="Historical gap"),
            elapsed_seconds=0,
        )
    ]
    before = ledger.export()
    decision = CroppedFocusedRepair(state)._exhausted_decision(harness)
    assert decision is not None
    assert decision.calls[0].name == "submit_partial_answer"
    assert decision.calls[0].arguments["answer"] == draft
    assert explanation in draft
    assert ledger.export() == before
