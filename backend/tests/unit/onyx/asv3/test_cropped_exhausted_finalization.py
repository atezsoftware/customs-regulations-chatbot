"""Exhausted source attempts retain supported text and still require publication checks."""

import copy
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.cropped_article_bindings import cropped_named_authority_gap
from onyx.asv3.cropped_focused_repair import (
    CroppedFocusedRepair,
    source_gap_sentence,
    unresolved_governing_projection,
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
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.registry import CapabilityRegistry
from tests.unit.onyx.asv3.test_cropped_authority_5309 import OPTIONS, captured, setup

_SUPPORTED = "Başvuruda belge sunulur [1]."
_UNSUPPORTED = "6455 sayılı Kanun m.11 uyarınca ceza uygulanmaz."
_DRAFT = _SUPPORTED + "\n\n" + _UNSUPPORTED


def _setup() -> tuple[Harness, CroppedFocusedRepair, CroppedRepairTarget]:
    context = RunContext(services={"asv3_exp_cropped": True})
    ledger = EvidenceLedger()
    ledger.add(
        [EvidenceItem(source_id="source", text="Başvuruda belge sunulur.")], context
    )
    harness = Harness(
        request="Başvuru ve ceza koşulları?",
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    target = CroppedRepairTarget(
        check_kind="governing_original",
        instrument_number="6455",
        article="11",
        unit_ids=[assertion_inventory(_DRAFT)[1]["unit_id"]],
    )
    state = CroppedRepairState(context, harness.request)
    state.reserve(target)
    state.finish(target.key, "not_found", [], "Focused original was not obtained.")
    state.reserve_integration([target], "coordinator")
    harness.last_draft = _DRAFT
    harness.publication_gap = _gap(target)
    return harness, CroppedFocusedRepair(state), target


def _gap(target: CroppedRepairTarget) -> ToolOutcome:
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Obtain the governing original.",
        data={
            "named_authority_gaps": [
                {
                    "unit_id": target.unit_ids[0],
                    "instrument_number": target.instrument_number,
                    "article": target.article,
                }
            ]
        },
    )


def test_exhausted_projection_preserves_supported_units_and_checks_whole_answer() -> (
    None
):
    harness, repair, target = _setup()
    notice = source_gap_sentence(target, "not_found")
    expected = _SUPPORTED + "\n\n" + notice
    guard = MagicMock(
        side_effect=lambda answer: None if answer == expected else _gap(target)
    )
    harness.draft_guard = guard
    decision = repair._exhausted_decision(harness)
    assert decision is not None
    assert decision.calls[0].name == "submit_partial_answer"
    assert decision.calls[0].arguments == {"answer": expected}
    assert [call.args[0] for call in guard.call_args_list] == [_DRAFT, expected]
    assert repair.state.reserve(target) is False
    assert repair.state.reserve_integration([target], "coordinator") is False


def test_unblocked_retained_partial_stays_exact_and_partial() -> None:
    harness, repair, _target = _setup()
    harness.draft_guard = MagicMock(return_value=None)
    harness.receipts = [
        ToolReceipt(
            call=CapabilityCall(
                name="submit_partial_answer", arguments={"answer": _DRAFT}
            ),
            outcome=ToolOutcome(status=OutcomeStatus.PARTIAL, summary="Rejected"),
            elapsed_seconds=0,
        )
    ]
    decision = repair._exhausted_decision(harness)
    assert decision is not None
    assert decision.calls[0].name == "submit_partial_answer"
    assert decision.calls[0].arguments["answer"] == _DRAFT


def test_projection_cannot_override_unrelated_publication_defect() -> None:
    harness, repair, target = _setup()
    guard = MagicMock(
        side_effect=[
            _gap(target),
            ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="An independent assertion remains unsupported.",
                data={"unknown_citations": [999]},
            ),
        ]
    )
    harness.draft_guard = guard
    assert repair._exhausted_decision(harness) is None
    assert guard.call_count == 2


@pytest.mark.parametrize("change", ["found", "generic", "stale", "only_gap"])
def test_projection_requires_unresolved_mapped_governing_units_and_supported_body(
    change: str,
) -> None:
    harness, repair, target = _setup()
    records = repair.state.records_for([target])
    draft = _DRAFT
    if change == "found":
        records[0]["status"] = "found"
        records[0]["evidence_numbers"] = [1]
    elif change == "generic":
        target = target.model_copy(update={"check_kind": "generic:unsupported"})
        records[0]["key"] = target.key
    elif change == "stale":
        target = target.model_copy(update={"unit_ids": ["au1-stale"]})
    else:
        draft = _UNSUPPORTED
        target = target.model_copy(
            update={"unit_ids": [assertion_inventory(draft)[0]["unit_id"]]}
        )
    assert unresolved_governing_projection(draft, [target], records) is None


@pytest.mark.parametrize("tamper", ["hash", "call", "draft"])
def test_cropped_restored_delivery_cannot_change_draft_or_claim_a_new_model_call(
    tamper: str,
) -> None:
    harness, _repair, _target = _setup()
    item = harness.evidence.get(1)
    assert item is not None
    rows: list[dict[str, JsonValue]] = [
        {
            "citation": 1,
            "source_id": "source",
            "chunk_id": None,
            "text_hash": item.text_hash,
            "text": "Başvuruda belge sunulur.",
        }
    ]
    harness.evidence.record_delivery("existing-call", "asv3_coordinator", rows)
    harness.context.services["last_model_call_id"] = "existing-call"
    saved = copy.deepcopy(harness.snapshot())
    binding = saved["cropped_draft_delivery"]
    assert isinstance(binding, dict)
    if tamper == "hash":
        binding["draft_sha256"] = "0" * 64
    elif tamper == "call":
        binding["model_call_id"] = "invented-call"
    else:
        saved["last_draft"] = _SUPPORTED
    with pytest.raises(ValueError, match="Cropped draft delivery checkpoint changed"):
        harness.restore(saved)


def test_legacy_cropped_checkpoint_does_not_invent_draft_delivery() -> None:
    harness, _repair, _target = _setup()
    saved = harness.snapshot()
    assert "cropped_draft_delivery" not in saved
    harness.context.services["last_model_call_id"] = "unbound-call"
    harness.restore(saved)
    assert "last_model_call_id" not in harness.context.services


def test_captured_5309_recheck_keeps_complete_units_without_new_model_or_reads() -> (
    None
):
    context, ledger, bindings, receipt = setup()
    draft = "\n\n".join(captured()[1])
    gap = canonical_named_authority_gap(draft, ledger, **OPTIONS)
    assert gap is not None
    targets = extract_repair_targets(gap, draft, ledger)
    state = CroppedRepairState(context, "Captured 5309 question")
    for target in targets:
        state.reserve(target)
        state.finish(
            target.key,
            "not_found",
            [],
            "Historical identity check missed the original.",
        )
    state.reserve_integration(targets, "coordinator")
    bindings.record_receipt(receipt, context, ledger)

    def guard(answer: str) -> ToolOutcome | None:
        current = cropped_named_authority_gap(answer, ledger, context, **OPTIONS)
        return (
            ToolOutcome(
                status=OutcomeStatus.PARTIAL, summary="Source identity", data=current
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
    harness.publication_gap = ToolOutcome(
        status=OutcomeStatus.PARTIAL, summary="Historical source identity", data=gap
    )
    harness.receipts = [
        ToolReceipt(
            call=CapabilityCall(
                name="submit_partial_answer", arguments={"answer": draft}
            ),
            outcome=harness.publication_gap,
            elapsed_seconds=0,
        )
    ]
    model = MagicMock()
    decision = CroppedFocusedRepair(state).decide(harness, model)
    assert decision.calls[0].name == "submit_partial_answer"
    assert decision.calls[0].arguments["answer"] == draft
    assert model.mock_calls == []
    assert len(harness.receipts) == 1
