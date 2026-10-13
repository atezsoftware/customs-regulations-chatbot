"""Changed shared facts require a fresh topic assessment, not another source read."""

import copy

import pytest

from onyx.asv3.cropped_task_packets import packet_for_view
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext, ToolReceipt
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import (
    ResearchState,
    ResearchUpdate,
    build_research_specs,
)
from tests.unit.onyx.asv3.test_cropped_task_packets import (
    finding,
    fixture,
    mapping,
    need,
    receipt,
    rows,
    view,
)

CHANGED_FACTS = ["Shared scenario date changed to 2025"]


def _parked_topic(
    status: str,
) -> tuple[RunContext, EvidenceLedger, ResearchState, list[ToolReceipt]]:
    context, ledger, state, _store = fixture(2)
    current_need = need("B", complete=status == "candidate").model_copy(
        update={"status": status}
    )
    state.update(
        ResearchUpdate(
            needs=[need("A"), current_need],
            findings=[finding("B", 2, ledger)] if status == "candidate" else [],
            active_need_ids=["A"],
        ),
        ledger,
    )
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    first = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert first is not None
    assert mapping(rows(first[1], "detached_need_index")[0]["need"])["need_id"] == "B"
    changed = view(ledger, receipts, facts=CHANGED_FACTS)
    assert packet_for_view(changed, receipts, state, context, ledger) is None
    assert packet_for_view(changed, receipts, state, context, ledger) is None
    return context, ledger, state, receipts


def _assess(
    context: RunContext,
    ledger: EvidenceLedger,
    state: ResearchState,
    status: str,
    *,
    new_finding: bool,
) -> ToolReceipt:
    registry = CapabilityRegistry()
    for spec in build_research_specs(state, ledger):
        registry.register(spec)
    assessed_need = need("B", complete=status == "candidate").model_copy(
        update={"status": status}
    )
    update = ResearchUpdate(
        needs=[assessed_need],
        findings=(
            [finding("B", 2, ledger, suffix=" Revised scenario explicitly reassessed.")]
            if new_finding
            else [finding("B", 2, ledger)]
            if status == "candidate"
            else []
        ),
    )
    call = CapabilityCall(
        name="update_research",
        call_id="actual-current-fact-reassessment",
        arguments=update.model_dump(mode="json", exclude_none=True),
    )
    outcome = registry.dispatch(call, context)
    assert outcome.status == OutcomeStatus.FOUND
    return ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0)


@pytest.mark.parametrize("status", ["candidate", "open"])
def test_changed_facts_plain_reread_does_not_reseal_old_topic(status: str) -> None:
    context, ledger, state, receipts = _parked_topic(status)
    before = copy.deepcopy(state.export())
    reread = receipt(2, "B", ledger, call_id="actual-same-original-reread")
    actual = [*receipts, reread]
    result = packet_for_view(
        view(ledger, actual, facts=CHANGED_FACTS), actual, state, context, ledger
    )
    assert state.export() == before
    assert result is None, (
        "A new read receipt cannot discharge changed-fact invalidation"
    )


@pytest.mark.parametrize("status", ["candidate", "open"])
def test_actual_reassessment_can_retain_same_finding_under_new_facts(
    status: str,
) -> None:
    context, ledger, state, receipts = _parked_topic(status)
    old_findings = copy.deepcopy(state.export()["findings"])
    reassessed = _assess(context, ledger, state, status, new_finding=False)
    assert state.export()["findings"] == old_findings
    actual = [*receipts, reassessed]
    result = packet_for_view(
        view(ledger, actual, facts=CHANGED_FACTS), actual, state, context, ledger
    )
    assert result is not None
    parked = rows(result[1], "detached_need_index")[0]
    assert mapping(parked["need"])["need_id"] == "B"
    assert mapping(parked["need"])["status"] == status
    receipt_ids = parked["receipt_ids"]
    assert isinstance(receipt_ids, list)
    assert reassessed.call.call_id in receipt_ids
    assert result[1]["legal_approval"] is False


@pytest.mark.parametrize("status", ["candidate", "open"])
def test_actual_new_source_bound_finding_reassesses_changed_facts(status: str) -> None:
    context, ledger, state, receipts = _parked_topic(status)
    old_findings = copy.deepcopy(state.export()["findings"])
    reassessed = _assess(context, ledger, state, status, new_finding=True)
    assert state.export()["findings"] != old_findings
    actual = [*receipts, reassessed]
    result = packet_for_view(
        view(ledger, actual, facts=CHANGED_FACTS), actual, state, context, ledger
    )
    assert result is not None
    parked = rows(result[1], "detached_need_index")[0]
    assert mapping(parked["need"])["status"] == status
    assert result[1]["legal_approval"] is False
    assert ledger.citation_numbers() == (1, 2)
