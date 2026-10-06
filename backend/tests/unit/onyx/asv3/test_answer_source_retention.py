"""Selected source coverage is independent of titles and unselected reads."""

import json

import pytest

from onyx.asv3.answer_source_retention import selected_outcome_source_gap
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from tests.unit.onyx.asv3.test_outcome_map import (
    condition,
    map_pair,
    outcome,
    resolution,
)


def setup(
    *, status: str = "supported", mode: str = "parallel"
) -> tuple[RunContext, EvidenceLedger, OutcomeMap]:
    context, ledger, state = map_pair()
    context.services.update(research_profile="experimental", experimental_parallel=True)
    if mode == "hosted":
        context.services.update(
            experimental_parallel=False,
            serial_session_diagnostics=True,
            lean_native_mode=True,
            task_id="owned-task",
        )
    elif mode == "plain":
        context.services["experimental_parallel"] = False
    context.services["outcome_map"] = state
    second = {
        **condition("exception"),
        "detail": "A second material source qualifies this effect.",
        "witnesses": [{"citation": 2, "start_char": 0, "end_char": 21}],
    }
    state.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [outcome()],
                "conditions": [condition(), second],
                "resolutions": [
                    resolution(
                        status=status,
                        condition_ids=["approval", "exception"],
                        evidence_numbers=[1, 2],
                        gap="The later effect remains unresolved."
                        if status == "unresolved"
                        else "",
                    )
                ],
            }
        ),
        ledger,
    )
    ledger.record_delivery(
        "actual-call", "asv3_coordinator", json.loads(ledger.serialize_records([1, 2]))
    )
    return context, ledger, state


@pytest.mark.parametrize("mode", ["parallel", "hosted"])
@pytest.mark.parametrize("status", ["supported", "conditional"])
def test_missing_one_selected_material_source_returns_exact_repair_bindings(
    mode: str, status: str
) -> None:
    context, ledger, state = setup(mode=mode, status=status)
    before, budget = state.export(), context.budget.snapshot()
    gap = selected_outcome_source_gap(
        "The result remains qualified [1].", "actual-call", context, ledger, state
    )
    assert gap is not None and gap.status == OutcomeStatus.PARTIAL
    assert gap.data["outcome_ids"] == ["release"]
    assert gap.data["condition_ids"] == ["exception"]
    assert gap.data["missing_inline_citations"] == [2]
    assert gap.data["undelivered_citations"] == []
    rows = gap.data["retained_outcome_sources"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["required_evidence_numbers"] == [1, 2]
    witnesses = rows[0]["witnesses"]
    assert isinstance(witnesses, list) and isinstance(witnesses[1], dict)
    item = ledger.get(2)
    assert item is not None
    assert witnesses[1] == {
        "condition_id": "exception",
        "citation": 2,
        "start_char": 0,
        "end_char": 21,
        "text_hash": item.text_hash,
    }
    assert state.export() == before and context.budget.snapshot() == budget


def test_complete_selected_sources_pass_without_requiring_unrelated_read() -> None:
    context, ledger, state = setup()
    ledger.add(
        [
            EvidenceItem(
                source_id="unrelated", chunk_id="unrelated", text="An unrelated rule."
            )
        ],
        context,
    )
    ledger.record_delivery(
        "actual-call",
        "asv3_coordinator",
        json.loads(ledger.serialize_records([1, 2, 3])),
    )
    assert (
        selected_outcome_source_gap(
            "The first condition applies [1]; the second qualifies it [2].",
            "actual-call",
            context,
            ledger,
            state,
        )
        is None
    )


@pytest.mark.parametrize("call_id", [None, "earlier-call", "partial-current-call"])
def test_inline_citation_does_not_replace_current_complete_delivery(
    call_id: str | None,
) -> None:
    context, ledger, state = setup()
    ledger.record_delivery(
        "partial-current-call",
        "asv3_coordinator",
        json.loads(ledger.serialize_records([1])),
    )
    gap = selected_outcome_source_gap(
        "The effect uses both [1][2].", call_id, context, ledger, state
    )
    assert gap is not None and gap.status == OutcomeStatus.PARTIAL
    assert gap.data["missing_inline_citations"] == []
    assert gap.data["undelivered_citations"] == (
        [2] if call_id == "partial-current-call" else [1, 2]
    )


@pytest.mark.parametrize(
    "kind", ["empty", "declaration", "unassessed_condition", "unresolved"]
)
def test_no_positive_resolution_creates_no_new_obligation_or_assessment(
    kind: str,
) -> None:
    if kind == "unresolved":
        context, ledger, state = setup(status="unresolved")
    else:
        context, ledger, state = map_pair()
        context.services.update(
            research_profile="experimental", experimental_parallel=True
        )
        payload = {} if kind == "empty" else {"outcomes": [outcome()]}
        if kind == "unassessed_condition":
            payload["conditions"] = [condition()]
        state.update(OutcomeUpdate.model_validate(payload), ledger)
    before = state.export()
    assert (
        selected_outcome_source_gap(
            "A precise unresolved effect.", None, context, ledger, state
        )
        is None
    )
    assert state.export() == before
    if kind == "unresolved":
        assert state.view()["resolutions"] == before["resolutions"]


@pytest.mark.parametrize(
    "override",
    [
        {"experimental_parallel": False},
        {"research_profile": "normal"},
        {"research_profile": "deep"},
        {"experimental_parallel": "true"},
    ],
)
def test_protected_profiles_ignore_this_parallel_only_retention_check(
    override: dict[str, object],
) -> None:
    context, ledger, state = setup()
    context.services.update(override)
    assert (
        selected_outcome_source_gap(
            "Only the first source [1].", None, context, ledger, state
        )
        is None
    )


@pytest.mark.parametrize("mismatch", ["run", "scope"])
def test_selected_outcome_map_cannot_move_to_another_run_or_scope(
    mismatch: str,
) -> None:
    context, ledger, state = setup()
    foreign = RunContext(
        run_id="another-run" if mismatch == "run" else context.run_id,
        scope=context.scope if mismatch == "run" else {"source": "other"},
        services=context.services,
    )
    gap = selected_outcome_source_gap(
        "Both sources [1][2].", "actual-call", foreign, ledger, state
    )
    assert gap is not None and gap.status == OutcomeStatus.DENIED
    assert gap.data["retained_outcome_scope_mismatch"] is True


def test_retained_condition_hash_cannot_silently_rebind_to_changed_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, ledger, state = setup()
    item = ledger.get(2)
    assert item is not None
    real_get = ledger.get

    def changed(number: int) -> EvidenceItem | None:
        return (
            item.model_copy(update={"text_hash": "a" * 64})
            if number == 2
            else real_get(number)
        )

    monkeypatch.setattr(ledger, "get", changed)
    gap = selected_outcome_source_gap(
        "Both selected sources [1][2].", "actual-call", context, ledger, state
    )
    assert gap is not None
    assert gap.data["missing_inline_citations"] == []
    assert gap.data["condition_ids"] == ["exception"]
    assert gap.data["invalid_source_witnesses"]


def test_another_requests_map_cannot_replace_the_bound_map_in_the_same_run() -> None:
    context, ledger, state = setup()
    unrelated = OutcomeMap(["Another request"], context)
    gap = selected_outcome_source_gap(
        "Both selected originals [1][2].", "actual-call", context, ledger, unrelated
    )
    assert gap is not None and gap.status == OutcomeStatus.DENIED
    assert gap.data["retained_outcome_request_mismatch"] is True
    assert context.services["outcome_map"] is state
