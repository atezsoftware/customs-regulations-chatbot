"""Outcome metadata preserves source conditions without making external calls."""

import json

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate


def map_pair() -> tuple[RunContext, EvidenceLedger, OutcomeMap]:
    context = RunContext(scope={"source": "owned"})
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="law", chunk_id="condition", text="Request and approval."
            ),
            EvidenceItem(
                source_id="law", chunk_id="exception", text="An exception applies."
            ),
        ],
        context,
    )
    state = OutcomeMap(
        ["Eligibility and release?", "Applicable sanction?"],
        context,
        factual_context="A prior request was filed. Eligibility and release? Applicable sanction?",
    )
    return context, ledger, state


def update(state: OutcomeMap, ledger: EvidenceLedger, **payload: JsonValue) -> None:
    state.update(OutcomeUpdate.model_validate(payload), ledger)


def outcome(identity: str = "release", question: str = "q0") -> dict[str, JsonValue]:
    return {
        "outcome_id": identity,
        "question_ids": [question],
        "detail": "Whether the subsequent release follows",
        "decisive_facts": ["A prior request was filed."],
    }


def condition(
    identity: str = "approval", outcomes: list[str] | None = None
) -> dict[str, JsonValue]:
    return {
        "condition_id": identity,
        "outcome_ids": outcomes or ["release"],
        "detail": "Approval remains a separate requirement",
        "witnesses": [{"citation": 1, "start_char": 0, "end_char": 21}],
    }


def resolution(**changes: JsonValue) -> dict[str, JsonValue]:
    return {
        "outcome_id": "release",
        "status": "supported",
        "condition_ids": ["approval"],
        "evidence_numbers": [1],
        **changes,
    }


def test_native_metadata_is_optional_and_preserves_conditions_without_calls() -> None:
    context, ledger, state = map_pair()
    counters = context.budget.snapshot()
    assert state.outcome_ids() == []
    update(
        state,
        ledger,
        outcomes=[outcome()],
        conditions=[condition()],
        resolutions=[resolution()],
    )
    assert state.revision == 1
    assert context.budget.snapshot() == counters
    view = state.view(delivered_citations=set())
    assert view["undelivered_evidence_numbers"] == [1]
    assert "Request and approval." not in json.dumps(view)
    assert state.view(delivered_citations={1})["undelivered_evidence_numbers"] == []


@pytest.mark.parametrize(
    "invalid",
    [
        {"outcomes": [outcome(question="q9")]},
        {"outcomes": [{**outcome(), "decisive_facts": ["Approval was granted."]}]},
        {"outcomes": [outcome(), outcome()]},
        {"conditions": [condition()]},
    ],
)
def test_invented_facts_unknown_bindings_and_duplicates_leave_state_unchanged(
    invalid: dict[str, JsonValue],
) -> None:
    _, ledger, state = map_pair()
    before = state.export()
    with pytest.raises(ValueError):
        state.update(OutcomeUpdate.model_validate(invalid), ledger)
    assert state.export() == before


@pytest.mark.parametrize("detailed", [False, True])
def test_literal_fact_feedback_identifies_only_the_invalid_entry_without_relaxing_match(
    detailed: bool,
) -> None:
    context, ledger, _state = map_pair()
    state = OutcomeMap(
        ["Eligibility and release?", "Applicable sanction?"],
        context,
        factual_context="A prior request was filed.",
        detailed_fact_errors=detailed,
    )
    before = state.export()
    invalid = {
        **outcome("sanction", "q1"),
        "decisive_facts": ["A prior request was filed.", "A prior Request was filed."],
    }
    with pytest.raises(ValueError) as error:
        update(state, ledger, outcomes=[outcome(), invalid])
    assert state.export() == before
    message = str(error.value)
    if detailed:
        assert "_outcomes[1].decisive_facts[1]" in message
        assert "outcome_id=sanction" in message
        assert "exact contiguous quote" in message
        assert "Do not normalize or paraphrase" in message
    else:
        assert message == "Outcome facts must quote supplied scenario text"
    update(
        state,
        ledger,
        outcomes=[
            outcome(),
            {**invalid, "decisive_facts": ["A prior request was filed."]},
        ],
    )
    assert state.outcome_ids() == ["release", "sanction"]
    assert state.revision == 1


def test_existing_outcome_cannot_move_but_clarification_reopens_its_resolution() -> (
    None
):
    _, ledger, state = map_pair()
    update(
        state,
        ledger,
        outcomes=[outcome()],
        conditions=[condition()],
        resolutions=[resolution()],
    )
    before = state.export()
    with pytest.raises(ValueError, match="cannot move"):
        update(state, ledger, outcomes=[outcome(question="q1")])
    assert state.export() == before
    update(
        state,
        ledger,
        outcomes=[{**outcome(), "detail": "Does the approved release follow?"}],
    )
    assert state.view()["unassessed_outcome_ids"] == ["release"]
    assert state.preferred_citations() == [1]


@pytest.mark.parametrize(
    "invalid",
    [
        resolution(condition_ids=[]),
        resolution(evidence_numbers=[2]),
        resolution(gap="Governing source not obtained"),
        resolution(status="unresolved", gap=""),
        resolution(evidence_numbers=[999]),
        resolution(condition_ids=["other"]),
    ],
)
def test_completeness_metadata_cannot_drop_a_known_condition_or_hide_a_gap(
    invalid: dict[str, JsonValue],
) -> None:
    _, ledger, state = map_pair()
    update(state, ledger, outcomes=[outcome()], conditions=[condition()])
    before = state.export()
    with pytest.raises(ValueError):
        update(state, ledger, resolutions=[invalid])
    assert state.export() == before
    update(
        state,
        ledger,
        resolutions=[
            resolution(status="unresolved", gap="Approval effect remains unread")
        ],
    )
    assert "Approval effect remains unread" in str(state.view()["resolutions"])


def test_parallel_subsets_share_originals_and_cannot_replace_retained_condition() -> (
    None
):
    _, ledger, state = map_pair()
    update(
        state,
        ledger,
        outcomes=[outcome(), outcome("sanction", "q1")],
        conditions=[condition(outcomes=["release", "sanction"])],
    )
    assert state.outcome_ids(["q1"]) == ["sanction"]
    assert state.preferred_citations(["q1"], outcome_ids=["sanction"]) == [1]
    view = state.view(question_ids=["q1"])
    selected_outcomes, selected_conditions = view["outcomes"], view["conditions"]
    assert isinstance(selected_outcomes, list) and len(selected_outcomes) == 1
    assert isinstance(selected_conditions, list)
    selected_condition = selected_conditions[0]
    assert isinstance(selected_condition, dict)
    assert selected_condition["outcome_ids"] == ["sanction"]
    before = state.export()
    update(state, ledger, conditions=[condition(outcomes=["release"])])
    assert state.export() == before
    with pytest.raises(ValueError, match="cannot be replaced"):
        update(
            state, ledger, conditions=[{**condition(), "detail": "A different rule"}]
        )
    assert state.export() == before
    with pytest.raises(ValueError, match="Unknown outcome"):
        state.view(outcome_ids=["invented"])


def test_new_condition_binding_keeps_already_assessed_sibling_outcome() -> None:
    _, ledger, state = map_pair()
    update(
        state,
        ledger,
        outcomes=[outcome(), outcome("sanction", "q1")],
        conditions=[condition()],
        resolutions=[resolution()],
    )
    update(state, ledger, conditions=[condition(outcomes=["sanction"])])
    assert state.view()["unassessed_outcome_ids"] == ["sanction"]
    assert state.view()["resolutions"] == [resolution(gap="")]


@pytest.mark.parametrize("detailed", [False, True])
def test_condition_feedback_preserves_immutable_record_and_allows_targeted_repair(
    detailed: bool,
) -> None:
    context, ledger, _state = map_pair()
    state = OutcomeMap(
        ["Eligibility and release?", "Applicable sanction?"],
        context,
        factual_context="A prior request was filed.",
        reuse_retained_conditions=detailed,
    )
    update(state, ledger, outcomes=[outcome()], conditions=[condition()])
    before = state.export()
    additional = condition("exception")
    additional["witnesses"] = [{"citation": 2, "start_char": 0, "end_char": 21}]
    modified = {**condition(), "witnesses": additional["witnesses"]}
    with pytest.raises(ValueError) as error:
        update(state, ledger, conditions=[additional, modified])
    assert state.export() == before
    if detailed:
        assert "_coverage.conditions[1]" in str(error.value)
        assert "condition_id=approval" in str(error.value)
        assert "changed fields: witnesses, source_hashes" in str(error.value)
        assert "Omit this retained condition" in str(error.value)
        assert "Preserve the answer edits" in str(error.value)
    else:
        assert str(error.value) == "Retained source conditions cannot be replaced"
    update(
        state,
        ledger,
        conditions=[additional],
        resolutions=[
            resolution(condition_ids=["approval", "exception"], evidence_numbers=[1, 2])
        ],
    )
    assert state.view()["unassessed_outcome_ids"] == []
    retained = state.export()["conditions"]
    previous = before["conditions"]
    assert isinstance(retained, list) and isinstance(previous, list)
    assert retained[0] == previous[0]


def test_tuned_duplicate_condition_keeps_original_witnesses_and_new_resolution() -> (
    None
):
    context, ledger, _state = map_pair()
    state = OutcomeMap(
        ["Eligibility and release?"],
        context,
        factual_context="A prior request was filed.",
        reuse_retained_conditions=True,
    )
    update(state, ledger, outcomes=[outcome()], conditions=[condition()])
    before = state.export()["conditions"]
    additional = {"citation": 2, "start_char": 0, "end_char": 21}
    repeated = condition()
    witnesses = repeated["witnesses"]
    assert isinstance(witnesses, list)
    repeated["witnesses"] = [additional, *witnesses]
    update(
        state,
        ledger,
        conditions=[repeated],
        resolutions=[resolution(evidence_numbers=[1, 2])],
    )
    assert state.export()["conditions"] == before
    assert state.view()["resolutions"] == [resolution(evidence_numbers=[1, 2], gap="")]
    saved = state.export()
    restored = OutcomeMap(
        list(state.questions),
        context,
        factual_context="A prior request was filed.",
        reuse_retained_conditions=True,
    )
    restored.restore(saved, ledger)
    update(restored, ledger, conditions=[repeated])
    assert restored.export() == saved
    unchanged = restored.export()
    for invalid in (
        {**repeated, "detail": "Approval no longer required"},
        {**repeated, "witnesses": [additional]},
        {**repeated, "witnesses": [*witnesses, {**additional, "end_char": 999}]},
    ):
        with pytest.raises(ValueError):
            update(restored, ledger, conditions=[invalid])
        assert restored.export() == unchanged


def test_source_ranges_and_checkpoint_provenance_are_validated_atomically() -> None:
    context, ledger, state = map_pair()
    update(state, ledger, outcomes=[outcome()])
    before = state.export()
    with pytest.raises(ValueError, match="recorded original range"):
        update(
            state,
            ledger,
            conditions=[
                {**condition(), "witnesses": [{"citation": 1, "end_char": 999}]}
            ],
        )
    assert state.export() == before
    update(state, ledger, conditions=[condition()], resolutions=[resolution()])
    saved = state.export()
    restored = OutcomeMap(
        list(state.questions),
        context,
        factual_context="A prior request was filed. Eligibility and release? Applicable sanction?",
    )
    restored.restore(saved, ledger)
    assert restored.export() == saved
    another_request = OutcomeMap(
        list(state.questions), context, factual_context="Changed request"
    )
    with pytest.raises(ValueError, match="request or scope"):
        another_request.restore(saved, ledger)
    changed_ledger = EvidenceLedger()
    changed_ledger.add(
        [
            EvidenceItem(
                source_id="law", chunk_id="condition", text="Altered operative law text"
            )
        ],
        context,
    )
    before = restored.export()
    with pytest.raises(ValueError, match="original identity changed"):
        restored.restore(saved, changed_ledger)
    assert restored.export() == before


def test_boolean_citation_and_extra_metadata_are_rejected() -> None:
    with pytest.raises(ValidationError):
        OutcomeUpdate.model_validate(
            {"resolutions": [resolution(evidence_numbers=[True])]}
        )
    with pytest.raises(ValidationError):
        OutcomeUpdate.model_validate(
            {"outcomes": [{**outcome(), "statute": "Guessed law"}]}
        )
