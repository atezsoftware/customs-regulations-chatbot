import copy
import json
from datetime import date

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, SharedBudget
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.session_research import (
    retain_session_research,
    session_research_checkpoint,
)
from onyx.db.asv3_corpus import CorpusScopeUnavailable


def previous(*items: EvidenceItem) -> dict[str, JsonValue]:
    return {
        "request": "İlk sorunun tam olguları",
        "scope": {"document_set": 15},
        "session_research": {"requests": ["Daha eski senaryo"]},
        "evidence": {
            "records": [
                {"citation": index, "item": item.model_dump(mode="json")}
                for index, item in enumerate(items, 1)
            ],
        },
        "workers": {"do_not_restore": True},
        "budget": {"decisions": 10},
    }


def previous_with_outcomes() -> dict[str, JsonValue]:
    original = EvidenceItem(
        source_id="law", chunk_id="approval", text="An operative approval requirement."
    )
    context, ledger = RunContext(scope={"document_set": 15}), EvidenceLedger()
    ledger.add([original], context)
    outcomes = OutcomeMap(["Release?", "Remaining question?"], context)
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [
                    {
                        "outcome_id": "release",
                        "question_ids": ["q0"],
                        "detail": "Release after approval",
                    },
                    {
                        "outcome_id": "remaining",
                        "question_ids": ["q1"],
                        "detail": "Separate remaining result",
                    },
                ],
                "conditions": [
                    {
                        "condition_id": "approval",
                        "outcome_ids": ["release"],
                        "detail": "Approval is a separate step",
                        "witnesses": [{"citation": 1, "start_char": 0, "end_char": 33}],
                    }
                ],
                "resolutions": [
                    {
                        "outcome_id": "release",
                        "status": "supported",
                        "condition_ids": ["approval"],
                        "evidence_numbers": [1],
                    },
                    {
                        "outcome_id": "remaining",
                        "status": "unresolved",
                        "gap": "The subsequent effect has not been established.",
                    },
                ],
            }
        ),
        ledger,
    )
    context.services["outcome_map"] = outcomes
    snapshot = previous(original)
    snapshot["session_research"] = session_research_checkpoint(context, "Release?")
    return snapshot


def navigation_of(context: RunContext) -> dict[str, JsonValue]:
    memory = context.services["session_research"]
    assert isinstance(memory, dict)
    navigation = memory["prior_outcomes"]
    assert isinstance(navigation, dict)
    return navigation


def test_revalidated_session_originals_keep_full_text_and_fresh_run_state() -> None:
    item = EvidenceItem(
        source_id="law",
        chunk_id="provision",
        text="Özgün hüküm\n" * 500,
        question_ids=["Eski soru"],
    )
    snapshot = previous(item)
    unchanged = copy.deepcopy(snapshot)
    context = RunContext(
        scope={"document_set": 15},
        timeout_seconds=float("inf"),
        budget=SharedBudget(unlimited_execution=True),
    )
    ledger = EvidenceLedger()
    checked: list[EvidenceItem] = []
    retain_session_research(
        snapshot, context, ledger, lambda items, _context: checked.extend(items)
    )
    retained = ledger.get(1)
    assert retained is not None
    assert retained.text == item.text and retained.text_hash == item.text_hash
    assert retained.question_ids == []
    assert checked == [retained]
    assert ledger.completely_delivered("new-model-call") == set()
    assert context.budget.snapshot()["decisions"] == 0
    assert "workers" not in context.services
    assert snapshot == unchanged
    memory = session_research_checkpoint(context, "Yeni devam sorusu")
    assert memory["requests"] == [
        "Daha eski senaryo",
        "İlk sorunun tam olguları",
        "Yeni devam sorusu",
    ]
    assert memory["reused_evidence_numbers"] == [1]


@pytest.mark.parametrize("error", [PermissionError, CorpusScopeUnavailable])
def test_unavailable_session_source_does_not_hide_other_authorized_originals(
    error: type[Exception],
) -> None:
    denied = EvidenceItem(source_id="denied", chunk_id="old", text="Erişilemeyen metin")
    allowed = EvidenceItem(
        source_id="allowed", chunk_id="current", text="Geçerli tam hüküm"
    )
    context, ledger = RunContext(scope={"document_set": 15}), EvidenceLedger()

    def revalidate(items: list[EvidenceItem], _context: RunContext) -> None:
        if items[0].source_id == "denied":
            raise error("Unavailable")

    retain_session_research(previous(denied, allowed), context, ledger, revalidate)
    assert ledger.citation_numbers() == (1,)
    assert ledger.get(1) == allowed
    assert context.services["session_research"]["source_gaps"] == [
        {
            "source_id": "denied",
            "chunk_id": "old",
            "status": "retained_original_unavailable",
        }
    ]


def test_changed_scope_and_external_originals_require_new_authorized_research() -> None:
    context, ledger = RunContext(scope={"document_set": 16}), EvidenceLedger()

    def unexpected(_items: list[EvidenceItem], _context: RunContext) -> None:
        raise AssertionError("Outside captured scope")

    snapshot = previous(
        EvidenceItem(source_id="outside", text="External", metadata={"external": True})
    )
    retain_session_research(snapshot, context, ledger, unexpected)
    assert ledger.citation_numbers() == ()
    assert context.services["session_research"]["status"] == "scope_changed"
    context.scope = {"document_set": 15}
    retain_session_research(snapshot, context, ledger, unexpected)
    assert ledger.citation_numbers() == ()
    assert context.services["session_research"]["source_gaps"] == [
        {"source_id": "outside", "status": "fresh_authorized_read_required"}
    ]


def test_changed_retained_hash_is_not_accepted_as_an_original() -> None:
    snapshot = previous(EvidenceItem(source_id="law", text="Original"))
    evidence = snapshot["evidence"]
    assert isinstance(evidence, dict)
    records = evidence["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    item = records[0]["item"]
    assert isinstance(item, dict)
    item["text"] = "Changed"
    context, ledger = RunContext(scope={"document_set": 15}), EvidenceLedger()
    retain_session_research(snapshot, context, ledger, lambda _items, _context: None)
    assert ledger.citation_numbers() == ()
    assert context.services["session_research"]["source_gaps"] == [
        {"status": "invalid_retained_original"}
    ]


@pytest.mark.parametrize("historical", [None, "2020-01-01"])
def test_followup_checks_the_current_or_explicit_historical_snapshot(
    historical: str | None,
) -> None:
    item = EvidenceItem(
        source_id="law",
        chunk_id="provision",
        text="Unchanged operative text",
        metadata={"read_as_of_date": "2019-01-01"},
    )
    snapshot = previous(item)
    scope: dict[str, JsonValue] = {"document_set": 15, "as_of_date": historical}
    snapshot["scope"] = scope
    context, ledger = RunContext(scope=scope), EvidenceLedger()
    expected = historical or date.today().isoformat()

    def revalidate(items: list[EvidenceItem], _context: RunContext) -> None:
        assert items[0].metadata["read_as_of_date"] == expected

    retain_session_research(snapshot, context, ledger, revalidate)
    retained = ledger.get(1)
    assert retained is not None and retained.metadata["read_as_of_date"] == expected
    assert item.metadata["read_as_of_date"] == "2019-01-01"


def test_one_changed_provision_does_not_discard_other_originals_in_the_same_source() -> (
    None
):
    changed = EvidenceItem(source_id="law", chunk_id="changed", text="Old rule")
    unchanged = EvidenceItem(source_id="law", chunk_id="unchanged", text="Valid rule")
    context, ledger = RunContext(scope={"document_set": 15}), EvidenceLedger()

    def revalidate(items: list[EvidenceItem], _context: RunContext) -> None:
        if changed in items:
            raise CorpusScopeUnavailable("This provision changed")

    retain_session_research(previous(changed, unchanged), context, ledger, revalidate)
    assert ledger.get(1) == unchanged
    assert ledger.citation_numbers() == (1,)
    assert context.services["session_research"]["source_gaps"] == [
        {
            "source_id": "law",
            "chunk_id": "changed",
            "status": "retained_original_unavailable",
        }
    ]


def test_followup_remaps_prior_witnesses_without_carrying_supported_completion() -> (
    None
):
    snapshot = previous_with_outcomes()
    context, ledger = RunContext(scope={"document_set": 15}), EvidenceLedger()
    ledger.add([EvidenceItem(source_id="unrelated", text="A new original")], context)
    before = context.budget.snapshot()
    retain_session_research(snapshot, context, ledger, lambda _items, _context: None)
    navigation = navigation_of(context)
    conditions = navigation["conditions"]
    assert isinstance(conditions, list) and len(conditions) == 1
    condition = conditions[0]
    assert isinstance(condition, dict)
    assert condition["witnesses"] == [{"citation": 2, "start_char": 0, "end_char": 33}]
    assert navigation["open_gaps"] == [
        {
            "outcome_id": "remaining",
            "gap": "The subsequent effect has not been established.",
        }
    ]
    serialized = json.dumps(navigation)
    assert "supported" not in serialized and "question_ids" not in serialized
    assert "An operative approval requirement." not in serialized
    assert "resolutions" not in navigation
    assert context.budget.snapshot()["decisions"] == before["decisions"]
    new_outcomes = OutcomeMap(["New applicability?"], context)
    assert new_outcomes.outcome_ids() == []
    assert new_outcomes.view()["resolutions"] == []
    # A conversational turn preserves navigation using this run's new numbers.
    remembered = session_research_checkpoint(context, "Hello")
    assert remembered["outcome_navigation"] == navigation


@pytest.mark.parametrize("changed_scope", [False, True])
def test_prior_source_conditions_require_revalidated_scope_and_originals(
    changed_scope: bool,
) -> None:
    snapshot = previous_with_outcomes()
    context = RunContext(scope={"document_set": 16 if changed_scope else 15})
    ledger = EvidenceLedger()
    checks: list[str] = []

    def unavailable(_items: list[EvidenceItem], _context: RunContext) -> None:
        checks.append("checked")
        raise PermissionError("Denied")

    retain_session_research(snapshot, context, ledger, unavailable)
    navigation = navigation_of(context)
    assert navigation["conditions"] == []
    assert navigation["open_gaps"] == [
        {
            "outcome_id": "remaining",
            "gap": "The subsequent effect has not been established.",
        }
    ]
    assert checks if not changed_scope else not checks
    assert ledger.citation_numbers() == ()


@pytest.mark.parametrize("tamper", ["hash", "range"])
def test_changed_or_invalid_prior_original_binding_is_omitted(tamper: str) -> None:
    snapshot = previous_with_outcomes()
    if tamper == "hash":
        evidence = snapshot["evidence"]
        assert isinstance(evidence, dict)
        records = evidence["records"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        item = records[0]["item"]
        assert isinstance(item, dict)
        item["text"] = "A changed original"
    else:
        remembered = snapshot["session_research"]
        assert isinstance(remembered, dict)
        navigation = remembered["outcome_navigation"]
        assert isinstance(navigation, dict)
        conditions = navigation["conditions"]
        assert isinstance(conditions, list) and isinstance(conditions[0], dict)
        witnesses = conditions[0]["witnesses"]
        assert isinstance(witnesses, list) and isinstance(witnesses[0], dict)
        witnesses[0]["end_char"] = 999
    context, ledger = RunContext(scope={"document_set": 15}), EvidenceLedger()
    retain_session_research(snapshot, context, ledger, lambda _items, _context: None)
    assert navigation_of(context)["conditions"] == []
    assert navigation_of(context)["open_gaps"]
