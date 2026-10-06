"""Keep task-owned serial publication proofs while sibling memory advances."""

import copy
import hashlib
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.asv3.serial_experimental_session import (
    accepted_serial_memory_state,
    merge_serial_session_memory,
    serial_session_history,
)
from onyx.asv3.session_research import (
    retain_session_research,
    session_research_checkpoint,
)
from onyx.configs.constants import MessageType
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    FOCUS,
    SCENARIO,
    outer,
    serial_run,
    session,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def original() -> EvidenceItem:
    return EvidenceItem(
        source_id="canonical-source",
        chunk_id="canonical-chunk",
        text="The operative original has a condition and a continuation.",
        metadata={
            "source_sha256": "original-source-sha",
            "canonical_metadata": {
                "document_type": "law",
                "heading_path": ["Example Law", "MADDE 27"],
                "publication_revision_id": "revision-a",
                "validity_start": "2026-01-01",
            },
            "index": {
                "query_index_uuid": "query-index-a",
                "payload_sha256": "payload-a",
            },
            "provenance": {"binding_sha256": "binding-a", "external": False},
        },
    )


def records(ledger: EvidenceLedger, number: int = 1) -> list[dict[str, JsonValue]]:
    item = ledger.get(number)
    assert item is not None
    return [{"citation": number, "text": item.text}]


@pytest.mark.parametrize("role", [MessageType.USER.value, "USER"])
def test_full_final_scenario_is_kept_byte_exact_once(role: str) -> None:
    history = f"assistant: Önceki bağlam.\n{role}: {SCENARIO}"
    assert serial_session_history(history, SCENARIO) == history
    assert serial_session_history(history, SCENARIO).count(SCENARIO) == 1
    assert serial_session_history("", SCENARIO) == "USER: " + SCENARIO
    old_occurrence = f"{role}: {SCENARIO}\nassistant: Yeni bir cevap."
    assert (
        serial_session_history(old_occurrence, SCENARIO)
        == old_occurrence + "\nUSER: " + SCENARIO
    )


@pytest.mark.parametrize(
    "container,field,value",
    [
        (None, "source_sha256", "source-sha-changed"),
        ("canonical_metadata", "publication_revision_id", "revision-b"),
        ("canonical_metadata", "validity_start", "2026-02-01"),
        ("canonical_metadata", "document_type", "guidance"),
        ("canonical_metadata", "heading_path", ["Another Law", "MADDE 28"]),
        ("index", "query_index_uuid", "query-index-b"),
        ("index", "payload_sha256", "payload-b"),
        ("provenance", "binding_sha256", "binding-b"),
        ("provenance", "external", True),
    ],
)
def test_resume_rejects_changed_canonical_provenance_without_mutating_siblings(
    monkeypatch: pytest.MonkeyPatch, container: str | None, field: str, value: JsonValue
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    ledger = EvidenceLedger()
    envelope = outer(run)
    ledger.add([original()], envelope)
    child = session(run, envelope, ledger)
    snapshot = child.snapshot()
    current = ledger.export()
    rows = cast(list[dict[str, JsonValue]], current["records"])
    item = cast(dict[str, JsonValue], rows[0]["item"])
    metadata = cast(dict[str, JsonValue], item["metadata"])
    target = (
        metadata
        if container is None
        else cast(dict[str, JsonValue], metadata[container])
    )
    target[field] = value
    ledger.restore(current, envelope)
    unchanged = copy.deepcopy(ledger.export())
    restored = session(run, envelope, ledger)
    try:
        with pytest.raises(ValueError, match="provenance or version"):
            restored.restore(snapshot)
        assert ledger.export() == unchanged
    finally:
        child.workers.close()
        restored.workers.close()


def test_resume_keeps_new_sibling_originals_pins_and_safe_metadata_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    ledger = EvidenceLedger()
    envelope = outer(run)
    ledger.add([original()], envelope)
    ledger.record_delivery("old-call", "coordinator", records(ledger))
    child = session(run, envelope, ledger)
    snapshot = child.snapshot()
    current = ledger.export()
    rows = cast(list[dict[str, JsonValue]], current["records"])
    item = cast(dict[str, JsonValue], rows[0]["item"])
    metadata = cast(dict[str, JsonValue], item["metadata"])
    metadata.update(
        asv3_citation_preview_url="owned-preview",
        article_closure_complete=True,
        article_closure_remaining_count=0,
    )
    ledger.restore(current, envelope)
    sibling = original().model_copy(
        update={
            "source_id": "sibling-source",
            "chunk_id": "sibling-chunk",
            "text": "A new sibling source.",
            "text_hash": "",
        }
    )
    sibling = EvidenceItem.model_validate(sibling.model_dump(mode="json"))
    assert ledger.add([sibling], envelope) == [2]
    ledger.record_delivery("sibling-accepted", "coordinator", records(ledger, 2))
    ledger.pin_delivery("sibling-accepted")
    unchanged = copy.deepcopy(ledger.export())
    restored = session(run, envelope, ledger)
    try:
        restored.restore(snapshot)
        assert ledger.export() == unchanged
        assert (
            restored.context.services["evidence"] is restored.harness.evidence is ledger
        )
        assert ledger.completely_delivered("sibling-accepted") == {2}
    finally:
        child.workers.close()
        restored.workers.close()


def test_accepted_call_stays_valid_when_unrelated_unpinned_history_is_pruned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(
        monkeypatch, "originals", "Tam özgün koşul bu işlemi destekler [1]."
    )
    run.calls.clear()
    envelope = outer(run)
    ledger = EvidenceLedger()
    child = session(run, envelope, ledger)
    result = child.run()
    call = child.model.last_call_id
    assert call
    envelope.services["last_model_call_id"] = call
    ledger.record_delivery("unrelated-old-call", "coordinator", records(ledger))
    snapshot = child.snapshot()
    root = RunContext(run_id=envelope.run_id, scope=envelope.scope)
    assignment: dict[str, JsonValue] = {
        "question_id": "independent-a",
        "question": FOCUS,
        "task_id": "owned-task",
        "parent_question_ids": [1],
    }
    receipts = ParallelAnswerReceipts(root, SCENARIO, user_id="owner")
    receipt = receipts.seal(
        envelope,
        assignment=assignment,
        answer=result.summary,
        status=result.status,
        model_call_id=call,
        ledger=ledger,
        validate_body=lambda: child.validate_accepted(
            result.summary, call, result.status
        ),
        source_state=child.source_state(),
    )
    for index in range(201):
        ledger.record_delivery(f"later-sibling-{index}", "coordinator", records(ledger))
    assert ledger.completely_delivered("unrelated-old-call") == set()
    assert ledger.completely_delivered(call) == {1}
    unchanged = copy.deepcopy(ledger.export())
    run.selected.invoke.side_effect = AssertionError("Restore must not run the model")
    restored = session(run, envelope, ledger)
    try:
        restored.restore(snapshot)
        replay = restored.run()
        assert replay.summary == result.summary
        assert replay.status == result.status
        assert ledger.export() == unchanged
        receipts.verify(
            root,
            receipt_id=receipt,
            task_id="owned-task",
            assignment=assignment,
            answer=result.summary,
            status=result.status,
            ledger=ledger,
            validate_body=lambda: restored.validate_accepted(
                result.summary, call, result.status
            ),
            source_state=restored.source_state(),
        )
    finally:
        restored.workers.close()


def test_restored_run_rejects_a_body_with_a_real_serial_publication_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "originals", "Tam özgün koşul uygulanır [1].")
    run.calls.clear()
    child = session(run, outer(run), EvidenceLedger())
    child.run()
    snapshot = child.snapshot()
    body = "8917 sayılı Faaliyet Kanunu m.27 gereğince kesin izin vardır [1]."
    snapshot["last_draft"] = body
    state = cast(dict[str, JsonValue], snapshot["serial_experimental_session"])
    accepted = cast(dict[str, JsonValue], state["accepted"])
    accepted["answer_hash"] = hashlib.sha256(body.encode()).hexdigest()
    run.selected.invoke.side_effect = AssertionError(
        "Rejected replay must not invoke the model"
    )
    restored = session(run, child.outer_context, child.ledger)
    try:
        restored.restore(snapshot)
        with pytest.raises(ValueError, match="no longer supported"):
            restored.run()
        assert restored.harness.last_draft == body
    finally:
        restored.workers.close()


def test_sealed_local_navigation_is_qualified_without_merging_live_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "originals", "İşlem özgün koşula tabidir [1].")
    root = RunContext(
        run_id=run.harness.context.run_id, scope=run.harness.context.scope
    )
    ledger = EvidenceLedger()
    source = run.broker.sources[0]
    item = evidence_for_chunk(source, run.broker.chunks[str(source.id)])
    assert ledger.add([item], root) == [1]
    length = len(item.text)
    update = OutcomeUpdate.model_validate(
        {
            "outcomes": [
                {"outcome_id": "local-result", "question_ids": ["q0"], "detail": FOCUS}
            ],
            "conditions": [
                {
                    "condition_id": "local-condition",
                    "outcome_ids": ["local-result"],
                    "detail": "Özgün koşul",
                    "witnesses": [{"citation": 1, "start_char": 0, "end_char": length}],
                }
            ],
            "resolutions": [
                {
                    "outcome_id": "local-result",
                    "status": "unresolved",
                    "condition_ids": ["local-condition"],
                    "gap": "Sonraki hüküm doğrulanmadı.",
                }
            ],
        }
    )
    root_map = OutcomeMap([SCENARIO], root)
    root_map.update(update, ledger)
    root.services["outcome_map"] = root_map
    original_root_map = root_map.export()
    receipts = ParallelAnswerReceipts(root, SCENARIO, user_id="owner")
    states: list[dict[str, JsonValue]] = []
    for owner, assignment_id in [
        ("first-task", "first-question"),
        ("second-task", "second-question"),
    ]:
        run.calls.clear()
        envelope = outer(run, owner=owner, assignment=assignment_id)
        child = session(run, envelope, ledger)
        child.outcomes.update(update, ledger)
        result = child.run()
        call = child.model.last_call_id
        assert call
        envelope.services["last_model_call_id"] = call
        assignment: dict[str, JsonValue] = {
            "question_id": assignment_id,
            "question": FOCUS,
            "task_id": owner,
            "parent_question_ids": [1],
        }
        receipt_id = receipts.seal(
            envelope,
            assignment=assignment,
            answer=result.summary,
            status=result.status,
            model_call_id=call,
            ledger=ledger,
            validate_body=lambda: child.validate_accepted(
                result.summary, call, result.status
            ),
            source_state=child.source_state(),
        )
        receipt_rows = cast(list[dict[str, JsonValue]], receipts.export()["receipts"])
        receipt = next(row for row in receipt_rows if row["receipt_id"] == receipt_id)
        state = accepted_serial_memory_state(
            child.snapshot(), assignment, receipt, root, SCENARIO, run.history
        )
        assert state["session_research"] == child.source_state()["session_research"]
        states.append(state)
        corrupted = copy.deepcopy(receipt)
        corrupted["source_state_hash"] = "0" * 64
        with pytest.raises(ValueError, match="sealed assignment"):
            accepted_serial_memory_state(
                child.snapshot(), assignment, corrupted, root, SCENARIO, run.history
            )
    memory = merge_serial_session_memory(
        session_research_checkpoint(root, SCENARIO), states
    )
    assert merge_serial_session_memory(memory, states) == memory
    assert root_map.export() == original_root_map
    navigation = cast(dict[str, JsonValue], memory["outcome_navigation"])
    outcomes = cast(list[dict[str, JsonValue]], navigation["outcomes"])
    conditions = cast(list[dict[str, JsonValue]], navigation["conditions"])
    gaps = cast(list[dict[str, JsonValue]], navigation["open_gaps"])
    assert len(outcomes) == len(conditions) == len(gaps) == 3
    assert outcomes[0]["outcome_id"] == "local-result"
    assert len({str(row["outcome_id"]) for row in outcomes}) == 3
    assert len({str(row["condition_id"]) for row in conditions}) == 3
    assert all(
        row["witnesses"] == [{"citation": 1, "start_char": 0, "end_char": length}]
        for row in conditions
    )
    assert all(
        row["outcome_ids"] == [outcomes[index]["outcome_id"]]
        for index, row in enumerate(conditions)
    )
    assert all(
        row["outcome_id"] == outcomes[index]["outcome_id"]
        for index, row in enumerate(gaps)
    )
    followup = RunContext(scope=root.scope)
    new_ledger = EvidenceLedger()
    new_ledger.add([original()], followup)
    retain_session_research(
        {"scope": root.scope, "evidence": ledger.export(), "session_research": memory},
        followup,
        new_ledger,
        lambda _items, _context: None,
    )
    remembered = cast(dict[str, JsonValue], followup.services["session_research"])
    prior = cast(dict[str, JsonValue], remembered["prior_outcomes"])
    remapped = cast(list[dict[str, JsonValue]], prior["conditions"])
    assert len(remapped) == 3
    assert all(
        row["witnesses"] == [{"citation": 2, "start_char": 0, "end_char": length}]
        for row in remapped
    )
