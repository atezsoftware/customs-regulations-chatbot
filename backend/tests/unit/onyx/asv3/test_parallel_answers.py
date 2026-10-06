import copy
from typing import Callable, Literal, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc

REQUEST = "Explain permission and its conditions for the supplied transaction."
ASSIGNMENT: dict[str, JsonValue] = {
    "question_id": "q0",
    "question": "Explain the permission.",
    "outcome_ids": ["permission"],
}


def original(text: str = "Permission requires a signed application.") -> EvidenceItem:
    return EvidenceItem(
        source_id="instrument",
        chunk_id="provision",
        text=text,
        metadata={"source_sha256": "a" * 64, "publication_revision": 4},
        search_doc=SearchDoc(
            document_id="instrument",
            chunk_ind=0,
            semantic_identifier="Official instrument",
            link="https://example.test/instrument",
            blurb=text,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": "provision"},
            match_highlights=[],
        ),
    )


def fixture() -> tuple[RunContext, RunContext, EvidenceLedger, ParallelAnswerReceipts]:
    root = RunContext(run_id="run", scope={"document_sets": [15]})
    child = root.independent_child()
    child.services.update(task_id="task-a", last_model_call_id="accepted-child")
    ledger = EvidenceLedger()
    ledger.add([original()], child)
    item = ledger.get(1)
    assert item is not None
    ledger.record_delivery(
        "accepted-child", "asv3_researcher", [{"citation": 1, "text": item.text}]
    )
    return root, child, ledger, ParallelAnswerReceipts(root, REQUEST, user_id="owner")


def seal(
    store: ParallelAnswerReceipts,
    child: RunContext,
    ledger: EvidenceLedger,
    *,
    answer: str = "A signed application is required [1].",
    status: OutcomeStatus = OutcomeStatus.FOUND,
    source_state: dict[str, JsonValue] | None = None,
    guard: Callable[[], ToolOutcome | None] = lambda: None,
) -> str:
    return store.seal(
        child,
        assignment=ASSIGNMENT,
        answer=answer,
        status=status,
        model_call_id="accepted-child",
        ledger=ledger,
        validate_body=guard,
        source_state=source_state,
    )


def verify(
    store: ParallelAnswerReceipts,
    context: RunContext,
    ledger: EvidenceLedger,
    receipt_id: str,
    *,
    answer: str = "A signed application is required [1].",
    status: OutcomeStatus = OutcomeStatus.FOUND,
    assignment: dict[str, JsonValue] = ASSIGNMENT,
    task_id: str = "task-a",
    source_state: dict[str, JsonValue] | None = None,
    guard: Callable[[], ToolOutcome | None] = lambda: None,
) -> None:
    store.verify(
        context,
        receipt_id=receipt_id,
        task_id=task_id,
        assignment=assignment,
        answer=answer,
        status=status,
        ledger=ledger,
        validate_body=guard,
        source_state=source_state,
    )


def test_lossless_host_assembly_uses_accepted_child_delivery_without_parent_reread() -> (
    None
):
    root, child, ledger, store = fixture()
    answer = "\n".join(f"Supported condition {i} [1]." for i in range(900))
    assert len(answer) > 12000
    called: list[str] = []

    def guard() -> None:
        called.append("target-owner")

    receipt_id = seal(store, child, ledger, answer=answer, guard=guard)
    assert ledger.completely_delivered("parent-assembly") == set()
    verify(store, root, ledger, receipt_id, answer=answer, guard=guard)
    snapshot = store.export()
    receipts = cast(list[dict[str, JsonValue]], snapshot["receipts"])
    assert receipts[0]["answer"] == answer
    assert called == ["target-owner", "target-owner"]

    restored_ledger = EvidenceLedger()
    restored_ledger.restore(ledger.export(), root)
    restored = ParallelAnswerReceipts(root, REQUEST, user_id="owner")
    restored.restore(snapshot, root, REQUEST, restored_ledger)
    verify(restored, root, restored_ledger, receipt_id, answer=answer)


@pytest.mark.parametrize(
    "change", ["answer", "status", "task_id", "assignment", "source_state"]
)
def test_assembly_rejects_changed_body_status_task_assignment_or_source_state(
    change: Literal["answer", "status", "task_id", "assignment", "source_state"],
) -> None:
    root, child, ledger, store = fixture()
    receipt_id = seal(store, child, ledger)
    with pytest.raises(ValueError, match="changed"):
        verify(
            store,
            root,
            ledger,
            receipt_id,
            answer=(
                "A generic permission exists [1]."
                if change == "answer"
                else "A signed application is required [1]."
            ),
            status=OutcomeStatus.PARTIAL if change == "status" else OutcomeStatus.FOUND,
            task_id="sibling" if change == "task_id" else "task-a",
            assignment=(
                {**ASSIGNMENT, "outcome_ids": ["other"]}
                if change == "assignment"
                else ASSIGNMENT
            ),
            source_state=(
                {"open_authority_ids": ["new-open"]}
                if change == "source_state"
                else None
            ),
        )


def test_publication_guard_rejection_is_not_sealed_or_pinned() -> None:
    root, child, ledger, store = fixture()

    def rejected() -> ToolOutcome:
        return ToolOutcome(status=OutcomeStatus.PARTIAL, summary="Unread basis")

    with pytest.raises(ValueError, match="guard rejected"):
        seal(store, child, ledger, guard=rejected)
    assert store.export()["receipts"] == []
    assert ledger.export()["pinned_delivery_calls"] == []
    receipt_id = seal(store, child, ledger)
    with pytest.raises(ValueError, match="guard rejected"):
        verify(store, root, ledger, receipt_id, guard=rejected)


def test_sealing_cannot_bind_a_sibling_assignment_to_an_accepted_child_call() -> None:
    _root, child, ledger, store = fixture()
    child.services.update(assignment_id="q0", task_outcome_ids=["permission"])
    for changed in [
        {**ASSIGNMENT, "task_id": "sibling"},
        {**ASSIGNMENT, "question_id": "q1"},
        {**ASSIGNMENT, "outcome_ids": ["sibling"]},
    ]:
        with pytest.raises(ValueError, match="does not belong"):
            store.seal(
                child,
                assignment=changed,
                answer="A signed application is required [1].",
                status=OutcomeStatus.FOUND,
                model_call_id="accepted-child",
                ledger=ledger,
                validate_body=lambda: None,
            )


def test_source_and_exact_delivery_tampering_cannot_pass_restore() -> None:
    root, child, ledger, store = fixture()
    seal(store, child, ledger)
    changed = EvidenceLedger()
    changed.add([original("An application alone is insufficient.")], root)
    item = changed.get(1)
    assert item is not None
    changed.record_delivery(
        "accepted-child", "asv3_researcher", [{"citation": 1, "text": item.text}]
    )
    restored = ParallelAnswerReceipts(root, REQUEST, user_id="owner")
    with pytest.raises(ValueError, match="changed"):
        restored.restore(store.export(), root, REQUEST, changed)

    ledger_snapshot = copy.deepcopy(ledger.export())
    deliveries = cast(list[dict[str, JsonValue]], ledger_snapshot["deliveries"])
    records = cast(list[dict[str, JsonValue]], deliveries[0]["records"])
    records[0]["end_char"] = 3
    malformed = EvidenceLedger()
    malformed.restore(ledger_snapshot, root)
    with pytest.raises(ValueError, match="delivery identity"):
        restored.restore(store.export(), root, REQUEST, malformed)


@pytest.mark.parametrize("field", ["source_id", "chunk_id", "metadata"])
def test_canonical_identity_and_provenance_are_checked_against_actual_ledger(
    field: str,
) -> None:
    root, child, ledger, store = fixture()
    seal(store, child, ledger)
    item = original()
    if field == "source_id":
        item.source_id = "other-instrument"
    elif field == "chunk_id":
        item.chunk_id = "other-provision"
    else:
        item.metadata["publication_revision"] = 5
    changed = EvidenceLedger()
    changed.add([item], root)
    changed.record_delivery(
        "accepted-child", "asv3_researcher", [{"citation": 1, "text": item.text}]
    )
    restored = ParallelAnswerReceipts(root, REQUEST, user_id="owner")
    with pytest.raises(ValueError, match="canonical original changed"):
        restored.restore(store.export(), root, REQUEST, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer", "Changed body [1]."),
        ("status", "partial"),
        ("task_id", "sibling"),
        ("model_call_id", "different-call"),
    ],
)
def test_checkpoint_integrity_rejects_changed_receipt(field: str, value: str) -> None:
    root, child, ledger, store = fixture()
    seal(store, child, ledger)
    snapshot = copy.deepcopy(store.export())
    receipts = cast(list[dict[str, JsonValue]], snapshot["receipts"])
    receipts[0][field] = value
    restored = ParallelAnswerReceipts(root, REQUEST, user_id="owner")
    with pytest.raises(ValueError, match="integrity"):
        restored.restore(snapshot, root, REQUEST, ledger)


def test_restore_fences_owner_scope_run_and_exact_request() -> None:
    root, child, ledger, store = fixture()
    seal(store, child, ledger)
    for context, request, owner in [
        (root, REQUEST, "other-owner"),
        (RunContext(run_id="other", scope=root.scope), REQUEST, "owner"),
        (RunContext(run_id="run", scope={"document_sets": [16]}), REQUEST, "owner"),
        (root, REQUEST + " ", "owner"),
    ]:
        restored = ParallelAnswerReceipts(context, request, user_id=owner)
        with pytest.raises(ValueError, match="ownership"):
            restored.restore(store.export(), context, request, ledger)


def test_uncited_partial_requires_actual_call_and_host_guard_but_not_invented_delivery() -> (
    None
):
    root, child, ledger, store = fixture()
    child.services["last_model_call_id"] = "uncited-last-call"
    answer = "The governing original for this permission could not be obtained."
    receipt_id = store.seal(
        child,
        assignment=ASSIGNMENT,
        answer=answer,
        status=OutcomeStatus.PARTIAL,
        model_call_id="uncited-last-call",
        ledger=ledger,
        validate_body=lambda: None,
        source_state={"open_authority_ids": ["permission-basis"]},
    )
    verify(
        store,
        root,
        ledger,
        receipt_id,
        answer=answer,
        status=OutcomeStatus.PARTIAL,
        source_state={"open_authority_ids": ["permission-basis"]},
    )
    with pytest.raises(ValueError, match="actual last"):
        seal(store, child, ledger)


def test_sealed_delivery_survives_history_rotation_and_restore_defaults_remain_bounded() -> (
    None
):
    root, child, ledger, store = fixture()
    receipt_id = seal(store, child, ledger)
    item = ledger.get(1)
    assert item is not None
    ordinary = EvidenceLedger()
    ordinary.add([original()], root)
    for i in range(250):
        row: dict[str, JsonValue] = {"citation": 1, "text": item.text}
        ledger.record_delivery(f"later-{i}", "asv3_coordinator", [row])
        ordinary.record_delivery(f"later-{i}", "asv3_coordinator", [row])
    assert len(cast(list[JsonValue], ledger.export()["deliveries"])) == 201
    assert len(cast(list[JsonValue], ordinary.export()["deliveries"])) == 200
    verify(store, root, ledger, receipt_id)
    restored = EvidenceLedger()
    restored.restore(ledger.export(), root)
    assert restored.completely_delivered("accepted-child") == {1}
    verify(store, root, restored, receipt_id)
    with pytest.raises(ValueError, match="unrecorded"):
        ledger.pin_delivery("invented-call")
    broken = copy.deepcopy(ledger.export())
    broken["pinned_delivery_calls"] = ["invented-call"]
    with pytest.raises(ValueError, match="pinned"):
        restored.restore(broken, root)
