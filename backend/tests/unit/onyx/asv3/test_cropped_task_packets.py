"""Actual receipt bindings, not bounded need lists, determine task context."""

import copy
import hashlib
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_task_packets import (
    CroppedTaskPacketPolicy,
    CroppedTaskPacketStore,
    expand_task_packet_receipts,
    packet_for_view,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    HarnessView,
    OriginalEvidenceRead,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.outcome_map import (
    OutcomeCondition,
    OutcomeMap,
    OutcomeUpdate,
    OutcomeWitness,
    RequestedOutcome,
)
from onyx.asv3.research_state import (
    ResearchFinding,
    ResearchNeed,
    ResearchState,
    ResearchUpdate,
    SourceWitness,
)
from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall, ToolMessage

REQUEST = "Compare topic A with topic B, including exceptions and later steps."


def rows(data: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    value = data[key]
    assert isinstance(value, list) and all(isinstance(row, dict) for row in value)
    return cast(list[dict[str, JsonValue]], value)


def mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def fixture(
    count: int = 3,
) -> tuple[RunContext, EvidenceLedger, ResearchState, CroppedTaskPacketStore]:
    context = RunContext(
        scope={"document_set": "Türkçe", "as_of": "2026-10-13"},
        services={"asv3_exp_cropped": True, "cropped_user_context": REQUEST},
    )
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id=f"source-{number}",
                chunk_id=f"chunk-{number}",
                text=f"Original {number}: condition, exception and subsequent stage.",
                metadata={"version": "2026", "article_no": str(number)},
            )
            for number in range(1, count + 1)
        ],
        context,
    )
    state = ResearchState([REQUEST], context)
    store = CroppedTaskPacketStore(
        context,
        REQUEST,
        policy=CroppedTaskPacketPolicy(original_records=2),
    )
    context.services["cropped_task_packet_store"] = store
    context.services["research_state"] = state
    context.services["evidence"] = ledger
    return context, ledger, state, store


def need(
    key: str, *, complete: bool = False, depends: list[str] | None = None
) -> ResearchNeed:
    return ResearchNeed(
        need_id=key,
        question_ids=["q0"],
        purpose=f"Determine topic {key}",
        completion_test="Obtain the original and assess conditions and exceptions",
        status="candidate" if complete else "researching",
        depends_on=depends or [],
    )


def finding(
    key: str, number: int, ledger: EvidenceLedger, *, suffix: str = ""
) -> ResearchFinding:
    item = ledger.get(number)
    assert item is not None
    return ResearchFinding(
        finding_id=f"finding_{key}",
        need_id=key,
        statement=f"Candidate topic {key}: exception and subsequent stage.{suffix}",
        kind="rule",
        witnesses=[SourceWitness(citation=number, end_char=len(item.text))],
    )


def receipt(
    number: int,
    binding: str | None,
    ledger: EvidenceLedger,
    *,
    call_id: str | None = None,
    data: dict[str, JsonValue] | None = None,
) -> ToolReceipt:
    item = ledger.get(number)
    assert item is not None
    return ToolReceipt(
        call=CapabilityCall(
            name="read_provision",
            call_id=call_id or f"read-{number}",
            arguments={
                "source_id": item.source_id,
                "article": str(number),
                **({"_need_id": binding} if binding else {}),
            },
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actually read original",
            data=data
            if data is not None
            else cast(dict[str, JsonValue], {"has_more": False}),
            original_reads=[
                OriginalEvidenceRead(
                    citation=number,
                    text_hash=item.text_hash,
                    start_char=0,
                    end_char=len(item.text),
                )
            ],
        ),
        elapsed_seconds=0,
        evidence_ids=[number],
    )


def view(
    ledger: EvidenceLedger,
    receipts: list[ToolReceipt],
    *,
    facts: list[str] | None = None,
) -> HarnessView:
    records = cast(
        list[dict[str, JsonValue]],
        json.loads(ledger.serialize_records(ledger.citation_numbers(), max_chars=None)),
    )
    return HarnessView(
        request=REQUEST,
        questions=[REQUEST],
        facts=facts or ["Shared scenario date is 2026-10-13"],
        receipts=receipts,
        evidence=[
            {"citation": row["citation"], "source_id": row["source_id"]}
            for row in records
        ],
        tools=[],
        original_evidence=records,
    )


def topics(
    state: ResearchState,
    ledger: EvidenceLedger,
    *,
    depends: list[str] | None = None,
    third: bool = False,
) -> None:
    state.update(
        ResearchUpdate(
            needs=[
                need("A", depends=depends),
                need("B", complete=True),
                *([need("C", complete=True)] if third else []),
            ],
            findings=[
                finding("B", 2, ledger),
                *([finding("C", 3, ledger)] if third else []),
            ],
            active_need_ids=["A"],
        ),
        ledger,
    )


def test_two_topic_packet_detaches_only_source_bound_candidate() -> None:
    context, ledger, state, store = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    original = view(ledger, receipts)
    before = original.model_dump(mode="json")
    result = packet_for_view(original, receipts, state, context, ledger)
    assert result is not None
    selected, manifest = result
    assert selected.required_evidence_numbers == [1]
    assert [row["citation"] for row in selected.original_evidence] == [1]
    assert selected.facts == original.facts and selected.request == REQUEST
    assert selected.turns == []
    assert [r.call.call_id for r in selected.receipts] == ["read-1"]
    assert mapping(rows(manifest, "detached_need_index")[0]["need"])["need_id"] == "B"
    assert manifest["legal_approval"] is False
    assert original.model_dump(mode="json") == before
    assert ledger.citation_numbers() == (1, 2)
    restored = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    restored.restore(store.export(), ledger)
    assert restored.export() == store.export()


def test_shared_need_dependency_retains_complete_originals() -> None:
    context, ledger, state, _ = fixture()
    topics(state, ledger, depends=["B"], third=True)
    receipts = [
        receipt(1, "A", ledger),
        receipt(2, "B", ledger),
        receipt(3, "C", ledger),
    ]
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    selected, manifest = result
    assert selected.required_evidence_numbers == [1, 2]
    assert manifest["active_need_ids"] == ["A", "B"]
    assert [
        mapping(r["need"])["need_id"] for r in rows(manifest, "detached_need_index")
    ] == ["C"]


def test_shared_outcome_condition_connects_both_topics() -> None:
    context, ledger, state, _ = fixture()
    topics(state, ledger, third=True)
    outcomes = OutcomeMap([REQUEST], context)
    context.services["outcome_map"] = outcomes
    item = ledger.get(2)
    assert item is not None
    outcomes.update(
        OutcomeUpdate(
            outcomes=[
                RequestedOutcome(
                    outcome_id=key, question_ids=["q0"], detail=f"Determine {key}"
                )
                for key in ("A", "B", "C")
            ],
            conditions=[
                OutcomeCondition(
                    condition_id="shared",
                    outcome_ids=["A", "B"],
                    detail="Shared exception changes both results",
                    witnesses=[OutcomeWitness(citation=2, end_char=len(item.text))],
                )
            ],
        ),
        ledger,
    )
    receipts = [receipt(n, key, ledger) for n, key in enumerate(("A", "B", "C"), 1)]
    for row in receipts:
        row.call.arguments["outcome_ids"] = [str(row.call.arguments["_need_id"])]
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    assert result[0].required_evidence_numbers == [1, 2]
    assert result[1]["selected_outcome_ids"] == ["A", "B"]


def test_worker_task_binding_is_retained_without_source_duplication_to_other_owner() -> (
    None
):
    context, ledger, state, store = fixture(2)
    topics(state, ledger)
    child = context.child()
    child.services["task_id"] = "worker"
    child.services["task_need_ids"] = ["B"]
    store.record_receipts(
        "worker", [receipt(2, None, ledger)], state, ledger, context=child
    )
    assert store.seal_eligible(
        state, child, ["Shared scenario date is 2026-10-13"], ledger
    ) == ["B"]
    root_receipts = [receipt(1, "A", ledger)]
    # Foreign originals are unbound in the root's own state until explicitly related.
    result = packet_for_view(
        view(ledger, root_receipts), root_receipts, state, context, ledger
    )
    assert result is None
    with pytest.raises(ValueError, match="owner"):
        store.record_receipts(
            "root", [receipt(2, None, ledger)], state, ledger, context=child
        )


def test_pending_source_continuation_cannot_detach_candidate() -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    receipts = [
        receipt(1, "A", ledger),
        receipt(
            2,
            "B",
            ledger,
            data={"has_more": True, "cursor": 2000, "absence_proven": False},
        ),
    ]
    assert (
        packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
        is None
    )


def test_all_actual_reads_survive_the_forty_reference_need_cap() -> None:
    context, ledger, state, _ = fixture(55)
    topics(state, ledger)
    receipts = [
        receipt(1, "A", ledger),
        *[receipt(n, "B", ledger) for n in range(2, 56)],
    ]
    state.attach("B", list(range(2, 56)))
    assert (
        len(cast(list[int], rows(state.export(), "needs")[1]["evidence_numbers"])) == 40
    )
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    selected, manifest = result
    assert selected.required_evidence_numbers == [1]
    assert rows(manifest, "detached_need_index")[0]["original_ids"] == list(
        range(2, 56)
    )
    context.services["task_need_ids"] = ["B"]
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    assert result[0].required_evidence_numbers == list(range(2, 56))
    parked = rows(result[1], "detached_need_index")[0]
    assert mapping(parked["need"])["need_id"] == "A"
    assert parked["status"] == "retained_open_need_not_completed"


def test_open_topics_keep_pending_work_and_restore_exact_receipts() -> None:
    context, ledger, state, store = fixture(2)
    state.update(
        ResearchUpdate(needs=[need("A"), need("B")], active_need_ids=["A"]), ledger
    )
    pending = receipt(
        2,
        "B",
        ledger,
        data={"has_more": True, "next_position": 12, "absence_proven": False},
    )
    pending.outcome.status = OutcomeStatus.PARTIAL
    receipts = [receipt(1, "A", ledger), pending]
    before = state.export()
    first = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert first is not None and first[0].required_evidence_numbers == [1]
    parked = rows(first[1], "detached_need_index")[0]
    assert parked["status"] == "retained_open_need_not_completed"
    assert mapping(parked["need"])["status"] == "researching"
    operation = rows(parked, "pending_operations")[0]
    assert operation["status"] == "partial"
    assert operation["continuation"] == pending.outcome.data
    assert operation["call"] == pending.call.model_dump(mode="json")
    assert state.export() == before
    restored = CroppedTaskPacketStore(context, REQUEST, policy=store.policy)
    restored.restore(store.export(), ledger)
    context.services["cropped_task_packet_store"] = restored
    state.update(ResearchUpdate(active_need_ids=["B"]), ledger)
    second = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert second is not None and second[0].required_evidence_numbers == [2]
    assert second[0].receipts == [pending]
    assert ledger.citation_numbers() == (1, 2)


def test_changed_facts_reopen_parked_open_topic_until_reassessed() -> None:
    context, ledger, state, _ = fixture(2)
    state.update(
        ResearchUpdate(needs=[need("A"), need("B")], active_need_ids=["A"]), ledger
    )
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    assert packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    changed = view(ledger, receipts, facts=["Common facts changed"])
    assert packet_for_view(changed, receipts, state, context, ledger) is None
    assert packet_for_view(changed, receipts, state, context, ledger) is None


def test_open_topic_dependencies_remain_in_active_packet() -> None:
    context, ledger, state, _ = fixture(3)
    state.update(
        ResearchUpdate(
            needs=[need("A", depends=["B"]), need("B"), need("C")],
            active_need_ids=["A"],
        ),
        ledger,
    )
    receipts = [receipt(n, key, ledger) for n, key in enumerate(("A", "B", "C"), 1)]
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    assert result[0].required_evidence_numbers == [1, 2]
    assert result[1]["active_need_ids"] == ["A", "B"]


@pytest.mark.parametrize("binding", [None, "unknown"])
def test_unknown_bindings_and_unbound_originals_remain_in_packet(
    binding: str | None,
) -> None:
    context, ledger, state, _ = fixture(4)
    topics(state, ledger)
    receipts = [
        receipt(1, "A", ledger),
        receipt(2, "B", ledger),
        receipt(3, binding, ledger),
    ]
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    assert result[0].required_evidence_numbers == [1, 3, 4]
    assert result[1]["unbound_original_ids"] == [3, 4]


def test_changed_common_facts_reopen_without_resealing_stale_candidate() -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    first = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert first is not None
    changed = view(ledger, receipts, facts=["Shared scenario date changed to 2025"])
    assert packet_for_view(changed, receipts, state, context, ledger) is None
    assert packet_for_view(changed, receipts, state, context, ledger) is None
    state.update(
        ResearchUpdate(
            findings=[finding("B", 2, ledger, suffix=" New facts reassessed.")]
        ),
        ledger,
    )
    revised = packet_for_view(changed, receipts, state, context, ledger)
    assert revised is not None and revised[1]["packet_hash"] != first[1]["packet_hash"]


def test_new_dependency_reopens_retained_candidate() -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    assert packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    state.update(
        ResearchUpdate(needs=[need("B", complete=True, depends=["A"])]), ledger
    )
    assert (
        packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
        is None
    )


def test_actual_candidate_reassessment_receipt_can_reseal_after_fact_change() -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    assert packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    changed = view(ledger, receipts, facts=["Scenario facts changed"])
    assert packet_for_view(changed, receipts, state, context, ledger) is None
    reassessed = ToolReceipt(
        call=CapabilityCall(
            name="update_research",
            call_id="reassessed-B",
            arguments={
                "needs": [need("B", complete=True).model_dump(mode="json")],
                "findings": [finding("B", 2, ledger).model_dump(mode="json")],
            },
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual current-fact candidate assessment retained",
        ),
        elapsed_seconds=0,
    )
    assert packet_for_view(changed, [*receipts, reassessed], state, context, ledger)


def test_exact_operation_errors_cursors_and_unmatched_recommendations_survive() -> None:
    context, ledger, state, _ = fixture(3)
    topics(state, ledger)
    receipts = [
        receipt(
            1,
            "A",
            ledger,
            data={
                "cursor": 2000,
                "has_more": True,
                "failed_candidates": ["old source"],
                "absence_proven": False,
            },
        ),
        receipt(2, "B", ledger),
        receipt(3, None, ledger),
    ]
    original = view(ledger, receipts)
    original.turns = [
        ResearchTurn(
            assistant=AssistantMessage(
                content="Consider the exception under the same scenario.",
                tool_calls=[
                    ToolCall(
                        id="not-executed",
                        function=FunctionCall(
                            name="read_source_range",
                            arguments='{"source_id":"source-3","start":2000}',
                        ),
                    )
                ],
            ),
            results=[
                ToolMessage(
                    tool_call_id="not-executed",
                    content='{"outcome":{"status":"error","data":{"has_more":true}}}',
                )
            ],
        )
    ]
    result = packet_for_view(original, receipts, state, context, ledger)
    assert result is not None
    manifest = result[1]
    assert rows(manifest, "local_receipts")[0] == receipts[0].model_dump(mode="json")
    assert rows(manifest, "assistant_observations")[0]["assistant"] == original.turns[
        0
    ].assistant.model_dump(mode="json")
    assert rows(rows(manifest, "assistant_observations")[0], "unmatched_results")[
        0
    ] == original.turns[0].results[0].model_dump(mode="json")


@pytest.mark.parametrize("changed", ["scope", "request", "run", "source", "receipt"])
def test_changed_scope_source_or_actual_receipt_is_rejected(changed: str) -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    original = view(ledger, receipts)
    assert packet_for_view(original, receipts, state, context, ledger)
    if changed == "scope":
        context.scope["as_of"] = "2024"
    elif changed == "request":
        context.services["cropped_user_context"] = "A different scenario"
    elif changed == "run":
        context.run_id = "different"
    elif changed == "source":
        original.original_evidence[0]["text_hash"] = "0" * 64
    else:
        receipts[0].outcome.data["has_more"] = True
    with pytest.raises(ValueError):
        packet_for_view(original, receipts, state, context, ledger)


@pytest.mark.parametrize("variant", ["small", "normal", "cheap", "draft"])
def test_small_and_protected_paths_are_unchanged(variant: str) -> None:
    context, ledger, state, store = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    original = view(ledger, receipts)
    if variant == "small":
        store = CroppedTaskPacketStore(context, REQUEST)
        context.services["cropped_task_packet_store"] = store
    elif variant == "normal":
        context.services.pop("asv3_exp_cropped")
    elif variant == "cheap":
        context.services["asv3_exp_cheap"] = True
    else:
        original.draft_to_repair = "Candidate legal answer"
    before = copy.deepcopy(original.model_dump(mode="json"))
    assert packet_for_view(original, receipts, state, context, ledger) is None
    assert original.model_dump(mode="json") == before


def test_measured_cumulative_input_triggers_without_a_call_or_source_cap() -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    store = CroppedTaskPacketStore(context, REQUEST)
    context.services["cropped_task_packet_store"] = store
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    store.record_input("root", 875_934)
    assert (
        packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
        is None
    )
    store.record_input("worker", 124_066)
    assert packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert ledger.citation_numbers() == (1, 2)


def test_checkpoint_tamper_does_not_replace_the_store() -> None:
    context, ledger, state, store = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    assert packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    before = store.export()
    forged = copy.deepcopy(before)
    rows(rows(forged, "receipts")[0], "sources")[0]["text_hash"] = "0" * 64
    forged["integrity"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in forged.items() if k != "integrity"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="original version"):
        store.restore(forged, ledger)
    assert store.export() == before


def test_actual_evidence_and_data_bodies_use_exact_same_message_references() -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    item = ledger.get(1)
    assert item is not None
    receipts[0].outcome.evidence = [item]
    receipts[0].outcome.data["original_evidence"] = json.loads(
        ledger.serialize_records([1], max_chars=None)
    )
    receipts[0].outcome.data["unknown_material"] = {
        "later_step": "Keep exact unknown fields",
        "threshold": None,
        "flag": False,
    }
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    selected, manifest = result
    assert item.text not in json.dumps(manifest["local_receipts"], ensure_ascii=False)
    assert expand_task_packet_receipts(manifest, selected.original_evidence) == [
        receipts[0].model_dump(mode="json")
    ]
    assert selected.receipts[0].outcome.evidence[0].text == item.text
    assert manifest["receipt_text_transport"] is not None


@pytest.mark.parametrize(
    "tamper", ["missing", "boolean", "range", "hash", "field", "count"]
)
def test_receipt_text_transport_rejects_unread_or_tampered_originals(
    tamper: str,
) -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    receipts = [receipt(1, "A", ledger), receipt(2, "B", ledger)]
    item = ledger.get(1)
    assert item is not None
    receipts[0].outcome.evidence = [item]
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    selected, manifest = result
    altered = copy.deepcopy(manifest)
    originals = copy.deepcopy(selected.original_evidence)
    source = rows(mapping(rows(altered, "local_receipts")[0]["outcome"]), "evidence")[0]
    reference = mapping(source["text"])
    if tamper == "missing":
        originals.clear()
    elif tamper == "boolean":
        reference["cropped_original_text_ref"] = [True, 0, len(item.text)]
    elif tamper == "range":
        reference["cropped_original_text_ref"] = [1, 0, len(item.text) + 1]
    elif tamper == "hash":
        originals[0]["text_hash"] = "0" * 64
    elif tamper == "field":
        source["source_id"] = "other source"
    else:
        mapping(altered["receipt_text_transport"])["reference_count"] = 2
    with pytest.raises(ValueError):
        expand_task_packet_receipts(altered, originals)


def test_original_reference_marker_collision_is_preserved_literally() -> None:
    context, ledger, state, _ = fixture(2)
    topics(state, ledger)
    receipts = [
        receipt(
            1,
            "A",
            ledger,
            data={"custom": {"cropped_original_text_ref": [900, True, None]}},
        ),
        receipt(2, "B", ledger),
    ]
    result = packet_for_view(view(ledger, receipts), receipts, state, context, ledger)
    assert result is not None
    assert result[1]["receipt_text_transport"] is None
    assert expand_task_packet_receipts(result[1], result[0].original_evidence) == [
        receipts[0].model_dump(mode="json")
    ]
