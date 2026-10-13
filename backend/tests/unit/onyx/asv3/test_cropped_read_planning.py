"""Planning preserves failure semantics and never impersonates original delivery."""

import copy
import json
from dataclasses import replace
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_read_planning import (
    build_cropped_read_planning_packet,
    expand_cropped_planning_navigation,
    serialize_cropped_read_planning_packet,
    snapshot_cropped_planning_receipts,
)
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.cropped_worker_handoff import WORKER_HANDOFF_REASON, build_worker_handoff
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    HarnessResult,
    HarnessView,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.working_memory import WorkingMemory
from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall, ToolMessage


def fixture() -> tuple[HarnessView, RunContext, list[dict[str, JsonValue]], str]:
    request = "9876 sayılı Kanunun 7 inci maddesi; ödeme tarihleri farklıdır."
    context = RunContext(run_id="real-fixture-run", scope={"owner": "fixture-user"})
    ledger = EvidenceLedger()
    context.services.update(
        evidence=ledger,
        asv3_exp_cropped=True,
        cropped_research_controller=CroppedResearchLimit(context, request),
    )
    text = (
        "MADDE 2 — 9876 sayılı Kanunun 7 inci maddesi uygulanır. "
        + "Özgün istisna. " * 100
    )
    ledger.add(
        [
            EvidenceItem(
                source_id="own-source",
                chunk_id="own-chunk",
                text=text,
                metadata={
                    "title": "Özgün mevzuat",
                    "article_no": "2",
                    "version_unknown": True,
                },
            ),
            EvidenceItem(
                source_id="own-source",
                chunk_id="second-chunk",
                text=text + " Son hüküm.",
                metadata={
                    "title": "Özgün mevzuat",
                    "article_no": "2",
                    "version_unknown": True,
                },
            ),
        ],
        context,
    )
    call = CapabilityCall(
        name="compose_tool_calls",
        call_id="actual-batch",
        arguments={
            "steps": [
                {
                    "id": "missing",
                    "tool": "read_named_provision",
                    "arguments": {"source_name": "Özgün Kanun", "article": "7"},
                },
                {
                    "id": "resolve",
                    "tool": "resolve_source",
                    "arguments": {"name": "Özgün Kanun"},
                },
            ],
            "max_parallel": 2,
        },
    )
    receipt = ToolReceipt(
        call=call,
        elapsed_seconds=0.1,
        evidence_ids=[1, 2],
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Identity unresolved; no absence proof.",
            data={
                "steps": {
                    "missing": {
                        "status": "not_found",
                        "summary": "Exact identity not found",
                        "data": {
                            "sources": [],
                            "absence_proven": False,
                            "has_more": False,
                        },
                    },
                    "resolve": {
                        "status": "ambiguous",
                        "summary": "Two real candidates",
                        "data": {
                            "sources": [
                                {"source_id": "own-source", "title": "Özgün mevzuat"},
                                {"source_id": "other-source", "title": "Eski metin"},
                            ],
                            "has_more": True,
                            "next_offset": 2,
                            "absence_proven": False,
                        },
                    },
                },
                "full_original": text,
            },
        ),
    )
    turn = ResearchTurn(
        assistant=AssistantMessage(
            content="Kimliği çözmeden tecil sonucu verme; tam kanunu oku.",
            tool_calls=[
                ToolCall(
                    id="actual-batch",
                    function=FunctionCall(
                        name=call.name, arguments=json.dumps(call.arguments)
                    ),
                )
            ],
        ),
        results=[
            ToolMessage(
                tool_call_id="actual-batch", content=json.dumps(receipt.outcome.data)
            )
        ],
    )
    memory = WorkingMemory(context.scope)
    memory_state = memory.export()
    memory_state["locators"] = [
        {
            "locator_id": f"ref-{n}",
            "kind": "reference",
            "status": "found",
            "origin_source_id": "own-source",
            "origin_chunk_id": chunk,
            "instrument_number": "9876",
            "article": "7",
            "reference_text": "9876 sayılı Kanunun 7 inci maddesi",
        }
        for n, chunk in enumerate(("own-chunk", "second-chunk"))
    ]
    child = context.child()
    child.services.update(
        task_id="actual-worker",
        task_outcome_ids=["payment"],
        task_need_ids=["law"],
        assignment_id="assignment",
        parent_task_id="root",
    )
    result = HarnessResult(
        answer=None,
        status=OutcomeStatus.PARTIAL,
        receipts=[receipt],
        questions=[request],
        facts=["Tahsil belgesi henüz doğrulanmadı."],
        stop_reason=WORKER_HANDOFF_REASON,
    )
    handoff = build_worker_handoff(
        result=result,
        turns=[turn],
        ledger=ledger,
        context=child,
        research_state={"needs": [], "findings": [], "source_conditions": []},
        working_memory=memory_state,
        draft=None,
    )
    tasks: list[dict[str, JsonValue]] = [
        {
            "task_id": "actual-worker",
            "task": "Yalnız ödeme ve tecil koşullarını araştır.",
            "status": "partial",
            "outcome_ids": ["payment"],
            "need_ids": ["law"],
            "assignment_id": "assignment",
            "parent_task_id": "root",
            "outcome": {
                "status": "partial",
                "summary": "Actual handoff",
                "data": {"cropped_worker_handoff": handoff},
            },
        }
    ]
    view = HarnessView(
        request=request,
        questions=[request],
        facts=["Kullanıcının ödeme beyanı"],
        receipts=[],
        evidence=[],
        tools=[],
        research_state={"needs": [], "source_conditions": []},
    )
    return view, context, tasks, text


def test_actual_handoff_preserves_failed_batch_and_owner_without_body_delivery() -> (
    None
):
    view, context, tasks, text = fixture()
    ledger = cast(EvidenceLedger, context.services["evidence"])
    before = copy.deepcopy((view.model_dump(), tasks, ledger.export()))
    packet = build_cropped_read_planning_packet(view, context, research_tasks=tasks)
    operations = cast(list[dict[str, JsonValue]], packet["operations"])
    missing = next(
        operation
        for operation in operations
        if operation["name"] == "read_named_provision"
    )
    assert missing["arguments"] == {"source_name": "Özgün Kanun", "article": "7"}
    assert missing["call_id"] == missing["parent_call_id"] == "actual-batch"
    assert missing["step_id"] == "missing"
    assert missing["status"] == "not_found"
    assert missing["data"] == {
        "sources": [],
        "absence_proven": False,
        "has_more": False,
    }
    resolution = next(
        operation for operation in operations if operation["name"] == "resolve_source"
    )
    assert cast(dict[str, JsonValue], resolution["data"])["next_offset"] == 2
    assert (
        len(
            cast(
                list[JsonValue],
                cast(dict[str, JsonValue], resolution["data"])["sources"],
            )
        )
        == 2
    )
    assert packet["planning_targets_available"] is True
    assert packet["request"] == view.request
    assert packet["facts"] == view.facts
    assert (
        cast(list[dict[str, JsonValue]], packet["research_tasks"])[0]["task"]
        == tasks[0]["task"]
    )
    assert cast(list[dict[str, JsonValue]], packet["research_tasks"])[0][
        "outcome_ids"
    ] == ["payment"]
    assert (
        cast(list[dict[str, JsonValue]], packet["assistant_recommendations"])[0][
            "assistant_content"
        ]
        == "Kimliği çözmeden tecil sonucu verme; tam kanunu oku."
    )
    wire = serialize_cropped_read_planning_packet(packet)
    assert text not in wire
    assert "handoff_hash" not in wire and "source_bindings" not in wire
    assert "original_evidence" not in packet
    assert packet["excluded_original_bodies"] == {
        "citations": [1, 2],
        "unread_to_planner": True,
    }
    assert packet["legal_approval"] is packet["corpus_absence_proven"] is False
    visible = json.loads(wire)
    visible_operations = [
        dict(zip(visible["operation_fields"], row, strict=True))
        for row in visible["operations"]
    ]
    visible_missing = next(
        operation
        for operation in visible_operations
        if operation["name"] == "read_named_provision"
    )
    assert (
        visible_missing["call_id"]
        == visible_missing["parent_call_id"]
        == "recorded_call_1"
    )
    assert visible_missing["arguments"] == missing["arguments"]
    assert visible_missing["data"] == missing["data"]
    assert before == (view.model_dump(), tasks, ledger.export())


def test_reference_nav_deduplicates_target_retains_all_exact_source_witnesses() -> None:
    view, context, tasks, _ = fixture()
    packet = build_cropped_read_planning_packet(view, context, research_tasks=tasks)
    refs = cast(list[dict[str, JsonValue]], packet["source_reference_navigation"])
    assert len(refs) == 1
    assert refs[0]["target"] == {"instrument_number": "9876", "article": "7"}
    assert refs[0]["witness_citations"] == [1, 2]
    assert refs[0]["exact_witnesses"] == [
        {"citation": n, "source_quote": "9876 sayılı Kanunun 7 inci maddesi"}
        for n in (1, 2)
    ]
    cards = cast(list[dict[str, JsonValue]], packet["source_cards"])
    assert cards == [
        {
            "source_id": "own-source",
            "common_metadata": {
                "title": "Özgün mevzuat",
                "article_no": "2",
                "version_unknown": True,
            },
            "metadata_variants": [{}],
            "units": [[1, "own-chunk", 0], [2, "second-chunk", 0]],
        }
    ]


@pytest.mark.parametrize("tamper", ["hash", "task", "scope", "text"])
def test_foreign_or_changed_handoff_fails_before_planning(tamper: str) -> None:
    view, context, tasks, _ = fixture()
    data = cast(
        dict[str, JsonValue], cast(dict[str, JsonValue], tasks[0]["outcome"])["data"]
    )
    handoff = cast(dict[str, JsonValue], data["cropped_worker_handoff"])
    if tamper == "hash":
        handoff["handoff_hash"] = "0" * 64
    elif tamper == "task":
        tasks[0]["task_id"] = "foreign-worker"
    elif tamper == "scope":
        context.scope = {"owner": "foreign-user"}
    else:
        other = EvidenceLedger()
        other.add(
            [
                EvidenceItem(
                    source_id="own-source", chunk_id="own-chunk", text="Changed"
                )
            ],
            context,
        )
        context.services["evidence"] = other
    with pytest.raises(ValueError):
        build_cropped_read_planning_packet(view, context, research_tasks=tasks)


def test_cards_alone_do_not_trigger_planner_and_unmatched_result_survives() -> None:
    view, context, _, _ = fixture()
    view.request = "Genel değerlendirme."
    view.questions = [view.request]
    view.turns = [
        ResearchTurn(
            assistant=AssistantMessage(
                content="Eksik ödeme tarihi önemli.",
                tool_calls=[
                    ToolCall(
                        id="uncommitted",
                        function=FunctionCall(
                            name="read_named_provision",
                            arguments='{"source_name":"Gerçek ad","article":"3"}',
                        ),
                    )
                ],
            ),
            results=[
                ToolMessage(
                    tool_call_id="uncommitted",
                    content='{"status":"not_found","absence_proven":false}',
                )
            ],
        )
    ]
    packet = build_cropped_read_planning_packet(view, context, research_tasks=[])
    assert packet["planning_targets_available"] is False
    recommendations = cast(
        list[dict[str, JsonValue]], packet["assistant_recommendations"]
    )
    assert recommendations[0]["assistant_content"] == "Eksik ödeme tarihi önemli."
    assert "unexecuted_assistant_tool_call" in recommendations[1]
    assert "unmatched_recorded_result" in recommendations[2]


@pytest.mark.parametrize("depth,cropped", [(1, True), (0, False)])
def test_root_cropped_scope_is_required(depth: int, cropped: bool) -> None:
    view, context, _, _ = fixture()
    context.depth = depth
    context.services["asv3_exp_cropped"] = cropped
    with pytest.raises(ValueError):
        build_cropped_read_planning_packet(view, context, research_tasks=[])


def test_oversized_handoff_uses_committed_root_receipts_before_view_clipping() -> None:
    view, context, tasks, _ = fixture()
    ledger = cast(EvidenceLedger, context.services["evidence"])
    root_call = CapabilityCall(
        name="resolve_source",
        call_id="real-root-resolution",
        arguments={"query": "Özgün Kanun", "offset": 2},
    )
    root_receipt = ToolReceipt(
        call=root_call,
        elapsed_seconds=0.1,
        outcome=ToolOutcome(
            status=OutcomeStatus.AMBIGUOUS,
            summary="Identity candidates are not original law.",
            data={
                "sources": [{"source_id": "own-source", "name": "Özgün mevzuat"}],
                "has_more": True,
                "next_offset": 3,
                "absence_proven": False,
            },
        ),
    )
    latest = ToolReceipt(
        call=CapabilityCall(
            name="wait_researcher",
            call_id="real-root-wait",
            arguments={"task_id": "actual-worker"},
        ),
        elapsed_seconds=0.1,
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual worker state",
            data={"tasks": tasks, "host_diagnostic": "x" * 21000},
        ),
    )
    root_turn = ResearchTurn(
        assistant=AssistantMessage(
            content="Kimliği çözmeden ödeme sonucunu vermeyin.",
            tool_calls=[
                ToolCall(
                    id=root_call.call_id,
                    function=FunctionCall(
                        name=root_call.name, arguments=json.dumps(root_call.arguments)
                    ),
                ),
                ToolCall(
                    id="genuine-uncommitted",
                    function=FunctionCall(name="read_chunk", arguments='{"id":"lead"}'),
                ),
            ],
        ),
        results=[
            ToolMessage(
                tool_call_id=root_call.call_id,
                content=root_receipt.outcome.model_dump_json(),
            ),
            ToolMessage(
                tool_call_id="genuine-uncommitted",
                content='{"status":"partial","has_more":true,"next_offset":4}',
            ),
        ],
    )
    harness = Harness(
        request=view.request,
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _: Decision(answer="Not invoked"),
        evidence=ledger,
    )
    harness.questions = view.questions
    harness.facts = view.facts
    harness.receipts = [root_receipt, latest]
    harness.turns = [root_turn]
    clipped = harness.view()
    assert clipped.receipts == []
    before = copy.deepcopy((harness.snapshot(), ledger.export(), tasks))
    snapshot = snapshot_cropped_planning_receipts(harness)
    packet = build_cropped_read_planning_packet(
        clipped, context, research_tasks=tasks, committed_receipt_snapshot=snapshot
    )
    operations = cast(list[dict[str, JsonValue]], packet["operations"])
    actual = next(x for x in operations if x["call_id"] == root_call.call_id)
    assert actual["arguments"] == root_call.arguments
    assert actual["data"] == root_receipt.outcome.data
    assert actual["status"] == "ambiguous"
    recommendations = cast(
        list[dict[str, JsonValue]], packet["assistant_recommendations"]
    )
    assert any(
        x.get("assistant_content") == root_turn.assistant.content
        for x in recommendations
    )
    assert not any(
        root_call.call_id in json.dumps(x, ensure_ascii=False) for x in recommendations
    )
    assert any("genuine-uncommitted" in json.dumps(x) for x in recommendations)
    assert any("unmatched_recorded_result" in x for x in recommendations)
    assert packet["planning_targets_available"] is True
    assert before == (harness.snapshot(), ledger.export(), tasks)
    assert snapshot.receipts[0] == root_receipt
    assert snapshot.receipts[0] is not root_receipt


@pytest.mark.parametrize(
    "change", ["run", "scope", "request", "owner", "depth", "user"]
)
def test_committed_snapshot_fences_owner_run_request_and_scope(change: str) -> None:
    view, context, tasks, _ = fixture()
    harness = Harness(
        request=view.request,
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _: Decision(answer="Not invoked"),
        evidence=cast(EvidenceLedger, context.services["evidence"]),
    )
    snapshot = snapshot_cropped_planning_receipts(harness)
    if change == "run":
        snapshot = replace(snapshot, run_id="other-run")
    elif change == "scope":
        snapshot = replace(snapshot, scope_hash="other-scope")
    elif change == "request":
        snapshot = replace(snapshot, request="other-question")
    elif change == "owner":
        snapshot = replace(snapshot, owner_task_id="worker")
    elif change == "depth":
        snapshot = replace(snapshot, depth=1)
    else:
        snapshot = replace(snapshot, user_id="foreign-user")
    with pytest.raises(ValueError, match="Foreign Cropped"):
        build_cropped_read_planning_packet(
            view, context, research_tasks=tasks, committed_receipt_snapshot=snapshot
        )


def test_committed_receipt_conflicts_fail_and_default_projection_is_unchanged() -> None:
    view, context, tasks, _ = fixture()
    harness = Harness(
        request=view.request,
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _: Decision(answer="Not invoked"),
        evidence=cast(EvidenceLedger, context.services["evidence"]),
    )
    receipt = ToolReceipt(
        call=CapabilityCall(name="read_chunk", call_id="owned", arguments={"id": "1"}),
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Actual read", data={"has_more": False}
        ),
        elapsed_seconds=0,
    )
    harness.receipts = [receipt, receipt.model_copy(update={"evidence_ids": [1]})]
    with pytest.raises(ValueError, match="Conflicting committed"):
        build_cropped_read_planning_packet(
            view,
            context,
            research_tasks=tasks,
            committed_receipt_snapshot=snapshot_cropped_planning_receipts(harness),
        )
    assert build_cropped_read_planning_packet(view, context, research_tasks=tasks) == (
        build_cropped_read_planning_packet(
            view, context, research_tasks=tasks, committed_receipt_snapshot=None
        )
    )


def test_lossless_planning_navigation_keeps_all_candidates_types_and_cursors() -> None:
    import tiktoken

    view, context, _, _ = fixture()
    view.receipts = [
        ToolReceipt(
            call=CapabilityCall(
                name="resolve_source",
                call_id=f"actual-resolution-{n}",
                arguments={"query": "Exact historical source title", "offset": n},
            ),
            elapsed_seconds=0,
            outcome=ToolOutcome(
                status=OutcomeStatus.AMBIGUOUS,
                summary="Actual identity leads, never originals",
                data={
                    "sources": [
                        {
                            "source_id": "actual-canonical-source-id",
                            "name": "Full canonical source title with all scope and date qualifiers",
                            "version_unknown": True,
                            "validity_end": None,
                            "literal_marker": {"s": 0, "o": [0, []]},
                        }
                    ],
                    "has_more": True,
                    "next_offset": n + 1,
                    "absence_proven": False,
                    "empty_array": [],
                },
            ),
        )
        for n in range(40)
    ]
    packet = build_cropped_read_planning_packet(view, context, research_tasks=[])
    before = copy.deepcopy(packet)
    counter = tiktoken.get_encoding("cl100k_base")

    def count(text: str) -> int:
        return len(counter.encode(text))

    ordinary = serialize_cropped_read_planning_packet(packet)
    wire = serialize_cropped_read_planning_packet(packet, count)
    assert count(wire) < count(ordinary)
    encoded = json.loads(wire)
    assert "planning_navigation_codec" in encoded
    assert expand_cropped_planning_navigation(encoded) == json.loads(ordinary)
    assert before == packet
    assert "original_evidence" not in encoded
    assert encoded["legal_approval"] is False
    changed = copy.deepcopy(encoded)
    changed["operations"] = {"s": 1000000}
    with pytest.raises(ValueError, match="string reference"):
        expand_cropped_planning_navigation(changed)
    assert serialize_cropped_read_planning_packet(packet, lambda _: 0) == ordinary


def test_snapshot_allows_same_scope_resume_and_rejects_conflicting_visible_receipt() -> (
    None
):
    view, context, tasks, _ = fixture()
    harness = Harness(
        request=view.request,
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _: Decision(answer="Not invoked"),
        evidence=cast(EvidenceLedger, context.services["evidence"]),
    )
    actual = ToolReceipt(
        call=CapabilityCall(name="read_chunk", call_id="owned", arguments={"id": "1"}),
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual complete read",
            data={"has_more": False},
        ),
        elapsed_seconds=0,
    )
    harness.receipts = [actual]
    snapshot = snapshot_cropped_planning_receipts(harness)
    resumed = RunContext(
        run_id=context.run_id, scope=context.scope, services=context.services
    )
    snapshot.validate(resumed, view.request)
    view.receipts = [
        actual.model_copy(
            update={
                "call": actual.call.model_copy(update={"arguments": {"id": "foreign"}})
            }
        )
    ]
    with pytest.raises(ValueError, match="differs from committed"):
        build_cropped_read_planning_packet(
            view, context, research_tasks=tasks, committed_receipt_snapshot=snapshot
        )
