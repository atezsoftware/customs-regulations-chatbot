"""Worker lifecycle handoffs preserve actual observations without legal approval."""

import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3 import cropped_worker_handoff as handoff_module
from onyx.asv3.cropped_cost_budget import CroppedCostBudget
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.cropped_worker_handoff import (
    WORKER_HANDOFF_REASON,
    CroppedWorkerHandoff,
    build_worker_handoff,
    contains_worker_handoff,
    expand_worker_handoff,
    should_handoff_worker,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    HarnessResult,
    OriginalEvidenceRead,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall, ToolMessage


def setup() -> tuple[RunContext, CroppedResearchLimit, EvidenceLedger]:
    root = RunContext(run_id="actual-run", scope={"owner": "onur", "document_set": 15})
    controller = CroppedResearchLimit(root, "Actual question")
    ledger = EvidenceLedger()
    root.services.update(
        asv3_exp_cropped=True, cropped_research_controller=controller, evidence=ledger
    )
    child = root.child()
    child.services.update(
        task_id="actual-worker",
        task_need_ids=["governing"],
        task_outcome_ids=["outcome1"],
        parent_task_id="parent-worker",
        assignment_id="assigned",
    )
    return child, controller, ledger


def test_observed_volume_alone_never_hands_off_a_v13_worker() -> None:
    context, controller, _ = setup()
    for _ in range(12):
        controller.reserve_model("actual-worker", context=context)
    for index in range(4):
        controller.record_search(f"query-{index}")
    controller.record_sources(
        [
            EvidenceItem(
                source_id="law", chunk_id=str(index), text=f"Complete original {index}"
            )
            for index in range(300)
        ]
    )
    assert controller.long_tail and not controller.forced_finish
    assert not should_handoff_worker(context)
    controller.force_finish("actual input context cannot fit")
    assert should_handoff_worker(context)


def payload() -> tuple[
    RunContext,
    EvidenceLedger,
    HarnessResult,
    list[ResearchTurn],
    dict[str, JsonValue],
    dict[str, JsonValue],
    str,
]:
    context, controller, ledger = setup()
    controller.force_finish("actual input overflow")
    text = 'MADDE 8 — Türkçe özgün "kural".\nİstisna ve koşul korunur. ' * 40
    number = ledger.add(
        [
            EvidenceItem(
                source_id="own-law",
                chunk_id="own-chunk",
                text=text,
                metadata={"article_no": "8"},
            )
        ],
        context,
    )[0]
    item = ledger.get(number)
    assert item is not None
    receipt = ToolReceipt(
        call=CapabilityCall(
            name="read_provision",
            call_id="actual-read",
            arguments={"source_id": "own-law", "article": "8", "_need_id": "governing"},
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Actual closure incomplete",
            data={"exact_text": text, "has_more": True},
            original_reads=[
                OriginalEvidenceRead(
                    citation=number,
                    text_hash=item.text_hash,
                    start_char=0,
                    end_char=len(text),
                )
            ],
        ),
        elapsed_seconds=0.2,
        evidence_ids=[number],
    )
    facts: list[str] = ["Actual source-bound candidate observation. " * 450]
    result = HarnessResult(
        answer=None,
        status=OutcomeStatus.PARTIAL,
        receipts=[receipt],
        questions=["Actual subquestion"],
        facts=facts,
        stop_reason=WORKER_HANDOFF_REASON,
        publication_gap=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Actual unresolved original",
            data={"not_read": True, "absence_proven": False},
        ),
    )
    turns = [
        ResearchTurn(
            assistant=AssistantMessage(
                content="Actual model reasoning; exception remains unresolved.",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        function=FunctionCall(
                            name="read_provision",
                            arguments='{"article":"8","num_chars":512}',
                        ),
                    )
                    for call_id in ("actual-read", "escaped-read")
                ],
            ),
            results=[
                ToolMessage(
                    tool_call_id="actual-read",
                    content=' {"text":'
                    + json.dumps(text, ensure_ascii=False)
                    + ', "has_more": true}\n',
                ),
                ToolMessage(
                    tool_call_id="escaped-read",
                    content=json.dumps({"text": text}, ensure_ascii=True),
                ),
            ],
        )
    ]
    state: dict[str, JsonValue] = {
        "needs": [
            {"need_id": "governing", "status": "researching", "gap": "Closure unread"}
        ],
        "findings": [
            {"statement": "Actual candidate", "witnesses": [{"citation": number}]}
        ],
        "source_conditions": [
            {"status": "unknown", "exception": "Actual unresolved condition"}
        ],
    }
    memory: dict[str, JsonValue] = {
        "locators": [
            {"receipt_id": "actual-read", "has_more": True, "next_position": 10}
        ],
        "archived_locators": 2,
    }
    return (
        context,
        ledger,
        result,
        turns,
        state,
        memory,
        "Unapproved candidate only; no paragraph closure.",
    )


def build() -> tuple[
    dict[str, JsonValue],
    RunContext,
    EvidenceLedger,
    HarnessResult,
    list[ResearchTurn],
    dict[str, JsonValue],
    dict[str, JsonValue],
    str,
]:
    context, ledger, result, turns, state, memory, draft = payload()
    data = build_worker_handoff(
        context=context,
        result=result,
        ledger=ledger,
        turns=turns,
        research_state=state,
        working_memory=memory,
        draft=draft,
    )
    return data, context, ledger, result, turns, state, memory, draft


def test_only_ordinary_cropped_workers_handoff_after_shared_tail() -> None:
    context, controller, _ = setup()
    assert not should_handoff_worker(context)
    controller.force_finish("recorded long tail")
    assert should_handoff_worker(context)
    context.services["independent_question"] = True
    assert not should_handoff_worker(context)
    context.services["independent_question"] = False
    context.depth = 0
    assert not should_handoff_worker(context)
    context.depth = 1
    context.services["asv3_exp_cropped"] = False
    assert not should_handoff_worker(context)


def test_worker_controller_run_and_scope_fences_are_required() -> None:
    context, _, _ = setup()
    context.scope["owner"] = "foreign"
    with pytest.raises(ValueError, match="scope"):
        should_handoff_worker(context)
    context, _, _ = setup()
    context.run_id = "foreign"
    with pytest.raises(ValueError, match="run"):
        should_handoff_worker(context)
    context, _, _ = setup()
    context.services.pop("cropped_research_controller")
    with pytest.raises(ValueError, match="controller"):
        should_handoff_worker(context)


def test_same_scope_cost_budget_does_not_create_false_scope_mismatch() -> None:
    context, _, _ = setup()
    context.services["cropped_cost_budget"] = CroppedCostBudget(
        context, "Actual question"
    )
    assert not should_handoff_worker(context)
    context.services["cropped_cost_budget"] = CroppedCostBudget(
        RunContext(run_id="foreign"), "Other question"
    )
    with pytest.raises(ValueError, match="scope"):
        should_handoff_worker(context)


def test_exact_receipts_turn_strings_observations_draft_and_unresolved_states_roundtrip() -> (
    None
):
    data, context, ledger, result, turns, state, memory, draft = build()
    decoded = expand_worker_handoff(data, ledger=ledger, context=context)
    assert decoded["receipts"] == [
        row.model_dump(mode="json") for row in result.receipts
    ]
    assert decoded["turns"] == [turn.model_dump(mode="json") for turn in turns]
    assert decoded["facts"] == result.facts and len(result.facts[0]) > 12000
    assert decoded["research_state"] == state and decoded["working_memory"] == memory
    assert decoded["unapproved_draft"] == draft and decoded["draft_approved"] is False
    assert decoded["task_need_ids"] == ["governing"] and decoded[
        "task_outcome_ids"
    ] == ["outcome1"]
    assert result.publication_gap is not None
    assert decoded["publication_gap"] == result.publication_gap.model_dump(mode="json")
    assert (
        decoded["assignment_id"] == "assigned"
        and decoded["parent_task_id"] == "parent-worker"
    )
    assert data["absence_proven"] is False and data["research_complete"] is False
    assert data["publication_approved"] is False
    assert data["source_bindings_are_model_delivery_receipts"] is False
    packed = cast(dict[str, JsonValue], data["handoff"])
    references = cast(list[dict[str, JsonValue]], packed["source_string_references"])
    json_references = cast(list[dict[str, JsonValue]], packed["json_string_references"])
    nested_source_count = sum(
        len(
            cast(
                list[JsonValue],
                cast(dict[str, JsonValue], row["value"])["source_string_references"],
            )
        )
        for row in json_references
    )
    assert len(references) + nested_source_count == 3
    assert len(json.dumps(data, ensure_ascii=False)) < len(
        json.dumps(decoded, ensure_ascii=False)
    )


def test_source_references_fail_closed_against_changed_ledger_or_source_metadata() -> (
    None
):
    data, context, ledger, *_ = build()
    foreign = EvidenceLedger()
    context.services["evidence"] = foreign
    with pytest.raises(ValueError, match="unavailable"):
        expand_worker_handoff(data, ledger=foreign, context=context)
    context.services["evidence"] = ledger
    snapshot = ledger.export()
    records = cast(list[dict[str, JsonValue]], snapshot["records"])
    item = cast(dict[str, JsonValue], records[0]["item"])
    item["metadata"] = {"article_no": "9"}
    altered = EvidenceLedger()
    altered.restore(snapshot, context)
    context.services["evidence"] = altered
    with pytest.raises(ValueError, match="identity"):
        expand_worker_handoff(data, ledger=altered, context=context)


def test_content_reference_hash_tampering_and_foreign_scope_are_rejected() -> None:
    data, context, ledger, *_ = build()
    altered = copy.deepcopy(data)
    altered["facts"] = ["Invented finding"]
    with pytest.raises(ValueError, match="integrity"):
        expand_worker_handoff(altered, ledger=ledger, context=context)
    context.scope["owner"] = "foreign"
    with pytest.raises(ValueError, match="scope"):
        expand_worker_handoff(data, ledger=ledger, context=context)


@pytest.mark.parametrize(
    "answer,status,reason",
    [
        ("Fake completed answer", OutcomeStatus.FOUND, WORKER_HANDOFF_REASON),
        (None, OutcomeStatus.CANCELLED, WORKER_HANDOFF_REASON),
        (None, OutcomeStatus.PARTIAL, "deadline"),
    ],
)
def test_only_actual_partial_handoff_marker_accepts_payload(
    answer: str | None, status: OutcomeStatus, reason: str
) -> None:
    context, ledger, result, turns, state, memory, draft = payload()
    result.answer, result.status, result.stop_reason = answer, status, reason
    with pytest.raises(ValueError, match="actual nonpublication"):
        build_worker_handoff(
            context=context,
            result=result,
            ledger=ledger,
            turns=turns,
            research_state=state,
            working_memory=memory,
            draft=draft,
        )


def test_race_exception_is_explicit_handoff_not_cancellation() -> None:
    error = CroppedWorkerHandoff()
    assert str(error) == WORKER_HANDOFF_REASON
    assert "cancel" not in str(error)


def test_handoff_locations_are_explicit_task_scoped_and_fail_closed() -> None:
    handoff, context, ledger, *_ = build()
    direct: dict[str, JsonValue] = {"cropped_worker_handoff": handoff}
    assert contains_worker_handoff(direct, ledger=ledger, context=context)
    context.depth = 0
    tasks: dict[str, JsonValue] = {
        "tasks": [
            {
                "task_id": "actual-worker",
                "status": "completed",
                "outcome": {"status": "partial", "data": direct},
            }
        ]
    }
    assert contains_worker_handoff(tasks, ledger=ledger, context=context)
    assert not contains_worker_handoff(
        {"unrelated": direct}, ledger=ledger, context=context
    )
    wrong: dict[str, JsonValue] = {
        "tasks": [
            {
                "task_id": "another-worker",
                "outcome": {"status": "partial", "data": direct},
            }
        ]
    }
    with pytest.raises(ValueError, match="task identity"):
        contains_worker_handoff(wrong, ledger=ledger, context=context)
    with pytest.raises(ValueError, match="task identity"):
        contains_worker_handoff(
            {"cropped_worker_handoff": {}}, ledger=ledger, context=context
        )
    context.services["asv3_exp_cropped"] = False
    assert not contains_worker_handoff(tasks, ledger=ledger, context=context)


def rebuild(
    context: RunContext,
    ledger: EvidenceLedger,
    result: HarnessResult,
    turns: list[ResearchTurn],
    state: dict[str, JsonValue],
    memory: dict[str, JsonValue],
    draft: str,
) -> dict[str, JsonValue]:
    return build_worker_handoff(
        context=context,
        result=result,
        ledger=ledger,
        turns=turns,
        research_state=state,
        working_memory=memory,
        draft=draft,
    )


@pytest.mark.parametrize(
    "ensure_ascii,compact", [(False, False), (True, False), (False, True), (True, True)]
)
def test_json_tool_string_exact_formats_restore_every_original_byte(
    ensure_ascii: bool, compact: bool
) -> None:
    context, ledger, result, turns, state, memory, draft = payload()
    item = ledger.get(1)
    assert item is not None
    text = item.text
    literal = json.dumps(
        {
            "own_text": text,
            "metadata": {
                "law": "Türkçe özgün",
                "exception": "Koşul korunur",
                "navigation": ["8", "8/A"],
            },
        },
        ensure_ascii=ensure_ascii,
        separators=(",", ":") if compact else None,
    )
    turns[0].results[0].content = literal
    data = rebuild(context, ledger, result, turns, state, memory, draft)
    assert data["version"] == 2
    packed = cast(dict[str, JsonValue], data["handoff"])
    reference = next(
        row
        for row in cast(list[dict[str, JsonValue]], packed["json_string_references"])
        if row["path"] == ["turns", 0, "results", 0, "content"]
    )
    assert reference["ensure_ascii"] is ensure_ascii and reference["compact"] is compact
    nested = cast(dict[str, JsonValue], reference["value"])
    assert nested["source_string_references"]
    assert cast(dict[str, JsonValue], nested["content"])["metadata"] == {
        "law": "Türkçe özgün",
        "exception": "Koşul korunur",
        "navigation": ["8", "8/A"],
    }
    restored = expand_worker_handoff(data, ledger=ledger, context=context)
    assert restored["turns"] == [turn.model_dump(mode="json") for turn in turns]
    actual_turn = ResearchTurn.model_validate(
        cast(list[JsonValue], restored["turns"])[0]
    )
    assert actual_turn.results[0].content.encode() == literal.encode()


@pytest.mark.parametrize(
    "literal",
    [
        ' {"note": "free prose"} ' * 25,
        json.dumps({"data": "Özgün koşul " * 100}, indent=2),
        '{"same":"' + "x" * 600 + '","same":"second"}',
        json.dumps("JSON scalar remains prose " * 50),
    ],
)
def test_unrecognized_json_spelling_remains_exact_literal_not_structured(
    literal: str,
) -> None:
    context, ledger, result, turns, state, memory, draft = payload()
    turns[0].results[0].content = literal
    data = rebuild(context, ledger, result, turns, state, memory, draft)
    packed = cast(dict[str, JsonValue], data["handoff"])
    assert all(
        row["path"] != ["turns", 0, "results", 0, "content"]
        for row in cast(list[dict[str, JsonValue]], packed["json_string_references"])
    )
    assert expand_worker_handoff(data, ledger=ledger, context=context)["turns"] == [
        turn.model_dump(mode="json") for turn in turns
    ]


def test_json_string_resealed_format_or_placeholder_tampering_fails_closed() -> None:
    data, context, ledger, *_ = build()
    for change in ("format", "placeholder"):
        altered = copy.deepcopy(data)
        packed = cast(dict[str, JsonValue], altered["handoff"])
        reference = cast(list[dict[str, JsonValue]], packed["json_string_references"])[
            0
        ]
        if change == "format":
            reference["compact"] = not reference["compact"]
        else:
            reference["path"] = ["draft_approved"]
        packed["packed_hash"] = handoff_module._hash(
            {key: value for key, value in packed.items() if key != "packed_hash"}
        )
        altered["handoff_hash"] = handoff_module._hash(
            {key: value for key, value in altered.items() if key != "handoff_hash"}
        )
        with pytest.raises(
            ValueError, match="JSON string did not reconstruct|placeholder"
        ):
            expand_worker_handoff(altered, ledger=ledger, context=context)


def test_version_one_persisted_literal_handoff_still_restores_exactly() -> None:
    data, context, ledger, *_ = build()
    actual = expand_worker_handoff(data, ledger=ledger, context=context)
    packed: dict[str, JsonValue] = {
        "content": actual,
        "source_string_references": [],
        "original_hash": handoff_module._hash(actual),
    }
    packed["packed_hash"] = handoff_module._hash(packed)
    data.update(version=1, handoff=packed)
    data["handoff_hash"] = handoff_module._hash(
        {key: value for key, value in data.items() if key != "handoff_hash"}
    )
    assert expand_worker_handoff(data, ledger=ledger, context=context) == actual
    data["version"] = 2
    data["handoff_hash"] = handoff_module._hash(
        {key: value for key, value in data.items() if key != "handoff_hash"}
    )
    with pytest.raises(ValueError, match="grammar"):
        expand_worker_handoff(data, ledger=ledger, context=context)
