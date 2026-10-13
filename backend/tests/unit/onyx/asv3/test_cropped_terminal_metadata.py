"""Contradictory terminal metadata is corrected without legal answer regeneration."""

import copy
import json
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.cropped_repair_model import fresh_model
from onyx.asv3.cropped_terminal_metadata import (
    prepare_cropped_terminal_metadata_repair,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.retained_answer import resolve_retained_answer
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import UserMessage
from tests.unit.onyx.asv3.test_native_model_adapter import model, native_response

GAP = "Olay tarihi, transit beyanı ve tütünün gümrük statüsü bilinmiyor."
ANSWER = "İzin ve bildirim koşulları uygulanır [1]. Tarih ve eşyanın statüsü bilinmiyor.\n\nÖncelik ayrıca değerlendirilir [2]."


def fixture() -> tuple[
    ResearchModel,
    HarnessView,
    EvidenceLedger,
    OutcomeMap,
    CapabilityRegistry,
    MagicMock,
]:
    context = RunContext(
        scope={"document_set": "owned", "as_of": "2026-10-13"},
        budget=SharedBudget(unlimited_execution=True),
    )
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="law", chunk_id="own-rule", text="İzin ve bildirim koşulları."
            ),
            EvidenceItem(
                source_id="law",
                chunk_id="priority",
                text="Öncelik ayrı değerlendirilir.",
            ),
        ],
        context,
    )
    outcomes = OutcomeMap(["Uygulama?", "Öncelik?"], context)
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [
                    {"outcome_id": "uyg", "question_ids": ["q0"], "detail": "Uygulama"},
                    {"outcome_id": "onc", "question_ids": ["q1"], "detail": "Öncelik"},
                ],
                "conditions": [
                    {
                        "condition_id": "izin",
                        "outcome_ids": ["uyg"],
                        "detail": "İzin koşulu korunur",
                        "witnesses": [{"citation": 1, "end_char": 4}],
                    }
                ],
            }
        ),
        ledger,
    )
    registry = CapabilityRegistry()

    def terminal(arguments: dict[str, JsonValue], run: RunContext) -> ToolOutcome:
        answer = str(arguments["answer"])
        writer = run.services.get("last_model_call_id")
        if not isinstance(writer, str) or not set(
            extract_citation_numbers(answer)
        ).issubset(ledger.completely_delivered(writer)):
            return ToolOutcome(
                status=OutcomeStatus.DENIED, summary="Writer did not read original"
            )
        gap = native_named_authority_gap(answer, ledger)
        return (
            ToolOutcome(
                status=OutcomeStatus.PARTIAL, summary="Named authority gap", data=gap
            )
            if gap
            else ToolOutcome(status=OutcomeStatus.FOUND, summary=answer)
        )

    registry.register(
        ToolSpec(
            name="submit_answer",
            description="Publish a source-grounded answer",
            parameters={
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "minLength": 1},
                    "basis": {"type": "string", "enum": ["originals"]},
                },
                "required": ["answer", "basis"],
                "additionalProperties": False,
            },
            handler=terminal,
            consumes_tool_budget=False,
        )
    )
    context.services.update(
        asv3_exp_cropped=True,
        lean_native_mode=True,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
        evidence=ledger,
        outcome_map=outcomes,
        registry=registry,
    )
    selected = model(limit=1000000)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    adapter._native_output_capacity = 8192
    records = cast(
        list[dict[str, JsonValue]], json.loads(ledger.serialize_records([1, 2]))
    )
    view = HarnessView(
        request="Uygulama ve öncelik?",
        questions=["Uygulama?", "Öncelik?"],
        facts=["Tarih ve eşyanın statüsü bilinmiyor."],
        receipts=[],
        evidence=[],
        original_evidence=records,
        tools=registry.definitions(context),
    )
    return adapter, view, ledger, outcomes, registry, selected


def arguments(answer: str = ANSWER) -> dict[str, JsonValue]:
    return {
        "answer": answer,
        "basis": "originals",
        "_coverage": {
            "resolutions": [
                {
                    "outcome_id": "uyg",
                    "status": "conditional",
                    "condition_ids": ["izin"],
                    "evidence_numbers": [1],
                    "gap": GAP,
                },
                {
                    "outcome_id": "onc",
                    "status": "supported",
                    "evidence_numbers": [2],
                },
            ]
        },
    }


def repaired(status: str = "conditional") -> dict[str, JsonValue]:
    coverage = arguments()["_coverage"]
    assert isinstance(coverage, dict)
    resolutions = copy.deepcopy(coverage["resolutions"])
    assert isinstance(resolutions, list) and isinstance(resolutions[0], dict)
    resolutions[0].update(status=status, gap="" if status == "conditional" else GAP)
    return {"resolutions": resolutions}


def patch_response(patch: dict[str, JsonValue] | None) -> ModelResponse:
    return ModelResponse(
        id="metadata-patch",
        created="0",
        choice=Choice(
            message=Message(
                content=json.dumps(
                    {
                        "entries": [
                            {
                                "call_id": "metadata:0",
                                "arguments_json": json.dumps(patch)
                                if patch is not None
                                else None,
                                "explanation": "Keep the stated uncertainty and correct only metadata.",
                            }
                        ]
                    }
                )
            )
        ),
    )


def owned(
    adapter: ResearchModel, view: HarnessView, ledger: EvidenceLedger
) -> Decision:
    adapter.last_call_id = "original-writer"
    adapter.context.services["last_model_call_id"] = "original-writer"
    ledger.record_delivery(
        "original-writer", "asv3_coordinator", view.original_evidence
    )
    return Decision(calls=[CapabilityCall(name="submit_answer", arguments=arguments())])


@pytest.mark.parametrize("status", ["conditional", "unresolved"])
def test_model_selects_metadata_status_without_mutating_answer_or_outcome_state(
    status: str,
) -> None:
    adapter, view, ledger, outcomes, registry, _selected = fixture()
    decision = owned(adapter, view, ledger)
    initial = outcomes.export()
    contract = prepare_cropped_terminal_metadata_repair(
        decision, view, adapter.context, adapter.last_call_id
    )
    assert contract is not None and outcomes.export() == initial
    result = contract.apply(repaired(status), adapter.context, registry)
    assert result.calls[0].arguments["answer"] == ANSWER
    assert outcomes.export() == initial
    assert result.calls[0].arguments["_coverage"] == {
        "resolutions": repaired(status)["resolutions"]
    }
    payload = json.loads(contract.payload_json)
    assert payload["preserved_gap_observations"] == [GAP]
    assert "original_evidence" not in payload
    assert registry.outcome_metadata_gap(result.calls[0], adapter.context) is None
    assert outcomes.export() != initial


@pytest.mark.parametrize(
    "mutation", ["answer", "evidence", "conditions", "gap", "escalate", "valid_row"]
)
def test_patch_cannot_rewrite_legal_content_or_erase_source_conditions(
    mutation: str,
) -> None:
    adapter, view, ledger, _outcomes, registry, _selected = fixture()
    contract = prepare_cropped_terminal_metadata_repair(
        owned(adapter, view, ledger), view, adapter.context, adapter.last_call_id
    )
    assert contract is not None
    patch = repaired()
    rows = cast(list[dict[str, JsonValue]], patch["resolutions"])
    if mutation == "answer":
        patch["answer"] = "A new legal answer"
    elif mutation == "evidence":
        rows[0]["evidence_numbers"] = [2]
    elif mutation == "conditions":
        rows[0]["condition_ids"] = []
    elif mutation == "gap":
        rows[0].update(status="unresolved", gap="A newly invented gap")
    elif mutation == "escalate":
        rows[0]["status"] = "supported"
    else:
        rows[1]["status"] = "unresolved"
    with pytest.raises(ValueError):
        contract.apply(patch, adapter.context, registry)


@pytest.mark.parametrize("mutation", ["scope", "writer", "text"])
def test_original_writer_and_source_identity_are_rechecked_after_patch(
    mutation: str,
) -> None:
    adapter, view, ledger, _outcomes, registry, _selected = fixture()
    contract = prepare_cropped_terminal_metadata_repair(
        owned(adapter, view, ledger), view, adapter.context, adapter.last_call_id
    )
    assert contract is not None
    if mutation == "scope":
        adapter.context.scope["document_set"] = "other"
    elif mutation == "writer":
        adapter.context.services["last_model_call_id"] = "patch-writer"
    else:
        item = ledger._items[1]
        item.text = item.text.replace("İzin", "Yasa")
    with pytest.raises(ValueError):
        contract.apply(repaired(), adapter.context, registry)


def test_unresolved_metadata_cannot_borrow_an_undelivered_condition_witness() -> None:
    adapter, view, ledger, outcomes, _registry, _selected = fixture()
    decision = owned(adapter, view, ledger)
    number = ledger.add(
        [
            EvidenceItem(
                source_id="law", chunk_id="unread", text="Another necessary condition."
            )
        ],
        adapter.context,
    )[0]
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "conditions": [
                    {
                        "condition_id": "unread",
                        "outcome_ids": ["uyg"],
                        "detail": "A retained condition is not evidence of writer delivery",
                        "witnesses": [{"citation": number, "end_char": 7}],
                    }
                ]
            }
        ),
        ledger,
    )
    with pytest.raises(ValueError, match="borrow another writer's sources"):
        prepare_cropped_terminal_metadata_repair(
            decision, view, adapter.context, adapter.last_call_id
        )


def test_native_before_dispatch_patch_retains_actual_delivery_and_avoids_focused_research() -> (
    None
):
    adapter, view, ledger, outcomes, registry, selected = fixture()
    selected.invoke.side_effect = [
        native_response("submit_answer", json.dumps(arguments())),
        patch_response(repaired()),
    ]
    initial = outcomes.export()
    prompt, tools, output = adapter._fit_native_decision(view)
    decision = adapter._invoke_decision(view, prompt, tools, output)
    assert selected.invoke.call_count == 2 and outcomes.export() == initial
    writer_id = adapter.last_call_id
    assert writer_id == adapter.context.services["last_model_call_id"]
    assert writer_id and ledger.completely_delivered(writer_id) == {1, 2}
    patch_args = selected.invoke.call_args.kwargs
    assert patch_args["tools"] is None and patch_args["max_tokens"] <= 2048
    assert (
        patch_args["structured_response_format"]["json_schema"]["name"]
        == "ToolArgumentPatch"
    )
    assert not any(
        isinstance(message, UserMessage) and "original_evidence" in str(message.content)
        for message in patch_args["prompt"]
    )
    repair = MagicMock(
        side_effect=AssertionError("Metadata must not start source research")
    )
    harness = Harness(
        request=view.request,
        context=adapter.context,
        registry=registry,
        evidence=ledger,
        decide=lambda _view: decision,
        repair_decision=repair,
        complete_submission=lambda: ANSWER,
    )
    result = harness.run()
    assert result.answer == ANSWER
    repair.assert_not_called()
    assert selected.invoke.call_count == 2


def test_small_metadata_patch_does_not_bypass_real_named_authority_gap() -> None:
    adapter, view, _ledger, _outcomes, registry, selected = fixture()
    body = ANSWER + "\n\n5326 sayılı Kabahatler Kanunu m.15/3 uygulanır [1]."
    selected.invoke.side_effect = [
        native_response("submit_answer", json.dumps(arguments(body))),
        patch_response(repaired()),
    ]
    prompt, tools, output = adapter._fit_native_decision(view)
    decision = adapter._invoke_decision(view, prompt, tools, output)
    outcome = registry.dispatch(decision.calls[0], adapter.context)
    assert outcome.status != OutcomeStatus.FOUND
    assert outcome.data.get("named_authority_gaps")
    assert selected.invoke.call_count == 2


@pytest.mark.parametrize("kind", ["malformed", "null", "rewrite", "truncated"])
def test_one_failed_metadata_patch_stops_without_format_or_source_retry(
    kind: str,
) -> None:
    adapter, view, _ledger, _outcomes, _registry, selected = fixture()
    response = patch_response(None if kind == "null" else repaired())
    if kind == "malformed":
        response.choice.message.content = "{broken"
    elif kind == "rewrite":
        response = patch_response({**repaired(), "answer": "Replace legal text"})
    elif kind == "truncated":
        response.choice.finish_reason = "length"
    selected.invoke.side_effect = [
        native_response("submit_answer", json.dumps(arguments())),
        response,
    ]
    prompt, tools, output = adapter._fit_native_decision(view)
    with pytest.raises(RunStopped, match="metadata correction failed"):
        adapter._invoke_decision(view, prompt, tools, output)
    assert selected.invoke.call_count == 2


def test_single_call_focused_writer_can_use_separate_metadata_patcher() -> None:
    adapter, view, _ledger, _outcomes, _registry, selected = fixture()
    focused = fresh_model(adapter)
    assert focused.context.services["registry"] is adapter.context.services["registry"]
    selected.invoke.side_effect = [
        native_response("submit_answer", json.dumps(arguments())),
        patch_response(repaired()),
    ]
    prompt, tools, output = focused._fit_native_decision(view)
    decision = focused._invoke_decision(view, prompt, tools, output)
    assert decision.calls[0].arguments["answer"] == ANSWER
    assert selected.invoke.call_count == 2


def test_retained_native_commit_preserves_exact_owned_body_and_writer_origin() -> None:
    adapter, view, ledger, _outcomes, _registry, selected = fixture()
    view = view.model_copy(
        update={"draft_to_repair": ANSWER, "publication_gap": {"source_gap": True}}
    )
    focused = fresh_model(adapter)
    args = arguments()
    args.pop("answer")
    args.pop("basis")
    selected.invoke.side_effect = [
        native_response("submit_retained_answer", json.dumps(args)),
        patch_response(repaired()),
    ]
    prompt, tools, output = focused._fit_native_decision(view)
    decision = focused._invoke_decision(view, prompt, tools, output)
    assert decision.calls[0].name == "submit_retained_answer"
    expanded = resolve_retained_answer(
        decision, focused.context, ANSWER, request=view.request
    )
    assert expanded.calls[0].arguments["answer"] == ANSWER
    assert focused.last_call_id == focused.context.services["last_model_call_id"]
    assert focused.last_call_id and ledger.completely_delivered(
        focused.last_call_id
    ) == {1, 2}
    assert selected.invoke.call_count == 2


@pytest.mark.parametrize("profile", ["healthy", "experimental", "cheap"])
def test_healthy_or_protected_terminal_has_no_patch_call_or_pre_dispatch_mutation(
    profile: str,
) -> None:
    adapter, view, _ledger, outcomes, _registry, selected = fixture()
    args = arguments()
    if profile == "healthy":
        args["_coverage"] = {"resolutions": repaired()["resolutions"]}
    elif profile == "experimental":
        adapter.context.services.pop("asv3_exp_cropped")
    else:
        adapter.context.services.pop("asv3_exp_cropped")
        adapter.context.services["asv3_exp_cheap"] = True
    selected.invoke.return_value = native_response("submit_answer", json.dumps(args))
    initial = outcomes.export()
    prompt, tools, output = adapter._fit_native_decision(view)
    result = adapter._invoke_decision(view, prompt, tools, output)
    assert selected.invoke.call_count == 1 and outcomes.export() == initial
    assert result.calls[0].arguments == args
