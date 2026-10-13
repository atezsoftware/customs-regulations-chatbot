"""The real working-view and native paths share one selected source contract."""

import json
from typing import cast

import pytest
import tiktoken
from pydantic import JsonValue

from onyx.asv3.cropped_transport import prepare_cropped_writer_view
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.models import (
    Decision,
    EvidenceItem,
    HarnessView,
    OriginalEvidenceRead,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import (
    OutcomeCondition,
    OutcomeMap,
    OutcomeUpdate,
    OutcomeWitness,
    RequestedOutcome,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchState
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    FunctionCall,
    ToolCall,
    ToolMessage,
)
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cropped_task_packets import (
    REQUEST,
    fixture,
    receipt,
    topics,
    view,
)
from tests.unit.onyx.asv3.test_native_model_adapter import model


def harness_fixture(
    *, dependency: bool = False
) -> tuple[Harness, ResearchModel, HarnessView, EvidenceLedger]:
    context, _, _, store = fixture()
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="source-1",
                chunk_id="chunk-1",
                text="Topic A includes its operative condition and exception.",
            ),
            EvidenceItem(
                source_id="source-2",
                chunk_id="chunk-2",
                text="\n".join(
                    f"Topic B paragraph {n}: preserve the condition, actor, exception and later procedure."
                    for n in range(700)
                ),
            ),
            EvidenceItem(
                source_id="source-3",
                chunk_id="chunk-3",
                text="An unbound material condition must remain available.",
            ),
        ],
        context,
    )
    state = ResearchState([REQUEST], context)
    topics(state, ledger, depends=["B"] if dependency else None, third=dependency)
    context.services.update(
        research_state=state,
        evidence=ledger,
        lean_native_mode=True,
        research_profile="normal",
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
    )
    store.record_input("root", 1_000_000)
    selected = model(limit=1_000_000)
    adapter = ResearchModel(
        selected,
        context,
        lean_native_mode=True,
        token_counter=lambda text: len(
            tiktoken.get_encoding("o200k_base").encode(text)
        ),
    )
    registry = CapabilityRegistry()

    def read(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        number = int(str(arguments["article"]))
        item = ledger.get(number)
        assert item is not None
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual original read",
            data={
                "has_more": False,
                "original_evidence": json.loads(
                    ledger.serialize_records([number], max_chars=None)
                ),
            },
            evidence=[item],
            original_reads=[
                OriginalEvidenceRead(
                    citation=number,
                    text_hash=item.text_hash,
                    start_char=0,
                    end_char=len(item.text),
                )
            ],
        )

    registry.register(
        ToolSpec(
            name="read_provision",
            description="Read a complete original provision",
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "article": {"type": "string"},
                },
                "required": ["source_id", "article"],
                "additionalProperties": False,
            },
            handler=read,
        )
    )
    active = Harness(
        request=REQUEST,
        context=context,
        registry=registry,
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    active.facts = ["Shared supplied scenario date is unchanged"]
    calls = [
        receipt(1, "A", ledger).call,
        receipt(2, "B", ledger).call,
        receipt(3, "C" if dependency else None, ledger).call,
    ]
    active.dispatch_repair_reads(calls)
    turns: list[ResearchTurn] = []
    for recorded in active.receipts:
        number = recorded.evidence_ids[0]
        item = ledger.get(number)
        assert item is not None
        sources = json.loads(ledger.serialize_records([number], max_chars=None))
        turns.append(
            ResearchTurn(
                assistant=AssistantMessage(
                    content="The operative conditions must be assessed under the supplied facts.",
                    tool_calls=[
                        ToolCall(
                            id=recorded.call.call_id,
                            function=FunctionCall(
                                name=recorded.call.name,
                                arguments=json.dumps(recorded.call.arguments),
                            ),
                        )
                    ],
                ),
                results=[
                    ToolMessage(
                        tool_call_id=recorded.call.call_id,
                        content=json.dumps(
                            {
                                "outcome": recorded.outcome.model_dump(mode="json"),
                                "original_evidence": sources,
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
            )
        )
    active.turns = turns
    broad = view(ledger, active.receipts, facts=active.facts)
    broad.turns = turns
    broad.tools = registry.definitions(context)
    return active, adapter, broad, ledger


def actual_records(prompt: list[ChatCompletionMessage]) -> list[dict[str, JsonValue]]:
    # This is the production same-message decoder, not a host source substitution.
    catalogue = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    catalogue.validate(prompt)
    records: list[dict[str, JsonValue]] = []
    for message in prompt:
        if not isinstance(message.content, str):
            continue
        try:
            value = json.loads(message.content)
        except ValueError:
            continue
        from onyx.asv3.cropped_handoff_transport import expand_cropped_handoff_transport

        payload = expand_cropped_handoff_transport(value)
        originals = payload.get("original_evidence")
        if isinstance(originals, list):
            records.extend(cast(list[dict[str, JsonValue]], originals))
    return records


def test_actual_harness_view_and_native_fit_do_not_restore_detached_history() -> None:
    active, adapter, broad, ledger = harness_fixture()
    selected = active.view()
    assert selected.turns == [] and selected.required_evidence_numbers == [1, 3]
    broad_prompt, broad_tools, _ = adapter._fit_native_decision(broad)
    prompt, tools, _ = adapter._fit_native_decision(selected)
    large = ledger.get(2)
    assert large is not None
    assert large.text not in json.dumps(
        [m.model_dump(mode="json") for m in prompt], ensure_ascii=False
    )
    assert adapter._input_cost(prompt, tools) < adapter._input_cost(
        broad_prompt, broad_tools
    )
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    adapter._invoke_once(
        prompt,
        tools,
        LLMFlow.ASV3_COORDINATOR,
        max_tokens=100,
        research=True,
        structured=False,
        native_catalogue=bound,
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 3}
    assert selected.facts == broad.facts
    assert len(active.turns) == 3 and len(active.receipts) == 3
    manifest = cast(
        dict[str, JsonValue], selected.research_state["cropped_task_packet"]
    )
    receipt_wire = json.dumps(manifest["local_receipts"], ensure_ascii=False)
    for number in (1, 3):
        item = ledger.get(number)
        assert item is not None and item.text not in receipt_wire


def test_dependency_and_fresh_final_writer_keep_every_original() -> None:
    active, adapter, broad, ledger = harness_fixture(dependency=True)
    selected = active.view()
    assert selected.required_evidence_numbers == [1, 2]
    prompt, tools, _ = adapter._fit_native_decision(selected)
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    adapter._invoke_once(
        prompt,
        tools,
        LLMFlow.ASV3_COORDINATOR,
        max_tokens=100,
        research=True,
        structured=False,
        native_catalogue=bound,
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    final = prepare_cropped_writer_view(
        selected, adapter.context, adapter._native_original_records(broad.turns)
    )
    assert final.required_evidence_numbers == [1, 2, 3]
    assert {row["citation"] for row in final.original_evidence} == {1, 2, 3}
    assert final.facts == selected.facts and final.turns == []


@pytest.mark.parametrize("cropped", [False, True])
def test_small_path_native_wire_is_unchanged_by_packet_call(cropped: bool) -> None:
    active, adapter, broad, _ = harness_fixture()
    adapter.context.services["asv3_exp_cropped"] = cropped
    if cropped:
        from onyx.asv3.cropped_task_packets import CroppedTaskPacketStore

        adapter.context.services["cropped_task_packet_store"] = CroppedTaskPacketStore(
            adapter.context, REQUEST
        )
    before, before_tools, _ = adapter._fit_native_decision(active.view())
    current = active.view()
    after, after_tools, _ = adapter._fit_native_decision(current)
    # Harness uses the same actual current source ranges; partitioning was not triggered.
    assert len(current.turns) == 3
    assert json.dumps(
        [m.model_dump(mode="json") for m in before], sort_keys=True
    ) == json.dumps([m.model_dump(mode="json") for m in after], sort_keys=True)
    assert before_tools == after_tools


@pytest.mark.parametrize("stage", ["research", "final", "repair"])
def test_final_and_repair_restore_detached_outcome_conditions(stage: str) -> None:
    from onyx.asv3.cropped_handoff_transport import expand_cropped_handoff_transport

    active, adapter, _, ledger = harness_fixture()
    outcomes = OutcomeMap([REQUEST], active.context)
    active.context.services["outcome_map"] = outcomes
    item = ledger.get(2)
    assert item is not None
    requirement = (
        "Retain topic B actor, precise threshold and subsequent-stage qualification."
    )
    outcomes.update(
        OutcomeUpdate(
            outcomes=[
                RequestedOutcome(
                    outcome_id=key, question_ids=["q0"], detail=f"Determine topic {key}"
                )
                for key in ("A", "B")
            ],
            conditions=[
                OutcomeCondition(
                    condition_id="topic_B_condition",
                    outcome_ids=["B"],
                    detail=requirement,
                    witnesses=[OutcomeWitness(citation=2, end_char=len(item.text))],
                )
            ],
        ),
        ledger,
    )
    for row, key in zip(active.receipts[:2], ("A", "B"), strict=True):
        row.call.arguments["_outcomes"] = [
            RequestedOutcome(
                outcome_id=key, question_ids=["q0"], detail=f"Determine topic {key}"
            ).model_dump(mode="json")
        ]
    selected = active.view()
    packet = selected.research_state.get("cropped_task_packet")
    assert isinstance(packet, dict) and packet["selected_outcome_ids"] == ["A"]
    if stage == "final":
        selected = prepare_cropped_writer_view(selected, active.context, [])
        adapter._writing_answer = True
    elif stage == "repair":
        selected = prepare_cropped_writer_view(selected, active.context, [])
        selected = selected.model_copy(
            update={
                "draft_to_repair": "Supported topic A [1].",
                "publication_gap": {"missing_material_condition": True},
            }
        )
    prompt, _, _ = adapter._fit_native_decision(selected)
    assert isinstance(prompt[-1].content, str)
    payload = expand_cropped_handoff_transport(json.loads(prompt[-1].content))
    outcome_view = payload["outcome_map"]
    assert isinstance(outcome_view, dict)
    rows = outcome_view["outcomes"]
    assert isinstance(rows, list)
    visible = {row["outcome_id"] for row in rows if isinstance(row, dict)}
    serialized = json.dumps(outcome_view, ensure_ascii=False)
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    actual = {row["citation"] for row in bound.records}
    if stage == "research":
        assert visible == {"A"} and requirement not in serialized
        assert actual == {1, 3}
    else:
        assert visible == {"A", "B"} and requirement in serialized
        assert actual == {1, 2, 3}
