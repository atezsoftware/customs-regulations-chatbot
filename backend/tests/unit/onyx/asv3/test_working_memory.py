import json
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.supplemental_tools import ScenarioState, build_supplemental_specs
from onyx.asv3.working_memory import WorkingMemory
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import (
    ChatCompletionMessageToolCall,
    Choice,
    FunctionCall,
    Message,
    ModelResponse,
)
from onyx.llm.models import AssistantMessage, TextContentPart, ToolMessage


def receipt(
    call_id: str,
    *,
    query: str = "Gümrük Kanunu",
    need: str = "q1",
    sources: list[dict[str, JsonValue]] | None = None,
) -> ToolReceipt:
    return ToolReceipt(
        call=CapabilityCall(
            name="resolve_source",
            call_id=call_id,
            arguments={"query": query, "coverage_item": need},
        ),
        elapsed_seconds=0,
        outcome=ToolOutcome(
            status=OutcomeStatus.AMBIGUOUS,
            summary="Choose source",
            data={
                "sources": sources
                or [{"source_id": "law-id-exact", "name": "4458 Gümrük Kanunu"}],
                "has_more": True,
                "next_offset": 20,
                "absence_proven": False,
            },
        ),
    )


def model_response(
    name: str | None, call_id: str, args: dict[str, JsonValue]
) -> ModelResponse:
    return ModelResponse(
        id=call_id,
        created="0",
        choice=Choice(
            message=Message(
                content="Draft" if name is None else None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=call_id,
                        function=FunctionCall(name=name, arguments=json.dumps(args)),
                    )
                ]
                if name
                else None,
            )
        ),
    )


def test_real_decide_keeps_ambiguous_source_candidates_after_scenario_pair_and_resume() -> (
    None
):
    context = RunContext(
        scope={"tenant": "A", "pc_set": 15},
        services={"scenario_state": ScenarioState()},
    )
    registry = CapabilityRegistry(build_supplemental_specs())
    registry.register(
        ToolSpec(
            name="resolve_source",
            description="Resolve",
            parameters={"type": "object", "properties": {}},
            handler=lambda _args, _ctx: receipt("ignored").outcome,
        )
    )
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="selected",
        temperature=0,
        max_input_tokens=100000,
    )
    llm.invoke.side_effect = [
        model_response("resolve_source", "resolve-1", {}),
        model_response(
            "record_scenario",
            "scenario-1",
            {"questions": ["q1"], "facts": ["No exchange permission"]},
        ),
        model_response(None, "draft", {}),
    ]
    harness = Harness(
        request="Compare scenarios",
        context=context,
        registry=registry,
        decide=ResearchModel(llm, context).decide,
    )
    harness.run()
    actual_prompt = llm.invoke.call_args.kwargs["prompt"]
    assistants = [
        message for message in actual_prompt if isinstance(message, AssistantMessage)
    ]
    results = [message for message in actual_prompt if isinstance(message, ToolMessage)]
    assert len(assistants) == len(results) == 1
    assert assistants[0].tool_calls[0].id == results[0].tool_call_id == "scenario-1"
    assert "law-id-exact" not in results[0].content
    parts = actual_prompt[-1].content
    assert isinstance(parts, list) and isinstance(parts[0], TextContentPart)
    data = json.loads(parts[0].text)
    sources = [
        locator
        for locator in data["working_locators"]["locators"]
        if locator["kind"] == "source"
    ]
    assert (
        sources[0]["source_id"] == "law-id-exact"
        and sources[0]["name"] == "4458 Gümrük Kanunu"
    )
    assert sources[0]["receipt_id"] == "resolve-1"
    assert any(
        locator.get("next_offset") == 20
        for locator in data["working_locators"]["locators"]
    )
    saved = harness.snapshot()
    resumed = Harness(
        request=harness.request,
        context=RunContext(run_id=context.run_id, scope=context.scope),
        registry=registry,
        decide=lambda _view: pytest.fail("No automatic decision on restore"),
    )
    resumed.restore(saved)
    assert (
        resumed.working_memory.view()["locators"]
        == harness.working_memory.view()["locators"]
    )
    # Source candidates are navigation, not fabricated original evidence.
    assert resumed.evidence.export()["records"] == []
    isolated_child = Harness(
        request="Independent",
        context=context.child(),
        registry=registry,
        decide=lambda _view: pytest.fail("No automatic child execution"),
    )
    assert isolated_child.working_memory.view()["locators"] == []
    assert harness.working_memory.view()["locators"]
    other_scope = WorkingMemory({"tenant": "B", "pc_set": 15})
    with pytest.raises(ValueError, match="scope changed"):
        other_scope.restore(harness.working_memory.export())


def test_locator_view_is_byte_bounded_and_paged_without_truncating_ids_or_losing_independent_needs() -> (
    None
):
    memory = WorkingMemory({"tenant": "A"})
    for need in ("q1", "q2"):
        memory.observe(
            receipt(
                "receipt-" + need,
                need=need,
                sources=[
                    {
                        "source_id": f"exact-id-{need}-{n}-" + "Z" * 80,
                        "name": "Türkçe kaynak " + "ğ" * 80,
                    }
                    for n in range(150)
                ],
            )
        )
    first = json.loads(json.dumps(memory.view()))
    assert len(json.dumps(first, ensure_ascii=False).encode()) <= 24576
    assert first["omitted_locators"] > 0
    assert {
        item["information_need"]
        for item in first["locators"]
        if item["kind"] == "source"
    } == {"q1", "q2"}
    seen: set[str] = set()
    offset = 0
    while True:
        page = json.loads(json.dumps(memory.view(offset=offset)))
        assert len(json.dumps(page, ensure_ascii=False).encode()) <= 24576
        for item in page["locators"]:
            if item["kind"] == "source":
                assert item["source_id"].endswith("Z" * 80)
                seen.add(item["source_id"])
        following = page["next_offset"]
        if following >= page["total_locators"]:
            break
        assert following > offset
        offset = following
    assert len(seen) == 300
    exported = memory.export()
    assert "vector" not in json.dumps(exported) and "text" not in json.dumps(exported)


def test_no_progress_feedback_is_advisory_and_new_query_or_information_need_remains_allowed() -> (
    None
):
    memory = WorkingMemory({})
    memory.observe(receipt("call-A"))
    revision = memory.revision
    memory.observe(receipt("call-B"))
    assert memory.revision == revision and memory.unchanged_streak == 1
    assert "progress_hint" in memory.view()
    memory.observe(receipt("call-C", query="Gümrük Yönetmeliği"))
    assert memory.revision > revision and memory.unchanged_streak == 0
    memory.observe(receipt("call-D", need="q2"))
    assert {
        item.get("information_need")
        for item in json.loads(json.dumps(memory.view()))["locators"]
    } == {
        "q1",
        "q2",
    }
    # An exact repeated scenario write is successful but is not new research progress.
    context = RunContext(services={"scenario_state": ScenarioState()})
    registry = CapabilityRegistry(build_supplemental_specs())
    first = registry.dispatch(
        CapabilityCall(
            name="record_scenario",
            arguments={"questions": ["repair"], "facts": ["paid repair"]},
        ),
        context,
    )
    again = registry.dispatch(
        CapabilityCall(
            name="record_scenario",
            arguments={"questions": ["repair"], "facts": ["paid repair"]},
        ),
        context,
    )
    assert (
        first.data["research_changed"] is True
        and again.data["research_changed"] is False
    )
    assert again.status == OutcomeStatus.FOUND
    memory.observe(
        ToolReceipt(
            call=CapabilityCall(name="record_scenario"),
            outcome=again,
            elapsed_seconds=0,
        )
    )
    assert memory.unchanged_streak == 1


def test_research_state_reopens_exact_locator_and_old_receipt_without_full_audit_payload() -> (
    None
):
    context = RunContext(scope={"tenant": "A"})
    registry = CapabilityRegistry()
    harness = Harness(
        request="Read",
        context=context,
        registry=registry,
        decide=lambda _view: pytest.fail("No automatic execution"),
    )
    first = receipt("old-receipt")
    harness.working_memory.observe(first)
    harness.receipts = [first] + [
        receipt(f"receipt-{n}", query=f"query-{n}") for n in range(30)
    ]
    for spec in build_core_specs(registry, harness.evidence, harness.snapshot):
        registry.register(spec)
    locators = harness.working_memory.view()["locators"]
    assert isinstance(locators, list) and isinstance(locators[0], dict)
    locator_id = locators[0]["locator_id"]
    outcome = registry.dispatch(
        CapabilityCall(
            name="read_research_state",
            arguments={"locator_ids": [locator_id], "receipt_ids": ["old-receipt"]},
        ),
        context,
    )
    data = json.loads(outcome.model_dump_json())["data"]
    assert outcome.status == OutcomeStatus.FOUND
    assert len(data["working_locators"]["locators"]) == 1
    assert data["working_locators"]["locators"][0]["locator_id"] == locator_id
    assert data["receipts"][0]["receipt_id"] == "old-receipt"
    assert data["receipts"][0]["data"]["sources"][0]["source_id"] == "law-id-exact"
    assert (
        "turns" not in data and "evidence" not in data and "working_memory" not in data
    )
