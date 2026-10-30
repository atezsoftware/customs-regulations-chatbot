"""Fast action research hands full state to a sticky, guarded primary decision."""

import json

import pytest
from pydantic import JsonValue

from onyx.asv3 import llm_adapter
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import NativeDecisionEnvelopeError, ResearchModel
from onyx.asv3.models import RunContext, RunStopped, SharedBudget
from onyx.llm.interfaces import LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import ChatCompletionMessage
from onyx.prompts.asv3.fast import (
    FAST_MATERIAL_OUTCOMES,
    FAST_TEK_ACTION_RESEARCH,
    FAST_TEK_MATERIAL_CONDITIONS,
)
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
    original,
)

MARKER = "request_primary_answer"


def prose(text: str) -> ModelResponse:
    return ModelResponse(
        id="text", created="0", choice=Choice(message=Message(content=text))
    )


def names(tools: list[dict[str, JsonValue]]) -> set[str]:
    return {
        str(function["name"])
        for tool in tools
        if isinstance(function := tool.get("function"), dict)
    }


def test_action_research_handoff_keeps_full_state_and_primary_may_research_then_repair() -> (
    None
):
    primary, secondary = model(), model()
    context = RunContext(services={"asv3_fast": True, "research_profile": "normal"})
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    text = "A AND B are required. Exception C permits a written application before reimport."
    record = original(ledger, context, text)
    research_state: dict[str, JsonValue] = {
        "needs": [{"need_id": "n1", "detail": "Apply the favorable procedure"}],
        "source_conditions": [{"detail": "Written application BEFORE reimport"}],
        "needs_omitted": 0,
        "findings_omitted": 0,
        "source_conditions_omitted": 0,
    }
    current = adaptive_tool_view(
        original_evidence=[record], research_state=research_state
    )
    primary.invoke.side_effect = [
        native_action("read_provision", {"source_id": "law", "article": "7"}),
        native_action("read_provision", {"source_id": "law", "article": "8"}),
        prose("A AND B; written application before reimport [1]."),
        prose("Corrected conditional result and remedy [1]."),
    ]
    secondary.invoke.side_effect = [
        native_action("read_provision", {"source_id": "law", "article": "9"}),
        native_action(MARKER, {}),
    ]
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    assert adapter.decide(current).calls[0].arguments["article"] == "7"
    assert adapter.decide(current).calls[0].arguments["article"] == "9"
    assert adapter.decide(current).calls[0].arguments["article"] == "8"
    assert (
        adapter.decide(current).answer
        == "A AND B; written application before reimport [1]."
    )
    repair = current.model_copy(
        update={
            "draft_to_repair": "Rejected [1].",
            "publication_gap": {"missing": "condition"},
        }
    )
    assert (
        adapter.decide(repair).answer == "Corrected conditional result and remedy [1]."
    )
    assert primary.invoke.call_count == 4 and secondary.invoke.call_count == 2
    assert adapter.research_llm is None
    assert adapter.native_sampling_snapshot()["fast_primary_answer_mode"] is True
    payload = last_payload(primary)
    assert payload["original_evidence"][0]["text"] == text
    assert payload["research_state"] == research_state
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    routine = secondary.invoke.call_args.kwargs
    assert MARKER in names(routine["tools"])
    assert not names(routine["tools"]) & {
        "submit_answer",
        "submit_partial_answer",
        "ask_user",
        "assemble_answers",
    }
    assert FAST_TEK_ACTION_RESEARCH in routine["prompt"][0].content
    assert all(
        MARKER not in str(call.kwargs["tools"])
        for call in primary.invoke.call_args_list
    )
    assert MARKER not in str(primary.invoke.call_args_list[1].kwargs["prompt"])


@pytest.mark.parametrize("flag", [True, False, 1, "true", None])
def test_only_strict_fast_root_discards_lite_prose_instead_of_rewriting_its_draft(
    flag: JsonValue,
) -> None:
    primary, secondary = model(), model()
    context = RunContext(services={"asv3_fast": flag, "research_profile": "normal"})
    primary.invoke.side_effect = [
        native_action("read_provision", {}),
        prose("Primary answer."),
    ]
    secondary.invoke.return_value = prose("Unapproved Lite provisional text.")
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    assert adapter.decide(adaptive_tool_view()).calls
    assert adapter.decide(adaptive_tool_view()).answer == "Primary answer."
    payload = last_payload(primary)
    assert ("draft_to_repair" not in payload) is (flag is True)
    assert secondary.invoke.call_count == 1
    assert (adapter.research_llm is None) is (flag is True)


@pytest.mark.parametrize(
    "depth,cropped,native", [(1, False, True), (0, True, True), (0, False, False)]
)
def test_secondary_role_and_tek_instruction_are_isolated(
    depth: int, cropped: bool, native: bool
) -> None:
    primary, secondary = model(), model()
    context = RunContext(
        depth=depth, services={"asv3_fast": True, "asv3_exp_cropped": cropped}
    )
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=native
    )
    assert adapter.research_llm is secondary
    settings = adapter.native_sampling_snapshot()["settings"]
    assert isinstance(settings, dict) and "fast_coordinator_routing" not in settings
    if native:
        prompt, _tools, _output = adapter._fit_native_decision(adaptive_tool_view())
        assert FAST_TEK_MATERIAL_CONDITIONS not in str(prompt[0].content)


@pytest.mark.parametrize("defect", ["nonempty", "mixed", "malformed"])
def test_marker_must_be_standalone_and_zero_argument_before_dispatch(
    defect: str,
) -> None:
    marker = native_action(
        MARKER, {"summary": "Not allowed"} if defect == "nonempty" else {}
    )
    if defect == "mixed":
        assert marker.choice.message.tool_calls is not None
        other = native_action("read_provision", {})
        assert other.choice.message.tool_calls is not None
        marker.choice.message.tool_calls.extend(other.choice.message.tool_calls)
    if defect == "malformed":
        assert marker.choice.message.tool_calls is not None
        marker.choice.message.tool_calls[0].function.arguments = "not JSON"
    tools = ResearchModel._fast_action_tools(adaptive_tool_view().tools)
    with pytest.raises(NativeDecisionEnvelopeError, match="Primary answer routing"):
        ResearchModel._decision(marker, tools)


@pytest.mark.parametrize("limit,question_started", [(0, False), (3, False), (3, True)])
def test_handoff_charges_another_decision_without_charging_a_tool(
    limit: int, question_started: bool
) -> None:
    primary, secondary = model(), model()
    context = RunContext(
        services={"asv3_fast": True, "question_research_started": question_started},
        budget=SharedBudget(
            max_decisions=limit,
            final_decision_reserve=0,
            coordinator_decision_reserve=0,
        ),
    )
    primary.invoke.side_effect = [
        native_action("read_provision", {}),
        prose("Primary answer."),
    ]
    secondary.invoke.return_value = native_action(MARKER, {})
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    adapter.decide(adaptive_tool_view())
    if limit == 0:
        with pytest.raises(RunStopped, match="decision"):
            adapter.decide(adaptive_tool_view())
        assert primary.invoke.call_count == 1
    else:
        assert adapter.decide(adaptive_tool_view()).answer == "Primary answer."
    assert context.budget.snapshot()["decisions"] == (0 if limit == 0 else 1)
    assert context.budget.snapshot()["tools"] == 0


def test_cancellation_at_marker_cannot_start_primary() -> None:
    primary, secondary = model(), model()
    context = RunContext(services={"asv3_fast": True})
    primary.invoke.return_value = native_action("read_provision", {})
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    adapter.decide(adaptive_tool_view())

    def stopped(**_kwargs: object) -> ModelResponse:
        context.cancel()
        return native_action(MARKER, {})

    secondary.invoke.side_effect = stopped
    with pytest.raises(RunStopped, match="cancelled"):
        adapter.decide(adaptive_tool_view())
    assert primary.invoke.call_count == 1


def test_capacity_sampling_and_pre_post_handoff_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, secondary = model(400000), model(100000)
    primary.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0,
        max_input_tokens=400000,
    )
    monkeypatch.setattr(llm_adapter, "get_model_map", lambda: {})
    monkeypatch.setattr(
        llm_adapter,
        "get_llm_max_output_tokens",
        lambda _map, name, _provider: 8000 if name == "gemini-3.8-flash" else 2000,
    )
    primary.with_seed.return_value = primary
    primary.invoke.side_effect = [
        native_action("read_provision", {}),
        native_action("read_provision", {"article": "more"}),
    ]
    secondary.invoke.return_value = native_action(MARKER, {})
    context = RunContext(services={"asv3_fast": True})
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    assert adapter._native_output_limit() == 2000 and adapter._limits(8000) == (
        92000,
        8000,
    )
    adapter.decide(adaptive_tool_view())
    before = adapter.native_sampling_snapshot()
    assert before["fast_primary_answer_mode"] is False
    assert primary.with_seed.call_args.args == (
        llm_adapter.NATIVE_COORDINATOR_FIRST_SEED,
    )
    adapter.decide(adaptive_tool_view())
    after = adapter.native_sampling_snapshot()
    assert primary.with_seed.call_args.args == (
        llm_adapter.NATIVE_COORDINATOR_CONTINUATION_SEED,
    )
    assert adapter._native_output_limit() == 8000 and adapter._limits(8000) == (
        392000,
        8000,
    )
    for snapshot, expected_secondary in [(before, secondary), (after, None)]:
        restored = ResearchModel(
            primary,
            RunContext(services={"asv3_fast": True}),
            research_llm=secondary,
            lean_native_mode=True,
        )
        restored.restore_native_sampling({"native_coordinator_sampling": snapshot})
        assert restored.research_llm is expected_secondary
        assert restored.native_sampling_snapshot() == snapshot
    old = {
        **after,
        "settings": {
            "mode": "native_coordinator",
            "version": 1,
            "first_seed": llm_adapter.NATIVE_COORDINATOR_FIRST_SEED,
            "continuation_seed": llm_adapter.NATIVE_COORDINATOR_CONTINUATION_SEED,
            "fast_coordinator_routing": "selected-primary-v1",
        },
    }
    with pytest.raises(ValueError, match="same coordinator sampling"):
        adapter.restore_native_sampling({"native_coordinator_sampling": old})
    for invalid in [None, 1, "true"]:
        with pytest.raises(ValueError, match="routing checkpoint"):
            adapter.restore_native_sampling(
                {
                    "native_coordinator_sampling": {
                        **after,
                        "fast_primary_answer_mode": invalid,
                    }
                }
            )


def test_primary_failure_keeps_sticky_mode_for_checkpoint_resume() -> None:
    primary, secondary = model(), model()
    context = RunContext(services={"asv3_fast": True})
    primary.invoke.side_effect = [
        native_action("read_provision", {}),
        RunStopped("primary stopped"),
    ]
    secondary.invoke.return_value = native_action(MARKER, {})
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    adapter.decide(adaptive_tool_view())
    with pytest.raises(RunStopped, match="primary stopped"):
        adapter.decide(adaptive_tool_view())
    assert adapter.native_sampling_snapshot()["fast_primary_answer_mode"] is True
    assert adapter.research_llm is None


def test_response_policy_can_follow_fresh_repair_without_enabling_fast_routing() -> (
    None
):
    from onyx.prompts.asv3.fast import (
        FAST_RESPONSE_POLICY,
        FAST_RESPONSE_POLICY_SERVICE,
    )

    context = RunContext(services={FAST_RESPONSE_POLICY_SERVICE: FAST_RESPONSE_POLICY})
    adapter = ResearchModel(model(), context, lean_native_mode=True)
    prompt, _tools, _output = adapter._fit_native_decision(adaptive_tool_view())
    assert FAST_MATERIAL_OUTCOMES in str(prompt[0].content)
    assert FAST_TEK_MATERIAL_CONDITIONS not in str(prompt[0].content)
    assert context.services.get("asv3_fast") is None
    assert "fast_coordinator_routing" not in json.dumps(
        adapter.native_sampling_snapshot()
    )


def test_actual_lite_packet_overflow_switches_to_primary_without_a_lite_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, secondary = model(), model()
    context = RunContext(services={"asv3_fast": True})
    primary.invoke.side_effect = [
        native_action("read_provision", {}),
        prose("Primary answer."),
    ]
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    count = adapter._input_cost

    def actual_cost(
        prompt: list[ChatCompletionMessage], tools: list[dict[str, JsonValue]]
    ) -> int:
        if MARKER in names(tools):
            return 100001
        return count(prompt, tools)

    monkeypatch.setattr(adapter, "_input_cost", actual_cost)
    adapter.decide(adaptive_tool_view())
    assert adapter.decide(adaptive_tool_view()).answer == "Primary answer."
    assert secondary.invoke.call_count == 0
    assert adapter.research_llm is None
    assert adapter.native_sampling_snapshot()["fast_primary_answer_mode"] is True
