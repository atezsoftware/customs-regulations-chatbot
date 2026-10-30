"""Fast root routing removes a candidate rewrite while preserving actual originals."""

import json

import pytest
from pydantic import JsonValue

from onyx.asv3 import llm_adapter
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.llm.interfaces import LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.prompts.asv3.fast import FAST_MATERIAL_OUTCOMES
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
    original,
)


@pytest.mark.parametrize("flag", [True, False, 1, "true", None])
def test_fast_root_primary_preserves_full_originals_without_secondary_draft(
    flag: JsonValue,
) -> None:
    primary, secondary = model(), model()
    context = RunContext(services={"asv3_fast": flag, "research_profile": "normal"})
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    text = "A AND B are required. Exception C permits a written application before reimport."
    record = original(ledger, context, text)
    current = adaptive_tool_view(original_evidence=[record])
    answer = (
        "A AND B apply; exception C permits a written application before reimport [1]."
    )
    primary.invoke.side_effect = [
        native_action("read_provision", {"source_id": "law", "article": "7"}),
        ModelResponse(
            id="final", created="0", choice=Choice(message=Message(content=answer))
        ),
    ]
    secondary.invoke.return_value = ModelResponse(
        id="proposal",
        created="0",
        choice=Choice(message=Message(content="Candidate [1].")),
    )
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    assert adapter.decide(current).calls[0].name == "read_provision"
    result = adapter.decide(current)
    assert result.answer == answer
    assert primary.invoke.call_count == 2
    assert secondary.invoke.call_count == (0 if flag is True else 1)
    assert (adapter.research_llm is None) is (flag is True)
    payload = last_payload(primary)
    assert payload["original_evidence"][0]["text"] == text
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert ("draft_to_repair" not in payload) is (flag is True)


@pytest.mark.parametrize(
    "depth,cropped,native", [(1, False, True), (0, True, True), (0, False, False)]
)
def test_secondary_role_is_preserved_outside_fast_native_root(
    depth: int,
    cropped: bool,
    native: bool,
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
    assert isinstance(settings, dict)
    assert "fast_coordinator_routing" not in settings


def test_primary_route_uses_selected_capacity_and_preserves_sampling_resume_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary, secondary = model(40000), model(10000)
    monkeypatch.setattr(llm_adapter, "get_model_map", lambda: {})
    primary.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0,
        max_input_tokens=40000,
    )
    monkeypatch.setattr(
        llm_adapter,
        "get_llm_max_output_tokens",
        lambda _map, name, _provider: 8000 if name == "gemini-3.8-flash" else 2000,
    )
    context = RunContext(services={"asv3_fast": True})
    adapter = ResearchModel(
        primary, context, research_llm=secondary, lean_native_mode=True
    )
    assert adapter._native_output_limit() == 8000
    assert adapter._limits(8000) == (32000, 8000)
    snapshot = adapter.native_sampling_snapshot()
    settings = snapshot["settings"]
    assert isinstance(settings, dict)
    assert settings["mode"] == "native_coordinator"
    assert settings["fast_coordinator_routing"] == "selected-primary-v1"
    current = adaptive_tool_view()
    adapter._native_decision_llm(current)
    assert primary.with_seed.call_args.args == (
        llm_adapter.NATIVE_COORDINATOR_FIRST_SEED,
    )
    adapter._native_first_decision_completed = True
    adapter._native_decision_llm(current)
    assert primary.with_seed.call_args.args == (
        llm_adapter.NATIVE_COORDINATOR_CONTINUATION_SEED,
    )
    restored = ResearchModel(
        primary,
        RunContext(services={"asv3_fast": True}),
        research_llm=secondary,
        lean_native_mode=True,
    )
    restored.restore_native_sampling({"native_coordinator_sampling": snapshot})
    old_settings = {
        k: v for k, v in settings.items() if k != "fast_coordinator_routing"
    }
    with pytest.raises(ValueError, match="same coordinator sampling"):
        restored.restore_native_sampling(
            {"native_coordinator_sampling": {**snapshot, "settings": old_settings}}
        )


def test_response_policy_can_follow_fresh_repair_without_enabling_other_fast_behaviors() -> (
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
    assert context.services.get("asv3_fast") is None
    assert "fast_coordinator_routing" not in json.dumps(
        adapter.native_sampling_snapshot()
    )
