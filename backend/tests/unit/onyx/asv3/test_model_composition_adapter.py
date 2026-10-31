"""Explicit roles fit their own models and preserve guarded original-source handoffs."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3 import llm_adapter
from onyx.asv3.cropped_repair_model import fresh_model, integrate_repair
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.model_composition.models import (
    COMPOSITION_ROLES,
    MODEL_COMPOSITION_SERVICE,
    CompositionProfileId,
    CompositionRole,
    ModelBinding,
    ModelComposition,
)
from onyx.asv3.models import HarnessView, RunContext, RunStopped, SharedBudget
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import ToolChoiceOptions
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span
from tests.unit.onyx.asv3.test_fast_primary_coordinator import names, prose
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record, original, recorded

CAPACITIES = {
    "coordinator": 2048,
    "routine": 1024,
    "search": 512,
    "author": 4096,
    "repair": 6144,
    "verifier": 3072,
}


def composition_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cropped: bool = False,
    profile: CompositionProfileId = "flash_sol_v1",
) -> tuple[ResearchModel, dict[CompositionRole, MagicMock], ModelComposition]:
    models: dict[CompositionRole, MagicMock] = {}
    bindings: dict[CompositionRole, ModelBinding] = {}
    for index, role in enumerate(COMPOSITION_ROLES):
        selected = model(200_000)
        selected.config = LLMConfig(
            model_provider="openai",
            model_name=f"role-{role}",
            temperature=0,
            max_input_tokens=200_000,
        )
        models[role] = selected
        bindings[role] = ModelBinding(
            llm=selected,
            token_counter=MagicMock(side_effect=lambda text: len(text)),
            provider_id=index + 10,
        )
    composition = ModelComposition(profile, 1, bindings)
    ledger = recorded(
        [
            original("rule", "The procedure requires A AND B."),
            original("exception", "Exception C requires written consent before entry."),
        ]
    )
    context = RunContext(
        budget=SharedBudget(unlimited_execution=True),
        services={
            "asv3_fast": True,
            "research_profile": "normal",
            "evidence": ledger,
            MODEL_COMPOSITION_SERVICE: composition,
        },
    )
    if cropped:
        context.services.update(
            asv3_exp_cropped=True, asv3_workflow_variant=ASV3_TUNED_VARIANT
        )
    monkeypatch.setattr(
        llm_adapter,
        "get_llm_max_output_tokens",
        lambda _map, name, _provider: CAPACITIES[name.removeprefix("role-")],
    )
    adapter = ResearchModel(models["coordinator"], context, lean_native_mode=True)
    return adapter, models, composition


def original_view(adapter: ResearchModel) -> HarnessView:
    ledger = cast(EvidenceLedger, adapter.context.services["evidence"])
    return adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)],
        required_evidence_numbers=[1, 2],
        research_state={"needs": [{"detail": "Preserve exception C."}]},
    )


@pytest.mark.parametrize(
    "profile",
    [
        "sol_reference_v1",
        "flash_sol_v1",
        "terra_sol_v1",
        "argon_sonnet_v1",
        "luna_sol_v1",
    ],
)
@pytest.mark.parametrize("cropped", [False, True])
def test_every_profile_uses_same_action_handoff_and_full_author_originals(
    monkeypatch: pytest.MonkeyPatch,
    profile: CompositionProfileId,
    cropped: bool,
) -> None:
    adapter, models, composition = composition_fixture(
        monkeypatch, cropped=cropped, profile=profile
    )
    current = original_view(adapter)
    models["coordinator"].invoke.return_value = native_action(
        "read_provision", {"source_id": "authorized-statute", "article": "17"}
    )
    models["routine"].invoke.return_value = native_action("request_primary_answer", {})
    models["author"].invoke.return_value = prose(
        "A AND B [1]. Exception C requires written consent before entry [2]."
    )
    assert adapter.decide(current).calls[0].name == "read_provision"
    result = adapter.decide(current)
    assert result.answer == models["author"].invoke.return_value.choice.message.content
    assert all(
        models[role].invoke.call_count == 1
        for role in ("coordinator", "routine", "author")
    )
    assert (
        models["repair"].invoke.call_count == models["verifier"].invoke.call_count == 0
    )
    routine = models["routine"].invoke.call_args.kwargs
    assert "request_primary_answer" in names(routine["tools"])
    assert not names(routine["tools"]) & {
        "submit_answer",
        "ask_user",
        "submit_partial_answer",
    }
    author = models["author"].invoke.call_args.kwargs
    assert "request_primary_answer" not in names(author["tools"])
    payload = last_payload(models["author"])
    wire = json.dumps(payload, ensure_ascii=False)
    assert "The procedure requires A AND B." in wire
    assert "Exception C requires written consent before entry." in wire
    assert "draft_to_repair" not in payload
    ledger = adapter.context.services["evidence"]
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}
    for role in ("coordinator", "routine", "author"):
        assert models[role].invoke.call_args.kwargs["max_tokens"] == CAPACITIES[role]
        counter = cast(MagicMock, composition.binding(role).token_counter)
        assert counter.call_count > 0
    assert adapter.llm is models["coordinator"]
    assert adapter.token_counter is composition.binding("coordinator").token_counter


@pytest.mark.parametrize("cropped", [False, True])
def test_bootstrap_candidate_cannot_publish_without_author(
    monkeypatch: pytest.MonkeyPatch, cropped: bool
) -> None:
    adapter, models, _composition = composition_fixture(monkeypatch, cropped=cropped)
    models["coordinator"].invoke.return_value = prose(
        "Unapproved incomplete claim [1]."
    )
    models["author"].invoke.return_value = prose(
        "A AND B, subject to exception C [1][2]."
    )
    result = adapter.decide(original_view(adapter))
    assert result.answer == "A AND B, subject to exception C [1][2]."
    assert models["author"].invoke.call_count == 1
    assert "Unapproved incomplete claim" not in json.dumps(
        last_payload(models["author"])
    )
    models["routine"].invoke.assert_not_called()


def test_role_failure_restores_model_counter_and_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, composition = composition_fixture(monkeypatch)
    adapter._native_first_decision_completed = True
    adapter._native_output_capacity = 123
    models["routine"].invoke.side_effect = ValueError("provider failed")
    with pytest.raises(ValueError, match="provider failed"):
        adapter.decide(original_view(adapter))
    assert adapter.llm is models["coordinator"]
    assert adapter.token_counter is composition.binding("coordinator").token_counter
    assert adapter._native_output_capacity == 123
    assert adapter._active_composition_role is None


def test_small_routine_capacity_falls_back_before_provider_and_keeps_required_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, _composition = composition_fixture(monkeypatch)
    adapter._native_first_decision_completed = True
    models["routine"].config = models["routine"].config.model_copy(
        update={"max_input_tokens": 1000}
    )
    models["author"].invoke.return_value = prose("Full supported result [1][2].")
    assert (
        adapter.decide(original_view(adapter)).answer == "Full supported result [1][2]."
    )
    models["routine"].invoke.assert_not_called()
    assert "Exception C requires written consent before entry." in json.dumps(
        last_payload(models["author"])
    )


def test_sticky_author_resume_and_repair_route_preserve_exact_owned_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, _composition = composition_fixture(monkeypatch)
    adapter._native_first_decision_started = (
        adapter._native_first_decision_completed
    ) = True
    adapter._fast_primary_answer_mode = True
    checkpoint: dict[str, JsonValue] = {
        "native_coordinator_sampling": adapter.native_sampling_snapshot()
    }
    resumed = ResearchModel(
        models["coordinator"], adapter.context, lean_native_mode=True
    )
    resumed.restore_native_sampling(checkpoint)
    models["author"].invoke.return_value = prose("Author after resume [1][2].")
    assert (
        resumed.decide(original_view(adapter)).answer == "Author after resume [1][2]."
    )
    models["routine"].invoke.assert_not_called()
    current = original_view(adapter).model_copy(
        update={
            "draft_to_repair": "Owned exact draft [1][2].",
            "publication_gap": {"missing": "condition"},
        }
    )
    models["repair"].invoke.return_value = prose("Repaired conditional draft [1][2].")
    assert resumed.decide(current).answer == "Repaired conditional draft [1][2]."
    assert "Owned exact draft" in json.dumps(last_payload(models["repair"]))
    assert (
        models["repair"].invoke.call_args.kwargs["max_tokens"] == CAPACITIES["repair"]
    )


@pytest.mark.parametrize("role", ["routine", "verifier", "repair", "author"])
def test_fresh_cropped_model_uses_fixed_role_and_isolated_delivery(
    monkeypatch: pytest.MonkeyPatch, role: CompositionRole
) -> None:
    adapter, models, composition = composition_fixture(monkeypatch, cropped=True)
    adapter.last_call_id = "root-writer"
    adapter.context.services["last_model_call_id"] = "root-writer"
    adapter._native_output_capacity = 1
    fresh = fresh_model(adapter, role=role)
    assert fresh.llm is models[role]
    assert fresh.token_counter is composition.binding(role).token_counter
    assert fresh._native_output_capacity is None
    assert fresh.context.services[MODEL_COMPOSITION_SERVICE] is composition
    assert "last_model_call_id" not in fresh.context.services
    assert fresh.context.budget is adapter.context.budget
    assert fresh.context.services["evidence"] is adapter.context.services["evidence"]
    assert fresh._native_output_limit() == CAPACITIES[role]
    assert (
        adapter.last_call_id
        == adapter.context.services["last_model_call_id"]
        == "root-writer"
    )


def test_structural_recovery_stays_on_routine_and_role_tags_are_span_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, _composition = composition_fixture(monkeypatch)
    adapter._native_first_decision_completed = True
    models["routine"].invoke.side_effect = [
        native_action("unexposed", {}),
        native_action("read_provision", {"source_id": "authorized-statute"}),
    ]
    spans: list[MagicMock] = []

    @contextmanager
    def resources(
        _context: RunContext,
        selected: LLM,
        flow: LLMFlow,
        *_args: object,
        **_kwargs: object,
    ) -> Iterator[Span[GenerationSpanData]]:
        span = MagicMock(spec=Span)
        span.span_id = f"test-span-{len(spans)}"
        span.span_data = GenerationSpanData(
            model=selected.config.model_name,
            model_config={
                "flow": flow.value,
                "model_provider": selected.config.model_provider,
            },
        )
        spans.append(span)
        yield cast(Span[GenerationSpanData], span)

    monkeypatch.setattr(llm_adapter, "_generation_resources", resources)
    assert adapter.decide(original_view(adapter)).calls[0].name == "read_provision"
    assert models["routine"].invoke.call_count == 2
    models["coordinator"].invoke.assert_not_called()
    models["author"].invoke.assert_not_called()
    assert len(spans) == 2
    for span in spans:
        tags = span.span_data.model_config
        assert tags["asv3_composition_role"] == "routine"
        assert tags["asv3_composition_profile"] == "flash_sol_v1"
        assert tags["asv3_composition_provider_id"] == "11"
        assert span.span_data.model == "role-routine"
        assert "api_key" not in json.dumps(tags)
    assert spans[0].span_data.model_config is not spans[1].span_data.model_config


def test_explicit_verifier_has_own_capacity_and_sonnet_keeps_auto_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, composition = composition_fixture(monkeypatch)
    verifier = ResearchModel(
        models["coordinator"], adapter.context, composition_role="verifier"
    )

    class Report(BaseModel):
        complete: bool

    models["verifier"].invoke.return_value = prose('{"complete":true}')
    assert (
        verifier.invoke_text(
            "Check", "{}", LLMFlow.ASV3_VERIFICATION, response_model_override=Report
        )
        == '{"complete":true}'
    )
    assert verifier.llm is models["verifier"]
    assert verifier.token_counter is composition.binding("verifier").token_counter
    models["author"].config = models["author"].config.model_copy(
        update={"model_provider": "anthropic"}
    )
    adapter._fast_primary_answer_mode = True
    models["author"].invoke.return_value = prose("Supported answer [1][2].")
    adapter.decide(original_view(adapter))
    assert (
        models["author"].invoke.call_args.kwargs["tool_choice"]
        == ToolChoiceOptions.AUTO
    )


def test_explicit_role_without_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="resolved model composition"):
        ResearchModel(model(), RunContext(), composition_role="verifier")


def test_cropped_retained_integration_uses_repair_and_preserves_other_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, _composition = composition_fixture(monkeypatch, cropped=True)
    ledger = cast(EvidenceLedger, adapter.context.services["evidence"])
    ledger.add(
        [original("later", "The later declaration is due after approval.")],
        adapter.context,
    )
    old = original_view(adapter).model_copy(
        update={
            "required_evidence_numbers": [1, 2, 3],
            "draft_to_repair": "The rule always applies [1].\n\nA later declaration is due [3].",
            "publication_gap": {"missing_condition": True},
        }
    )

    def patch(**_kwargs: object) -> ModelResponse:
        payload = last_payload(models["repair"])
        unit_id = payload["draft_to_repair"]["units"][0]["unit_id"]
        return native_action(
            "submit_retained_answer",
            {
                "retained_answer_edits": [
                    {"unit_id": unit_id, "replacement": "A AND B are required [1]."}
                ]
            },
        )

    models["repair"].invoke.side_effect = patch
    result = integrate_repair(
        adapter,
        old,
        [{"status": "found", "evidence_numbers": [1, 2], "absence_proven": False}],
    )
    assert (
        result.calls[0].arguments["answer"]
        == "A AND B are required [1].\n\nA later declaration is due [3]."
    )
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2, 3}
    assert models["repair"].invoke.call_count == 1
    models["author"].invoke.assert_not_called()
    models["coordinator"].invoke.assert_not_called()


def test_cancellation_at_composed_marker_prevents_author_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, _composition = composition_fixture(monkeypatch)
    adapter._native_first_decision_completed = True

    def stopped(**_kwargs: object) -> ModelResponse:
        adapter.context.cancel()
        return native_action("request_primary_answer", {})

    models["routine"].invoke.side_effect = stopped
    with pytest.raises(RunStopped, match="cancelled"):
        adapter.decide(original_view(adapter))
    models["author"].invoke.assert_not_called()
    assert adapter.llm is models["coordinator"]


def test_cropped_author_stays_on_full_source_writer_after_read_and_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, _composition = composition_fixture(monkeypatch, cropped=True)
    view = original_view(adapter)
    models["coordinator"].invoke.return_value = native_action("read_provision", {})
    models["routine"].invoke.return_value = native_action("request_primary_answer", {})
    models["author"].invoke.side_effect = [
        native_action(
            "read_provision", {"source_id": "authorized-statute", "article": "18"}
        ),
        prose("Full answer with both conditions and exception [1][2]."),
    ]
    adapter.decide(view)
    assert adapter.decide(view).calls[0].arguments["article"] == "18"
    saved: dict[str, JsonValue] = {
        "native_coordinator_sampling": adapter.native_sampling_snapshot()
    }
    assert adapter._fast_primary_answer_mode
    resumed = ResearchModel(
        models["coordinator"], adapter.context, lean_native_mode=True
    )
    resumed.restore_native_sampling(saved)
    assert (
        resumed.decide(view).answer
        == "Full answer with both conditions and exception [1][2]."
    )
    assert models["routine"].invoke.call_count == 1
    assert models["author"].invoke.call_count == 2
    for call in models["author"].invoke.call_args_list:
        text = str(call.kwargs["prompt"])
        assert "The procedure requires A AND B." in text
        assert "Exception C requires written consent before entry." in text
