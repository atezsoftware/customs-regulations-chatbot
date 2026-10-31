"""Composition selection stays explicit, authorized and visible in the primary slot."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from onyx.asv3.model_composition.models import ModelComposition
from onyx.asv3.models import ASv3WorkflowSelection
from onyx.auth.schemas import UserRole
from onyx.chat import asv3_composition, process_message
from onyx.chat.chat_state import ChatTurnSetup
from onyx.chat.models import StreamingError
from onyx.db.models import Persona, User
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.server.query_and_chat.models import MessageOrigin, SendMessageRequest
from tests.unit.onyx.chat.test_multi_model_streaming import _make_setup


def request() -> SendMessageRequest:
    return SendMessageRequest.model_validate(
        {
            "message": "A source-bound question",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_fast": True,
            "asv3_model_profile": "sol_reference_v1",
            "origin": "benchmark",
        }
    )


@pytest.mark.parametrize("variant", ["asv3_fast", "asv3_fast_cropped"])
@pytest.mark.parametrize("api_normalized", [False, True])
def test_composition_passes_actual_caller_persona_and_workflow_to_resolver(
    monkeypatch: pytest.MonkeyPatch, variant: str, api_normalized: bool
) -> None:
    user, persona = MagicMock(spec=User), MagicMock(spec=Persona)
    user.role = UserRole.ADMIN
    selected = MagicMock(spec=LLM)
    selected.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-6.1-sol",
        temperature=1,
        max_input_tokens=920000,
    )
    composition = MagicMock(spec=ModelComposition)
    composition.binding.return_value.llm = selected
    resolver = MagicMock(return_value=composition)
    monkeypatch.setattr(asv3_composition, "resolve_model_composition", resolver)
    workflow = ASv3WorkflowSelection.model_validate(
        {
            "workflow_variant": variant,
            "research_profile": "normal",
            "parallel_research": False,
        }
    )
    selected_request = request()
    if api_normalized:
        selected_request.origin = MessageOrigin.API
    result = asv3_composition.resolve_chat_model_composition(
        selected_request, user, persona, workflow, selected
    )
    assert result is composition
    resolver.assert_called_once_with(
        profile_id="sol_reference_v1",
        user=user,
        persona=persona,
        workflow_variant=variant,
    )


@pytest.mark.parametrize("defect", ["role", "origin", "workflow", "model", "sampling"])
def test_composition_cannot_bypass_authorization_or_silently_replace_primary(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    user, persona = MagicMock(spec=User), MagicMock(spec=Persona)
    user.role = UserRole.BASIC if defect == "role" else UserRole.ADMIN
    selected = MagicMock(spec=LLM)
    selected.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-6.1-sol",
        temperature=1,
        max_input_tokens=920000,
    )
    expected = MagicMock(spec=LLM)
    expected.config = selected.config.model_copy()
    if defect == "model":
        selected.config = selected.config.model_copy(
            update={"model_name": "gpt-6.1-terra"}
        )
    if defect == "sampling":
        selected.config = selected.config.model_copy(update={"temperature": 0.2})
    composition = MagicMock(spec=ModelComposition)
    composition.binding.return_value.llm = expected
    resolver = MagicMock(return_value=composition)
    monkeypatch.setattr(asv3_composition, "resolve_model_composition", resolver)
    selected_request = request()
    if defect == "origin":
        selected_request = selected_request.model_copy(
            update={"origin": MessageOrigin.WEBAPP}
        )
    workflow = ASv3WorkflowSelection.model_validate(
        {
            "workflow_variant": "standard" if defect == "workflow" else "asv3_fast",
            "research_profile": "normal",
            "parallel_research": False,
        }
    )
    with pytest.raises(OnyxError):
        asv3_composition.resolve_chat_model_composition(
            selected_request, user, persona, workflow, selected
        )
    assert resolver.call_count == (defect in {"model", "sampling"})


def test_default_requests_never_resolve_composition_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = MagicMock(side_effect=AssertionError("No model resolution expected"))
    monkeypatch.setattr(asv3_composition, "resolve_model_composition", resolver)
    selected = request().model_copy(update={"asv3_model_profile": None})
    workflow = ASv3WorkflowSelection(research_profile="normal", parallel_research=False)
    assert (
        asv3_composition.resolve_chat_model_composition(
            selected,
            MagicMock(spec=User),
            MagicMock(spec=Persona),
            workflow,
            MagicMock(spec=LLM),
        )
        is None
    )
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "defect", ["unknown", "not_fast", "origin", "api_origin", "parallel"]
)
def test_request_schema_rejects_unregistered_or_wrong_scope_profiles(
    defect: str,
) -> None:
    payload = request().model_dump(mode="json")
    changes = {
        "unknown": {"asv3_model_profile": "arbitrary-model-routing"},
        "not_fast": {"asv3_fast": False},
        "origin": {"origin": "webapp"},
        "api_origin": {"origin": "api"},
        "parallel": {"asv3_parallel_research": True},
    }
    with pytest.raises(ValidationError):
        SendMessageRequest.model_validate({**payload, **changes[defect]})


@pytest.mark.parametrize(
    "role", [UserRole.BASIC, UserRole.CURATOR, UserRole.GLOBAL_CURATOR]
)
@pytest.mark.parametrize("api_normalized", [False, True])
def test_nonadmin_remains_rejected_before_resolution_for_both_internal_origins(
    monkeypatch: pytest.MonkeyPatch, role: UserRole, api_normalized: bool
) -> None:
    selected_request = request()
    if api_normalized:
        selected_request.origin = MessageOrigin.API
    user = MagicMock(spec=User)
    user.role = role
    resolver = MagicMock(side_effect=AssertionError("Unauthorized model resolution"))
    monkeypatch.setattr(asv3_composition, "resolve_model_composition", resolver)
    with pytest.raises(OnyxError, match="administrator"):
        asv3_composition.resolve_chat_model_composition(
            selected_request,
            user,
            MagicMock(spec=Persona),
            ASv3WorkflowSelection(
                research_profile="normal",
                parallel_research=False,
                workflow_variant="asv3_fast",
            ),
            MagicMock(spec=LLM),
        )
    resolver.assert_not_called()


@pytest.mark.parametrize("cropped", [False, True])
def test_actual_chat_worker_uses_resolved_profile_without_legacy_model_lookup(
    cropped: bool,
) -> None:
    setup = _make_setup()
    setup.persona.id = process_message.DEFAULT_PERSONA_ID
    setup.new_msg_req = request().model_copy(
        update={"asv3_fast": not cropped, "asv3_fast_cropped": cropped}
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-6.1-sol",
        temperature=1,
        max_input_tokens=920000,
    )
    routine = MagicMock(spec=LLM)
    routine.config = selected.config.model_copy()
    composition = MagicMock(spec=ModelComposition)
    composition.binding.return_value.llm = routine
    setup.user_message.id, setup.reserved_messages[0].id = 1, 2
    calls: list[dict[str, Any]] = []
    with (
        patch(
            "onyx.chat.asv3_composition.resolve_chat_model_composition",
            return_value=composition,
        ),
        patch(
            "onyx.asv3.runtime.run_asv3_loop", side_effect=lambda **kw: calls.append(kw)
        ),
        patch(
            "onyx.chat.process_message.get_llm_for_persona",
            side_effect=AssertionError("Legacy resolver must not run"),
        ) as legacy,
        patch("onyx.chat.process_message.check_llm_cost_limit_for_provider") as cost,
        patch("onyx.chat.process_message.construct_tools", return_value={}),
        patch("onyx.chat.process_message.get_llm_token_counter", return_value=len),
        patch(
            "onyx.chat.process_message.load_settings",
            return_value=SimpleNamespace(auto_detect_search_filters=False),
        ),
        patch("onyx.chat.process_message.llm_loop_completion_handle"),
        patch("onyx.chat.process_message.record_final_answer_message"),
        patch("onyx.chat.process_message.set_processing_status"),
    ):
        result = list(
            process_message._run_models(cast(ChatTurnSetup, setup), MagicMock())
        )
    assert not any(isinstance(item, StreamingError) for item in result)
    assert len(calls) == 1 and calls[0]["model_composition"] is composition
    assert calls[0]["llm"] is selected and calls[0]["research_llm"] is routine
    assert calls[0]["workflow_variant"] == (
        "asv3_fast_cropped" if cropped else "asv3_fast"
    )
    legacy.assert_not_called()
    cost.assert_not_called()
    selected.with_model.assert_not_called()
