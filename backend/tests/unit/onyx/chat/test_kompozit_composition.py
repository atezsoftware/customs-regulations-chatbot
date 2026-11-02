"""Product presets retain caller access, fixed models and owned resume identity."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from onyx.asv3.model_composition import resolver
from onyx.asv3.model_composition.models import ModelComposition
from onyx.asv3.models import ASv3WorkflowSelection
from onyx.auth.schemas import UserRole
from onyx.chat import asv3_composition, process_message
from onyx.chat.chat_state import ChatTurnSetup
from onyx.chat.models import StreamingError
from onyx.db.models import Persona, User
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.models import ReasoningEffort
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.models import MessageOrigin, SendMessageRequest
from tests.unit.onyx.chat.test_multi_model_streaming import _make_setup


def _request(cropped: bool = False, **changes: object) -> SendMessageRequest:
    return SendMessageRequest.model_validate(
        {
            "message": "Antrepo rejiminin şartları nelerdir?",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_kompozit": not cropped,
            "asv3_kompozit_cropped": cropped,
            "origin": "webapp",
            **changes,
        }
    )


def _workflow(cropped: bool) -> ASv3WorkflowSelection:
    return ASv3WorkflowSelection(
        research_profile="normal",
        parallel_research=False,
        workflow_variant="asv3_kompozit_cropped" if cropped else "asv3_kompozit",
    )


def _llm(provider: str, model: str) -> MagicMock:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider=provider,
        model_name=model,
        temperature=1,
        max_input_tokens=200000,
    )
    return llm


@pytest.mark.parametrize("cropped", [False, True])
def test_basic_web_user_resolves_all_fixed_roles_through_authorized_factory(
    monkeypatch: pytest.MonkeyPatch, cropped: bool
) -> None:
    user, persona = MagicMock(spec=User), MagicMock(spec=Persona)
    user.role = UserRole.BASIC
    models: dict[str, MagicMock] = {}

    def create(
        actual_persona: Persona | None,
        actual_user: User,
        *,
        llm_override: LLMOverride,
    ) -> LLM:
        assert actual_persona is persona and actual_user is user
        assert llm_override.model_provider_id == 17
        assert llm_override.model_provider_type and llm_override.model_version
        llm = _llm(llm_override.model_provider_type, llm_override.model_version)
        models[llm_override.model_version] = llm
        return llm

    factory, cost = MagicMock(side_effect=create), MagicMock()
    monkeypatch.setattr(resolver, "resolve_asv3_model_provider_id", lambda *_: 17)
    monkeypatch.setattr(resolver, "get_llm_for_persona", factory)
    monkeypatch.setattr(resolver, "check_asv3_secondary_model_cost_limit", cost)
    monkeypatch.setattr(resolver, "get_llm_token_counter", lambda _: len)
    selected = _llm("openai", "ui-selected-model")
    composition = asv3_composition.resolve_chat_model_composition(
        _request(cropped), user, persona, _workflow(cropped), selected
    )
    assert composition is not None
    assert composition.profile_id == "kompozit_v1"
    expected = {
        "coordinator": "gemini-3.8-flash",
        "routine": "gemini-3.8-flash",
        "search": "gemini-3.5-flash-lite",
        "author": "claude-sonnet-5-5",
        "repair": "gpt-6.1-terra",
        "verifier": "gemini-4-argon",
    }
    assert factory.call_count == cost.call_count == 5
    for role, model in expected.items():
        assert composition.binding(cast(Any, role)).llm is models[model]
    for llm in [selected, *models.values()]:
        llm.invoke.assert_not_called()
        llm.stream.assert_not_called()


@pytest.mark.parametrize("stage", ["provider", "access", "cost"])
def test_missing_or_inaccessible_preset_model_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    user = MagicMock(spec=User)
    user.role = UserRole.BASIC
    selected = _llm("vertex_ai", "gemini-3.8-flash")
    provider, factory, cost = (
        MagicMock(return_value=17),
        MagicMock(return_value=selected),
        MagicMock(),
    )
    boundary = {"provider": provider, "access": factory, "cost": cost}[stage]
    failure = PermissionError("Required configured model unavailable")
    boundary.side_effect = failure
    monkeypatch.setattr(resolver, "resolve_asv3_model_provider_id", provider)
    monkeypatch.setattr(resolver, "get_llm_for_persona", factory)
    monkeypatch.setattr(resolver, "check_asv3_secondary_model_cost_limit", cost)
    monkeypatch.setattr(resolver, "get_llm_token_counter", lambda _: len)
    with pytest.raises(PermissionError) as error:
        asv3_composition.resolve_chat_model_composition(
            _request(), user, MagicMock(spec=Persona), _workflow(False), selected
        )
    assert error.value is failure
    assert provider.call_count == 1
    assert factory.call_count == int(stage != "provider")
    selected.invoke.assert_not_called()
    selected.stream.assert_not_called()


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        {"atez_search_v3": False},
        {"asv3_research_profile": "deep"},
        {"asv3_research_profile": "experimental"},
        {"asv3_parallel_research": True},
        {"asv3_fast": True},
        {"asv3_fast_cropped": True},
        {"asv3_exp_cropped": True},
        {"asv3_guardrails_v3": True},
        {"deep_research": True},
        {"legal_review": True},
        {"supersearch": True},
        {"legal_composite": True},
        {"asv3_kompozit": True, "asv3_kompozit_cropped": True},
        {"asv3_model_profile": "factorial_v1_fna", "origin": "benchmark"},
    ],
)
def test_product_request_requires_exclusive_normal_preset(
    cropped: bool, changes: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        _request(cropped, **changes)


@pytest.mark.parametrize(
    "role,origin",
    [(UserRole.BASIC, MessageOrigin.BENCHMARK), (UserRole.ADMIN, MessageOrigin.WEBAPP)],
)
def test_product_access_does_not_relax_old_fast_benchmark_authorization(
    monkeypatch: pytest.MonkeyPatch, role: UserRole, origin: MessageOrigin
) -> None:
    request = SendMessageRequest.model_validate(
        {
            "message": "Soru",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_fast": True,
            "asv3_model_profile": "sol_reference_v1",
            "origin": "benchmark",
        }
    )
    request.origin = origin
    user = MagicMock(spec=User)
    user.role = role
    resolve = MagicMock()
    monkeypatch.setattr(asv3_composition, "resolve_model_composition", resolve)
    with pytest.raises(OnyxError, match="administrator"):
        asv3_composition.resolve_chat_model_composition(
            request,
            user,
            MagicMock(spec=Persona),
            ASv3WorkflowSelection(
                research_profile="normal",
                parallel_research=False,
                workflow_variant="asv3_fast",
            ),
            MagicMock(spec=LLM),
        )
    resolve.assert_not_called()


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_owned_resume_restores_fixed_primary_with_or_without_explicit_preset(
    monkeypatch: pytest.MonkeyPatch, cropped: bool, explicit: bool
) -> None:
    user = MagicMock(spec=User)
    saved_variant = _workflow(cropped).workflow_variant
    checkpoint = MagicMock(return_value={"asv3_workflow_variant": saved_variant})
    monkeypatch.setattr(asv3_composition, "load_asv3_checkpoint", checkpoint)
    monkeypatch.setattr(
        asv3_composition, "resolve_asv3_model_provider_id", lambda *_: 17
    )
    request = _request(
        cropped,
        asv3_kompozit=explicit and not cropped,
        asv3_kompozit_cropped=explicit and cropped,
        asv3_resume_message_id=41,
        llm_override={
            "model_provider_type": "openai",
            "model_version": "ui-selected-model",
        },
    )
    override = asv3_composition.kompozit_llm_override(request, user)
    assert override is not None
    assert (
        override.model_provider_id,
        override.model_provider_type,
        override.model_version,
    ) == (17, "vertex_ai", "gemini-3.8-flash")
    assert override.display_name == ("Kompozit Cropped" if cropped else "Kompozit")
    checkpoint.assert_called_once_with(message_id=41, user_id=user.id)
    assert request.llm_override is not None
    assert request.llm_override.model_version == "ui-selected-model"


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize(
    "saved_variant",
    [None, "standard", "asv3_fast", "asv3_fast_cropped", "other_kompozit"],
)
def test_resume_rejects_missing_or_other_workflow_before_model_resolution(
    monkeypatch: pytest.MonkeyPatch, cropped: bool, saved_variant: str | None
) -> None:
    if saved_variant == "other_kompozit":
        saved_variant = _workflow(not cropped).workflow_variant
    monkeypatch.setattr(
        asv3_composition,
        "load_asv3_checkpoint",
        lambda **_: {"asv3_workflow_variant": saved_variant} if saved_variant else None,
    )
    provider = MagicMock()
    monkeypatch.setattr(asv3_composition, "resolve_asv3_model_provider_id", provider)
    with pytest.raises(OnyxError, match="checkpoint|same Kompozit"):
        asv3_composition.kompozit_llm_override(
            _request(cropped, asv3_resume_message_id=41), MagicMock(spec=User)
        )
    provider.assert_not_called()


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize(
    "old_variant_flag",
    [
        "asv3_fast",
        "asv3_fast_cropped",
        "asv3_exp_cropped",
        "asv3_guarded_experimental",
        "asv3_guardrails_v2",
        "asv3_guardrails_v3",
    ],
)
def test_owned_preset_resume_rejects_explicit_old_variant_before_resolution(
    monkeypatch: pytest.MonkeyPatch, cropped: bool, old_variant_flag: str
) -> None:
    user = MagicMock(spec=User)
    checkpoint = MagicMock(
        return_value={"asv3_workflow_variant": _workflow(cropped).workflow_variant}
    )
    provider = MagicMock()
    monkeypatch.setattr(asv3_composition, "load_asv3_checkpoint", checkpoint)
    monkeypatch.setattr(asv3_composition, "resolve_asv3_model_provider_id", provider)
    request = _request(
        cropped,
        asv3_kompozit=False,
        asv3_kompozit_cropped=False,
        asv3_resume_message_id=41,
        **{old_variant_flag: True},
    )
    with pytest.raises(OnyxError, match="cannot select another workflow"):
        asv3_composition.kompozit_llm_override(request, user)
    checkpoint.assert_called_once_with(message_id=41, user_id=user.id)
    provider.assert_not_called()


@pytest.mark.parametrize("cropped", [False, True])
def test_chat_setup_fixes_primary_before_any_reservation(
    cropped: bool,
) -> None:
    session_id = uuid4()
    request = _request(
        cropped,
        chat_session_id=session_id,
        llm_override={"model_version": "ui-selected-model"},
    )
    session = SimpleNamespace(
        id=session_id,
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        project_id=None,
        llm_override=LLMOverride(model_version="session-model"),
        reasoning_effort_override=ReasoningEffort.HIGH,
    )
    user = SimpleNamespace(id=uuid4(), email="test@example.com", is_anonymous=False)
    with (
        patch.object(process_message, "get_chat_session_by_id", return_value=session),
        patch.object(process_message, "mt_cloud_telemetry"),
        patch.object(
            asv3_composition, "resolve_asv3_model_provider_id", return_value=17
        ),
        patch.object(
            process_message,
            "get_llm_for_persona",
            side_effect=RuntimeError("selection probe"),
        ) as factory,
        patch.object(process_message, "reserve_message_id") as reserve,
    ):
        with pytest.raises(RuntimeError, match="selection probe"):
            next(
                process_message.build_chat_turn(
                    request, cast(User, user), MagicMock(), None
                )
            )
    override = factory.call_args.kwargs["llm_override"]
    assert (override.model_provider_id, override.model_version) == (
        17,
        "gemini-3.8-flash",
    )
    assert override.display_name == ("Kompozit Cropped" if cropped else "Kompozit")
    assert session.llm_override.model_version == "session-model"
    assert session.reasoning_effort_override == ReasoningEffort.HIGH
    reserve.assert_not_called()


@pytest.mark.parametrize("preset", [None, "asv3_kompozit", "asv3_kompozit_cropped"])
def test_setup_low_is_turn_local_and_does_not_replace_saved_user_settings(
    monkeypatch: pytest.MonkeyPatch, preset: str | None
) -> None:
    session_id = uuid4()
    request = (
        _request(preset == "asv3_kompozit_cropped", chat_session_id=session_id)
        if preset
        else SendMessageRequest(message="Soru", chat_session_id=session_id)
    )
    persona = SimpleNamespace(
        id=process_message.DEFAULT_PERSONA_ID, system_prompt="", user_files=[]
    )
    session = SimpleNamespace(
        id=session_id,
        persona=persona,
        project_id=None,
        llm_override=LLMOverride(model_version="saved-ui-model"),
        reasoning_effort_override=ReasoningEffort.HIGH,
    )
    user = MagicMock(spec=User)
    user.id, user.email, user.is_anonymous = uuid4(), "basic@example.com", False
    user.use_memories = True
    message = SimpleNamespace(id=1, message_type=process_message.MessageType.USER)
    llm = _llm("vertex_ai", "gemini-3.8-flash")
    factory = MagicMock(return_value=llm)
    values = {
        "get_chat_session_by_id": session,
        "mt_cloud_telemetry": None,
        "check_llm_cost_limit_for_provider": None,
        "get_llm_token_counter": len,
        "verify_user_files": None,
        "_collect_available_file_ids": MagicMock(),
        "find_summary_for_branch": None,
        "is_last_assistant_message_clarification": False,
        "get_memories": SimpleNamespace(
            user_info=SimpleNamespace(placeholder_values={})
        ),
        "get_custom_agent_prompt": None,
        "calculate_reserved_tokens": 100,
        "extract_context_files": SimpleNamespace(
            publication_evidence=None, image_files=[]
        ),
        "determine_search_params": SimpleNamespace(
            search_usage=process_message.SearchToolUsage.AUTO
        ),
        "get_tools": [],
        "reserve_message_id": SimpleNamespace(id=2),
        "get_effective_persona_tools": [],
        "convert_chat_history": SimpleNamespace(
            simple_messages=[], all_injected_file_metadata={}
        ),
        "get_cache_backend": MagicMock(),
        "reset_cancel_status": None,
        "set_processing_status": None,
    }
    for name, value in values.items():
        monkeypatch.setattr(process_message, name, MagicMock(return_value=value))
    monkeypatch.setattr(process_message, "get_llm_for_persona", factory)
    monkeypatch.setattr(
        asv3_composition, "resolve_asv3_model_provider_id", lambda *_: 17
    )
    with patch(
        "onyx.chat.chat_utils.load_chat_history_for_turn",
        return_value=([message], message, None),
    ):
        turn = process_message.build_chat_turn(request, user, MagicMock(), None)
        next(turn)
        with pytest.raises(StopIteration) as completed:
            next(turn)
    setup = completed.value.value
    assert isinstance(setup, ChatTurnSetup)
    assert setup.reasoning_effort == (
        ReasoningEffort.LOW if preset else ReasoningEffort.HIGH
    )
    assert session.reasoning_effort_override == ReasoningEffort.HIGH
    assert session.llm_override.model_version == "saved-ui-model"
    override = factory.call_args.kwargs["llm_override"]
    assert override.model_version == (
        "gemini-3.8-flash" if preset else "saved-ui-model"
    )
    llm.invoke.assert_not_called()
    llm.stream.assert_not_called()


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_worker_uses_fixed_coordinator_and_retains_owned_resume_identity(
    cropped: bool, resume: bool
) -> None:
    setup = _make_setup()
    setup.persona.id = process_message.DEFAULT_PERSONA_ID
    setup.new_msg_req = _request(
        cropped,
        asv3_kompozit=not resume and not cropped,
        asv3_kompozit_cropped=not resume and cropped,
        asv3_resume_message_id=41 if resume else None,
    )
    user = MagicMock(spec=User)
    user.role = UserRole.BASIC
    setup.reasoning_effort = ReasoningEffort.LOW
    setup.user_message.id, setup.reserved_messages[0].id = 1, 2
    setup.llms[0].config = _llm("openai", "stale-ui-model").config
    fixed = _llm("vertex_ai", "gemini-3.8-flash")
    composition = MagicMock(spec=ModelComposition)
    composition.binding.return_value.llm = fixed
    calls: list[dict[str, Any]] = []
    variant = _workflow(cropped).workflow_variant
    with (
        patch.object(
            asv3_composition, "resolve_model_composition", return_value=composition
        ) as resolve,
        patch.object(
            process_message,
            "load_asv3_checkpoint",
            return_value={"asv3_workflow_variant": variant},
        ),
        patch.object(process_message, "retain_asv3_response_workflow") as retain,
        patch(
            "onyx.asv3.runtime.run_asv3_loop", side_effect=lambda **kw: calls.append(kw)
        ),
        patch.object(
            process_message,
            "get_llm_for_persona",
            side_effect=AssertionError("No legacy fallback"),
        ),
        patch.object(process_message, "construct_tools", return_value={}),
        patch.object(process_message, "get_llm_token_counter", return_value=len),
        patch.object(
            process_message,
            "load_settings",
            return_value=SimpleNamespace(auto_detect_search_filters=False),
        ),
        patch.object(process_message, "llm_loop_completion_handle"),
        patch.object(process_message, "record_final_answer_message"),
        patch.object(process_message, "set_processing_status"),
    ):
        result = list(process_message._run_models(cast(ChatTurnSetup, setup), user))
    assert not any(isinstance(item, StreamingError) for item in result)
    assert len(calls) == 1
    assert calls[0]["llm"] is calls[0]["research_llm"] is fixed
    assert calls[0]["model_composition"] is composition
    assert calls[0]["workflow_variant"] == variant
    assert calls[0]["reasoning_effort"] == ReasoningEffort.LOW
    resolve.assert_called_once_with(
        profile_id="kompozit_v1",
        user=user,
        persona=setup.persona,
        workflow_variant=variant,
    )
    if resume:
        retain.assert_called_once_with(
            message_id=2, user_id=user.id, workflow_variant=variant
        )
        assert setup.reserved_messages[0].workflow == variant
    else:
        retain.assert_not_called()
