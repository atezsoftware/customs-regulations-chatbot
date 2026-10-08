"""Verify isolated opt-in routing without provider or persistence calls."""

from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.chat import process_message
from onyx.chat.chat_state import ChatTurnSetup
from onyx.chat.models import StreamingError
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import BaseFilters
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLMConfig
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.models import SendMessageRequest
from tests.unit.onyx.chat.test_multi_model_streaming import _make_setup


def test_composite_defaults_off_and_preserves_existing_request() -> None:
    request = SendMessageRequest(message="Antrepo nedir?")
    assert request.legal_composite is False
    assert request.atez_search_v3 is False
    assert request.deep_research is False


@pytest.mark.parametrize(
    "other_workflow",
    ["atez_search", "atez_search_v2", "atez_search_v3", "deep_research"],
)
def test_composite_rejects_competing_workflow(other_workflow: str) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SendMessageRequest.model_validate(
            {"message": "Soru", "legal_composite": True, other_workflow: True}
        )


def test_composite_rejects_multiple_answering_models() -> None:
    with pytest.raises(ValueError, match="one answering model"):
        SendMessageRequest.model_validate(
            {
                "message": "Soru",
                "legal_composite": True,
                "llm_overrides": [
                    {"model_provider": "openai", "model_version": "gpt-5-mini"},
                    {
                        "model_provider": "anthropic",
                        "model_version": "claude-haiku-4-5",
                    },
                ],
            }
        )


def test_composite_rejects_direct_multi_model_dispatch() -> None:
    request = SendMessageRequest.model_validate(
        {"message": "Soru", "legal_composite": True}
    )
    result = next(
        process_message.handle_multi_model_stream(
            request,
            MagicMock(),
            [
                LLMOverride(model_provider="openai", model_version="gpt-5-mini"),
                LLMOverride(
                    model_provider="anthropic", model_version="claude-haiku-4-5"
                ),
            ],
        )
    )
    assert isinstance(result, StreamingError)
    assert result.error_code == "VALIDATION_ERROR"
    assert "Legal Composite" in result.error


@pytest.mark.parametrize("explicit_label_opt_out", [False, True])
def test_composite_scope_preserves_user_filters(explicit_label_opt_out: bool) -> None:
    filters = BaseFilters(document_set=["Dar kapsam"], as_of_date=date(2026, 7, 1))
    if explicit_label_opt_out:
        filters = filters.model_copy(update={"regulatory_label_search_enabled": False})
    setup = SimpleNamespace(
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        new_msg_req=SendMessageRequest(
            message="Antrepo rejiminin şartları nelerdir?",
            legal_composite=True,
            internal_search_filters=filters,
        ),
    )
    actual = process_message._global_regulatory_search_filters(
        cast(ChatTurnSetup, setup)
    )
    assert actual is not None
    assert actual.regulatory_chunks_only is True
    assert actual.source_type == [DocumentSource.USER_FILE]
    assert actual.document_set == filters.document_set
    assert actual.as_of_date == filters.as_of_date
    assert actual.regulatory_label_search_enabled is not explicit_label_opt_out
    assert filters.regulatory_chunks_only is False


@pytest.mark.parametrize("invalid_scope", [None, "agent", "project"])
@pytest.mark.parametrize("selected_provider", ["openai", "vertex_ai", "gemini"])
def test_actual_chat_worker_routes_only_composite_request(
    invalid_scope: str | None,
    selected_provider: str,
) -> None:
    setup = _make_setup()
    setup.persona.id = (
        process_message.DEFAULT_PERSONA_ID if invalid_scope != "agent" else 42
    )
    setup.search_params.project_id_filter = (
        "project" if invalid_scope == "project" else None
    )
    setup.new_msg_req = SendMessageRequest(
        message="İlgili mevzuat nedir?", legal_composite=True
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider=selected_provider,
        model_name="user-selected",
        max_input_tokens=200000,
        temperature=0,
    )
    secondary = MagicMock()
    secondary.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.5-flash-lite",
        max_input_tokens=1048576,
        temperature=0,
    )
    setup.user_message.id = 1
    setup.reserved_messages[0].id = 2
    user = MagicMock()
    with (
        patch("onyx.legal_composite.runtime.run_legal_composite_loop") as composite,
        patch("onyx.asv3.runtime.run_asv3_loop") as existing,
        patch("onyx.chat.process_message.run_llm_loop") as ordinary,
        patch("onyx.chat.process_message.construct_tools", return_value={}),
        patch(
            "onyx.chat.process_message.get_llm_for_persona", return_value=secondary
        ) as factory,
        patch("onyx.chat.process_message.get_llm_token_counter", return_value=len),
        patch(
            "onyx.chat.process_message.load_settings",
            return_value=SimpleNamespace(auto_detect_search_filters=False),
        ),
        patch("onyx.chat.process_message.llm_loop_completion_handle"),
        patch("onyx.chat.process_message.record_final_answer_message"),
        patch("onyx.chat.process_message.set_processing_status"),
    ):
        packets = list(process_message._run_models(cast(ChatTurnSetup, setup), user))
    existing.assert_not_called()
    ordinary.assert_not_called()
    selected.with_model.assert_not_called()
    if invalid_scope:
        composite.assert_not_called()
        factory.assert_not_called()
        assert any(isinstance(packet, StreamingError) for packet in packets)
        return
    assert not any(isinstance(packet, StreamingError) for packet in packets)
    composite.assert_called_once()
    actual = composite.call_args.kwargs
    assert actual["llm"] is selected
    assert actual["research_llm"] is secondary
    assert actual["filters"].regulatory_chunks_only is True
    assert actual["user_message_id"] == 1
    assert actual["assistant_message_id"] == 2
    assert "research_profile" not in actual
    assert "workflow_variant" not in actual
    assert "resume_message_id" not in actual
    factory.assert_called_once_with(
        persona=setup.persona,
        user=user,
        llm_override=LLMOverride(
            model_provider_type="vertex_ai", model_version="gemini-3.5-flash-lite"
        ),
    )


@pytest.mark.parametrize("route_failure", ["unavailable_model", "usage_limit"])
def test_composite_honors_research_model_access_and_usage_limits(
    route_failure: str,
) -> None:
    setup = _make_setup()
    setup.persona.id = process_message.DEFAULT_PERSONA_ID
    setup.new_msg_req = SendMessageRequest.model_validate(
        {"message": "İlgili mevzuat nedir?", "legal_composite": True}
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="selected-expensive-model",
        max_input_tokens=200000,
        temperature=0,
        api_key="selected-key",
    )
    secondary = MagicMock()
    secondary.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.5-flash-lite",
        max_input_tokens=1048576,
        temperature=0,
        api_key="research-key",
    )
    rejection = OnyxError(OnyxErrorCode.INVALID_INPUT, "Configured model unavailable")
    usage_rejection = OnyxError(OnyxErrorCode.RATE_LIMITED, "Research usage limit")
    with (
        patch("onyx.legal_composite.runtime.run_legal_composite_loop") as composite,
        patch("onyx.chat.process_message.run_llm_loop") as ordinary,
        patch("onyx.chat.process_message.construct_tools", return_value={}),
        patch(
            "onyx.chat.process_message.get_llm_for_persona",
            return_value=secondary,
            side_effect=rejection if route_failure == "unavailable_model" else None,
        ) as factory,
        patch("onyx.chat.process_message.get_session_with_current_tenant"),
        patch(
            "onyx.chat.process_message.check_llm_cost_limit_for_provider",
            side_effect=usage_rejection,
        ) as cost_check,
        patch(
            "onyx.chat.process_message.load_settings",
            return_value=SimpleNamespace(auto_detect_search_filters=False),
        ),
        patch("onyx.chat.process_message.llm_loop_completion_handle"),
        patch("onyx.chat.process_message.record_final_answer_message"),
        patch("onyx.chat.process_message.set_processing_status"),
    ):
        packets = list(
            process_message._run_models(cast(ChatTurnSetup, setup), MagicMock())
        )
    factory.assert_called_once()
    composite.assert_not_called()
    ordinary.assert_not_called()
    selected.with_model.assert_not_called()
    selected.invoke.assert_not_called()
    selected.stream.assert_not_called()
    errors = [packet for packet in packets if isinstance(packet, StreamingError)]
    assert len(errors) == 1
    if route_failure == "unavailable_model":
        cost_check.assert_not_called()
        assert errors[0].error_code == "INVALID_INPUT"
        assert errors[0].is_retryable is False
        assert (
            "requires access to the configured gemini-3.5-flash-lite" in errors[0].error
        )
    else:
        cost_check.assert_called_once()
        assert errors[0].error_code == "RATE_LIMITED"
        assert cost_check.call_args.kwargs["llm_provider_api_key"] == "research-key"
        assert "Research usage limit" in errors[0].error
