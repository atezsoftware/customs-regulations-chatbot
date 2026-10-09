"""Opt-in isolation, fixed model and saved-run identity for Legal Review."""

import json
from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.chat import process_message
from onyx.chat.chat_state import ChatTurnSetup
from onyx.chat.models import StreamingError
from onyx.context.search.models import BaseFilters
from onyx.error_handling.exceptions import OnyxError
from onyx.legal_review.provider import legal_review_override, require_legal_review_model
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.server.query_and_chat.session_loading import create_asv3_progress_packets
from onyx.server.query_and_chat.streaming_models import ASv3Progress


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"legal_composite": True},
        {"supersearch": True},
        {"deep_research": True},
        {"atez_search_v3": True, "asv3_research_profile": "normal"},
        {"atez_search_v3": True, "asv3_research_profile": "experimental"},
        {
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_guardrails_v2": True,
        },
    ],
)
def test_old_requests_leave_new_workflow_disabled(options: dict[str, object]) -> None:
    assert not SendMessageRequest.model_validate(
        {"message": "Soru", **options}
    ).legal_review


@pytest.mark.parametrize(
    "options",
    [
        {"legal_composite": True},
        {"supersearch": True},
        {"deep_research": True},
        {"atez_search": True},
        {"atez_search_v2": True},
        {"atez_search_v3": True},
        {"asv3_resume_message_id": 42},
        {"asv3_guardrails_v2": True},
        {"asv3_allow_external": True},
        {"asv3_research_profile": "normal"},
    ],
)
def test_new_request_rejects_other_workflow_options(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Legal Review is mutually exclusive"):
        SendMessageRequest.model_validate(
            {"message": "Soru", "legal_review": True, **options}
        )


def test_model_is_fixed_and_an_unexpected_fallback_is_rejected() -> None:
    override = legal_review_override()
    assert override.model_version == "gemini-3.8-flash"
    assert override.model_provider_type == "vertex_ai"
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0,
        max_input_tokens=200000,
    )
    require_legal_review_model(llm)
    llm.config = llm.config.model_copy(update={"model_name": "gemini-3.5-flash-lite"})
    with pytest.raises(OnyxError, match="Gemini 3.8 Flash"):
        require_legal_review_model(llm)


def test_chat_setup_resolves_flash_before_using_stale_session_or_request_models() -> (
    None
):
    session_id = uuid4()
    request = SendMessageRequest(
        message="Antrepo şartları",
        chat_session_id=session_id,
        legal_review=True,
        llm_override=LLMOverride(model_provider="openai", model_version="gpt-5-mini"),
    )
    session = SimpleNamespace(
        id=session_id,
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        project_id=None,
        llm_override=LLMOverride(
            model_provider="anthropic", model_version="claude-haiku-4-5"
        ),
    )
    user = SimpleNamespace(id=uuid4(), email="test@example.com", is_anonymous=False)
    with (
        patch.object(process_message, "get_chat_session_by_id", return_value=session),
        patch.object(process_message, "mt_cloud_telemetry"),
        patch.object(
            process_message,
            "get_llm_for_persona",
            side_effect=RuntimeError("selection probe"),
        ) as factory,
    ):
        with pytest.raises(RuntimeError, match="selection probe"):
            next(process_message.build_chat_turn(request, user, MagicMock(), None))
    selected = factory.call_args.kwargs["llm_override"]
    assert selected.model_version == "gemini-3.8-flash"
    assert selected.model_provider_type == "vertex_ai"


@pytest.mark.parametrize(
    "options",
    [
        {"file_descriptors": [{"id": "file", "type": "plain_text"}]},
        {"additional_context": "browser text"},
        {"forced_tool_id": 3},
    ],
)
def test_new_mode_rejects_sources_outside_indexed_corpus(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="indexed corpus sources only"):
        SendMessageRequest.model_validate(
            {"message": "Soru", "legal_review": True, **options}
        )


def test_multi_model_dispatch_does_not_start_any_workflow() -> None:
    request = SendMessageRequest(message="Soru", legal_review=True)
    with patch.object(process_message, "_stream_chat_turn") as dispatch:
        result = next(
            process_message.handle_multi_model_stream(
                request,
                MagicMock(),
                [LLMOverride(model_version="one"), LLMOverride(model_version="two")],
            )
        )
    dispatch.assert_not_called()
    assert isinstance(result, StreamingError)
    assert "Hukuki İnceleme" in result.error


def test_new_scope_preserves_user_date_and_explicit_label_opt_out() -> None:
    filters = BaseFilters(
        document_set=["Subset"],
        as_of_date=date(2025, 7, 1),
        regulatory_label_search_enabled=False,
    )
    setup = SimpleNamespace(
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        new_msg_req=SendMessageRequest(
            message="Antrepo şartları",
            legal_review=True,
            internal_search_filters=filters,
        ),
    )
    actual = process_message._global_regulatory_search_filters(
        cast(ChatTurnSetup, setup)
    )
    assert actual is not None and actual.regulatory_chunks_only
    assert (
        actual.document_set == filters.document_set
        and actual.as_of_date == filters.as_of_date
    )
    assert not actual.regulatory_label_search_enabled
    assert not filters.regulatory_chunks_only


@pytest.mark.parametrize("variant", [None, "legal_review"])
def test_history_preserves_new_identity_and_revokes_resume(variant: str | None) -> None:
    event = ASv3Progress(
        workflow="legal_review",
        run_id="r",
        event_id="e",
        sequence=1,
        language="tr",
        phase="interrupted",
        status="failed",
        title="İnceleme durdu",
        resume_label="Araştırmaya devam et",
    )
    packets = create_asv3_progress_packets(
        json.dumps(
            {
                "version": 1,
                "asv3_workflow_variant": variant,
                "progress": [event.model_dump(mode="json")],
            }
        )
    )
    assert len(packets) == 1 and isinstance(packets[0].obj, ASv3Progress)
    assert packets[0].obj.workflow == "legal_review"
    assert packets[0].obj.resume_label is None
