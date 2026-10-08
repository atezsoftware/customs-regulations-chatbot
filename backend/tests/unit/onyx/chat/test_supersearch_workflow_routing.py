"""Check isolated PC-only dispatch and explicit model selection without providers."""

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
from onyx.llm.interfaces import LLMConfig
from onyx.llm.override_models import LLMOverride
from onyx.server.query_and_chat.models import SendMessageRequest
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from tests.unit.onyx.chat.test_multi_model_streaming import _make_setup


def test_supersearch_defaults_off_and_old_profiles_remain_valid() -> None:
    for options in (
        {},
        {"legal_composite": True},
        {"atez_search_v3": True, "asv3_research_profile": "normal"},
        {"atez_search_v3": True, "asv3_research_profile": "experimental"},
        {"deep_research": True},
    ):
        request = SendMessageRequest.model_validate({"message": "Soru", **options})
        assert request.supersearch is False


@pytest.mark.parametrize(
    "options",
    [
        {"atez_search": True},
        {"atez_search_v2": True},
        {"atez_search_v3": True},
        {"legal_composite": True},
        {"deep_research": True},
        {"asv3_allow_external": True},
        {"asv3_parallel_research": True},
        {"asv3_guarded_experimental": True},
        {"asv3_resume_message_id": 42},
        {"asv3_research_profile": "normal"},
        {"asv3_research_profile": "experimental"},
        {"atez_search_v2_labels": True},
    ],
)
def test_supersearch_rejects_competing_or_external_options(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Supersearch.*mutually exclusive"):
        SendMessageRequest.model_validate(
            {"message": "Soru", "supersearch": True, **options}
        )


@pytest.mark.parametrize(
    "options",
    [
        {"additional_context": "Web page text"},
        {"forced_tool_id": 3},
        {"file_descriptors": [{"id": "outside-pc", "type": "plain_text"}]},
    ],
)
def test_supersearch_rejects_non_corpus_context(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Supersearch accepts sources only"):
        SendMessageRequest.model_validate(
            {"message": "Soru", "supersearch": True, **options}
        )


def test_supersearch_rejects_multiple_or_ambiguous_model_overrides() -> None:
    models = [
        {"model_provider": "openai", "model_version": "gpt-5-mini"},
        {"model_provider": "anthropic", "model_version": "claude-haiku-4-5"},
    ]
    with pytest.raises(ValueError, match="one selected model"):
        SendMessageRequest.model_validate(
            {"message": "Soru", "supersearch": True, "llm_overrides": models}
        )
    with pytest.raises(ValueError, match="one model override format"):
        SendMessageRequest.model_validate(
            {
                "message": "Soru",
                "supersearch": True,
                "llm_override": models[0],
                "llm_overrides": [models[1]],
            }
        )


def test_supersearch_rejects_direct_multi_model_dispatch() -> None:
    request = SendMessageRequest(message="Soru", supersearch=True)
    with patch("onyx.chat.process_message._stream_chat_turn") as dispatch:
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
    dispatch.assert_not_called()
    assert isinstance(result, StreamingError)
    assert result.error_code == "VALIDATION_ERROR"
    assert "Supersearch" in result.error


@pytest.mark.parametrize("label_opt_out", [False, True])
def test_supersearch_scope_keeps_explicit_narrowing(label_opt_out: bool) -> None:
    filters = BaseFilters(document_set=["Subset"], as_of_date=date(2025, 12, 31))
    if label_opt_out:
        filters = filters.model_copy(update={"regulatory_label_search_enabled": False})
    setup = SimpleNamespace(
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        new_msg_req=SendMessageRequest(
            message="Antrepo koşulları nedir?",
            supersearch=True,
            internal_search_filters=filters,
        ),
    )
    actual = process_message._global_regulatory_search_filters(
        cast(ChatTurnSetup, setup)
    )
    assert actual is not None
    assert actual.document_set == ["Subset"]
    assert actual.as_of_date == date(2025, 12, 31)
    assert actual.regulatory_chunks_only is True
    assert actual.regulatory_label_search_enabled is not label_opt_out
    assert filters.regulatory_chunks_only is False


def test_supersearch_does_not_erase_disjoint_source_type_before_scope_binding() -> None:
    filters = BaseFilters(source_type=[DocumentSource.WEB])
    setup = SimpleNamespace(
        persona=SimpleNamespace(id=process_message.DEFAULT_PERSONA_ID),
        new_msg_req=SendMessageRequest(
            message="Soru", supersearch=True, internal_search_filters=filters
        ),
    )
    actual = process_message._global_regulatory_search_filters(
        cast(ChatTurnSetup, setup)
    )
    assert actual is not None and actual.source_type == [DocumentSource.WEB]


@pytest.mark.parametrize("invalid_scope", [None, "agent", "project", "multi"])
@pytest.mark.parametrize("selected_provider", ["openai", "vertex_ai", "gemini"])
def test_actual_chat_worker_uses_only_selected_supersearch_model(
    invalid_scope: str | None, selected_provider: str
) -> None:
    setup = _make_setup(n_models=2 if invalid_scope == "multi" else 1)
    setup.persona.id = (
        process_message.DEFAULT_PERSONA_ID if invalid_scope != "agent" else 42
    )
    setup.search_params.project_id_filter = 7 if invalid_scope == "project" else None
    setup.new_msg_req = SendMessageRequest(
        message="İlgili mevzuat nedir?", supersearch=True
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider=selected_provider,
        model_name="user-selected",
        max_input_tokens=200000,
        temperature=0,
    )
    setup.user_message.id = 1
    setup.reserved_messages[0].id = 2
    with (
        patch("onyx.supersearch.runtime.run_supersearch_loop") as supersearch,
        patch("onyx.legal_composite.runtime.run_legal_composite_loop") as composite,
        patch("onyx.asv3.runtime.run_asv3_loop") as existing,
        patch("onyx.chat.process_message.run_llm_loop") as ordinary,
        patch(
            "onyx.chat.process_message.construct_tools", return_value={}
        ) as general_tools,
        patch(
            "onyx.chat.process_message.construct_internal_search_tool",
            return_value=MagicMock(spec=SearchTool, id=1),
        ) as internal_tools,
        patch("onyx.chat.process_message.get_llm_for_persona") as factory,
        patch("onyx.chat.process_message.get_llm_token_counter", return_value=len),
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
    for other in (existing, ordinary, composite, factory):
        other.assert_not_called()
    selected.with_model.assert_not_called()
    general_tools.assert_not_called()
    if invalid_scope:
        supersearch.assert_not_called()
        internal_tools.assert_not_called()
        assert any(isinstance(packet, StreamingError) for packet in packets)
        return
    assert not any(isinstance(packet, StreamingError) for packet in packets)
    supersearch.assert_called_once()
    internal_tools.assert_called_once()
    actual = supersearch.call_args.kwargs
    assert actual["llm"] is selected
    assert "research_llm" not in actual and "allow_external" not in actual
    assert actual["filters"].regulatory_chunks_only is True
    assert actual["user_message_id"] == 1
    assert actual["assistant_message_id"] == 2


def test_supersearch_constructs_canonical_search_without_custom_or_memory_tools() -> (
    None
):
    from onyx.db.models import Tool as DbTool
    from onyx.tools.tool_constructor import (
        SearchToolConfig,
        construct_internal_search_tool,
    )

    canonical = DbTool(
        id=73, name="internal_search", in_code_tool_id="SearchTool", enabled=True
    )
    external = DbTool(id=74, name="external MCP", enabled=True, mcp_server_id=1)
    persona = MagicMock(
        id=process_message.DEFAULT_PERSONA_ID,
        name="default",
        tools=[canonical, external],
    )
    persona.document_sets = []
    persona.attached_documents = []
    persona.hierarchy_nodes = []
    persona.search_start_date = None
    user = MagicMock(oauth_accounts=[], enable_memory_tool=True)
    filters = BaseFilters(as_of_date=date(2025, 7, 1), document_set=["User subset"])
    with (
        patch(
            "onyx.tools.tool_constructor.get_builtin_tool", return_value=canonical
        ) as builtin,
        patch("onyx.tools.tool_constructor.get_current_search_settings"),
        patch("onyx.tools.tool_constructor.get_default_document_index"),
        patch("onyx.tools.tool_constructor.get_effective_persona_tools") as effective,
        patch(
            "onyx.tools.tool_constructor.resolve_mcp_credentials"
        ) as external_credentials,
        patch("onyx.tools.tool_constructor.MemoryTool") as memory,
        patch(
            "onyx.tools.tool_constructor._resolve_document_set_file_ids"
        ) as catalogue,
        patch.object(SearchTool, "is_available", return_value=True),
    ):
        actual = construct_internal_search_tool(
            persona=persona,
            emitter=MagicMock(),
            user=user,
            llm=MagicMock(),
            db_session=MagicMock(),
            search_tool_config=SearchToolConfig(
                user_selected_filters=filters,
                document_set_names_override=["PC Külliyatı"],
                auto_detect_filters=True,
                bypass_acl=True,
                enable_slack_search=True,
            ),
        )
    assert isinstance(actual, SearchTool) and actual.id == 73
    assert actual.user_selected_filters == filters
    assert actual.auto_detect_filters is False
    assert actual.enable_slack_search is False
    assert actual.bypass_acl is False
    builtin.assert_called_once()
    for uncalled in (effective, external_credentials, memory, catalogue):
        uncalled.assert_not_called()
