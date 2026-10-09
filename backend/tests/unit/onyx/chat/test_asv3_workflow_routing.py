"""Drive the actual chat worker while isolating provider and persistence boundaries."""

from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.asv3.workflow_variant import (
    ASV3_GUARDED_EXPERIMENTAL_VARIANT,
    ASV3_GUARDRAILS_V2_VARIANT,
    ASV3_GUARDRAILS_V3_VARIANT,
    ASV3_TUNED_VARIANT,
)
from onyx.chat import process_message
from onyx.chat.chat_state import ChatTurnSetup
from onyx.chat.models import StreamingError
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.server.query_and_chat.models import SendMessageRequest
from tests.unit.onyx.chat.test_multi_model_streaming import _make_setup


@pytest.mark.parametrize(
    "requested_profile,parallel,guarded,deep,effective_profile,variant,lite",
    [
        ("experimental", True, False, False, "normal", ASV3_TUNED_VARIANT, False),
        ("experimental", False, False, False, "experimental", "standard", False),
        (
            "normal",
            False,
            True,
            False,
            "normal",
            ASV3_GUARDED_EXPERIMENTAL_VARIANT,
            True,
        ),
        ("normal", False, False, False, "normal", "standard", True),
        ("deep", False, False, True, "deep", "standard", True),
    ],
)
def test_actual_chat_worker_maps_variant_before_lite_construction(
    requested_profile: Literal["normal", "deep", "experimental"],
    parallel: bool,
    guarded: bool,
    deep: bool,
    effective_profile: str,
    variant: str,
    lite: bool,
) -> None:
    setup = _make_setup()
    setup.persona.id = process_message.DEFAULT_PERSONA_ID
    setup.new_msg_req = SendMessageRequest.model_validate(
        {
            "message": "İlgili mevzuat nedir?",
            "atez_search_v3": not deep,
            "deep_research": deep,
            "asv3_research_profile": requested_profile,
            "asv3_parallel_research": parallel,
            "asv3_guarded_experimental": guarded,
        }
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider="openai",
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
    invocations: list[dict[str, Any]] = []

    with (
        patch(
            "onyx.asv3.runtime.run_asv3_loop",
            side_effect=lambda **args: invocations.append(args),
        ),
        patch(
            "onyx.chat.process_message.construct_tools", return_value={}
        ) as construct,
        patch(
            "onyx.chat.process_message.get_llm_for_persona", return_value=secondary
        ) as lite_factory,
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
    assert len(invocations) == 1
    actual = invocations[0]
    assert actual["research_profile"] == effective_profile
    assert actual["parallel_research"] is False
    assert actual["workflow_variant"] == variant
    assert actual["llm"] is selected
    assert actual["research_llm"] is (secondary if lite else None)
    assert lite_factory.call_count == int(lite)
    selected.with_model.assert_not_called()
    assert construct.call_args.kwargs["llm"] is selected
    assert setup.new_msg_req.asv3_research_profile == requested_profile
    assert setup.new_msg_req.asv3_parallel_research is parallel


@pytest.mark.parametrize(
    "saved_variant", ["standard", ASV3_GUARDED_EXPERIMENTAL_VARIANT]
)
def test_actual_chat_worker_restores_variant_from_resume_checkpoint(
    saved_variant: str,
) -> None:
    setup = _make_setup()
    setup.persona.id = process_message.DEFAULT_PERSONA_ID
    setup.new_msg_req = SendMessageRequest.model_validate(
        {
            "message": "İlgili mevzuat nedir?",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_resume_message_id": 41,
        }
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider="openai",
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
    invocations: list[dict[str, Any]] = []

    with (
        patch(
            "onyx.chat.process_message.load_asv3_checkpoint",
            return_value={"asv3_workflow_variant": saved_variant},
        ),
        patch(
            "onyx.asv3.runtime.run_asv3_loop",
            side_effect=lambda **args: invocations.append(args),
        ),
        patch("onyx.chat.process_message.construct_tools", return_value={}),
        patch(
            "onyx.chat.process_message.get_llm_for_persona", return_value=secondary
        ) as lite_factory,
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
    assert setup.new_msg_req.asv3_guarded_experimental is False
    assert invocations[0]["workflow_variant"] == saved_variant
    assert invocations[0]["research_profile"] == "normal"
    assert invocations[0]["parallel_research"] is False
    assert invocations[0]["research_llm"] is secondary
    lite_factory.assert_called_once()


def test_actual_chat_worker_builds_isolated_guardrails_v2_repair_model() -> None:
    setup = _make_setup()
    setup.persona.id = process_message.DEFAULT_PERSONA_ID
    setup.new_msg_req = SendMessageRequest.model_validate(
        {
            "message": "İlgili mevzuat nedir?",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_guardrails_v2": True,
        }
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="user-selected",
        max_input_tokens=200000,
        temperature=0,
    )
    research = MagicMock(spec=LLM)
    research.config = selected.config.model_copy(
        update={"model_name": "gemini-3.5-flash-lite"}
    )
    repair = MagicMock(spec=LLM)
    repair.config = selected.config.model_copy(
        update={"model_name": "gemini-3.8-flash"}
    )
    selected.with_model.side_effect = [research, repair]
    setup.user_message.id = 1
    setup.reserved_messages[0].id = 2
    invocations: list[dict[str, Any]] = []

    with (
        patch(
            "onyx.asv3.runtime.run_asv3_loop",
            side_effect=lambda **args: invocations.append(args),
        ),
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
    assert len(invocations) == 1
    actual = invocations[0]
    assert actual["workflow_variant"] == ASV3_GUARDRAILS_V2_VARIANT
    assert actual["research_llm"] is research
    assert actual["repair_llm"] is repair
    assert [call.args[0] for call in selected.with_model.call_args_list] == [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    ]


def test_actual_chat_worker_restores_guardrails_v2_from_resume_checkpoint() -> None:
    setup = _make_setup()
    setup.persona.id = process_message.DEFAULT_PERSONA_ID
    setup.new_msg_req = SendMessageRequest.model_validate(
        {
            "message": "İlgili mevzuat nedir?",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_guarded_experimental": True,
            "asv3_resume_message_id": 42,
        }
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider="openai",
        model_name="user-selected",
        max_input_tokens=200000,
        temperature=0,
    )
    research = MagicMock(spec=LLM)
    research.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.5-flash-lite",
        max_input_tokens=1_048_576,
        temperature=0,
    )
    repair = MagicMock(spec=LLM)
    repair.config = research.config.model_copy(
        update={"model_name": "gemini-3.8-flash"}
    )
    setup.user_message.id = 1
    setup.reserved_messages[0].id = 2
    invocations: list[dict[str, Any]] = []

    with (
        patch(
            "onyx.chat.process_message.load_asv3_checkpoint",
            return_value={"asv3_workflow_variant": ASV3_GUARDRAILS_V2_VARIANT},
        ),
        patch(
            "onyx.asv3.runtime.run_asv3_loop",
            side_effect=lambda **args: invocations.append(args),
        ),
        patch("onyx.chat.process_message.construct_tools", return_value={}),
        patch(
            "onyx.chat.process_message.get_llm_for_persona",
            side_effect=[research, repair],
        ) as model_factory,
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
    assert invocations[0]["workflow_variant"] == ASV3_GUARDRAILS_V2_VARIANT
    assert invocations[0]["research_llm"] is research
    assert invocations[0]["repair_llm"] is repair
    assert [
        call.kwargs["llm_override"].model_version
        for call in model_factory.call_args_list
    ] == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]


def test_actual_chat_worker_restores_guardrails_v3_from_resume_checkpoint() -> None:
    setup = _make_setup()
    setup.persona.id = process_message.DEFAULT_PERSONA_ID
    setup.new_msg_req = SendMessageRequest.model_validate(
        {
            "message": "İlgili mevzuat nedir?",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_guardrails_v2": True,
            "asv3_resume_message_id": 43,
        }
    )
    selected = setup.llms[0]
    selected.config = LLMConfig(
        model_provider="openai",
        model_name="user-selected",
        max_input_tokens=200000,
        temperature=0,
    )
    research = MagicMock(spec=LLM)
    research.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.5-flash-lite",
        max_input_tokens=1_048_576,
        temperature=0,
    )
    repair = MagicMock(spec=LLM)
    repair.config = research.config.model_copy(
        update={"model_name": "gemini-3.8-flash"}
    )
    setup.user_message.id = 1
    setup.reserved_messages[0].id = 2
    invocations: list[dict[str, Any]] = []

    with (
        patch(
            "onyx.chat.process_message.load_asv3_checkpoint",
            return_value={"asv3_workflow_variant": ASV3_GUARDRAILS_V3_VARIANT},
        ),
        patch(
            "onyx.asv3.runtime.run_asv3_loop",
            side_effect=lambda **args: invocations.append(args),
        ),
        patch("onyx.chat.process_message.construct_tools", return_value={}),
        patch(
            "onyx.chat.process_message.get_llm_for_persona",
            side_effect=[research, repair],
        ) as model_factory,
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
    assert invocations[0]["workflow_variant"] == ASV3_GUARDRAILS_V3_VARIANT
    assert invocations[0]["research_llm"] is research
    assert invocations[0]["repair_llm"] is repair
    assert [
        call.kwargs["llm_override"].model_version
        for call in model_factory.call_args_list
    ] == ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
