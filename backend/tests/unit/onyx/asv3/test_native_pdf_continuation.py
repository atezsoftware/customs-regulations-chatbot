from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponseStream

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext, SharedBudget
from onyx.asv3.session_attachments import (
    NATIVE_SESSION_ATTACHMENTS,
    session_attachment_prompt,
)
from onyx.chat.models import ChatLoadedFile
from onyx.chat.session_attachments import build_asv3_attachment_inputs
from onyx.file_store.models import ChatFileType
from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.llm.multi_llm import LitellmLLM
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.file_processing.test_chat_attachments import _pdf


def test_selected_claude_continuation_accepts_pdf_with_native_sampling_defaults() -> (
    None
):
    _, files = build_asv3_attachment_inputs(
        [
            ChatLoadedFile(
                file_id="pdf",
                file_type=ChatFileType.DOC,
                filename="scan.pdf",
                content=_pdf(""),
                content_text=None,
                token_count=0,
            )
        ],
        len,
        100_000,
    )
    llm = LitellmLLM(
        api_key="test",
        model_provider="anthropic",
        model_name="claude-opus-4-7",
        max_input_tokens=100_000,
        temperature=0,
    )
    context = RunContext(
        budget=SharedBudget(unlimited_execution=True),
        services={NATIVE_SESSION_ATTACHMENTS: files},
    )
    adapter = ResearchModel(llm, context, reasoning_effort=ReasoningEffort.OFF)
    chunk = ModelResponseStream.model_validate(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Belge alındı."},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    with patch("litellm.completion", return_value=iter([chunk])) as completion:
        adapter._invoke_once(
            [UserMessage(content="Sorunun devamını belgeye göre yanıtla.")],
            [],
            LLMFlow.ASV3_COORDINATOR,
            max_tokens=2000,
            research=True,
            structured=False,
        )
    sent = completion.call_args.kwargs
    assert sent["model"] == "anthropic/claude-opus-4-7"
    assert "temperature" not in sent
    assert any(
        part.get("type") == "file"
        for message in sent["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )
    assert llm.config.temperature == 0
    with pytest.raises(ValueError, match="pdf_model_does_not_support_temperature_zero"):
        llm._completion(
            session_attachment_prompt([UserMessage(content="transcribe")], files),
            tools=None,
            tool_choice=None,
            stream=False,
            parallel_tool_calls=False,
        )


@pytest.mark.usefixtures("empty_source_inventory")
def test_first_flash_lite_decision_then_selected_claude_get_same_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64
    from typing import Any

    from onyx.asv3 import runtime
    from onyx.asv3.workflow_variant import ASV3_STANDARD_VARIANT
    from onyx.llm.models import FileContentPart
    from tests.unit.onyx.asv3.test_runtime import response, setup_run

    kwargs, broker, _, _, _ = setup_run(monkeypatch)
    kwargs.pop("test_language")
    original = _pdf("")
    context, files = build_asv3_attachment_inputs(
        [
            ChatLoadedFile(
                file_id="pdf",
                file_type=ChatFileType.DOC,
                filename="scan.pdf",
                content=original,
                content_text=None,
                token_count=0,
            )
        ],
        len,
        500_000,
    )
    initial = LitellmLLM(
        api_key="test",
        model_provider="vertex_ai",
        model_name="gemini-3.5-flash-lite",
        max_input_tokens=500_000,
    )
    continuation = LitellmLLM(
        api_key="test",
        model_provider="anthropic",
        model_name="claude-opus-4-7",
        max_input_tokens=500_000,
        temperature=0,
    )
    first_calls = []

    def first(**arguments: Any) -> Any:
        first_calls.append(arguments)
        parts = [
            part
            for message in arguments["prompt"]
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, FileContentPart)
        ]
        assert base64.b64decode(parts[0].file.file_data.split(",", 1)[1]) == original
        return response(
            calls=[
                (
                    "read_source_range",
                    {
                        "source_id": str(broker.sources[0].id),
                        "_language": "tr",
                        "_public_update": [
                            "Belge incelemesi",
                            "Belgeye ilişkin kaynak koşullarını inceliyorum.",
                        ],
                    },
                )
            ]
        )

    monkeypatch.setattr(initial, "invoke", first)
    kwargs.update(
        research_profile="normal",
        workflow_variant=ASV3_STANDARD_VARIANT,
        llm=continuation,
        attachment_context=context,
        native_attachments=files,
        initial_attachment_llm=initial,
    )
    chunk = ModelResponseStream.model_validate(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "ask-1",
                                "type": "function",
                                "function": {
                                    "name": "ask_user",
                                    "arguments": '{"question":"Hangi belge alanı incelenecek?"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    with patch("litellm.completion", return_value=iter([chunk])) as completion:
        runtime.run_asv3_loop(**kwargs)
    assert len(first_calls) == 1
    assert completion.call_count == 1
    sent = completion.call_args.kwargs
    assert sent["model"] == "anthropic/claude-opus-4-7"
    native = [
        part
        for message in sent["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "file"
    ]
    assert base64.b64decode(native[0]["file"]["file_data"].split(",", 1)[1]) == original
