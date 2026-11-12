import copy
import hashlib
import json
from typing import Any

import pytest

from onyx.asv3 import runtime
from onyx.asv3.workflow_variant import ASV3_EXP_CROPPED_VARIANT, ASV3_STANDARD_VARIANT
from onyx.chat.models import ChatLoadedFile
from onyx.chat.session_attachments import build_asv3_attachment_context
from onyx.file_store.models import ChatFileType
from tests.unit.onyx.asv3.test_runtime import response, setup_run, user_payload

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize("variant", [ASV3_STANDARD_VARIANT, ASV3_EXP_CROPPED_VARIANT])
def test_full_session_files_reach_native_model_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
    text = "Fatura satırı: 1200 EUR.\n" * 100 + "EXACT_ATTACHMENT_TAIL_123"
    context = build_asv3_attachment_context(
        [
            ChatLoadedFile(
                file_id="private-file",
                file_type=ChatFileType.TABULAR,
                filename="fatura.csv",
                content=b"",
                content_text=text,
                token_count=100,
            )
        ],
        len,
        100_000,
    )
    kwargs["attachment_context"] = context

    def answer(**arguments: Any) -> Any:
        payload = user_payload(arguments["prompt"][1])
        assert context in payload["conversation"]
        assert "EXACT_ATTACHMENT_TAIL_123" in payload["conversation"]
        return response(
            calls=[("ask_user", {"question": "Hangi fatura satırını inceleyelim?"})]
        )

    llm.invoke.side_effect = answer
    runtime.run_asv3_loop(**kwargs)
    fingerprint = hashlib.sha256(context.encode()).hexdigest()
    assert checkpoints[-1]["session_attachment_sha256"] == fingerprint
    if variant == ASV3_EXP_CROPPED_VARIANT:
        request = kwargs["simple_chat_history"][0].message + "\n\n" + context
        assert (
            checkpoints[-1]["cropped_repair_state"]["request_hash"]
            == hashlib.sha256(
                json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
        )

    previous = copy.deepcopy(checkpoints[-1])
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_: previous)
    kwargs["resume_message_id"] = kwargs["assistant_message_id"]
    kwargs["attachment_context"] = context + "changed"
    calls_before = llm.invoke.call_count
    with pytest.raises(ValueError, match="same session attachments"):
        runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == calls_before


@pytest.mark.parametrize("variant", [ASV3_STANDARD_VARIANT, ASV3_EXP_CROPPED_VARIANT])
def test_original_pdf_reaches_first_question_request_without_pre_reading(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    import base64

    from onyx.chat import session_attachments
    from onyx.llm.models import FileContentPart
    from onyx.llm.multi_llm import LitellmLLM
    from tests.unit.onyx.file_processing.test_chat_attachments import _pdf

    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
    original = _pdf("")
    file = ChatLoadedFile(
        file_id="private-pdf",
        file_type=ChatFileType.DOC,
        filename="scan.pdf",
        content=original,
        content_text=None,
        token_count=0,
    )
    context, native_files = session_attachments.build_asv3_attachment_inputs(
        [file], len, 100_000
    )
    kwargs["attachment_context"] = context
    kwargs["native_attachments"] = native_files
    initial_llm = LitellmLLM(
        api_key=None,
        model_provider="vertex_ai",
        model_name="gemini-3.5-flash-lite",
        max_input_tokens=500_000,
    )
    kwargs["initial_attachment_llm"] = initial_llm
    requests = []

    def answer(**arguments: Any) -> Any:
        requests.append(arguments)
        prompt = arguments["prompt"]
        parts = [
            part
            for message in prompt
            if isinstance(message.content, list)
            for part in message.content
            if isinstance(part, FileContentPart)
        ]
        assert len(parts) == 1
        assert parts[0].file.filename == "scan.pdf"
        assert base64.b64decode(parts[0].file.file_data.split(",", 1)[1]) == original
        payload = user_payload(prompt[-1])
        assert kwargs["simple_chat_history"][0].message == payload["request"]
        assert (
            hashlib.sha256(original).hexdigest()
            in user_payload(prompt[1])["conversation"]
        )
        return response(calls=[("ask_user", {"question": "Hangi belge alanı?"})])

    monkeypatch.setattr(initial_llm, "invoke", answer)
    llm.invoke.side_effect = AssertionError("Initial PDF request must use Flash Lite")
    runtime.run_asv3_loop(**kwargs)
    assert len(requests) == 1
    assert (
        checkpoints[-1]["session_attachment_sha256"]
        == hashlib.sha256(context.encode()).hexdigest()
    )
    assert "data:application/pdf" not in json.dumps(checkpoints[-1])

    # Changing the actual original must invalidate a resume, even without text.
    previous = copy.deepcopy(checkpoints[-1])
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_: previous)
    changed = file.model_copy(update={"content": _pdf("replacement")})
    changed_context, changed_native = session_attachments.build_asv3_attachment_inputs(
        [changed], len, 100_000
    )
    kwargs.update(
        attachment_context=changed_context,
        native_attachments=changed_native,
        resume_message_id=kwargs["assistant_message_id"],
    )
    with pytest.raises(ValueError, match="same session attachments"):
        runtime.run_asv3_loop(**kwargs)
    assert len(requests) == 1
