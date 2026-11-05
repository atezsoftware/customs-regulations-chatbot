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
