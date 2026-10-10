from unittest.mock import MagicMock, patch

import httpx
import pytest

from onyx.legal_review.streaming import Deadline, DeadlineStream


def test_every_socket_read_clamps_to_absolute_deadline() -> None:
    wire = MagicMock()
    wire.read.return_value = b"chunk"
    active = MagicMock()
    stream = DeadlineStream(wire, Deadline(100, 45, active))
    with patch("onyx.legal_review.streaming.time.monotonic", side_effect=[10, 80, 101]):
        assert stream.read(10) == b"chunk"
        assert stream.read(10) == b"chunk"
        with pytest.raises(TimeoutError):
            stream.read(10)
    assert [call.kwargs["timeout"] for call in wire.read.call_args_list] == [45, 20]
    assert active.call_count == 3


def test_cancel_prevents_next_network_operation() -> None:
    wire = MagicMock()
    active = MagicMock(side_effect=RuntimeError("cancelled"))
    stream = DeadlineStream(wire, Deadline(100, 45, active))
    with pytest.raises(RuntimeError, match="cancelled"):
        stream.write(b"data")
    wire.write.assert_not_called()


def test_incomplete_stream_is_rejected_and_received_usage_is_accounted() -> None:
    import time

    from litellm.types.utils import Delta, StreamingChoices
    from litellm.types.utils import ModelResponseStream as RawChunk

    from onyx.legal_review.streaming import read_stream
    from onyx.llm.models import ReasoningEffort

    selected, usage = MagicMock(), MagicMock()
    selected._completion.return_value = iter(
        [
            RawChunk(
                id="reader",
                created=1,
                model="gemini-3.8-flash",
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(role="assistant", content='{"incomplete"'),
                        finish_reason="length",
                    )
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            )
        ]
    )
    with pytest.raises(ValueError, match="complete response"):
        read_stream(
            selected,
            deadline=Deadline(time.monotonic() + 20, 5, lambda: None),
            prompt=[],
            tools=None,
            tool_choice=None,
            structured_response_format=None,
            max_tokens=50,
            reasoning_effort=ReasoningEffort.LOW,
            user_identity=None,
            record_usage=usage,
        )
    usage.assert_called_once_with(10, 3)
    selected._completion.assert_called_once()
    assert selected._completion.call_args.kwargs["provider_compatibility_attempts"] == 1


def test_provider_read_timeout_uses_workflow_failure_path_without_retry() -> None:
    from onyx.legal_review.streaming import read_stream
    from onyx.llm.models import ReasoningEffort

    selected = MagicMock()
    selected._completion.side_effect = httpx.ReadTimeout("Network stalled")
    with pytest.raises(TimeoutError, match="reader deadline exceeded"):
        read_stream(
            selected,
            deadline=Deadline(float("inf"), 45, lambda: None),
            prompt=[],
            tools=None,
            tool_choice=None,
            structured_response_format=None,
            max_tokens=50,
            reasoning_effort=ReasoningEffort.LOW,
            user_identity=None,
            record_usage=MagicMock(),
        )
    selected._completion.assert_called_once()
