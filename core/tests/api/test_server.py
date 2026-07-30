import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from fs_explorer_api.agent import _normalize_indexed_text
from fs_explorer_api.server import (
    _await_with_websocket_heartbeat,
    _completion_is_incomplete,
    _decode_html_entities,
)


def test_decode_html_entities_preserves_turkish_text() -> None:
    value = "Transit s&uuml;resinin a&#351;&#305;lmas&#305; &ldquo;otomatik&rdquo; de&#287;ildir."

    assert (
        _decode_html_entities(value)
        == "Transit süresinin aşılması “otomatik” değildir."
    )


def test_normalize_indexed_text_decodes_entities_before_model_context() -> None:
    value = "G&uuml;mr&uuml;k &amp; transit: a&#351;&#305;lma"

    assert _normalize_indexed_text(value) == "Gümrük & transit: aşılma"


def test_completion_surfaces_terminal_multi_agent_evidence_gaps() -> None:
    agent = cast(
        Any,
        SimpleNamespace(forced_stop=False, multi_agent_incomplete=True),
    )

    assert _completion_is_incomplete(agent=agent, result_error=None) is True
    assert (
        _completion_is_incomplete(agent=agent, result_error="provider failed") is False
    )


@pytest.mark.asyncio
async def test_slow_work_emits_transport_heartbeats_until_result() -> None:
    class RecordingWebSocket:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        async def send_json(self, message: dict[str, object]) -> None:
            self.messages.append(message)

    async def slow_result() -> str:
        await asyncio.sleep(0.035)
        return "finished"

    websocket = RecordingWebSocket()
    result = await _await_with_websocket_heartbeat(
        slow_result(),
        cast(Any, websocket),
        phase="research",
        interval_seconds=0.01,
    )

    assert result == "finished"
    assert len(websocket.messages) >= 2
    assert all(message["type"] == "heartbeat" for message in websocket.messages)
