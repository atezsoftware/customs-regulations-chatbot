"""Tests for the in-memory resumable-run registry (runs.py)."""

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from fs_explorer_api.agent import FsExplorerAgent
from fs_explorer_api.exploration_trace import ExplorationTrace
from fs_explorer_api.runs import (
    RunRecord,
    get_run,
    new_run_id,
    register_run,
    remove_run,
)
from fs_explorer_api.server import _finish_run
from fs_explorer_api import runs as runs_mod
from .conftest import make_mock_llm_client


def _record(run_id: str, **overrides) -> RunRecord:
    defaults = dict(
        run_id=run_id,
        agent=FsExplorerAgent(llm_client=make_mock_llm_client()),
        trace=ExplorationTrace(root_directory="."),
        step_number=1,
        folder=".",
        use_index=False,
        enable_semantic=False,
        enable_metadata=False,
        effort="medium",
        index_folders=[],
        database_url=None,
        original_task="find the readme",
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
class TestRunRegistry:
    def test_register_then_get_round_trips(self) -> None:
        run_id = new_run_id()
        register_run(_record(run_id))

        fetched = get_run(run_id)

        assert fetched is not None
        assert fetched.run_id == run_id
        assert fetched.original_task == "find the readme"

    def test_get_missing_run_returns_none(self) -> None:
        assert get_run("does-not-exist") is None

    def test_remove_run_makes_it_unresumable(self) -> None:
        run_id = new_run_id()
        register_run(_record(run_id))

        remove_run(run_id)

        assert get_run(run_id) is None

    def test_expired_run_is_swept_on_next_access(self) -> None:
        run_id = new_run_id()
        record = _record(run_id)
        register_run(record)
        # Simulate the record having gone stale well past the TTL.
        record.updated_at = time.monotonic() - runs_mod.RUN_TTL_SECONDS - 1

        assert get_run(run_id) is None


class _SingleFlightFinalClient:
    model = "test-final"

    def __init__(self) -> None:
        self.calls = 0
        self.release = asyncio.Event()

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        self.calls += 1
        yield "first "
        await self.release.wait()
        yield "second"

    def last_stream_usage(self):
        return None


class _DisconnectOnFirstDelta:
    async def send_json(self, message) -> None:
        if message["type"] == "answer_delta":
            raise ConnectionError("WebSocket disconnected")


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.messages = []

    async def send_json(self, message) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_finish_run_replays_single_flight_final_after_disconnect() -> None:
    client = _SingleFlightFinalClient()
    agent = FsExplorerAgent(llm_client=client)
    agent.configure_task("evidence")
    trace = ExplorationTrace(root_directory=".")

    async def flush_llm_calls() -> None:
        return None

    with pytest.raises(ConnectionError, match="disconnected"):
        await _finish_run(
            _DisconnectOnFirstDelta(),
            run_id="replay-run",
            agent=agent,
            trace=trace,
            step_number=1,
            folder_path=Path("."),
            use_index=True,
            final_result="fallback",
            result_error=None,
            run_started_at=time.monotonic(),
            flush_llm_calls=flush_llm_calls,
        )

    client.release.set()
    assert agent._final_answer_task is not None
    await asyncio.wait_for(agent._final_answer_task, timeout=1)

    resumed_websocket = _RecordingWebSocket()
    await _finish_run(
        resumed_websocket,
        run_id="replay-run",
        agent=agent,
        trace=trace,
        step_number=1,
        folder_path=Path("."),
        use_index=True,
        final_result="",
        result_error=None,
        run_started_at=time.monotonic(),
        flush_llm_calls=flush_llm_calls,
    )

    answer_done = next(
        message
        for message in resumed_websocket.messages
        if message["type"] == "answer_done"
    )
    assert answer_done["data"]["final_result"] == "first second"
    assert client.calls == 1
