"""Tests for the in-memory resumable-run registry (runs.py)."""

import os
import time
from unittest.mock import patch

from fs_explorer_api.agent import FsExplorerAgent
from fs_explorer_api.exploration_trace import ExplorationTrace
from fs_explorer_api.runs import RunRecord, get_run, new_run_id, register_run, remove_run
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
