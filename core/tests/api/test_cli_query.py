"""CLI tests for the api service's indexed `query` command.

Building/managing the index itself is the indexer service's CLI
(`fs_explorer_indexer.main`), covered in tests/indexer/test_cli_indexing.py.
"""

from pathlib import Path

import fs_explorer_api.main as main_module
from typer.testing import CliRunner


def test_query_command_runs_indexed_workflow(tmp_path: Path, monkeypatch) -> None:
    called: dict[str, object] = {}

    async def fake_run_workflow(
        task: str,
        folder: str = ".",
        *,
        database_url: str | None = None,
    ) -> None:
        called["task"] = task
        called["folder"] = folder
        called["database_url"] = database_url

    monkeypatch.setattr(main_module, "run_workflow", fake_run_workflow)

    runner = CliRunner()
    result = runner.invoke(
        main_module.app,
        [
            "--task",
            "purchase price?",
            "--folder",
            str(tmp_path),
            "--database-url",
            "postgresql://user:pass@localhost:5432/tmp",
        ],
    )

    assert result.exit_code == 0
    assert called["task"] == "purchase price?"
    assert called["folder"] == str(tmp_path)
    assert called["database_url"] == "postgresql://user:pass@localhost:5432/tmp"


def test_query_is_the_default_command(tmp_path: Path, monkeypatch) -> None:
    """`query` is the app's only command, so Typer collapses it to the default
    — `explore --task ... --folder ...` works without typing `query`."""
    called: dict[str, object] = {}

    async def fake_run_workflow(
        task: str,
        folder: str = ".",
        *,
        database_url: str | None = None,
    ) -> None:
        called["task"] = task
        called["folder"] = folder

    monkeypatch.setattr(main_module, "run_workflow", fake_run_workflow)

    runner = CliRunner()
    result = runner.invoke(
        main_module.app,
        ["--task", "who is the CTO?", "--folder", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert called["task"] == "who is the CTO?"
    assert called["folder"] == str(tmp_path)
