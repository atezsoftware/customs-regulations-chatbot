"""The indexing worker must run in the parser-free lightweight image.

Markdown is read as plain text, so ingesting it needs none of the
source-document parsers that image omits. These tests pin that the worker
actually starts there — a registration that reaches for markitdown or
unstructured at import time would crash the container on boot, not at upload.
"""

import os
import subprocess
import sys
from pathlib import Path


def _backend_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "onyx").is_dir() and (parent / "tests").is_dir()
    )


def _run_with_parsers_blocked(verification: str) -> subprocess.CompletedProcess[str]:
    backend_root = _backend_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(backend_root), env.get("PYTHONPATH")) if value
    )
    preamble = """
import sys

blocked_import_roots = {
    "markitdown",
    "playwright",
    "pypdfium2",
    "unstructured",
    "unstructured_client",
}


class BlockedImportFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked_import_roots:
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, BlockedImportFinder())
"""
    return subprocess.run(
        [sys.executable, "-c", preamble + verification],
        cwd=str(backend_root),
        env=env,
        capture_output=True,
        text=True,
    )


def test_processing_worker_registers_indexing_without_the_parser_stack() -> None:
    result = _run_with_parsers_blocked(
        """
from onyx.background.celery.versioned_apps.user_file_processing import app
from onyx.configs.constants import OnyxCeleryTask

app.loader.import_default_modules()
app.finalize()

assert OnyxCeleryTask.PROCESS_SINGLE_USER_FILE in app.tasks, sorted(app.tasks)
assert blocked_import_roots.isdisjoint(sys.modules), blocked_import_roots.intersection(
    sys.modules
)
"""
    )

    assert result.returncode == 0, result.stderr


def test_markdown_extraction_needs_no_parser_stack() -> None:
    """The path a .md upload takes, exercised with the parsers unavailable."""

    result = _run_with_parsers_blocked(
        """
from io import BytesIO
from unittest.mock import patch

# The Unstructured cloud key lives in the KV store; stub the lookup so this
# exercises parsing rather than requiring a database.
patch(
    "onyx.file_processing.extract_file_text.get_unstructured_api_key",
    return_value=None,
).start()

from onyx.file_processing.extract_file_text import extract_file_text

text = extract_file_text(
    file=BytesIO("# Madde 1\\n\\nGumruk kiymeti beyan edilir.\\n".encode()),
    file_name="madde.md",
    break_on_unprocessable=True,
    extension=".md",
)
assert "Gumruk kiymeti" in text, text
"""
    )

    assert result.returncode == 0, result.stderr


def _lite_supervisord() -> str:
    return (_backend_root() / "supervisord-lite.conf").read_text(encoding="utf-8")


def test_lite_image_runs_the_user_file_processing_worker() -> None:
    """Without this worker, uploads sit in PROCESSING forever: the API enqueues
    onto a queue nothing in the image consumes."""

    config = _lite_supervisord()

    assert "[program:celery_worker_user_file_processing]" in config
    assert "versioned_apps.user_file_processing worker" in config


def test_lite_user_file_queues_have_exactly_one_consumer_each() -> None:
    """Two workers on one queue would race for the same task."""

    queues_by_program: dict[str, set[str]] = {}
    program: str | None = None
    for line in _lite_supervisord().splitlines():
        stripped = line.strip()
        if stripped.startswith("[program:"):
            program = stripped.removeprefix("[program:").removesuffix("]")
        elif stripped.startswith("-Q ") and program is not None:
            queues_by_program[program] = set(stripped.removeprefix("-Q ").split(","))

    user_file_queues = [
        (program, queue)
        for program, queues in queues_by_program.items()
        for queue in queues
        if queue.startswith("user_file")
    ]
    consumed = [queue for _program, queue in user_file_queues]

    assert sorted(consumed) == sorted(set(consumed)), queues_by_program
    assert "user_file_processing" in consumed
