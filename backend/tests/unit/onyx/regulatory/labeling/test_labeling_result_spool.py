import json
import random
import sqlite3
import stat
import tracemalloc
import zlib
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from onyx.db import regulatory_labeling_results as result_spool
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchResultError


def _hash(index: int) -> str:
    return f"{index:064x}"


def _line(index: int, text: str = '{"labels":[],"abstained":true}') -> str:
    return (
        json.dumps(
            {
                "key": _hash(index),
                "response": {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": text}]},
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )
        + "\n"
    )


@pytest.fixture
def scratch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Path]:
    paths: list[Path] = []

    def temporary_directory() -> TemporaryDirectory[str]:
        directory = TemporaryDirectory(dir=tmp_path)
        paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(result_spool, "TemporaryDirectory", temporary_directory)
    return paths


def test_stages_out_of_order_results_and_leaves_missing_results_absent() -> None:
    progress_calls = 0

    def on_progress() -> None:
        nonlocal progress_calls
        progress_calls += 1

    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected(_hash(index) for index in range(3))
        spool.stage(iter([_line(1), "\n", _line(0)]), on_progress=on_progress)

        results = spool.get_many([_hash(0), _hash(1), _hash(2)])

    assert set(results) == {_hash(0), _hash(1)}
    assert results[_hash(1)].context == '{"labels":[],"abstained":true}'
    assert results[_hash(0)].error is None
    assert progress_calls >= 3


@pytest.mark.parametrize("late_line", [_line(0), _line(2), "not json\n"])
def test_late_invalid_row_prevents_all_reads_and_closes_source(
    late_line: str, scratch_paths: list[Path]
) -> None:
    closed = False

    def lines() -> Iterator[str]:
        nonlocal closed
        try:
            yield _line(0)
            yield late_line
        finally:
            closed = True

    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected([_hash(0), _hash(1)])
        with pytest.raises(ValueError):
            spool.stage(lines(), on_progress=lambda: None)
        with pytest.raises(ValueError, match="validated"):
            spool.get_many([_hash(0)])
        assert closed

    assert scratch_paths
    assert all(not path.exists() for path in scratch_paths)


def test_duplicate_expected_hash_across_pages_is_rejected() -> None:
    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected([_hash(0), _hash(1)])
        with pytest.raises(ValueError, match="duplicate"):
            spool.add_expected([_hash(1)])


@pytest.mark.parametrize("request_hash", ["", "z" * 64, "a" * 63])
def test_invalid_expected_hash_is_rejected(request_hash: str) -> None:
    with result_spool.LabelingResultSpool() as spool:
        with pytest.raises(ValueError, match="hash"):
            spool.add_expected([request_hash])


def test_per_row_bound_also_rejects_oversized_blank_lines() -> None:
    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected([_hash(0)])
        with pytest.raises(ValueError, match="row.*size limit"):
            spool.stage(iter([" " * (1024 * 1024 + 1)]), on_progress=lambda: None)


@pytest.mark.parametrize("over_limit", [False, True])
def test_total_limit_counts_utf8_bytes_and_rejects_late_overflow(
    over_limit: bool,
) -> None:
    lines = [_line(0, "Türkçe"), _line(1)]
    byte_count = sum(len(line.encode()) for line in lines)
    with result_spool.LabelingResultSpool(
        max_result_bytes=byte_count - int(over_limit)
    ) as spool:
        spool.add_expected([_hash(0), _hash(1)])
        if over_limit:
            with pytest.raises(ValueError, match="size limit"):
                spool.stage(iter(lines), on_progress=lambda: None)
            with pytest.raises(ValueError, match="validated"):
                spool.get_many([_hash(0)])
        else:
            spool.stage(iter(lines), on_progress=lambda: None)
            assert len(spool.get_many([_hash(0), _hash(1)])) == 2


def test_result_limit_accepts_more_than_old_one_gib_cap() -> None:
    with result_spool.LabelingResultSpool(max_result_bytes=1024**3 + 1) as spool:
        spool.add_expected([_hash(0)])
        spool.stage(iter([_line(0)]), on_progress=lambda: None)
        assert spool.get_many([_hash(0)])[_hash(0)].context is not None


def test_scaled_large_output_is_compressed_and_still_byte_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(result_spool, "_MAX_RESULT_BYTES", 8 * 1024)
    line = _line(0, "x" * 4096)
    assert 1024 < len(line.encode()) < 8 * 1024
    with result_spool.LabelingResultSpool(max_result_bytes=8 * 1024) as spool:
        spool.add_expected([_hash(0)])
        spool.stage(iter([line]), on_progress=lambda: None)
        assert spool.get_many([_hash(0)])[_hash(0)].context == "x" * 4096
    with result_spool.LabelingResultSpool(max_result_bytes=8 * 1024) as spool:
        spool.add_expected([_hash(0), _hash(1)])
        with pytest.raises(ValueError, match="size limit"):
            spool.stage(iter([line, _line(1, "x" * 4096)]), on_progress=lambda: None)


def test_existing_provider_row_errors_are_preserved() -> None:
    line = json.loads(_line(0))
    line["response"]["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected([_hash(0)])
        spool.stage(iter([json.dumps(line)]), on_progress=lambda: None)
        assert (
            spool.get_many([_hash(0)])[_hash(0)].error
            is VertexBatchResultError.REMOTE_ERROR
        )


def test_result_reads_are_page_bounded() -> None:
    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected(_hash(index) for index in range(129))
        spool.stage(iter(()), on_progress=lambda: None)
        with pytest.raises(ValueError, match="page"):
            spool.get_many([_hash(index) for index in range(129)])


def test_large_result_stream_uses_private_disk_storage_with_bounded_python_memory(
    scratch_paths: list[Path],
) -> None:
    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected(_hash(index) for index in range(2000))
        tracemalloc.start()
        try:
            spool.stage(
                (_line(index, "x" * 8192) for index in range(2000)),
                on_progress=lambda: None,
            )
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert peak_bytes < 4 * 1024 * 1024
        directory = scratch_paths[0]
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        databases = list(directory.iterdir())
        assert len(databases) == 1
        assert databases[0].stat().st_size < 4 * 1024 * 1024
        assert stat.S_IMODE(databases[0].stat().st_mode) == 0o600
        with sqlite3.connect(databases[0]) as connection:
            stored_result = connection.execute(
                "SELECT result FROM expected WHERE request_hash = ?", (_hash(1999),)
            ).fetchone()[0]
        assert isinstance(stored_result, bytes)
        assert json.loads(zlib.decompress(stored_result))["context"] == "x" * 8192
        assert spool.get_many([_hash(1999)])[_hash(1999)].context == "x" * 8192

    assert not scratch_paths[0].exists()


def test_sqlite_file_limit_fails_closed_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, scratch_paths: list[Path]
) -> None:
    monkeypatch.setattr(result_spool, "_MAX_DATABASE_BYTES", 64 * 1024)
    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected([_hash(0)])
        with pytest.raises(ValueError, match="size limit"):
            spool.stage(
                iter([_line(0, random.Random(0).randbytes(64 * 1024).hex())]),
                on_progress=lambda: None,
            )
        with pytest.raises(ValueError, match="validated"):
            spool.get_many([_hash(0)])
        assert all(
            path.stat().st_size <= 64 * 1024 for path in scratch_paths[0].iterdir()
        )

    assert not scratch_paths[0].exists()


def test_heartbeat_failure_closes_source_and_removes_scratch_files(
    scratch_paths: list[Path],
) -> None:
    closed = False

    def lines() -> Iterator[str]:
        nonlocal closed
        try:
            yield _line(0)
        finally:
            closed = True

    def on_progress() -> None:
        raise RuntimeError("lease lost")

    with pytest.raises(RuntimeError, match="lease lost"):
        with result_spool.LabelingResultSpool() as spool:
            spool.add_expected([_hash(0)])
            spool.stage(lines(), on_progress=on_progress)

    assert closed
    assert all(not path.exists() for path in scratch_paths)


@pytest.mark.parametrize(
    "stored_bytes,error_match",
    [
        (b"not zlib", "compressed"),
        (zlib.compress(b"{}")[:-1], "compressed"),
        (zlib.compress(b"{}") + b"trailing", "compressed"),
        (zlib.compress(b"x" * (2 * 1024 * 1024 + 1)), "size limit"),
        (
            zlib.compress(
                json.dumps({"request_hash": _hash(1), "context": "different"}).encode()
            ),
            "hash",
        ),
    ],
    ids=["invalid", "truncated", "trailing", "oversized", "wrong_hash"],
)
def test_corrupted_or_oversized_compressed_rows_fail_closed(
    stored_bytes: bytes, error_match: str, scratch_paths: list[Path]
) -> None:
    with result_spool.LabelingResultSpool() as spool:
        spool.add_expected([_hash(0)])
        spool.stage(iter([_line(0)]), on_progress=lambda: None)
        database_path = next(scratch_paths[0].iterdir())
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                "UPDATE expected SET result = ? WHERE request_hash = ?",
                (stored_bytes, _hash(0)),
            )
        with pytest.raises(ValueError, match=error_match):
            spool.get_many([_hash(0)])
