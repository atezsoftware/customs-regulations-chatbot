"""Disk-backed, fully validated Batch results for bounded labeling application."""

from __future__ import annotations

import json
import re
import sqlite3
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Self, cast

from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchResult
from onyx.regulatory.labeling.provider import (
    MAX_RESPONSE_BYTES,
    parse_labeling_batch_output,
)

_MAX_DATABASE_BYTES = 2 * 1024**3
_MAX_RESULT_BYTES = 8 * 1024**3
_MAX_SERIALIZED_RESULT_BYTES = 2 * MAX_RESPONSE_BYTES
_DATABASE_PAGE_BYTES = 4096
_MAX_RESULT_PAGE = 128
_REQUEST_HASH = re.compile(r"[0-9a-f]{64}")


@contextmanager
def _database_errors() -> Iterator[None]:
    try:
        yield
    except sqlite3.Error as error:
        if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
            raise ValueError(
                "Labeling result spool exceeds its disk size limit"
            ) from None
        raise ValueError("Labeling result scratch database is unavailable") from None


def _validate_hash(value: object) -> str:
    if not isinstance(value, str) or _REQUEST_HASH.fullmatch(value) is None:
        raise ValueError("Labeling result has an invalid request hash")
    return value


def _close_iterator(iterator: Iterator[str]) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _decode_result(data: bytes, *, request_hash: str) -> VertexBatchResult:
    if not isinstance(data, bytes):
        raise ValueError("Labeling result contains an invalid compressed record")
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(data, _MAX_SERIALIZED_RESULT_BYTES + 1)
    except zlib.error:
        raise ValueError(
            "Labeling result contains an invalid compressed record"
        ) from None
    if len(decoded) > _MAX_SERIALIZED_RESULT_BYTES or decoder.unconsumed_tail:
        raise ValueError("Decompressed labeling result exceeds its size limit")
    if not decoder.eof or decoder.unused_data:
        raise ValueError("Labeling result contains an invalid compressed record")
    try:
        result = VertexBatchResult.model_validate_json(decoded)
    except ValueError:
        raise ValueError(
            "Labeling result contains an invalid compressed record"
        ) from None
    if result.request_hash != request_hash:
        raise ValueError("Labeling result contains a mismatched request hash")
    return result


class LabelingResultSpool:
    def __init__(self, *, max_result_bytes: int = _MAX_RESULT_BYTES) -> None:
        if not 1 <= max_result_bytes <= _MAX_RESULT_BYTES:
            raise ValueError("Labeling result byte limit must be between 1 and 8 GiB")
        self._max_result_bytes = max_result_bytes
        self._directory: TemporaryDirectory[str] | None = None
        self._connection: sqlite3.Connection | None = None
        self._entered = False
        self._stage_started = False
        self._validated = False
        self._failed = False

    def __enter__(self) -> Self:
        if self._entered:
            raise ValueError("Labeling result spool cannot be reopened")
        self._entered = True
        try:
            self._directory = TemporaryDirectory()
            database_path = Path(self._directory.name) / "results.sqlite3"
            with _database_errors():
                self._connection = sqlite3.connect(database_path)
                database_path.chmod(0o600)
                connection = self._connection
                connection.execute(f"PRAGMA page_size={_DATABASE_PAGE_BYTES}")
                connection.execute(
                    f"PRAGMA max_page_count={_MAX_DATABASE_BYTES // _DATABASE_PAGE_BYTES}"
                )
                connection.execute("PRAGMA cache_size=-4096")
                connection.execute("PRAGMA mmap_size=0")
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute(
                    "CREATE TABLE expected (request_hash TEXT PRIMARY KEY, result BLOB) "
                    "WITHOUT ROWID"
                )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        try:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
        finally:
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None

    def _active_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ValueError("Labeling result spool is not open")
        return self._connection

    def add_expected(self, hashes: Iterable[str]) -> None:
        connection = self._active_connection()
        if self._stage_started or self._failed:
            raise ValueError("Expected labeling requests cannot change after staging")
        try:
            with _database_errors():
                for request_hash in hashes:
                    key = _validate_hash(request_hash)
                    try:
                        connection.execute(
                            "INSERT INTO expected (request_hash) VALUES (?)", (key,)
                        )
                    except sqlite3.IntegrityError:
                        raise ValueError(
                            "Labeling expected requests contain a duplicate hash"
                        ) from None
                connection.commit()
        except BaseException:
            self._failed = True
            raise

    def stage(self, lines: Iterable[str], *, on_progress: Callable[[], None]) -> None:
        connection = self._active_connection()
        if self._stage_started or self._failed:
            raise ValueError("Labeling result spool cannot be staged again")
        self._stage_started = True
        iterator = iter(lines)
        used_bytes = 0
        try:
            with _database_errors():
                for line in iterator:
                    on_progress()
                    if not isinstance(line, str):
                        raise ValueError("Labeling output is not valid text")
                    if len(line) > MAX_RESPONSE_BYTES:
                        raise ValueError("Labeling output row exceeds its size limit")
                    line_bytes = len(line.encode())
                    if line_bytes > MAX_RESPONSE_BYTES:
                        raise ValueError("Labeling output row exceeds its size limit")
                    used_bytes += line_bytes
                    if used_bytes > self._max_result_bytes:
                        raise ValueError("Labeling output exceeds its total size limit")
                    if not line.strip():
                        continue
                    try:
                        value: object = json.loads(line)
                    except ValueError:
                        raise ValueError("Labeling output is not valid JSON") from None
                    if not isinstance(value, dict):
                        raise ValueError("Labeling output is not an object")
                    key = _validate_hash(cast(dict[str, object], value).get("key"))
                    existing = connection.execute(
                        "SELECT result FROM expected WHERE request_hash = ?", (key,)
                    ).fetchone()
                    if existing is None:
                        raise ValueError(
                            "Labeling output has an unexpected request hash"
                        )
                    if existing[0] is not None:
                        raise ValueError("Labeling output has a duplicate request hash")
                    result = parse_labeling_batch_output(iter([line]), {key})[key]
                    serialized = result.model_dump_json().encode()
                    if len(serialized) > _MAX_SERIALIZED_RESULT_BYTES:
                        raise ValueError(
                            "Serialized labeling result exceeds its size limit"
                        )
                    connection.execute(
                        "UPDATE expected SET result = ? WHERE request_hash = ?",
                        (zlib.compress(serialized), key),
                    )
                connection.commit()
            _close_iterator(iterator)
            on_progress()
        except BaseException:
            self._failed = True
            with suppress(Exception):
                connection.rollback()
            with suppress(Exception):
                _close_iterator(iterator)
            raise
        self._validated = True

    def get_many(self, hashes: Sequence[str]) -> dict[str, VertexBatchResult]:
        connection = self._active_connection()
        if not self._validated:
            raise ValueError("Labeling output has not been fully validated")
        if len(hashes) > _MAX_RESULT_PAGE:
            raise ValueError("Labeling result page exceeds 128 requests")
        keys = list(dict.fromkeys(_validate_hash(value) for value in hashes))
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        with _database_errors():
            # Only generated placeholders enter SQL; every request hash is bound.
            rows = connection.execute(
                f"SELECT request_hash, result FROM expected WHERE request_hash IN ({placeholders})",  # noqa: S608
                keys,
            ).fetchall()
        if len(rows) != len(keys):
            raise ValueError("Labeling result page contains an unexpected request hash")
        return {
            request_hash: _decode_result(result, request_hash=request_hash)
            for request_hash, result in rows
            if result is not None
        }
