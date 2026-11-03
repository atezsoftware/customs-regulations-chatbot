"""Actual ACK bytes, exact echo, default-off spool retention and fixed caps."""

import json
import zlib
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import regulatory_labeling_duplicate_evidence as evidence_db
from onyx.db.regulatory_labeling_results import LabelingResultSpool
from onyx.file_store.file_store import FileStore
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.duplicate_ack_evidence import (
    MAX_ACK_BYTES,
    DuplicateAckEvidence,
)
from onyx.regulatory.labeling.provider import MAX_RESPONSE_BYTES


@pytest.fixture
def frozen() -> dict[str, Any]:
    return json.loads(
        (
            Path(__file__).parent / "fixtures/legacy_duplicate_label_context.json"
        ).read_text()
    )


def setup_store(
    monkeypatch: pytest.MonkeyPatch, frozen: dict[str, Any], *, raw: bytes | None = None
) -> tuple[Mock, Mock, SimpleNamespace, bytes]:
    raw = frozen["actual_base_cache_blob"].encode() if raw is None else raw
    record = SimpleNamespace(
        file_origin=FileOrigin.OTHER,
        file_type="application/json",
        file_metadata={
            "sha256": sha256(raw).hexdigest(),
            "binding_sha256": frozen["actual_binding_fingerprint"],
        },
    )
    session, store = Mock(spec=Session), Mock(spec=FileStore)
    session.get.return_value = record
    store.read_file.side_effect = lambda _file_id: BytesIO(raw)
    monkeypatch.setattr(
        evidence_db,
        "read_direct_ack",
        lambda _session, file_id, *, max_bytes: store.read_file(file_id).read(
            max_bytes + 1
        ),
    )
    return session, store, record, raw


def load(
    session: Mock, frozen: dict[str, Any], *, line: str | None = None
) -> DuplicateAckEvidence:
    return evidence_db.load_duplicate_ack(
        cast(Session, session),
        binding_fingerprint=frozen["actual_binding_fingerprint"],
        submission_key=frozen["actual_base_submission_key"],
        request=VertexBatchRequest.model_validate(frozen["base_request"]),
        expected_line=json.loads(frozen["actual_base_cache_blob"])["line"]
        if line is None
        else line,
    )


def test_actual_blob_and_line_bytes_are_hash_bound_and_header_checks_read_no_blob(
    frozen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    session, store, record, raw = setup_store(monkeypatch, frozen)
    proof = load(session, frozen)
    stored = json.loads(raw)
    assert proof.blob_sha256 == frozen["actual_base_cache_blob_sha256"]
    assert proof.blob_byte_count == len(raw)
    assert proof.line_sha256 == sha256(stored["line"].encode()).hexdigest()
    assert proof.line_byte_count == len(stored["line"].encode())
    assert "line" not in proof.model_dump()
    evidence_db.validate_duplicate_ack_header(cast(Session, session), proof)
    store.read_file.assert_called_once_with(proof.file_id)
    record.file_metadata = {**record.file_metadata, "sha256": "f" * 64}
    with pytest.raises(ValueError):
        evidence_db.validate_duplicate_ack_header(cast(Session, session), proof)
    store.read_file.assert_called_once()


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "binding",
        "origin",
        "mime",
        "absent",
        "line-bytes",
        "request",
        "echo",
        "key",
        "stop",
        "candidate-count",
        "size",
    ],
)
def test_cache_identity_echo_and_bounded_bytes_cannot_be_substituted(
    frozen: dict[str, Any], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    data = json.loads(frozen["actual_base_cache_blob"])
    line = data["line"]
    if change == "request":
        data["request_payload"]["prompt"] += " changed"
    elif change in {"echo", "key", "stop", "candidate-count"}:
        row = json.loads(line)
        if change == "echo":
            row["request"]["contents"][0]["parts"][0]["text"] += " changed"
        elif change == "key":
            row["key"] = "f" * 64
        elif change == "stop":
            row["response"]["candidates"][0]["finishReason"] = "MAX_TOKENS"
        else:
            row["response"]["candidates"].append(row["response"]["candidates"][0])
        line = json.dumps(row)
        data["line"] = line
        data["line_sha256"] = sha256(line.encode()).hexdigest()
    raw = b" " * (MAX_ACK_BYTES + 1) if change == "size" else json.dumps(data).encode()
    session, store, record, _ = setup_store(monkeypatch, frozen, raw=raw)
    if change == "hash":
        record.file_metadata["sha256"] = "f" * 64
    elif change == "binding":
        record.file_metadata["binding_sha256"] = "f" * 64
    elif change == "origin":
        record.file_origin = FileOrigin.CONNECTOR
    elif change == "mime":
        record.file_type = "text/plain"
    elif change == "absent":
        session.get.return_value = None
    elif change == "line-bytes":
        line += "\n"
    with pytest.raises(ValueError):
        load(session, frozen, line=line)
    if change in {"binding", "origin", "mime", "absent"}:
        store.read_file.assert_not_called()


@pytest.mark.parametrize("retained", [False, True])
def test_spool_retains_the_actual_parsed_row_only_when_explicitly_enabled(
    frozen: dict[str, Any], retained: bool
) -> None:
    line = json.loads(frozen["actual_base_cache_blob"])["line"]
    request = VertexBatchRequest.model_validate(frozen["base_request"])
    with LabelingResultSpool(retain_raw_rows=retained) as spool:
        spool.add_expected([request.request_hash])
        spool.stage([line], on_progress=lambda: None)
        assert (
            spool.get_many([request.request_hash])[request.request_hash].context
            == frozen["base_response"]
        )
        assert spool.get_raw_line(request.request_hash) == (line if retained else None)
        if retained:
            raw_line = spool.get_raw_line(request.request_hash)
            assert raw_line is not None
            assert (
                sha256(raw_line.encode()).hexdigest()
                == json.loads(frozen["actual_base_cache_blob"])["line_sha256"]
            )


def test_default_off_and_existing_total_cap_remain_unchanged(
    frozen: dict[str, Any],
) -> None:
    line = json.loads(frozen["actual_base_cache_blob"])["line"]
    request = VertexBatchRequest.model_validate(frozen["base_request"])
    with LabelingResultSpool() as spool:
        spool.add_expected([request.request_hash])
        spool.stage([line], on_progress=lambda: None)
        assert spool.get_raw_line(request.request_hash) is None
    with LabelingResultSpool(
        retain_raw_rows=True, max_result_bytes=len(line.encode()) - 1
    ) as spool:
        spool.add_expected([request.request_hash])
        with pytest.raises(ValueError):
            spool.stage([line], on_progress=lambda: None)
        with pytest.raises(ValueError):
            spool.get_raw_line(request.request_hash)


@pytest.mark.parametrize(
    "raw",
    [
        b"bad-zlib",
        zlib.compress(b"x" * (MAX_RESPONSE_BYTES + 1)),
        zlib.compress(b"x") + b"trailing",
    ],
)
def test_raw_spool_getter_preserves_decompression_and_frame_caps(
    frozen: dict[str, Any], raw: bytes
) -> None:
    line = json.loads(frozen["actual_base_cache_blob"])["line"]
    request = VertexBatchRequest.model_validate(frozen["base_request"])
    with LabelingResultSpool(retain_raw_rows=True) as spool:
        spool.add_expected([request.request_hash])
        spool.stage([line], on_progress=lambda: None)
        spool._active_connection().execute(
            "UPDATE expected SET raw_line=? WHERE request_hash=?",
            (raw, request.request_hash),
        )
        with pytest.raises(ValueError):
            spool.get_raw_line(request.request_hash)
