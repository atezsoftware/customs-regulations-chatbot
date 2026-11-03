"""Direct ACKs use bounded same-session PG reads even when source storage is S3."""

import json
from collections.abc import Generator
from hashlib import sha256
from io import BytesIO
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.configs import app_configs
from onyx.configs.constants import FileOrigin
from onyx.db import document_intake_metadata_contract_retry as contract
from onyx.db import regulatory_labeling_duplicate_evidence as duplicate
from onyx.db import regulatory_labeling_reviewed_fragment as reviewed
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    FileContent,
    FileRecord,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
)
from onyx.db.regulatory_direct_ack import read_direct_ack
from onyx.file_store import file_store
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


@pytest.fixture
def cache(
    intake_session: Session, monkeypatch: pytest.MonkeyPatch
) -> Generator[dict[str, Any], None, None]:
    monkeypatch.setattr(app_configs, "FILE_STORE_BACKEND", "s3")
    forbidden = Mock(
        side_effect=AssertionError("ACK must not access configured S3 source storage")
    )
    monkeypatch.setattr(file_store, "get_s3_file_store", forbidden)
    assert app_configs.FILE_STORE_BACKEND == "s3"
    binding = LabelingProviderBinding(
        model_configuration_id=1,
        provider_id=2,
        project="local-ack-storage",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="workload_identity",
        model="gemini-2.5-flash",
        transport="vertex_direct_v1",
    )
    request = VertexBatchRequest(
        prompt='{"instruction":"Exact retained request"}',
        system_instruction="Return the source-bound result",
        generation_config={"maxOutputTokens": 128},
    )
    submission = "regulatory-labeling-" + sha256(uuid4().bytes).hexdigest()
    file_id = f"regulatory-labeling-direct/{binding.fingerprint}/{submission.removeprefix('regulatory-labeling-')}/{request.request_hash}.json"
    response = '{"labels":[],"abstained":true,"reason":"insufficient_evidence"}'
    line = json.dumps(
        {
            "key": request.request_hash,
            "request": request.to_generate_content_request(),
            "response": {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": response}]}}
                ]
            },
            "status": "",
        }
    )
    raw = json.dumps(
        {
            "binding_sha256": binding.fingerprint,
            "submission_key": submission,
            "request_hash": request.request_hash,
            "request_payload": request.model_dump(
                mode="json", exclude={"request_hash"}
            ),
            "line_sha256": sha256(line.encode()).hexdigest(),
            "line": line,
        }
    ).encode()
    PostgresBackedFileStore().save_file(
        BytesIO(raw),
        "direct ACK",
        FileOrigin.OTHER,
        "application/json",
        file_metadata={
            "sha256": sha256(raw).hexdigest(),
            "binding_sha256": binding.fingerprint,
        },
        file_id=file_id,
        db_session=intake_session,
    )
    # The fixture's outer transaction is still uncommitted to other connections.
    with get_session_with_current_tenant() as independent:
        assert independent.get(FileRecord, file_id) is None
    yield_data = {
        "id": file_id,
        "raw": raw,
        "line": line,
        "request": request,
        "submission": submission,
        "binding": binding,
        "forbidden": forbidden,
    }
    yield yield_data
    forbidden.assert_not_called()


def load(session: Session, data: dict[str, Any], mode: str) -> bytes | str:
    if mode == "contract":
        raw = contract._read_ack(session, data["id"])
        contract._ack(
            raw,
            binding=data["binding"],
            cache_id=data["id"],
            expected_hash=data["request"].request_hash,
        )
        return raw
    line = data["line"]
    if mode == "reviewed":
        line = reviewed._line_from_cache(
            session,
            run=RegulatoryLabelingRun(
                provider_binding=data["binding"].model_dump(mode="json")
            ),
            source=RegulatoryLabelingItem(),
            request=data["request"],
            submission_key=data["submission"],
        )
    result = duplicate.load_duplicate_ack(
        session,
        binding_fingerprint=data["binding"].fingerprint,
        submission_key=data["submission"],
        request=data["request"],
        expected_line=line,
    )
    assert result.blob_sha256 == sha256(data["raw"]).hexdigest()
    return line


@pytest.mark.parametrize("mode", ["contract", "duplicate", "reviewed"])
def test_actual_uncommitted_pg_ack_never_uses_configured_s3(
    intake_session: Session, cache: dict[str, Any], mode: str
) -> None:
    assert load(intake_session, cache, mode) == (
        cache["raw"] if mode == "contract" else cache["line"]
    )


@pytest.mark.parametrize("mode", ["contract", "duplicate", "reviewed"])
@pytest.mark.parametrize(
    "change",
    [
        "missing_record",
        "missing_content",
        "wrong_pointer",
        "wrong_backend",
        "tampered_hash",
        "oversize",
    ],
)
def test_pg_ack_identity_bytes_and_size_remain_fail_closed(
    intake_session: Session, cache: dict[str, Any], mode: str, change: str
) -> None:
    record = intake_session.get_one(FileRecord, cache["id"])
    content = intake_session.get_one(FileContent, cache["id"])
    if change == "missing_record":
        intake_session.delete(record)
    elif change == "missing_content":
        intake_session.delete(content)
    elif change == "wrong_pointer":
        record.object_key = "0"
    elif change == "wrong_backend":
        record.bucket_name = "source-s3-bucket"
    elif change == "tampered_hash":
        record.file_metadata = {
            "sha256": "0" * 64,
            "binding_sha256": cache["binding"].fingerprint,
        }
    else:
        content.file_size = (
            8 * 1024 * 1024 if mode == "contract" else duplicate.MAX_ACK_BYTES
        ) + 1
    intake_session.flush()
    with pytest.raises((ValueError, contract.intake.IntakeConflictError)):
        load(intake_session, cache, mode)


def test_pg_lo_read_is_bounded_even_if_record_size_lies(
    intake_session: Session, cache: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    content = intake_session.get_one(FileContent, cache["id"])
    content.file_size = 1
    intake_session.flush()
    actual_scalar = intake_session.scalar
    fetched: list[bytes] = []

    def bounded_result(statement: Any, *args: Any, **kwargs: Any) -> Any:
        value = actual_scalar(statement, *args, **kwargs)
        fetched.append(bytes(value))
        return value

    monkeypatch.setattr(intake_session, "scalar", bounded_result)
    with pytest.raises(ValueError, match="byte count"):
        read_direct_ack(intake_session, cache["id"], max_bytes=64)
    assert len(fetched) == 1 and len(fetched[0]) == 65


def test_configured_raw_source_reader_is_still_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = Mock()
    monkeypatch.setattr(file_store, "get_s3_file_store", default)
    monkeypatch.setattr(app_configs, "FILE_STORE_BACKEND", "s3")
    assert reviewed.get_default_file_store() is default.return_value
    default.assert_called_once_with()
