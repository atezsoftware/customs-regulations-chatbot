"""Read exact cache acknowledgments once; durable consumers check their headers."""

import hashlib
import json
import re
from collections.abc import Mapping
from typing import cast

from botocore.exceptions import ClientError
from google.api_core.exceptions import GoogleAPICallError
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

from onyx.configs.constants import FileOrigin
from onyx.db.models import FileRecord
from onyx.db.regulatory_direct_ack import read_direct_ack
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.duplicate_ack_evidence import (
    MAX_ACK_BYTES,
    DuplicateAckEvidence,
    complete_ack_response,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_duplicate_ack_header(
    session: Session, evidence: DuplicateAckEvidence
) -> None:
    row = session.get(FileRecord, evidence.file_id, populate_existing=True)
    metadata: Mapping[str, object] | None = (
        cast(Mapping[str, object], row.file_metadata)
        if row is not None and isinstance(row.file_metadata, Mapping)
        else None
    )
    if (
        evidence.file_id != evidence.expected_file_id()
        or row is None
        or row.file_origin != FileOrigin.OTHER
        or row.file_type != "application/json"
        or not isinstance(metadata, Mapping)
        or metadata.get("sha256") != evidence.blob_sha256
        or metadata.get("binding_sha256") != evidence.binding_fingerprint
    ):
        raise ValueError("The retained duplicate-label ACK identity changed")


def load_duplicate_ack(
    session: Session,
    *,
    binding_fingerprint: str,
    submission_key: str,
    request: VertexBatchRequest,
    expected_line: str,
) -> DuplicateAckEvidence:
    if (
        re.fullmatch(r"[0-9a-f]{64}", binding_fingerprint) is None
        or re.fullmatch(r"regulatory-labeling-[0-9a-f]{64}", submission_key) is None
    ):
        raise ValueError("The duplicate-label cache identity is invalid")
    file_id = f"regulatory-labeling-direct/{binding_fingerprint}/{submission_key.removeprefix('regulatory-labeling-')}/{request.request_hash}.json"
    row = session.get(FileRecord, file_id, populate_existing=True)
    metadata: Mapping[str, object] | None = (
        cast(Mapping[str, object], row.file_metadata)
        if row is not None and isinstance(row.file_metadata, Mapping)
        else None
    )
    if (
        row is None
        or row.file_origin != FileOrigin.OTHER
        or row.file_type != "application/json"
        or not isinstance(metadata, Mapping)
        or metadata.get("binding_sha256") != binding_fingerprint
    ):
        raise ValueError("The original duplicate-label ACK is unavailable")
    try:
        raw = read_direct_ack(session, file_id, max_bytes=MAX_ACK_BYTES)
    except (OSError, ClientError, GoogleAPICallError, NoResultFound) as error:
        raise ValueError(
            "The original duplicate-label ACK bytes are unavailable"
        ) from error
    if not 0 < len(raw) <= MAX_ACK_BYTES or _sha(raw) != metadata.get("sha256"):
        raise ValueError("The original duplicate-label ACK bytes changed")
    stored = json.loads(raw)
    line = stored.get("line") if isinstance(stored, dict) else None
    if (
        not isinstance(stored, dict)
        or stored.get("binding_sha256") != binding_fingerprint
        or stored.get("submission_key") != submission_key
        or stored.get("request_hash") != request.request_hash
        or VertexBatchRequest.model_validate(stored.get("request_payload")) != request
        or not isinstance(line, str)
        or line != expected_line
        or _sha(line.encode()) != stored.get("line_sha256")
    ):
        raise ValueError("The actual duplicate-label ACK correlation changed")
    response = complete_ack_response(line, request)
    evidence = DuplicateAckEvidence(
        file_id=file_id,
        binding_fingerprint=binding_fingerprint,
        submission_key=submission_key,
        request_hash=request.request_hash,
        blob_sha256=_sha(raw),
        blob_byte_count=len(raw),
        line_sha256=_sha(line.encode()),
        line_byte_count=len(line.encode()),
        response_sha256=_sha(response.encode()),
    )
    validate_duplicate_ack_header(session, evidence)
    return evidence
