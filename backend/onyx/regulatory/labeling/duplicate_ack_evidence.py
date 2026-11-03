"""Bounded immutable pointers to the actual direct-gateway acknowledgment bytes."""

import json

from pydantic import BaseModel, ConfigDict, Field

from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.provider import (
    MAX_RESPONSE_BYTES,
    parse_labeling_batch_output,
)

MAX_ACK_BYTES = 2 * MAX_RESPONSE_BYTES
SHA = r"^[0-9a-f]{64}$"


class DuplicateAckEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    file_id: str = Field(min_length=1, max_length=1024)
    binding_fingerprint: str = Field(pattern=SHA)
    submission_key: str = Field(pattern=r"^regulatory-labeling-[0-9a-f]{64}$")
    request_hash: str = Field(pattern=SHA)
    blob_sha256: str = Field(pattern=SHA)
    blob_byte_count: int = Field(gt=0, le=MAX_ACK_BYTES)
    line_sha256: str = Field(pattern=SHA)
    line_byte_count: int = Field(gt=0, le=MAX_RESPONSE_BYTES)
    response_sha256: str = Field(pattern=SHA)

    def expected_file_id(self) -> str:
        submission = self.submission_key.removeprefix("regulatory-labeling-")
        return f"regulatory-labeling-direct/{self.binding_fingerprint}/{submission}/{self.request_hash}.json"


def complete_ack_response(line: str, request: VertexBatchRequest) -> str:
    if not 0 < len(line.encode()) <= MAX_RESPONSE_BYTES:
        raise ValueError("Duplicate-label ACK row exceeds its byte budget")
    row = json.loads(line)
    response = row.get("response") if isinstance(row, dict) else None
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if (
        not isinstance(row, dict)
        or row.get("key") != request.request_hash
        or row.get("request") != request.to_generate_content_request()
        or not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], dict)
        or candidates[0].get("finishReason") != "STOP"
    ):
        raise ValueError("Duplicate-label ACK does not echo one complete request")
    parsed = parse_labeling_batch_output(iter([line]), {request.request_hash})[
        request.request_hash
    ]
    if parsed.error is not None or parsed.context is None:
        raise ValueError("Duplicate-label ACK provider response was rejected")
    return parsed.context
