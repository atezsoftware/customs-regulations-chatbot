from __future__ import annotations

import json
from collections.abc import Collection, Iterator, Sequence
from enum import StrEnum
from hashlib import sha256
from typing import NoReturn, Protocol, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

_SAFETY_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
        "MODEL_ARMOR",
    }
)


class VertexBatchContractError(ValueError):
    """A secret-safe violation of the Vertex input or output contract."""


def _raise_secret_safe_public_error(error: Exception) -> NoReturn:
    try:
        raise error from None
    except Exception:
        error.__context__ = None
        raise


class VertexBatchSubmissionConflictError(VertexBatchContractError):
    submission_key: str
    match_count: int

    def __init__(self, submission_key: str, match_count: int) -> None:
        self.submission_key = submission_key
        self.match_count = match_count
        super().__init__(
            f"Vertex submission {submission_key} matched {match_count} remote jobs"
        )


class VertexBatchResultError(StrEnum):
    EMPTY = "empty_response"
    SAFETY = "safety_blocked"
    REMOTE_ERROR = "remote_error"
    MALFORMED = "malformed_response"


class VertexBatchJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


def _request_contents(prompt: str) -> dict[str, object]:
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}


def _request_payload(prompt: str) -> dict[str, object]:
    return {
        **_request_contents(prompt),
        "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
    }


def _canonical_request_hash(prompt: str) -> str:
    canonical_contents = json.dumps(
        _request_contents(prompt),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical_contents.encode()).hexdigest()


class VertexBatchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str = Field(min_length=1)

    @computed_field
    @property
    def request_hash(self) -> str:
        return _canonical_request_hash(self.prompt)


class VertexBatchResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    context: str | None = None
    error: VertexBatchResultError | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "VertexBatchResult":
        if (self.context is None) == (self.error is None):
            raise ValueError("a result must contain exactly one outcome")
        return self


class VertexBatchState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    remote_job_name: str = Field(min_length=1)
    status: VertexBatchJobStatus
    input_uri: str | None = None
    output_uri: str | None = None
    error_code: int | None = None


class VertexReadOnlyAccessProbe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    credential_identity: str = Field(min_length=1)


class VertexBatchGateway(Protocol):
    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState: ...

    def get(self, remote_job_name: str) -> VertexBatchState: ...

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None: ...

    def read_results(self, output_uri: str) -> Iterator[str]: ...

    def cancel(self, remote_job_name: str) -> None: ...

    def delete(self, remote_job_name: str) -> None: ...

    def cleanup(self, prefix: str) -> None: ...


def vertex_batch_submission_key(
    requests: Sequence[VertexBatchRequest],
    *,
    tenant_id: str | None = None,
    job_id: UUID | None = None,
    output_prefix: str | None = None,
    submission_attempt: int | None = None,
) -> str:
    if not requests:
        raise VertexBatchContractError("Vertex batch requires at least one request")
    request_hashes = [request.request_hash for request in requests]
    if len(set(request_hashes)) != len(request_hashes):
        raise VertexBatchContractError("Vertex batch contains a duplicate request hash")
    identity_fields = (tenant_id, job_id, output_prefix, submission_attempt)
    if all(field is None for field in identity_fields):
        request_set_hash = sha256(
            "\n".join(sorted(request_hashes)).encode()
        ).hexdigest()
        return f"regulatory-context-{request_set_hash}"
    if (
        not isinstance(tenant_id, str)
        or not tenant_id.strip()
        or job_id is None
        or not isinstance(output_prefix, str)
        or not output_prefix.strip()
        or submission_attempt is None
        or submission_attempt < 1
    ):
        raise VertexBatchContractError(
            "Vertex submission identity requires tenant, job, prefix, and attempt"
        )
    identity = json.dumps(
        {
            "job_id": str(job_id),
            "output_prefix": output_prefix,
            "request_hashes": sorted(request_hashes),
            "submission_attempt": submission_attempt,
            "tenant_id": tenant_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"regulatory-context-{sha256(identity.encode()).hexdigest()}"


def _vertex_jsonl_line(request: VertexBatchRequest) -> str:
    return (
        json.dumps(
            {
                "key": request.request_hash,
                "request": _request_payload(request.prompt),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def vertex_jsonl_line_size(request: VertexBatchRequest) -> int:
    return len(_vertex_jsonl_line(request).encode("utf-8"))


def build_vertex_jsonl(
    requests: Sequence[VertexBatchRequest], *, max_bytes: int | None = None
) -> str:
    if not requests:
        raise VertexBatchContractError("Vertex batch requires at least one request")
    request_hashes = [request.request_hash for request in requests]
    if len(set(request_hashes)) != len(request_hashes):
        raise VertexBatchContractError("Vertex batch contains a duplicate request hash")
    if max_bytes is not None and max_bytes < 1:
        raise VertexBatchContractError("Vertex JSONL byte limit must be positive")
    parts: list[str] = []
    used_bytes = 0
    for request in requests:
        line = _vertex_jsonl_line(request)
        line_bytes = len(line.encode("utf-8"))
        if max_bytes is not None and used_bytes + line_bytes > max_bytes:
            if not parts:
                raise VertexBatchContractError(
                    "Vertex request exceeds the configured JSONL byte limit"
                )
            break
        parts.append(line)
        used_bytes += line_bytes
    return "".join(parts)


def _output_request_hash(value: object) -> str:
    if not isinstance(value, dict):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    typed_value = cast(dict[str, object], value)
    key = typed_value.get("key")
    if (
        isinstance(key, str)
        and len(key) == 64
        and all(character in "0123456789abcdef" for character in key)
    ):
        return key
    request = typed_value.get("request")
    if not isinstance(request, dict):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    contents = cast(dict[str, object], request).get("contents")
    if not isinstance(contents, list) or len(contents) != 1:
        raise VertexBatchContractError("Vertex output has no correlatable request")
    content = contents[0]
    if not isinstance(content, dict):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    typed_content = cast(dict[str, object], content)
    if typed_content.get("role") != "user":
        raise VertexBatchContractError("Vertex output has no correlatable request")
    parts = typed_content.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        raise VertexBatchContractError("Vertex output has no correlatable request")
    part = parts[0]
    if not isinstance(part, dict):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    typed_part = cast(dict[str, object], part)
    if not isinstance(typed_part.get("text"), str):
        raise VertexBatchContractError("Vertex output has no correlatable request")
    return _canonical_request_hash(cast(str, typed_part["text"]))


def _failure(request_hash: str, error: VertexBatchResultError) -> VertexBatchResult:
    return VertexBatchResult(request_hash=request_hash, error=error)


def _parse_output_result(
    value: dict[str, object], request_hash: str
) -> VertexBatchResult:
    if value.get("error") is not None:
        return _failure(request_hash, VertexBatchResultError.REMOTE_ERROR)
    status = value.get("status", "")
    if not isinstance(status, str):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    if status:
        return _failure(request_hash, VertexBatchResultError.REMOTE_ERROR)

    response = value.get("response")
    if not isinstance(response, dict):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    typed_response = cast(dict[str, object], response)
    prompt_feedback = typed_response.get("promptFeedback")
    if isinstance(prompt_feedback, dict):
        block_reason = cast(dict[str, object], prompt_feedback).get("blockReason")
        if isinstance(block_reason, str) and block_reason not in {
            "",
            "BLOCK_REASON_UNSPECIFIED",
        }:
            return _failure(request_hash, VertexBatchResultError.SAFETY)

    candidates = typed_response.get("candidates")
    if not isinstance(candidates, list):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    if not candidates:
        return _failure(request_hash, VertexBatchResultError.EMPTY)
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    typed_candidate = cast(dict[str, object], candidate)
    finish_reason = typed_candidate.get("finishReason")
    if isinstance(finish_reason, str) and finish_reason in _SAFETY_FINISH_REASONS:
        return _failure(request_hash, VertexBatchResultError.SAFETY)
    content = typed_candidate.get("content")
    if not isinstance(content, dict):
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    parts = cast(dict[str, object], content).get("parts")
    if not isinstance(parts, list) or not parts:
        return _failure(request_hash, VertexBatchResultError.MALFORMED)
    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            return _failure(request_hash, VertexBatchResultError.MALFORMED)
        typed_part = cast(dict[str, object], part)
        if not isinstance(typed_part.get("text"), str):
            return _failure(request_hash, VertexBatchResultError.MALFORMED)
        text_parts.append(cast(str, typed_part["text"]))
    context = "".join(text_parts).strip()
    if not context:
        return _failure(request_hash, VertexBatchResultError.EMPTY)
    return VertexBatchResult(request_hash=request_hash, context=context)


def parse_vertex_jsonl_output(
    output: str | Iterator[str],
    expected_request_hashes: Collection[str],
    *,
    require_complete: bool = True,
) -> dict[str, VertexBatchResult]:
    expected = set(expected_request_hashes)
    results: dict[str, VertexBatchResult] = {}
    lines = output.splitlines() if isinstance(output, str) else output
    for line in lines:
        if not line.strip():
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError:
            _raise_secret_safe_public_error(
                VertexBatchContractError("Vertex output has no correlatable request")
            )
        if not isinstance(value, dict):
            raise VertexBatchContractError("Vertex output has no correlatable request")
        typed_value = cast(dict[str, object], value)
        request_hash = _output_request_hash(typed_value)
        if request_hash not in expected:
            raise VertexBatchContractError(
                "Vertex output has an unexpected request hash"
            )
        if request_hash in results:
            raise VertexBatchContractError("Vertex output has a duplicate request hash")
        results[request_hash] = _parse_output_result(typed_value, request_hash)
    if require_complete and results.keys() != expected:
        raise VertexBatchContractError("Vertex output is missing a request hash")
    return results
