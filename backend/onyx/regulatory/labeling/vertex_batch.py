from __future__ import annotations

import json
import math
import re
import tempfile
from collections.abc import Callable, Iterator, Sequence
from hashlib import sha256
from typing import cast

from google.cloud import storage

from onyx.regulatory.indexing_jobs.legacy_vertex_batch import (
    GoogleVertexBatchGateway,
    _batch_state,
    _parse_gcs_uri,
    _translate_create_errors,
    _translate_gateway_errors,
)
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayIndeterminateSubmissionError,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchRequest,
    VertexBatchState,
    VertexBatchSubmissionConflictError,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call

_SUBMISSION_KEY = re.compile(r"regulatory-labeling-([0-9a-f]{64})")
_HEX_HASH = re.compile(r"[0-9a-f]{64}")
_MAX_RESULT_BLOBS = 1024
_BLOB_PAGE_SIZE = 100
_DELETE_BATCH_SIZE = 100
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_MANIFEST_REQUESTS = 1024
_MAX_RESULT_LINE_BYTES = 1024 * 1024


def _jsonl_line(request: VertexBatchRequest) -> bytes:
    return (
        json.dumps(
            {"request": request.to_generate_content_request()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _correlation_hash(request: object) -> str:
    if not isinstance(request, dict):
        raise VertexBatchContractError(
            "Vertex labeling output has no correlatable request"
        )
    typed_request = cast(dict[str, object], request)
    if not set(typed_request).issubset(
        {"contents", "generationConfig", "systemInstruction", "model"}
    ):
        raise VertexBatchContractError(
            "Vertex labeling output has an unknown request field"
        )
    contents = typed_request.get("contents")
    if not isinstance(contents, list) or len(contents) != 1:
        raise VertexBatchContractError(
            "Vertex labeling output has no correlatable request"
        )
    content = contents[0]
    if not isinstance(content, dict):
        raise VertexBatchContractError(
            "Vertex labeling output has no correlatable request"
        )
    typed_content = cast(dict[str, object], content)
    if set(typed_content) != {"role", "parts"} or typed_content.get("role") != "user":
        raise VertexBatchContractError(
            "Vertex labeling output has no correlatable request"
        )
    parts = typed_content.get("parts")
    if not isinstance(parts, list) or len(parts) != 1:
        raise VertexBatchContractError(
            "Vertex labeling output has no correlatable request"
        )
    part = parts[0]
    if not isinstance(part, dict):
        raise VertexBatchContractError(
            "Vertex labeling output has no correlatable request"
        )
    typed_part = cast(dict[str, object], part)
    if (
        not set(typed_part).issubset({"text", "fileData"})
        or typed_part.get("fileData") is not None
    ):
        raise VertexBatchContractError(
            "Vertex labeling output has no correlatable request"
        )
    text = typed_part.get("text")
    if not isinstance(text, str):
        raise VertexBatchContractError(
            "Vertex labeling output has no correlatable request"
        )
    canonical_contents = {"contents": [{"role": "user", "parts": [{"text": text}]}]}
    return sha256(
        json.dumps(
            canonical_contents,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _manifest_bytes(
    submission_key: str, requests: Sequence[VertexBatchRequest]
) -> bytes:
    if len(requests) > _MAX_MANIFEST_REQUESTS:
        raise VertexBatchContractError(
            "Vertex labeling batch contains too many requests"
        )
    correlations: dict[str, str] = {}
    for request in requests:
        correlation_hash = _correlation_hash(request.to_generate_content_request())
        if correlation_hash in correlations:
            raise VertexBatchContractError(
                "Vertex labeling batch has ambiguous request correlation"
            )
        correlations[correlation_hash] = request.request_hash
    payload = {
        "version": 1,
        "submission_key": submission_key,
        "requests": [
            {
                "correlation_hash": correlation_hash,
                "request_hash": request_hash,
            }
            for correlation_hash, request_hash in sorted(correlations.items())
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise VertexBatchContractError(
            "Vertex labeling correlation manifest exceeds its size limit"
        )
    return encoded


class LabelingVertexBatchGateway(GoogleVertexBatchGateway):
    """Native Vertex Batch transport isolated to labeling-owned GCS objects."""

    def __init__(
        self,
        *,
        config: VertexBatchConfig,
        staging_uri: str,
        credential_json_provider: Callable[[], str | None],
        request_timeout_seconds: float = 20,
        max_result_bytes: int = 64 * 1024 * 1024,
        max_reconciliation_seconds: float = 180,
    ) -> None:
        bucket, prefix = _parse_gcs_uri(staging_uri)
        if any(component in {".", ".."} for component in prefix.split("/")):
            raise VertexBatchContractError("Vertex labeling staging URI is invalid")
        if max_result_bytes < 1:
            raise VertexBatchContractError(
                "Vertex labeling result byte limit must be positive"
            )
        if (
            not math.isfinite(max_reconciliation_seconds)
            or max_reconciliation_seconds <= 0
        ):
            raise VertexBatchContractError(
                "Vertex reconciliation timeout must be positive and finite"
            )
        normalized_staging_uri = f"gs://{bucket}/{prefix}"
        runtime_config = config.model_copy(update={"gcs_uri": normalized_staging_uri})
        super().__init__(
            config=runtime_config,
            object_prefix="labeling",
            credential_json_provider=credential_json_provider,
            request_timeout_seconds=request_timeout_seconds,
        )
        self._max_result_bytes = max_result_bytes
        self._max_reconciliation_seconds = max_reconciliation_seconds
        self._labeling_root = f"{prefix}/labeling"

    def _submission_root(self, uri: str) -> tuple[str, str]:
        bucket_name, object_name = _parse_gcs_uri(uri)
        base_bucket, _ = _parse_gcs_uri(self._gcs_uri)
        match = re.fullmatch(
            rf"({re.escape(self._labeling_root)}/[0-9a-f]{{64}})(?:/.*)?",
            object_name,
        )
        if bucket_name != base_bucket or match is None:
            raise VertexBatchContractError(
                "Vertex labeling storage URI is outside its owned prefix"
            )
        return bucket_name, match.group(1)

    def _validate_remote_job_name(self, remote_job_name: str) -> None:
        match = re.fullmatch(
            rf"projects/([^/\s]+)/locations/{re.escape(self._config.location)}/"
            r"batchPredictionJobs/[^/\s]+",
            remote_job_name,
        )
        if match is None or (
            match.group(1) != self._config.project and not match.group(1).isdecimal()
        ):
            raise VertexBatchContractError(
                "Vertex labeling job is outside its configured project or location"
            )

    def _validate_owned_output_uri(
        self, output_uri: str, *, expected_output_uri: str | None = None
    ) -> None:
        _, submission_root = self._submission_root(output_uri)
        _, output_prefix = _parse_gcs_uri(output_uri)
        owned_output_root = f"{submission_root}/output"
        if output_prefix != owned_output_root and not output_prefix.startswith(
            f"{owned_output_root}/"
        ):
            raise VertexBatchContractError(
                "Vertex labeling output URI is outside its owned prefix"
            )
        if expected_output_uri is None:
            return
        expected_bucket, expected_prefix = _parse_gcs_uri(expected_output_uri)
        actual_bucket, _ = _parse_gcs_uri(output_uri)
        if actual_bucket != expected_bucket or (
            output_prefix != expected_prefix
            and not output_prefix.startswith(f"{expected_prefix}/")
        ):
            raise VertexBatchContractError(
                "Vertex labeling job returned an unexpected output URI"
            )

    def _load_manifest(
        self,
        storage_client: storage.Client,
        bucket_name: str,
        submission_root: str,
    ) -> dict[str, str]:
        blob = storage_client.bucket(bucket_name).blob(
            f"{submission_root}/manifest.json"
        )
        raw = blob.download_as_bytes(
            start=0,
            end=_MAX_MANIFEST_BYTES,
            timeout=self._request_timeout_seconds,
        )
        if len(raw) > _MAX_MANIFEST_BYTES:
            raise VertexBatchContractError(
                "Vertex labeling correlation manifest exceeds its size limit"
            )
        try:
            value: object = json.loads(raw)
        except ValueError:
            raise VertexBatchContractError(
                "Vertex labeling correlation manifest is invalid"
            ) from None
        if not isinstance(value, dict):
            raise VertexBatchContractError(
                "Vertex labeling correlation manifest is invalid"
            )
        manifest = cast(dict[str, object], value)
        expected_submission_key = (
            f"regulatory-labeling-{submission_root.rsplit('/', 1)[-1]}"
        )
        requests = manifest.get("requests")
        if (
            set(manifest) != {"version", "submission_key", "requests"}
            or manifest.get("version") != 1
            or manifest.get("submission_key") != expected_submission_key
            or not isinstance(requests, list)
            or not requests
            or len(requests) > _MAX_MANIFEST_REQUESTS
        ):
            raise VertexBatchContractError(
                "Vertex labeling correlation manifest is invalid"
            )
        correlations: dict[str, str] = {}
        request_hashes: set[str] = set()
        for entry in requests:
            if not isinstance(entry, dict):
                raise VertexBatchContractError(
                    "Vertex labeling correlation manifest is invalid"
                )
            typed_entry = cast(dict[str, object], entry)
            correlation_hash = typed_entry.get("correlation_hash")
            request_hash = typed_entry.get("request_hash")
            if (
                set(typed_entry) != {"correlation_hash", "request_hash"}
                or not isinstance(correlation_hash, str)
                or _HEX_HASH.fullmatch(correlation_hash) is None
                or not isinstance(request_hash, str)
                or _HEX_HASH.fullmatch(request_hash) is None
                or correlation_hash in correlations
                or request_hash in request_hashes
            ):
                raise VertexBatchContractError(
                    "Vertex labeling correlation manifest is invalid"
                )
            correlations[correlation_hash] = request_hash
            request_hashes.add(request_hash)
        return correlations

    def _normalize_output_line(self, line: str, correlations: dict[str, str]) -> str:
        try:
            raw: object = json.loads(line)
        except ValueError:
            raise VertexBatchContractError(
                "Vertex labeling output is not valid JSON"
            ) from None
        if not isinstance(raw, dict):
            raise VertexBatchContractError("Vertex labeling output is not an object")
        value = cast(dict[str, object], raw)
        request = value.get("request")
        if not isinstance(request, dict):
            raise VertexBatchContractError(
                "Vertex labeling output has no correlatable request"
            )
        echoed = dict(cast(dict[str, object], request))
        echoed_model = echoed.pop("model", None)
        if echoed_model is not None:
            if (
                not isinstance(echoed_model, str)
                or echoed_model.rsplit("/", 1)[-1]
                != self._config.model_name.rsplit("/", 1)[-1]
            ):
                raise VertexBatchContractError(
                    "Vertex labeling output references a different model"
                )
        correlation_hash = _correlation_hash(echoed)
        request_hash = correlations.get(correlation_hash)
        if request_hash is None:
            raise VertexBatchContractError(
                "Vertex output has an unexpected request hash"
            )
        normalized: dict[str, object] = {"key": request_hash}
        for field in ("response", "error", "status"):
            if field in value:
                normalized[field] = value[field]
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        from google.genai import types as genai_types

        match = _SUBMISSION_KEY.fullmatch(submission_key)
        if match is None:
            raise VertexBatchContractError("Vertex labeling submission key is invalid")
        if max_jsonl_bytes < 1 or not requests:
            raise VertexBatchContractError("Vertex labeling batch input is invalid")
        if len({request.request_hash for request in requests}) != len(requests):
            raise VertexBatchContractError(
                "Vertex batch contains a duplicate request hash"
            )
        manifest = _manifest_bytes(submission_key, requests)
        used_bytes = 0
        for request in requests:
            used_bytes += len(_jsonl_line(request))
            if used_bytes > max_jsonl_bytes:
                raise VertexBatchContractError(
                    "Vertex batch exceeds the configured JSONL byte limit"
                )

        submission_hash = match.group(1)
        batch_prefix = f"{self._gcs_uri.rstrip('/')}/labeling/{submission_hash}"
        input_uri = f"{batch_prefix}/input.jsonl"
        output_uri = f"{batch_prefix}/output"
        bucket_name, input_name = _parse_gcs_uri(input_uri)
        manifest_name = f"{input_name.rsplit('/', 1)[0]}/manifest.json"
        with _translate_gateway_errors():
            credentials = self._credentials()
            with self._managed_storage_client(credentials) as storage_client:
                with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as payload:
                    for request in requests:
                        payload.write(_jsonl_line(request))
                    storage_client.bucket(bucket_name).blob(
                        input_name
                    ).upload_from_file(
                        payload,
                        content_type="application/jsonl",
                        rewind=True,
                        timeout=self._request_timeout_seconds,
                    )
                storage_client.bucket(bucket_name).blob(
                    manifest_name
                ).upload_from_string(
                    manifest,
                    content_type="application/json",
                    timeout=self._request_timeout_seconds,
                )
            with self._managed_genai_client(credentials) as client:
                with _translate_create_errors(submission_key):
                    with traced_llm_call(
                        flow=LLMFlow.REGULATORY_LABELING_BATCH,
                        model=self._config.model_name,
                        provider="vertex_ai",
                        extra_config={"request_count": str(len(requests))},
                    ):
                        job = client.batches.create(
                            model=self._config.model_name,
                            src=input_uri,
                            config=genai_types.CreateBatchJobConfig(
                                display_name=submission_key,
                                dest=output_uri,
                            ),
                        )
        try:
            state = _batch_state(
                job, input_uri=input_uri, fallback_output_uri=output_uri
            )
            self._validate_remote_job_name(state.remote_job_name)
            if state.output_uri is None:
                raise VertexBatchContractError(
                    "Vertex labeling job returned no output URI"
                )
            self._validate_owned_output_uri(
                state.output_uri, expected_output_uri=output_uri
            )
        except VertexBatchContractError:
            raise IndexingGatewayIndeterminateSubmissionError(submission_key) from None
        return state

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        from google.genai import types as genai_types

        submission_match = _SUBMISSION_KEY.fullmatch(submission_key)
        if submission_match is None:
            raise VertexBatchContractError("Vertex labeling submission key is invalid")
        timeout_milliseconds = math.ceil(
            min(self._request_timeout_seconds, self._max_reconciliation_seconds) * 1000
        )
        with _translate_gateway_errors():
            credentials = self._credentials()
            with self._managed_genai_client(credentials) as client:
                pager = client.batches.list(
                    config=genai_types.ListBatchJobsConfig(
                        page_size=2,
                        filter=f'display_name="{submission_key}"',
                        http_options=genai_types.HttpOptions(
                            timeout=timeout_milliseconds
                        ),
                    )
                )
                matches = pager.page
                next_page_token = pager.config.get("page_token")
        if next_page_token:
            raise VertexBatchSubmissionConflictError(
                submission_key, max(2, len(matches) + 1)
            )
        if not matches:
            return None
        if len(matches) != 1:
            raise VertexBatchSubmissionConflictError(submission_key, len(matches))
        job = matches[0]
        if job.display_name != submission_key:
            raise VertexBatchContractError(
                "Vertex reconciliation returned a different submission"
            )
        submission_hash = submission_match.group(1)
        batch_prefix = f"{self._gcs_uri.rstrip('/')}/labeling/{submission_hash}"
        input_uri = f"{batch_prefix}/input.jsonl"
        output_uri = f"{batch_prefix}/output"
        state = _batch_state(job, input_uri=input_uri, fallback_output_uri=output_uri)
        self._validate_remote_job_name(state.remote_job_name)
        if state.output_uri is None:
            raise VertexBatchContractError("Vertex labeling job returned no output URI")
        self._validate_owned_output_uri(
            state.output_uri, expected_output_uri=output_uri
        )
        return state

    def get(self, remote_job_name: str) -> VertexBatchState:
        self._validate_remote_job_name(remote_job_name)
        state = super().get(remote_job_name)
        self._validate_remote_job_name(state.remote_job_name)
        if state.output_uri is not None:
            self._validate_owned_output_uri(state.output_uri)
        return state

    def read_results(self, output_uri: str) -> Iterator[str]:
        bucket_name, submission_root = self._submission_root(output_uri)
        _, output_prefix = _parse_gcs_uri(output_uri)
        self._validate_owned_output_uri(output_uri)

        def iter_lines() -> Iterator[str]:
            with _translate_gateway_errors():
                credentials = self._credentials()
                with self._managed_storage_client(credentials) as storage_client:
                    correlations = self._load_manifest(
                        storage_client, bucket_name, submission_root
                    )
                    blobs = list(
                        storage_client.list_blobs(
                            bucket_name,
                            prefix=f"{output_prefix.rstrip('/')}/",
                            max_results=_MAX_RESULT_BLOBS + 1,
                            page_size=_BLOB_PAGE_SIZE,
                            timeout=self._request_timeout_seconds,
                        )
                    )
                    if len(blobs) > _MAX_RESULT_BLOBS:
                        raise VertexBatchContractError(
                            "Vertex labeling output contains too many objects"
                        )
                    jsonl_blobs = sorted(
                        (blob for blob in blobs if blob.name.endswith(".jsonl")),
                        key=lambda blob: blob.name,
                    )
                    if not jsonl_blobs:
                        raise VertexBatchContractError(
                            "Vertex labeling output contains no JSONL files"
                        )
                    known_size = sum(
                        size
                        for blob in jsonl_blobs
                        if isinstance((size := getattr(blob, "size", None)), int)
                    )
                    if known_size > self._max_result_bytes:
                        raise VertexBatchContractError(
                            "Vertex labeling output exceeds its size limit"
                        )
                    used_bytes = 0
                    seen_hashes: set[str] = set()
                    for blob in jsonl_blobs:
                        with blob.open(
                            "rb", timeout=self._request_timeout_seconds
                        ) as stream:
                            while True:
                                remaining = self._max_result_bytes - used_bytes
                                read_limit = min(_MAX_RESULT_LINE_BYTES, remaining) + 1
                                raw_line = stream.readline(read_limit)
                                if not raw_line:
                                    break
                                if len(raw_line) > remaining:
                                    raise VertexBatchContractError(
                                        "Vertex labeling output exceeds its size limit"
                                    )
                                if len(raw_line) > _MAX_RESULT_LINE_BYTES:
                                    raise VertexBatchContractError(
                                        "Vertex labeling output row exceeds its size limit"
                                    )
                                used_bytes += len(raw_line)
                                try:
                                    line = raw_line.decode("utf-8")
                                except UnicodeDecodeError:
                                    raise VertexBatchContractError(
                                        "Vertex labeling output is not valid UTF-8"
                                    ) from None
                                if not line.strip():
                                    continue
                                normalized = self._normalize_output_line(
                                    line, correlations
                                )
                                normalized_value = cast(
                                    dict[str, object], json.loads(normalized)
                                )
                                request_hash = cast(str, normalized_value["key"])
                                if request_hash in seen_hashes:
                                    raise VertexBatchContractError(
                                        "Vertex output has a duplicate request hash"
                                    )
                                seen_hashes.add(request_hash)
                                yield normalized + "\n"

        return iter_lines()

    def cancel(self, remote_job_name: str) -> None:
        self._validate_remote_job_name(remote_job_name)
        super().cancel(remote_job_name)

    def delete(self, remote_job_name: str) -> None:
        self._validate_remote_job_name(remote_job_name)
        super().delete(remote_job_name)

    def cleanup(self, prefix: str) -> None:
        bucket_name, submission_root = self._submission_root(prefix)
        with _translate_gateway_errors():
            credentials = self._credentials()
            with self._managed_storage_client(credentials) as storage_client:
                blobs = list(
                    storage_client.list_blobs(
                        bucket_name,
                        prefix=f"{submission_root}/",
                        max_results=_MAX_RESULT_BLOBS + 1,
                        page_size=_BLOB_PAGE_SIZE,
                        timeout=self._request_timeout_seconds,
                    )
                )
                if len(blobs) > _MAX_RESULT_BLOBS:
                    raise VertexBatchContractError(
                        "Vertex labeling cleanup contains too many objects"
                    )
                bucket = storage_client.bucket(bucket_name)
                for offset in range(0, len(blobs), _DELETE_BATCH_SIZE):
                    bucket.delete_blobs(
                        blobs[offset : offset + _DELETE_BATCH_SIZE],
                        timeout=self._request_timeout_seconds,
                    )
