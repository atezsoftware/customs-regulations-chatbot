"""Bounded direct Vertex labeling with resumable results in the local file store.

Acknowledged responses are reused. An inference whose response was never stored
can repeat after a crash or timeout; this transport does not claim exactly once.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from io import BytesIO
from typing import TYPE_CHECKING, Literal, Protocol, cast

from google.auth.credentials import Credentials
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.configs.constants import FileOrigin
from onyx.file_store.file_store import FileStore
from onyx.regulatory.indexing_jobs.legacy_vertex_batch import _translated_gateway_error
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
    VertexReadOnlyAccessProbe,
    _raise_secret_safe_public_error,
    build_vertex_jsonl,
)
from onyx.regulatory.labeling.provider import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call

if TYPE_CHECKING:
    from google.genai import types as genai_types

_SUBMISSION = re.compile(r"regulatory-labeling-([0-9a-f]{64})")
_REMOTE = re.compile(r"vertex-direct/([0-9a-f]{64})/([0-9a-f]{64})")
_OUTPUT = re.compile(r"vertex-direct://([0-9a-f]{64})/([0-9a-f]{64})")
_MAX_REQUESTS = 64
_MAX_INPUT_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024
_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class _Models(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
    ) -> genai_types.GenerateContentResponse: ...

    def get(self, *, model: str) -> object: ...


class _Client(Protocol):
    models: _Models

    def close(self) -> None: ...


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submission_key: str
    request_hashes: list[str] = Field(min_length=1, max_length=_MAX_REQUESTS)
    complete: bool = False
    cancelled: bool = False


class _StoredResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    submission_key: str
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_payload: dict[str, JsonValue]
    line_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    line: str


@contextmanager
def _safe_sdk_errors() -> Iterator[None]:
    try:
        yield
    except Exception as error:
        translated = _translated_gateway_error(error)
        _raise_secret_safe_public_error(
            translated or VertexBatchContractError("Vertex direct request failed")
        )


class LabelingVertexDirectGateway(VertexBatchGateway):
    def __init__(
        self,
        *,
        config: VertexBatchConfig,
        binding_sha256: str,
        credential_json_provider: Callable[[], str | None],
        credential_identity: str,
        store: FileStore | None = None,
        request_timeout_seconds: float = 120,
        max_result_bytes: int = 64 * 1024 * 1024,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        flow: LLMFlow = LLMFlow.REGULATORY_LABELING_BATCH,
        client_factory: Callable[[Credentials], _Client] | None = None,
        credentials_factory: Callable[[], Credentials] | None = None,
    ) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", binding_sha256) is None:
            raise VertexBatchContractError("Vertex direct binding identity is invalid")
        if (
            not math.isfinite(request_timeout_seconds)
            or not 0 < request_timeout_seconds <= 240
        ):
            raise VertexBatchContractError(
                "Vertex direct timeout must be at most 240 seconds"
            )
        if max_result_bytes < 1:
            raise VertexBatchContractError(
                "Vertex direct result budget must be positive"
            )
        if not 1 <= max_request_bytes <= _MAX_INPUT_BYTES:
            raise VertexBatchContractError("Vertex direct request budget exceeds 8 MiB")
        if not 1 <= max_response_bytes <= 16 * 1024 * 1024:
            raise VertexBatchContractError(
                "Vertex direct response budget exceeds 16 MiB"
            )
        if store is None:
            from onyx.file_store.postgres_file_store import PostgresBackedFileStore

            store = PostgresBackedFileStore()
        self._config = config
        self._binding = binding_sha256
        self._credential_json_provider = credential_json_provider
        self._identity = credential_identity
        self._store = store
        self._timeout = request_timeout_seconds
        self._max_result_bytes = max_result_bytes
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._flow = flow
        self._client_factory = client_factory or self._create_client
        self._credentials_factory = credentials_factory or self._credentials

    def _credentials(self) -> Credentials:
        import google.auth
        from google.oauth2 import service_account

        with _safe_sdk_errors():
            if (
                self._config.authentication_mode
                is VertexAuthenticationMode.WORKLOAD_IDENTITY
            ):
                credentials, _project = google.auth.default(scopes=[_CLOUD_SCOPE])
                return credentials
            raw = self._credential_json_provider()
            if not raw:
                raise ValueError("Missing service account")
            return service_account.Credentials.from_service_account_info(
                json.loads(raw), scopes=[_CLOUD_SCOPE]
            )
        raise AssertionError("credential context did not return")

    def _create_client(self, credentials: Credentials) -> _Client:
        from google import genai
        from google.genai import types as genai_types

        return cast(
            _Client,
            genai.Client(
                vertexai=True,
                project=self._config.project,
                location=self._config.location,
                credentials=credentials,
                http_options=genai_types.HttpOptions(
                    api_version="v1",
                    timeout=int(self._timeout * 1000),
                    retry_options=genai_types.HttpRetryOptions(attempts=1),
                ),
            ),
        )

    @contextmanager
    def _client(self) -> Iterator[_Client]:
        with _safe_sdk_errors():
            client = self._client_factory(self._credentials_factory())
        try:
            yield client
        finally:
            with _safe_sdk_errors():
                client.close()

    def _submission_hash(self, submission_key: str) -> str:
        match = _SUBMISSION.fullmatch(submission_key)
        if match is None:
            raise VertexBatchContractError(
                "Vertex direct submission identity is invalid"
            )
        return match.group(1)

    def _prefix(self, submission_key: str) -> str:
        return f"regulatory-labeling-direct/{self._binding}/{self._submission_hash(submission_key)}"

    def _manifest_id(self, submission_key: str) -> str:
        return f"{self._prefix(submission_key)}/manifest.json"

    def _result_id(self, submission_key: str, request_hash: str) -> str:
        return f"{self._prefix(submission_key)}/{request_hash}.json"

    def _write(self, file_id: str, content: bytes) -> None:
        try:
            self._store.save_file(
                BytesIO(content),
                display_name="labeling-direct.json",
                file_origin=FileOrigin.OTHER,
                file_type="application/json",
                file_id=file_id,
                file_metadata={
                    "sha256": sha256(content).hexdigest(),
                    "binding_sha256": self._binding,
                },
            )
        except Exception:
            _raise_secret_safe_public_error(IndexingGatewayConnectionError())

    def _read(self, file_id: str, *, limit: int) -> bytes | None:
        try:
            if not self._store.has_file(file_id, FileOrigin.OTHER, "application/json"):
                return None
            record = self._store.read_file_record(file_id)
            metadata = record.file_metadata
            with self._store.read_file(file_id) as stream:
                content = stream.read(limit + 1)
        except Exception:
            _raise_secret_safe_public_error(IndexingGatewayConnectionError())
        if not isinstance(metadata, Mapping):
            raise VertexBatchContractError("Vertex direct stored metadata is malformed")
        typed_metadata = cast(Mapping[str, object], metadata)
        if (
            len(content) > limit
            or typed_metadata.get("binding_sha256") != self._binding
            or typed_metadata.get("sha256") != sha256(content).hexdigest()
        ):
            raise VertexBatchContractError(
                "Vertex direct stored result integrity changed"
            )
        return content

    def _manifest(self, submission_key: str) -> _Manifest | None:
        raw = self._read(self._manifest_id(submission_key), limit=_MAX_MANIFEST_BYTES)
        if raw is None:
            return None
        try:
            manifest = _Manifest.model_validate_json(raw)
        except ValueError:
            raise VertexBatchContractError(
                "Vertex direct manifest is malformed"
            ) from None
        if (
            manifest.binding_sha256 != self._binding
            or manifest.submission_key != submission_key
            or len(set(manifest.request_hashes)) != len(manifest.request_hashes)
            or any(
                re.fullmatch(r"[0-9a-f]{64}", item) is None
                for item in manifest.request_hashes
            )
        ):
            raise VertexBatchContractError("Vertex direct manifest identity changed")
        return manifest

    def _result(self, submission_key: str, request_hash: str) -> str | None:
        raw = self._read(
            self._result_id(submission_key, request_hash),
            limit=self._max_response_bytes * 4,
        )
        if raw is None:
            return None
        try:
            stored = _StoredResult.model_validate_json(raw)
        except ValueError:
            raise VertexBatchContractError(
                "Vertex direct result is malformed"
            ) from None
        if (
            stored.binding_sha256 != self._binding
            or stored.submission_key != submission_key
            or stored.request_hash != request_hash
            or len(stored.line.encode()) > self._max_response_bytes
            or sha256(stored.line.encode()).hexdigest() != stored.line_sha256
        ):
            raise VertexBatchContractError("Vertex direct result identity changed")
        try:
            line = json.loads(stored.line)
            rebuilt = VertexBatchRequest.model_validate(stored.request_payload)
        except (KeyError, IndexError, TypeError, ValueError):
            raise VertexBatchContractError(
                "Vertex direct correlation request is malformed"
            ) from None
        if (
            line.get("key") != request_hash
            or rebuilt.request_hash != request_hash
            or line.get("request") != rebuilt.to_generate_content_request()
        ):
            raise VertexBatchContractError("Vertex direct correlation request changed")
        return stored.line

    def _state(self, manifest: _Manifest) -> VertexBatchState:
        suffix = f"{self._binding}/{self._submission_hash(manifest.submission_key)}"
        return VertexBatchState(
            remote_job_name=f"vertex-direct/{suffix}",
            status=(
                VertexBatchJobStatus.CANCELLED
                if manifest.cancelled
                else VertexBatchJobStatus.SUCCEEDED
                if manifest.complete
                else VertexBatchJobStatus.PENDING
            ),
            output_uri=f"vertex-direct://{suffix}" if manifest.complete else None,
        )

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        return self.submit_stream(
            requests,
            submission_key=submission_key,
            max_jsonl_bytes=max_jsonl_bytes,
            on_progress=lambda: None,
        )

    def submit_stream(
        self,
        requests: Iterable[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
        on_progress: Callable[[], None],
    ) -> VertexBatchState:
        self._submission_hash(submission_key)
        prepared: list[VertexBatchRequest] = []
        for request in requests:
            if len(prepared) >= _MAX_REQUESTS:
                raise VertexBatchContractError(
                    "Vertex direct labeling is limited to 64 requests"
                )
            prepared.append(request)
        if (
            not prepared
            or max_jsonl_bytes < 1
            or len(build_vertex_jsonl(prepared).encode())
            > min(max_jsonl_bytes, _MAX_INPUT_BYTES)
            or any(
                len(
                    json.dumps(
                        request.to_generate_content_request(), ensure_ascii=False
                    ).encode()
                )
                > self._max_request_bytes
                for request in prepared
            )
        ):
            raise VertexBatchContractError(
                "Vertex direct input exceeds its bounded request budget"
            )
        hashes = sorted(request.request_hash for request in prepared)
        if len(set(hashes)) != len(hashes):
            raise VertexBatchContractError(
                "Vertex direct input contains duplicate requests"
            )
        manifest = self._manifest(submission_key)
        if manifest is None:
            manifest = _Manifest(
                binding_sha256=self._binding,
                submission_key=submission_key,
                request_hashes=hashes,
            )
            self._write(
                self._manifest_id(submission_key), manifest.model_dump_json().encode()
            )
        if manifest.request_hashes != hashes:
            raise VertexBatchContractError(
                "Vertex direct submission belongs to different requests"
            )
        if manifest.cancelled:
            return self._state(manifest)
        total_bytes = 0
        client: _Client | None = None
        with ExitStack() as clients:
            for request in prepared:
                on_progress()
                line = self._result(submission_key, request.request_hash)
                if line is None:
                    if client is None:
                        client = clients.enter_context(self._client())
                    line = self._generate(client, request)
                    stored = _StoredResult(
                        binding_sha256=self._binding,
                        submission_key=submission_key,
                        request_hash=request.request_hash,
                        request_payload=request.model_dump(
                            mode="json", exclude={"request_hash"}
                        ),
                        line_sha256=sha256(line.encode()).hexdigest(),
                        line=line,
                    )
                    self._write(
                        self._result_id(submission_key, request.request_hash),
                        stored.model_dump_json().encode(),
                    )
                total_bytes += len(line.encode())
                if total_bytes > self._max_result_bytes:
                    raise VertexBatchContractError(
                        "Vertex direct output exceeds its complete result budget"
                    )
                on_progress()
        manifest.complete = True
        self._write(
            self._manifest_id(submission_key), manifest.model_dump_json().encode()
        )
        return self._state(manifest)

    def _generate(self, client: _Client, request: VertexBatchRequest) -> str:
        from google.genai import types as genai_types

        payload = request.to_generate_content_request()
        options = dict(cast(dict[str, object], payload["generationConfig"]))
        if request.system_instruction is not None:
            options["systemInstruction"] = request.system_instruction
        config = genai_types.GenerateContentConfig.model_validate(options)
        with _safe_sdk_errors():
            with traced_llm_call(
                flow=self._flow,
                model=self._config.model_name,
                provider="vertex_ai",
                input_messages=payload["contents"],
                extra_config={
                    "transport": "vertex_direct_v1",
                    "request_hash": request.request_hash,
                },
            ) as span:
                response = client.models.generate_content(
                    model=self._config.model_name,
                    contents=[
                        genai_types.Content(
                            role="user", parts=[genai_types.Part(text=request.prompt)]
                        )
                    ],
                    config=config,
                )
                response_payload = response.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
                usage = response_payload.get("usageMetadata")
                if isinstance(usage, dict):
                    span.span_data.usage = usage
        line = json.dumps(
            {
                "key": request.request_hash,
                "request": payload,
                "response": response_payload,
                "status": "",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(line.encode()) > self._max_response_bytes:
            raise VertexBatchContractError(
                "Vertex direct complete response exceeds its row budget"
            )
        return line

    def _key_from_resource(self, resource: str, *, output: bool = False) -> str:
        match = (_OUTPUT if output else _REMOTE).fullmatch(resource)
        if match is None or match.group(1) != self._binding:
            raise VertexBatchContractError(
                "Vertex direct resource is outside its provider binding"
            )
        return f"regulatory-labeling-{match.group(2)}"

    def get(self, remote_job_name: str) -> VertexBatchState:
        key = self._key_from_resource(remote_job_name)
        manifest = self._manifest(key)
        if manifest is None:
            raise VertexBatchContractError("Vertex direct manifest is unavailable")
        return self._state(manifest)

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        manifest = self._manifest(submission_key)
        if manifest is None or not (manifest.complete or manifest.cancelled):
            return None
        if manifest.cancelled:
            return self._state(manifest)
        for request_hash in manifest.request_hashes:
            if self._result(submission_key, request_hash) is None:
                raise VertexBatchContractError(
                    "Vertex direct manifest has a missing response"
                )
        return self._state(manifest)

    def read_results(self, output_uri: str) -> Iterator[str]:
        key = self._key_from_resource(output_uri, output=True)
        manifest = self._manifest(key)
        if manifest is None or not manifest.complete or manifest.cancelled:
            raise VertexBatchContractError("Vertex direct results are incomplete")
        total = 0
        for request_hash in manifest.request_hashes:
            line = self._result(key, request_hash)
            if line is None:
                raise VertexBatchContractError("Vertex direct response is missing")
            total += len(line.encode())
            if total > self._max_result_bytes:
                raise VertexBatchContractError(
                    "Vertex direct output exceeds its complete result budget"
                )
            yield line

    def cancel(self, remote_job_name: str) -> None:
        key = self._key_from_resource(remote_job_name)
        manifest = self._manifest(key)
        if manifest is not None:
            manifest.cancelled = True
            self._write(self._manifest_id(key), manifest.model_dump_json().encode())

    def delete(self, remote_job_name: str) -> None:
        key = self._key_from_resource(remote_job_name)
        manifest = self._manifest(key)
        if manifest is not None:
            for request_hash in manifest.request_hashes:
                self._store.delete_file(
                    self._result_id(key, request_hash), error_on_missing=False
                )
            self._store.delete_file(self._manifest_id(key), error_on_missing=False)

    def cleanup(self, prefix: str) -> None:
        key = self._key_from_resource(
            prefix, output=prefix.startswith("vertex-direct://")
        )
        self.delete(f"vertex-direct/{self._binding}/{self._submission_hash(key)}")

    def probe_gemini_read_access(self) -> VertexReadOnlyAccessProbe:
        with self._client() as client:
            with _safe_sdk_errors():
                client.models.get(model=self._config.model_name)
        return VertexReadOnlyAccessProbe(credential_identity=self._identity)
