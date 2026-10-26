import json
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from google.auth.credentials import Credentials
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from onyx.file_store.file_store import FileStore
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayHTTPError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchJobStatus,
    VertexBatchRequest,
)
from onyx.regulatory.labeling import vertex_direct
from onyx.regulatory.labeling.provider import (
    DEFAULT_MODEL,
    LabelDefinition,
    TaxonomyDefinition,
    build_labeling_request,
    parse_labeling_batch_output,
)
from onyx.tracing.flows import LLMFlow

_KEY = "regulatory-labeling-" + "b" * 64
_BINDING = "a" * 64


class _Store:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, object]] = {}
        self.fail_final_manifest = False
        self.fail_result = False

    def has_file(self, file_id: str, *_args: object) -> bool:
        return file_id in self.files

    def read_file(self, file_id: str) -> BytesIO:
        return BytesIO(self.files[file_id])

    def read_file_record(self, file_id: str) -> object:
        return SimpleNamespace(file_metadata=self.metadata[file_id])

    def save_file(
        self,
        content: BytesIO,
        *,
        file_id: str,
        file_metadata: dict[str, object],
        **_kwargs: object,
    ) -> str:
        data = content.read()
        if self.fail_result and not file_id.endswith("manifest.json"):
            self.fail_result = False
            raise ConnectionError("local cache connection interrupted")
        if (
            self.fail_final_manifest
            and file_id.endswith("manifest.json")
            and json.loads(data)["complete"]
        ):
            self.fail_final_manifest = False
            raise ConnectionError("local manifest connection interrupted")
        self.files[file_id] = data
        self.metadata[file_id] = file_metadata
        return file_id

    def delete_file(self, file_id: str, *, error_on_missing: bool = True) -> None:
        del error_on_missing
        self.files.pop(file_id, None)
        self.metadata.pop(file_id, None)


def _request(text: str = "İthalatta belge ibraz edilir.") -> VertexBatchRequest:
    return build_labeling_request(
        chunk_id=sha256(text.encode()).hexdigest(),
        text=text,
        context="Full unmodified context.",
        taxonomy=TaxonomyDefinition(
            name="Labels",
            labels=[
                LabelDefinition(
                    id="customs", name="Customs", description="Customs documents"
                )
            ],
        ),
        source_hash=sha256(text.encode()).hexdigest(),
    )


def _response(*, finish: str = "STOP") -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "finishReason": finish,
                    "content": {
                        "role": "model",
                        "parts": [{"text": '{"labels":[],"abstained":true}'}],
                    },
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 25,
                "candidatesTokenCount": 9,
                "totalTokenCount": 34,
            },
            "modelVersion": DEFAULT_MODEL,
        }
    )


def _gateway(
    store: _Store, model: MagicMock | None = None
) -> tuple[vertex_direct.LabelingVertexDirectGateway, MagicMock]:
    client = model or MagicMock()
    if model is None:
        client.models.generate_content.return_value = _response()
    factory = MagicMock(return_value=client)
    gateway = vertex_direct.LabelingVertexDirectGateway(
        config=VertexBatchConfig(
            model_configuration_id=7,
            model_name=DEFAULT_MODEL,
            project="test-project",
            location="global",
            authentication_mode=VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
        ),
        binding_sha256=_BINDING,
        credential_identity="test@example.com",
        credential_json_provider=lambda: pytest.fail(
            "Mocked SDK requested raw credentials"
        ),
        credentials_factory=lambda: cast(Credentials, object()),
        client_factory=factory,
        store=cast(FileStore, store),
    )
    return gateway, factory


def test_exact_structured_request_full_response_and_trace_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces: list[dict[str, Any]] = []

    @contextmanager
    def traced(**kwargs: Any) -> Any:
        traces.append(kwargs)
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    store = _Store()
    gateway, factory = _gateway(store)
    request = _request()
    heartbeat = MagicMock()
    state = gateway.submit_stream(
        [request],
        submission_key=_KEY,
        max_jsonl_bytes=8 * 1024 * 1024,
        on_progress=heartbeat,
    )
    assert state.status is VertexBatchJobStatus.SUCCEEDED
    assert state.output_uri is not None
    sdk = factory.return_value.models.generate_content.call_args.kwargs
    assert sdk["model"] == DEFAULT_MODEL
    assert sdk["contents"][0].parts[0].text == request.prompt
    assert sdk["config"].system_instruction == request.system_instruction
    assert request.generation_config is not None
    assert (
        sdk["config"].response_json_schema
        == request.generation_config["responseJsonSchema"]
    )
    assert sdk["config"].max_output_tokens == 32768
    assert sdk["config"].thinking_config.thinking_level.value.lower() == "medium"
    assert traces[0]["flow"] is LLMFlow.REGULATORY_LABELING_BATCH
    assert traces[0]["provider"] == "vertex_ai"
    assert heartbeat.call_count == 2
    lines = list(gateway.read_results(state.output_uri))
    payload = json.loads(lines[0])
    assert payload["request"] == request.to_generate_content_request()
    assert payload["response"]["candidates"][0]["finishReason"] == "STOP"
    assert payload["response"]["usageMetadata"]["totalTokenCount"] == 34
    parsed = parse_labeling_batch_output(iter(lines), [request.request_hash])
    assert parsed[request.request_hash].context == '{"labels":[],"abstained":true}'
    assert gateway.reconcile_submission(_KEY) == state
    assert gateway.get(state.remote_job_name) == state
    assert all("gs://" not in name and "files/" not in name for name in store.files)


def test_partial_retry_reuses_acknowledged_results_and_keeps_original_submission() -> (
    None
):
    store = _Store()
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _response(),
        genai_errors.ServerError(
            503, {"error": {"message": "private provider detail"}}
        ),
        _response(),
    ]
    gateway, factory = _gateway(store, client)
    requests = [_request("First source."), _request("Second source.")]
    with pytest.raises(IndexingGatewayHTTPError) as failure:
        gateway.submit(requests, submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    assert failure.value.status_code == 503
    assert "private provider detail" not in str(failure.value)
    assert gateway.reconcile_submission(_KEY) is None
    state = gateway.submit(
        requests, submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024
    )
    assert client.models.generate_content.call_count == 3
    assert factory.call_count == 2
    assert len(list(gateway.read_results(state.output_uri or ""))) == 2


def test_manifest_save_failure_is_retryable_and_reuses_every_stored_response() -> None:
    store = _Store()
    store.fail_final_manifest = True
    gateway, factory = _gateway(store)
    request = _request()
    with pytest.raises(IndexingGatewayConnectionError):
        gateway.submit([request], submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    state = gateway.submit(
        [request], submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024
    )
    assert state.status is VertexBatchJobStatus.SUCCEEDED
    assert factory.call_count == 1
    assert factory.return_value.models.generate_content.call_count == 1


def test_unstored_inference_can_repeat_after_cache_failure_without_exactly_once_claim() -> (
    None
):
    store = _Store()
    store.fail_result = True
    gateway, factory = _gateway(store)
    request = _request()
    with pytest.raises(IndexingGatewayConnectionError):
        gateway.submit([request], submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    gateway.submit([request], submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    assert factory.return_value.models.generate_content.call_count == 2


def test_tampered_cache_is_rejected_without_another_generation() -> None:
    store = _Store()
    gateway, factory = _gateway(store)
    request = VertexBatchRequest(prompt="Default generation request.")
    state = gateway.submit(
        [request], submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024
    )
    assert state.output_uri is not None
    assert len(list(gateway.read_results(state.output_uri))) == 1
    name = next(name for name in store.files if not name.endswith("manifest.json"))
    store.files[name] += b"corruption"
    with pytest.raises(VertexBatchContractError, match="integrity"):
        gateway.submit([request], submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024)
    assert factory.return_value.models.generate_content.call_count == 1


def test_cardinality_duplicate_and_binding_boundaries_reject_before_inference() -> None:
    gateway, factory = _gateway(_Store())
    with pytest.raises(VertexBatchContractError, match="64"):
        gateway.submit(
            [_request(str(number)) for number in range(65)],
            submission_key=_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
        )
    with pytest.raises(VertexBatchContractError, match="duplicate"):
        gateway.submit(
            [_request(), _request()],
            submission_key=_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
        )
    with pytest.raises(VertexBatchContractError, match="binding"):
        list(gateway.read_results("vertex-direct://" + "c" * 64 + "/" + "b" * 64))
    factory.assert_not_called()


def test_incomplete_cancel_is_local_and_does_not_require_missing_results() -> None:
    gateway, factory = _gateway(_Store())
    calls = 0

    def heartbeat() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Cancelled locally")

    with pytest.raises(RuntimeError, match="Cancelled locally"):
        gateway.submit_stream(
            [_request("First."), _request("Second.")],
            submission_key=_KEY,
            max_jsonl_bytes=8 * 1024 * 1024,
            on_progress=heartbeat,
        )
    remote = f"vertex-direct/{_BINDING}/" + "b" * 64
    gateway.cancel(remote)
    state = gateway.reconcile_submission(_KEY)
    assert state is not None and state.status is VertexBatchJobStatus.CANCELLED
    assert factory.return_value.models.generate_content.call_count == 1
    gateway.cleanup(remote)
    assert gateway.reconcile_submission(_KEY) is None


def test_truncated_finish_reason_is_preserved_and_never_accepted_as_labels() -> None:
    client = MagicMock()
    client.models.generate_content.return_value = _response(finish="MAX_TOKENS")
    gateway, _factory = _gateway(_Store(), client)
    request = _request()
    state = gateway.submit(
        [request], submission_key=_KEY, max_jsonl_bytes=8 * 1024 * 1024
    )
    lines = list(gateway.read_results(state.output_uri or ""))
    assert (
        json.loads(lines[0])["response"]["candidates"][0]["finishReason"]
        == "MAX_TOKENS"
    )
    assert (
        parse_labeling_batch_output(iter(lines), [request.request_hash])[
            request.request_hash
        ].error
        is not None
    )


def test_probe_only_reads_vertex_model_and_sdk_automatic_retries_are_disabled() -> None:
    gateway, factory = _gateway(_Store())
    probe = gateway.probe_gemini_read_access()
    assert probe.credential_identity == "test@example.com"
    factory.return_value.models.get.assert_called_once_with(model=DEFAULT_MODEL)
    factory.return_value.models.generate_content.assert_not_called()
    with patch("google.genai.Client", autospec=True) as client:
        gateway._create_client(cast(Credentials, object()))
    options = client.call_args.kwargs
    assert options["vertexai"]
    assert options["http_options"].retry_options.attempts == 1
    assert options["http_options"].timeout == 120_000
