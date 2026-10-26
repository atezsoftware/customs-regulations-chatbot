"""Bounded metadata windows preserve full evidence and acknowledged responses."""

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from google.genai import types as genai_types

from onyx.regulatory.document_intake import metadata_job
from onyx.regulatory.document_intake.metadata import (
    TARGETS_PER_REQUEST,
    MetadataChunk,
    build_metadata_repair_request,
    build_metadata_request,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
)
from onyx.regulatory.labeling import vertex_direct
from tests.unit.onyx.regulatory.labeling.test_vertex_direct import _gateway, _Store


def _sources(count: int) -> list[MetadataChunk]:
    return [
        MetadataChunk(
            chunk_id=f"chunk-{index}",
            text=f"MADDE {index} — Bu kural uygulanır.",
            heading_path=[f"MADDE {index}"],
            chunk_type="article",
        )
        for index in range(count)
    ]


def _outcome(targets: Sequence[str]) -> str:
    return json.dumps(
        {
            "chunks": [{"chunk_id": target, "fields": []} for target in targets],
            "warnings": [],
        }
    )


def _envelope(request: VertexBatchRequest, finish: str, text: str) -> dict[str, Any]:
    return {
        "key": request.request_hash,
        "request": json.loads(json.dumps(request.to_generate_content_request())),
        "response": {
            "candidates": [
                {"finishReason": finish, "content": {"parts": [{"text": text}]}}
            ]
        },
        "status": "",
    }


class _Gateway:
    def __init__(self, respond: Callable[[VertexBatchRequest], dict[str, Any]]) -> None:
        self.respond = respond
        self.requests: list[VertexBatchRequest] = []
        self.lines: dict[str, str] = {}
        self.lock = Lock()

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        assert len(requests) == 1 and max_jsonl_bytes > 0
        request = requests[0]
        with self.lock:
            self.requests.append(request)
        envelope = self.respond(request)
        with self.lock:
            self.lines[submission_key] = json.dumps(envelope)
        return VertexBatchState(
            remote_job_name=submission_key,
            status=VertexBatchJobStatus.SUCCEEDED,
            output_uri=submission_key,
        )

    def read_results(self, output_uri: str) -> Iterator[str]:
        yield self.lines[output_uri]


def _run(gateway: object, sources: list[MetadataChunk]) -> metadata_job.MetadataResults:
    return metadata_job._extract_metadata_document(
        gateway=cast(VertexBatchGateway, gateway),
        chunks=sources,
        identity_prefix="metadata-v1:tenant:intake:file:binding",
        heartbeat=lambda: None,
    )


@pytest.mark.parametrize("interrupt_child", [False, True])
def test_max_tokens_splits_eight_to_four_plus_four_and_reuses_all_real_direct_acks(
    monkeypatch: pytest.MonkeyPatch,
    interrupt_child: bool,
) -> None:
    sources = _sources(TARGETS_PER_REQUEST)
    store, client = _Store(), MagicMock()
    payloads: list[dict[str, Any]] = []

    def generate(**kwargs: Any) -> genai_types.GenerateContentResponse:
        payload = json.loads(kwargs["contents"][0].parts[0].text)
        payloads.append(payload)
        targets = payload["target_chunk_ids"]
        if interrupt_child and len(payloads) == 3:
            raise RuntimeError("Interrupted second child")
        finish, text = (
            ("MAX_TOKENS", '{"chunks":[{"chunk_id":"unfinished')
            if len(targets) == 8
            else ("STOP", _outcome(targets))
        )
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "finishReason": finish,
                        "content": {"role": "model", "parts": [{"text": text}]},
                    }
                ]
            }
        )

    @contextmanager
    def traced(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    client.models.generate_content.side_effect = generate
    gateway, _factory = _gateway(store, client)
    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    retained_acks: dict[str, bytes] = {}
    if interrupt_child:
        with pytest.raises(ValueError):
            _run(gateway, sources)
        retained_acks = {
            key: value
            for key, value in store.files.items()
            if not key.endswith("manifest.json")
        }
    results = _run(gateway, sources)
    original_cache = dict(store.files)
    assert [len(payload["target_chunk_ids"]) for payload in payloads] == (
        [8, 4, 4, 4] if interrupt_child else [8, 4, 4]
    )
    if interrupt_child:
        assert all(store.files[key] == value for key, value in retained_acks.items())
    assert all(len(payload["complete_document"]) == 8 for payload in payloads)
    assert [chunk.chunk_id for _hash, result in results for chunk in result.chunks] == [
        chunk.chunk_id for chunk in sources
    ]
    assert [item[0] for item in results] == [
        build_metadata_request(
            sources, [chunk.chunk_id for chunk in subset]
        ).request_hash
        for subset in (sources[:4], sources[4:])
    ]
    assert _run(gateway, sources) == results
    assert len(payloads) == (4 if interrupt_child else 3)
    assert store.files == original_cache


def test_recursive_split_has_disjoint_ordered_leaves_and_complete_context() -> None:
    sources = _sources(8)

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        payload = json.loads(request.prompt)
        targets = payload["target_chunk_ids"]
        assert (
            payload["complete_document"]
            == json.loads(
                build_metadata_request(sources, [sources[0].chunk_id]).prompt
            )["complete_document"]
        )
        assert request.generation_config
        assert request.generation_config["thinkingConfig"] == {
            "thinkingLevel": "medium"
        }
        assert request.generation_config["maxOutputTokens"] == 32768
        return _envelope(
            request,
            "MAX_TOKENS" if len(targets) > 1 else "STOP",
            _outcome(targets),
        )

    gateway = _Gateway(respond)
    results = _run(gateway, sources)
    assert len(gateway.requests) == 15
    assert len({request.request_hash for request in gateway.requests}) == 15
    assert [chunk.chunk_id for _hash, result in results for chunk in result.chunks] == [
        chunk.chunk_id for chunk in sources
    ]


@pytest.mark.parametrize("text", ["", '{"chunks":', _outcome(["chunk-0"])])
def test_single_target_max_tokens_fails_without_accepting_even_valid_partial_json(
    text: str,
) -> None:
    gateway = _Gateway(lambda request: _envelope(request, "MAX_TOKENS", text))
    with pytest.raises(ValueError, match="token budget for one target"):
        _run(gateway, _sources(1))
    assert len(gateway.requests) == 1


@pytest.mark.parametrize("body", [None, {}, {"parts": []}])
def test_authoritative_max_tokens_without_visible_body_still_splits(
    body: dict[str, Any] | None,
) -> None:
    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        targets = json.loads(request.prompt)["target_chunk_ids"]
        value = _envelope(request, "STOP", _outcome(targets))
        if len(targets) == 8:
            value["response"]["candidates"][0].update(
                finishReason="MAX_TOKENS", content=body
            )
        return value

    gateway = _Gateway(respond)
    assert len(_run(gateway, _sources(8))) == 2
    assert len(gateway.requests) == 3


@pytest.mark.parametrize(
    "case",
    [
        "invalid-json",
        "missing-target",
        "duplicate-target",
        "unquoted-evidence",
        "safety",
        "remote-error",
        "invalid-status-without-body",
        "wrong-echo",
        "wrong-hash",
        "multiple-candidates",
        "malformed-content",
        "missing-finish",
        "max-tokens-only-in-generated-text",
    ],
)
def test_invalid_bodies_get_one_repair_but_envelope_failures_never_retry(
    case: str,
) -> None:
    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        targets = json.loads(request.prompt)["target_chunk_ids"]
        value = _envelope(request, "STOP", _outcome(targets))
        candidate = value["response"]["candidates"][0]
        if case == "invalid-json":
            candidate["content"]["parts"][0]["text"] = "{"
        elif case == "missing-target":
            candidate["content"]["parts"][0]["text"] = _outcome(targets[:-1])
        elif case == "duplicate-target":
            candidate["content"]["parts"][0]["text"] = _outcome([targets[0]] * 8)
        elif case == "unquoted-evidence":
            candidate["content"]["parts"][0]["text"] = json.dumps(
                {
                    "chunks": [
                        {
                            "chunk_id": target,
                            "fields": [
                                {
                                    "field": "article_title",
                                    "value": "Invented",
                                    "source_chunk_id": target,
                                    "evidence_quote": "not in source",
                                }
                            ],
                        }
                        for target in targets
                    ]
                }
            )
        elif case == "safety":
            candidate["finishReason"] = "MAX_TOKENS"
            value["response"]["promptFeedback"] = {"blockReason": "SAFETY"}
        elif case == "remote-error":
            candidate["finishReason"] = "MAX_TOKENS"
            value["status"] = "provider failed"
        elif case == "invalid-status-without-body":
            candidate.update(finishReason="MAX_TOKENS", content=None)
            value["status"] = {}
        elif case == "wrong-echo":
            candidate["finishReason"] = "MAX_TOKENS"
            value["request"]["generationConfig"]["temperature"] = 1
        elif case == "wrong-hash":
            candidate["finishReason"] = "MAX_TOKENS"
            value["key"] = "f" * 64
        elif case == "multiple-candidates":
            candidate["finishReason"] = "MAX_TOKENS"
            value["response"]["candidates"].append(candidate.copy())
        elif case == "malformed-content":
            candidate["finishReason"] = "MAX_TOKENS"
            candidate["content"] = {"parts": [{"text": 42}]}
        elif case == "missing-finish":
            del candidate["finishReason"]
        else:
            candidate["content"]["parts"][0]["text"] = '{"finishReason":"MAX_TOKENS"}'
        return value

    gateway = _Gateway(respond)
    with pytest.raises(ValueError):
        _run(gateway, _sources(8))
    repairable = {
        "invalid-json",
        "missing-target",
        "duplicate-target",
        "unquoted-evidence",
        "max-tokens-only-in-generated-text",
    }
    assert len(gateway.requests) == (2 if case in repairable else 1)
    assert all(
        len(json.loads(request.prompt)["target_chunk_ids"]) == 8
        for request in gateway.requests
    )


def test_strict_turkish_quote_repair_is_cached_with_valid_windows_and_own_leaf_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources(16)
    sources[0] = sources[0].model_copy(update={"text": "TEBLİĞ\n" + sources[0].text})
    store, client = _Store(), MagicMock()
    payloads: list[dict[str, Any]] = []
    invalid_response = ""

    def generate(**kwargs: Any) -> genai_types.GenerateContentResponse:
        nonlocal invalid_response
        payload = json.loads(kwargs["contents"][0].parts[0].text)
        payloads.append(payload)
        targets = payload["target_chunk_ids"]
        outcome = json.loads(_outcome(targets))
        if targets[0] == "chunk-8":
            repaired = "validation_feedback" in payload
            for chunk in outcome["chunks"]:
                chunk["fields"] = [
                    {
                        "field": "document_type",
                        "value": "teblig",
                        "source_chunk_id": "chunk-0",
                        "evidence_quote": "TEBLİĞ" if repaired else "TEBLIĞ",
                        "basis": "explicit",
                        "reference_chunk_id": "",
                        "reference_quote": "",
                    }
                ]
            if not repaired:
                invalid_response = json.dumps(outcome)
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [{"text": json.dumps(outcome)}],
                        },
                    }
                ],
            }
        )

    @contextmanager
    def traced(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    client.models.generate_content.side_effect = generate
    gateway, _factory = _gateway(store, client)
    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    monkeypatch.setattr(metadata_job, "METADATA_CONCURRENT_WINDOWS", 1)
    result = _run(gateway, sources)
    original_cache = dict(store.files)
    assert len(payloads) == 3
    feedback = payloads[-1]["validation_feedback"]
    base = build_metadata_request(sources, [chunk.chunk_id for chunk in sources[8:]])
    assert feedback["original_request_hash"] == base.request_hash
    assert (
        feedback["previous_response_sha256"]
        == sha256(invalid_response.encode()).hexdigest()
    )
    assert feedback["repair_attempt"] == 1
    assert feedback["category"] == "invalid_coverage_or_evidence"
    assert "İ/I/ı" in feedback["instruction"]
    assert all(
        payload["complete_document"] == payloads[0]["complete_document"]
        for payload in payloads
    )
    repair = build_metadata_repair_request(
        base, invalid_response, validation_category="invalid_coverage_or_evidence"
    )
    assert result[-1][0] == repair.request_hash != base.request_hash
    assert all(
        fact.evidence_quote == "TEBLİĞ"
        for chunk in result[-1][1].chunks
        for fact in chunk.fields
    )
    assert repair.generation_config == base.generation_config
    assert repair.system_instruction == base.system_instruction
    assert _run(gateway, sources) == result
    assert len(payloads) == 3 and store.files == original_cache


def test_repair_max_tokens_reduces_to_base_children_without_another_same_window_repair() -> (
    None
):
    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        payload = json.loads(request.prompt)
        targets = payload["target_chunk_ids"]
        if len(targets) == 8:
            return _envelope(
                request,
                "MAX_TOKENS" if "validation_feedback" in payload else "STOP",
                "{",
            )
        assert "validation_feedback" not in payload
        return _envelope(request, "STOP", _outcome(targets))

    gateway = _Gateway(respond)
    results = _run(gateway, _sources(8))
    assert [
        len(json.loads(request.prompt)["target_chunk_ids"])
        for request in gateway.requests
    ] == [8, 8, 4, 4]
    assert len(results) == 2


def test_single_target_repair_max_tokens_fails_after_two_requests() -> None:
    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        return _envelope(
            request,
            "MAX_TOKENS"
            if "validation_feedback" in json.loads(request.prompt)
            else "STOP",
            "{",
        )

    gateway = _Gateway(respond)
    with pytest.raises(ValueError, match="token budget for one target"):
        _run(gateway, _sources(1))
    assert len(gateway.requests) == 2


def test_two_outer_windows_propagate_fresh_context_and_return_target_order() -> None:
    tenant = ContextVar("metadata_test_tenant", default="missing")
    barrier, release_first = Barrier(2), Event()
    lock = Lock()
    active = maximum = 0
    contexts: list[str] = []

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        nonlocal active, maximum
        targets = json.loads(request.prompt)["target_chunk_ids"]
        with lock:
            active += 1
            maximum = max(maximum, active)
            contexts.append(tenant.get())
        tenant.set("worker-local")
        if targets[0] in {"chunk-0", "chunk-8"}:
            barrier.wait(timeout=5)
        if targets[0] == "chunk-0":
            assert release_first.wait(timeout=5)
        elif targets[0] == "chunk-8":
            release_first.set()
        with lock:
            active -= 1
        return _envelope(request, "STOP", _outcome(targets))

    token = tenant.set("tenant-and-trace")
    try:
        gateway = _Gateway(respond)
        results = _run(gateway, _sources(24))
        assert tenant.get() == "tenant-and-trace"
    finally:
        tenant.reset(token)
    assert maximum == 2
    assert contexts == ["tenant-and-trace"] * 3
    assert [chunk.chunk_id for _hash, result in results for chunk in result.chunks] == [
        f"chunk-{index}" for index in range(24)
    ]


def test_first_failure_stops_new_windows_and_waits_for_active_receipt() -> None:
    barrier, failed, active_receipt = Barrier(2), Event(), Event()

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        targets = json.loads(request.prompt)["target_chunk_ids"]
        barrier.wait(timeout=5)
        if targets[0] == "chunk-0":
            failed.set()
            raise RuntimeError("first provider failure")
        assert failed.wait(timeout=5)
        active_receipt.set()
        return _envelope(request, "STOP", _outcome(targets))

    gateway = _Gateway(respond)
    with pytest.raises(RuntimeError, match="first provider failure"):
        _run(gateway, _sources(32))
    assert active_receipt.is_set()
    assert len(gateway.lines) == 1
    assert len(gateway.requests) == 2


def test_recursive_children_do_not_expand_outer_concurrency() -> None:
    lock, barrier = Lock(), Barrier(2)
    active = maximum = 0

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        nonlocal active, maximum
        targets = json.loads(request.prompt)["target_chunk_ids"]
        with lock:
            active += 1
            maximum = max(maximum, active)
        if len(targets) == 8:
            barrier.wait(timeout=5)
        with lock:
            active -= 1
        return _envelope(
            request, "MAX_TOKENS" if len(targets) > 2 else "STOP", _outcome(targets)
        )

    gateway = _Gateway(respond)
    results = _run(gateway, _sources(16))
    assert maximum == 2
    assert len(gateway.requests) == 14
    assert len(results) == 8
