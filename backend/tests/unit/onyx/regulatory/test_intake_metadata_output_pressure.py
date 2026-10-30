"""Correlated output pressure permits one bounded reserve before target splitting."""

import json
from threading import Event
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.regulatory.document_intake import metadata, metadata_job
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchRequest,
)
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import _CachedGateway
from tests.unit.onyx.regulatory.test_intake_metadata_split_shapes import _parent
from tests.unit.onyx.regulatory.test_intake_metadata_value_recovery import _fact

_PRESSURE = {"thoughtsTokenCount": 31455, "candidatesTokenCount": 8}
_EMPTY = '{"chunks":[],"warnings":[]}'


def _config(request: VertexBatchRequest) -> dict[str, JsonValue]:
    assert request.generation_config is not None
    return request.generation_config


def _valid(request: VertexBatchRequest) -> str:
    return json.dumps(
        {
            "chunks": [
                {"chunk_id": target, "fields": []}
                for target in json.loads(request.prompt)["target_chunk_ids"]
            ],
            "warnings": [],
        }
    )


def _with_usage(envelope: dict[str, Any], usage: object = _PRESSURE) -> dict[str, Any]:
    if usage is not None:
        envelope["response"]["usageMetadata"] = usage
    return envelope


def _seed(
    gateway: _CachedGateway,
    request: VertexBatchRequest,
    response: str,
    usage: object = None,
) -> None:
    key = metadata_job._metadata_submission_key("sgm4-frozen", request)
    gateway.lines[key] = json.dumps(
        _with_usage(_envelope(request, "STOP", response), usage)
    )


def _fixture() -> tuple[
    list[metadata.MetadataChunk], VertexBatchRequest, str, VertexBatchRequest
]:
    sources, targets, invalid = _parent()
    base = metadata.build_metadata_request(sources, targets)
    repair = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )
    return sources, base, invalid, repair


def _run(
    gateway: _CachedGateway, sources: list[metadata.MetadataChunk]
) -> metadata_job.MetadataResults:
    return metadata_job._extract_metadata_targets(
        gateway=cast(VertexBatchGateway, gateway),
        chunks=sources,
        targets=[source.chunk_id for source in sources],
        identity_prefix="sgm4-frozen",
        heartbeat=lambda: None,
        stopped=Event(),
    )


def test_cached_empty_stop_repair_uses_one_exact_reserve_and_reuses_every_ack() -> None:
    sources, base, invalid, repair = _fixture()
    gateway = _CachedGateway(
        lambda request: _envelope(request, "STOP", _valid(request))
    )
    _seed(gateway, base, invalid)
    _seed(gateway, repair, _EMPTY, _PRESSURE)
    before = dict(gateway.lines)
    result = _run(gateway, sources)
    assert len(gateway.new_requests) == 1
    reserved = gateway.new_requests[0]
    assert reserved == metadata.build_metadata_output_reserve_request(repair)
    assert (
        reserved.prompt == repair.prompt
        and reserved.system_instruction == repair.system_instruction
    )
    assert reserved.generation_config == {
        **_config(repair),
        "maxOutputTokens": 65536,
    }
    assert all(
        target.fields[0].field == "publication_number"
        and target.fields[0].retained_request_hash == base.request_hash
        for target in result[0][1].chunks
    )
    assert base.request_hash in result[0][1].warnings[0]
    assert all(gateway.lines[key] == value for key, value in before.items())
    after = dict(gateway.lines)
    assert _run(gateway, sources) == result
    assert len(gateway.new_requests) == 1 and gateway.lines == after


def test_valid_cached_legacy_response_never_uses_the_reserve() -> None:
    sources, base, _invalid, _repair = _fixture()
    gateway = _CachedGateway(
        lambda _request: pytest.fail("Valid cached ACK needs no call")
    )
    _seed(gateway, base, _valid(base), _PRESSURE)
    assert _run(gateway, sources)[0][0] == base.request_hash
    assert not gateway.new_requests


def test_correlated_base_max_tokens_gets_one_reserve_before_target_splitting() -> None:
    sources, base, _invalid, _repair = _fixture()
    gateway = _CachedGateway(
        lambda request: _envelope(request, "STOP", _valid(request))
    )
    key = metadata_job._metadata_submission_key("sgm4-frozen", base)
    gateway.lines[key] = json.dumps(
        _with_usage(_envelope(base, "MAX_TOKENS", _valid(base)))
    )
    assert len(_run(gateway, sources)) == 1
    assert gateway.new_requests == [
        metadata.build_metadata_output_reserve_request(base)
    ]


def test_provider_exception_stops_without_output_reserve_or_child_requests() -> None:
    sources, base, invalid, _repair = _fixture()

    def fail(_request: VertexBatchRequest) -> dict[str, Any]:
        raise RuntimeError("Provider unavailable")

    gateway = _CachedGateway(fail)
    _seed(gateway, base, invalid, _PRESSURE)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        _run(gateway, sources)
    assert len(gateway.new_requests) == 1
    assert _config(gateway.new_requests[0])["maxOutputTokens"] == 32768


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"thoughtsTokenCount": "31455", "candidatesTokenCount": 8},
        {"thoughtsTokenCount": True, "candidatesTokenCount": 8},
        {"thoughtsTokenCount": 31455, "candidatesTokenCount": -1},
        {"thoughtsTokenCount": 31455, "candidatesTokenCount": 2000},
        {"thoughtsTokenCount": 100, "candidatesTokenCount": 8},
    ],
)
def test_missing_or_invalid_repair_usage_splits_without_borrowing_base_pressure(
    usage: object,
) -> None:
    sources, base, invalid, repair = _fixture()
    gateway = _CachedGateway(
        lambda request: _envelope(request, "STOP", _valid(request))
    )
    _seed(gateway, base, invalid, _PRESSURE)
    _seed(gateway, repair, _EMPTY, usage)
    result = _run(gateway, sources)
    assert len(gateway.new_requests) == 2
    assert all(
        _config(request)["maxOutputTokens"] == 32768
        and len(json.loads(request.prompt)["target_chunk_ids"]) == 4
        for request in gateway.new_requests
    )
    assert all(
        target.fields[0].retained_request_hash == base.request_hash
        for _hash, leaf in result
        for target in leaf.chunks
    )
    assert all(base.request_hash in leaf.warnings[0] for _hash, leaf in result)


def test_recursive_shape_tree_is_finite_keeps_full_context_and_stops_reboosting() -> (
    None
):
    sources, _base, _invalid, _repair = _fixture()

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        payload = json.loads(request.prompt)
        assert payload["complete_document"] == [
            source.model_dump(mode="json") for source in sources
        ]
        assert _config(request)["thinkingConfig"] == {"thinkingLevel": "medium"}
        return _with_usage(
            _envelope(
                request,
                "STOP",
                _EMPTY if len(payload["target_chunk_ids"]) > 1 else _valid(request),
            )
        )

    gateway = _CachedGateway(respond)
    result = _run(gateway, sources)
    assert len(result) == 8
    assert (
        len(gateway.new_requests)
        == len({request.request_hash for request in gateway.new_requests})
        <= 45
    )
    assert (
        sum(
            _config(request)["maxOutputTokens"] == 65536
            for request in gateway.new_requests
        )
        == 7
    )
    before = dict(gateway.lines)
    count = len(gateway.new_requests)
    assert _run(gateway, sources) == result
    assert len(gateway.new_requests) == count and gateway.lines == before


def test_single_target_invalid_shape_is_terminal_after_one_reserve() -> None:
    sources, _base, _invalid, _repair = _fixture()
    gateway = _CachedGateway(
        lambda request: _with_usage(_envelope(request, "STOP", _EMPTY))
    )
    with pytest.raises(ValueError, match="invalid structured output for one target"):
        _run(gateway, sources[:1])
    assert len(gateway.new_requests) == 3
    assert (
        sum(
            _config(request)["maxOutputTokens"] == 65536
            for request in gateway.new_requests
        )
        == 1
    )


@pytest.mark.parametrize("case", ["safety", "provider", "wrong_echo", "wrong_hash"])
def test_untrusted_provider_envelopes_never_authorize_reserve_or_splitting(
    case: str,
) -> None:
    sources, base, invalid, repair = _fixture()
    gateway = _CachedGateway(
        lambda _request: pytest.fail("Invalid envelope cannot authorize a call")
    )
    _seed(gateway, base, invalid)
    envelope = _with_usage(_envelope(repair, "STOP", _EMPTY))
    if case == "safety":
        envelope["response"]["candidates"][0]["finishReason"] = "SAFETY"
    elif case == "provider":
        envelope["status"] = "provider failed"
    elif case == "wrong_echo":
        envelope["request"]["generationConfig"]["maxOutputTokens"] = 65536
    else:
        envelope["key"] = "f" * 64
    gateway.lines[metadata_job._metadata_submission_key("sgm4-frozen", repair)] = (
        json.dumps(envelope)
    )
    with pytest.raises(ValueError):
        _run(gateway, sources)
    assert not gateway.new_requests


@pytest.mark.parametrize(
    "field,value",
    [
        ("article_no", "1"),
        ("validity_start_date", "2021-07-02"),
        ("document_title", "Invented title"),
    ],
)
def test_semantic_failures_never_trigger_output_reserve_or_target_splitting(
    field: str, value: str
) -> None:
    sources, base, invalid, repair = _fixture()
    evidence = next(
        entry
        for entry in metadata.build_metadata_evidence_catalog(sources)
        if entry.source_chunk_id == sources[0].chunk_id and entry.scope == "heading"
    )
    semantic = json.loads(_valid(base))
    for target in semantic["chunks"]:
        target["fields"] = [
            _fact(
                field,
                value,
                "f" * 64 if field == "article_no" else evidence.evidence_id,
            )
        ]
    response = json.dumps(semantic)
    gateway = _CachedGateway(
        lambda request: _with_usage(_envelope(request, "STOP", response))
    )
    _seed(gateway, base, invalid)
    _seed(gateway, repair, response, _PRESSURE)
    with pytest.raises(ValueError, match="after its single repair"):
        _run(gateway, sources)
    assert len(gateway.new_requests) <= 1
    assert all(
        _config(request)["maxOutputTokens"] == 32768
        and len(json.loads(request.prompt)["target_chunk_ids"]) == 8
        for request in gateway.new_requests
    )
