"""Exhausted semantic windows shrink without accepting unsupported facts."""

import json
from collections.abc import Sequence
from threading import Event
from typing import Literal, cast

import pytest

from onyx.regulatory.document_intake import metadata, metadata_job
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchRequest,
)
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
from tests.unit.onyx.regulatory.test_intake_metadata_cached_repair import (
    _entry,
    _fact,
    _sources,
)
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import _CachedGateway

FailureKind = Literal["catalog", "value", "scope"]
_TITLE = "Taşımacılık faaliyetlerinde genel kurallar"
_DATE_CLAUSE = "(1) Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer."


def _chunks() -> list[metadata.MetadataChunk]:
    article, closing = _sources("scoped")
    return [
        article.model_copy(update={"chunk_id": f"target-{index}"}) for index in range(8)
    ] + [closing]


def _response(
    chunks: Sequence[metadata.MetadataChunk],
    targets: Sequence[str],
    kind: FailureKind,
    *,
    valid: bool = False,
    original: bool = False,
) -> str:
    result = []
    for target in targets:
        fields = []
        if original:
            fields.append(
                _fact(
                    "article_no",
                    "5",
                    _entry(chunks, "MADDE 5 - " + _TITLE, "heading"),
                )
            )
        if valid or target == "target-0":
            if kind == "scope":
                fields.append(
                    _fact(
                        "validity_start_date",
                        "2022-01-01",
                        _entry(chunks, _DATE_CLAUSE, "body" if valid else "heading"),
                    )
                )
            else:
                fields.append(
                    _fact(
                        "article_title",
                        _TITLE if valid or kind == "catalog" else "Unsupported title",
                        _entry(chunks, "MADDE 5 - " + _TITLE, "heading")
                        if valid or kind == "value"
                        else "d570f2441372a556963db9f869080024ced1d",
                    )
                )
        result.append({"chunk_id": target, "fields": fields})
    return json.dumps(
        {
            "chunks": result,
            "warnings": ["Earlier source warning."] if original else [],
        }
    )


def _seed_chain(
    gateway: _CachedGateway,
    chunks: list[metadata.MetadataChunk],
    kind: FailureKind,
) -> tuple[VertexBatchRequest, list[str]]:
    targets = [chunk.chunk_id for chunk in chunks[:-1]]
    base = metadata.build_metadata_request(chunks, targets)
    invalid = _response(chunks, targets, kind, original=True)
    gateway.seed(base, invalid)
    primary = (
        metadata.build_metadata_body_scope_repair_request(base, invalid)
        if kind == "scope"
        else metadata.build_metadata_repair_request(
            base,
            invalid,
            validation_category=(
                "value_not_covered_by_evidence"
                if kind == "value"
                else "invalid_coverage_or_evidence"
            ),
        )
    )
    assert primary is not None
    invalid_repair = _response(chunks, targets, kind)
    gateway.seed(primary, invalid_repair)
    recovery = metadata.build_metadata_catalog_recovery_request(primary, invalid_repair)
    if recovery is not None:
        gateway.seed(recovery, invalid_repair)
    return base, targets


def _run(
    gateway: _CachedGateway,
    chunks: list[metadata.MetadataChunk],
    targets: list[str],
) -> metadata_job.MetadataResults:
    return metadata_job._extract_metadata_targets(
        gateway=cast(VertexBatchGateway, gateway),
        chunks=chunks,
        targets=targets,
        identity_prefix="sgm4-frozen",
        heartbeat=lambda: None,
        stopped=Event(),
    )


@pytest.mark.parametrize("kind", ["catalog", "value", "scope"])
@pytest.mark.parametrize("cached_children", [False, True])
def test_exhausted_semantics_split_targets_and_retain_validated_parent_facts(
    kind: FailureKind, cached_children: bool
) -> None:
    chunks = _chunks()

    def respond(request: VertexBatchRequest) -> dict[str, object]:
        payload = json.loads(request.prompt)
        assert len(payload["target_chunk_ids"]) == 4
        assert "validation_feedback" not in payload
        return _envelope(
            request,
            "STOP",
            _response(chunks, payload["target_chunk_ids"], kind, valid=True),
        )

    gateway = _CachedGateway(respond)
    base, targets = _seed_chain(gateway, chunks, kind)
    child_requests = [
        metadata.build_metadata_request(chunks, child)
        for child in (targets[:4], targets[4:])
    ]
    if cached_children:
        for request in child_requests:
            gateway.seed(
                request,
                _response(
                    chunks,
                    json.loads(request.prompt)["target_chunk_ids"],
                    kind,
                    valid=True,
                ),
            )
    before = dict(gateway.lines)
    results = _run(gateway, chunks, targets)
    assert [request_hash for request_hash, _result in results] == [
        request.request_hash for request in child_requests
    ]
    assert gateway.new_requests == ([] if cached_children else child_requests)
    original_payload = json.loads(base.prompt)
    for request in child_requests:
        payload = json.loads(request.prompt)
        for key in ("complete_document", "evidence_catalog", "input_sha256"):
            assert payload[key] == original_payload[key]
    for _request_hash, result in results:
        for chunk in result.chunks:
            facts = {fact.field: fact for fact in chunk.fields}
            assert facts["article_no"].value == "5"
            assert facts["article_no"].retained_request_hash == base.request_hash
            if kind != "scope":
                assert facts["article_title"].value == _TITLE
        assert any("Earlier source warning" in warning for warning in result.warnings)
    assert all(gateway.lines[key] == value for key, value in before.items())
    after = dict(gateway.lines)
    assert _run(gateway, chunks, targets) == results
    assert gateway.lines == after
    assert len(gateway.new_requests) == (0 if cached_children else 2)


@pytest.mark.parametrize("kind", ["catalog", "value", "scope"])
def test_semantic_splitting_stops_strictly_at_one_target_and_retry_reuses_acks(
    kind: FailureKind,
) -> None:
    chunks = _chunks()
    gateway = _CachedGateway(
        lambda request: _envelope(
            request,
            "STOP",
            _response(chunks, json.loads(request.prompt)["target_chunk_ids"], kind),
        )
    )
    _base, targets = _seed_chain(gateway, chunks, kind)
    before = dict(gateway.lines)
    with pytest.raises(ValueError, match="failed strict validation after bounded"):
        _run(gateway, chunks, targets)
    counts = [
        len(json.loads(request.prompt)["target_chunk_ids"])
        for request in gateway.new_requests
    ]
    assert set(counts) == {4, 2, 1}
    assert counts == sorted(counts, reverse=True)
    assert len(counts) <= 9
    assert all(gateway.lines[key] == value for key, value in before.items())
    after = dict(gateway.lines)
    with pytest.raises(ValueError, match="failed strict validation after bounded"):
        _run(gateway, chunks, targets)
    assert gateway.lines == after and len(gateway.new_requests) == len(counts)


@pytest.mark.parametrize("finish", ["SAFETY", "RECITATION"])
def test_blocked_provider_output_never_authorizes_semantic_split(
    finish: str,
) -> None:
    chunks = _chunks()
    gateway = _CachedGateway(lambda _request: pytest.fail("No provider call permitted"))
    base, targets = _seed_chain(gateway, chunks, "catalog")
    gateway.seed(base, _response(chunks, targets, "catalog"), finish=finish)
    before = dict(gateway.lines)
    with pytest.raises(ValueError, match="incomplete or invalid"):
        _run(gateway, chunks, targets)
    assert gateway.lines == before and not gateway.new_requests


def test_every_failed_window_can_reach_valid_leaves_with_finite_exact_coverage() -> (
    None
):
    chunks = _chunks()

    def respond(request: VertexBatchRequest) -> dict[str, object]:
        targets = json.loads(request.prompt)["target_chunk_ids"]
        response = json.loads(_response(chunks, targets, "catalog", valid=True))
        if len(targets) > 1:
            for chunk in response["chunks"]:
                chunk["fields"][0]["evidence_id"] = "not-a-catalog-id"
        return _envelope(request, "STOP", json.dumps(response))

    gateway = _CachedGateway(respond)
    base, targets = _seed_chain(gateway, chunks, "catalog")
    results = _run(gateway, chunks, targets)
    assert len(results) == len(targets)
    assert [
        chunk.chunk_id for _hash, result in results for chunk in result.chunks
    ] == targets
    windows = {
        tuple(json.loads(request.prompt)["target_chunk_ids"])
        for request in gateway.new_requests
    }
    assert len(windows) == 2 * len(targets) - 2
    assert len(gateway.new_requests) <= 3 * len(windows)
    for _hash, result in results:
        facts = {fact.field: fact for fact in result.chunks[0].fields}
        assert facts["article_no"].retained_request_hash == base.request_hash
        assert facts["article_title"].value == _TITLE
        assert any("Earlier source warning" in warning for warning in result.warnings)


@pytest.mark.parametrize("failure", ["request_mismatch", "provider_error"])
def test_provider_integrity_errors_never_authorize_semantic_split(failure: str) -> None:
    chunks = _chunks()
    gateway = _CachedGateway(lambda _request: pytest.fail("No provider call permitted"))
    base, targets = _seed_chain(gateway, chunks, "catalog")
    key = metadata_job._metadata_submission_key("sgm4-frozen", base)
    envelope = json.loads(gateway.lines[key])
    if failure == "request_mismatch":
        envelope["request"]["contents"][0]["parts"][0]["text"] += " tampered"
    else:
        envelope["status"] = "provider failed"
    gateway.lines[key] = json.dumps(envelope)
    before = dict(gateway.lines)
    with pytest.raises(ValueError):
        _run(gateway, chunks, targets)
    assert gateway.lines == before and not gateway.new_requests
