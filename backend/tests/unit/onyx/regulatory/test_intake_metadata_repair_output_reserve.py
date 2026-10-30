"""Every correlated repair MAX shares one output reserve before target splitting."""

import json
from typing import Any

import pytest

from onyx.regulatory.document_intake import metadata, metadata_job
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
from tests.unit.onyx.regulatory.test_intake_metadata_cached_repair import (
    _case,
)
from tests.unit.onyx.regulatory.test_intake_metadata_cached_repair import (
    _run as _run_one,
)
from tests.unit.onyx.regulatory.test_intake_metadata_output_pressure import (
    _config,
    _fixture,
    _run,
    _seed,
    _valid,
    _with_usage,
)
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import _CachedGateway
from tests.unit.onyx.regulatory.test_intake_metadata_value_recovery import (
    _entry,
    _fact,
    _response,
)

_USAGE = {"thoughtsTokenCount": 31454, "candidatesTokenCount": 1299}


def _maximum(request: VertexBatchRequest, usage: object = _USAGE) -> dict[str, Any]:
    raw = json.loads(_valid(request))
    catalog = json.loads(request.prompt)["evidence_catalog"]
    for target in raw["chunks"]:
        target["fields"] = [
            _fact(
                "document_type",
                "Never retain partial MAX output",
                next(
                    entry["evidence_id"]
                    for entry in catalog
                    if entry["source_chunk_id"] == target["chunk_id"]
                    and entry["scope"] == "body"
                ),
            )
        ]
    return _with_usage(_envelope(request, "MAX_TOKENS", json.dumps(raw)), usage)


def _seed_max(
    gateway: _CachedGateway, request: VertexBatchRequest, usage: object = _USAGE
) -> None:
    gateway.lines[metadata_job._metadata_submission_key("sgm4-frozen", request)] = (
        json.dumps(_maximum(request, usage))
    )


@pytest.mark.parametrize("cached", [False, True])
def test_fresh_and_cached_generic_max_reserve_same_prompt_and_actual_receipt(
    cached: bool,
) -> None:
    sources, base, invalid, primary = _fixture()
    reserved = metadata.build_metadata_output_reserve_request(primary)
    assert reserved is not None
    gateway = _CachedGateway(
        lambda request: (
            _maximum(request)
            if request.request_hash == primary.request_hash
            else _envelope(request, "STOP", _valid(request))
        )
    )
    gateway.seed(base, invalid)
    if cached:
        _seed_max(gateway, primary)
    before = dict(gateway.lines)
    result = _run(gateway, sources)
    assert result[0][0] == reserved.request_hash
    assert gateway.new_requests[-1] == reserved
    assert all(
        [fact.field for fact in target.fields] == ["publication_number"]
        for target in result[0][1].chunks
    )
    assert all(
        target.fields[0].retained_request_hash == base.request_hash
        for target in result[0][1].chunks
    )
    assert all(gateway.lines[key] == value for key, value in before.items())
    after = dict(gateway.lines)
    assert _run(gateway, sources) == result
    assert len(gateway.new_requests) == (1 if cached else 2) and gateway.lines == after


def test_cached_generic_max_in_first_body_branch_reuses_existing_64k_ack() -> None:
    sources, base, invalid, _scoped, _bad, valid = _case("scoped")
    generic = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )
    reserved = metadata.build_metadata_output_reserve_request(generic)
    assert reserved is not None
    gateway = _CachedGateway(
        lambda _request: pytest.fail("All required ACKs are cached")
    )
    gateway.seed(base, invalid)
    _seed_max(gateway, generic)
    gateway.seed(reserved, valid)
    before = dict(gateway.lines)
    assert _run_one(gateway, sources)[0][0] == reserved.request_hash
    assert gateway.new_requests == [] and gateway.lines == before


def test_fresh_targeted_feedback_max_reserves_and_keeps_latest_valid_fact_origin() -> (
    None
):
    sources, base, invalid, primary = _fixture()
    targeted = metadata.build_metadata_catalog_recovery_request(primary, invalid)
    assert targeted is not None
    reserved = metadata.build_metadata_output_reserve_request(targeted)
    assert reserved is not None

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        if request.request_hash == primary.request_hash:
            return _envelope(request, "STOP", invalid)
        if request.request_hash == targeted.request_hash:
            return _maximum(request)
        assert request == reserved
        return _envelope(request, "STOP", _valid(request))

    gateway = _CachedGateway(respond)
    gateway.seed(base, invalid)
    result = _run(gateway, sources)
    assert result[0][0] == reserved.request_hash and len(gateway.new_requests) == 3
    assert all(
        target.fields[0].retained_request_hash == primary.request_hash
        for target in result[0][1].chunks
    )
    before = dict(gateway.lines)
    assert _run(gateway, sources) == result
    assert len(gateway.new_requests) == 3 and gateway.lines == before


def test_primary_reserved_semantic_failure_uses_actual_64k_hash_for_body_feedback() -> (
    None
):
    sources, base, heading_date, _scoped, _bad, valid = _case("scoped")
    title = sources[0].heading_path[0]
    invalid = _response(
        [
            _fact(
                "document_title",
                title.replace("KULLANILACAK", "KULLANILACAL"),
                _entry(sources, title, "heading").evidence_id,
            )
        ],
        sources[0].chunk_id,
    )
    primary = metadata.build_metadata_repair_request(
        base, invalid, validation_category="value_not_covered_by_evidence"
    )
    reserved = metadata.build_metadata_output_reserve_request(primary)
    assert reserved is not None

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        if request == reserved:
            return _envelope(request, "STOP", heading_date)
        feedback = json.loads(request.prompt)["validation_feedback"]
        assert feedback["cached_repair_request_hash"] == reserved.request_hash
        assert request.generation_config == reserved.generation_config
        return _envelope(request, "STOP", valid)

    gateway = _CachedGateway(respond)
    gateway.seed(base, invalid)
    _seed_max(gateway, primary)
    result = _run_one(gateway, sources)
    assert (
        len(gateway.new_requests) == 2
        and result[0][0] == gateway.new_requests[-1].request_hash
    )


@pytest.mark.parametrize("usage", [None, _USAGE])
def test_unknown_pressure_splits_and_used_64k_reserve_never_boosts_again(
    usage: object,
) -> None:
    sources, base, invalid, primary = _fixture()

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        if len(json.loads(request.prompt)["target_chunk_ids"]) == 8:
            assert _config(request)["maxOutputTokens"] == 65536
            return _maximum(request)
        return _envelope(request, "STOP", _valid(request))

    gateway = _CachedGateway(respond)
    gateway.seed(base, invalid)
    _seed_max(gateway, primary, usage)
    result = _run(gateway, sources)
    assert len(result) == 2
    assert sum(
        _config(request)["maxOutputTokens"] == 65536 for request in gateway.new_requests
    ) == (0 if usage is None else 1)
    assert all(
        len(json.loads(request.prompt)["target_chunk_ids"]) == 4
        for request in gateway.new_requests[-2:]
    )
    assert all(
        [fact.field for fact in target.fields] == ["publication_number"]
        for _hash, leaf in result
        for target in leaf.chunks
    )


@pytest.mark.parametrize("corruption", ["provider", "hash", "echo"])
def test_uncorrelated_or_provider_failed_max_never_authorizes_a_reserve(
    corruption: str,
) -> None:
    sources, base, invalid, primary = _fixture()
    gateway = _CachedGateway(
        lambda _request: pytest.fail("Invalid response cannot authorize a call")
    )
    gateway.seed(base, invalid)
    envelope = _maximum(primary)
    if corruption == "provider":
        envelope["status"] = "provider failed"
    elif corruption == "hash":
        envelope["key"] = "f" * 64
    else:
        envelope["request"]["generationConfig"]["maxOutputTokens"] = 65536
    gateway.lines[metadata_job._metadata_submission_key("sgm4-frozen", primary)] = (
        json.dumps(envelope)
    )
    with pytest.raises(ValueError):
        _run(gateway, sources)
    assert gateway.new_requests == []


def test_full_recursive_repair_max_tree_stays_within_four_identities_per_node() -> None:
    sources, _base, invalid, _primary = _fixture()
    parent = json.loads(invalid)

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        payload = json.loads(request.prompt)
        targets = payload["target_chunk_ids"]
        assert payload["complete_document"] == [
            source.model_dump(mode="json") for source in sources
        ]
        if len(targets) == 1:
            return _envelope(request, "STOP", _valid(request))
        feedback = payload.get("validation_feedback")
        if feedback is None:
            body = {
                **parent,
                "chunks": [
                    target
                    for target in parent["chunks"]
                    if target["chunk_id"] in targets
                ],
            }
            return _envelope(request, "STOP", json.dumps(body))
        if (
            _config(request)["maxOutputTokens"] == 32768
            or "catalog_recovery_version" in feedback
        ):
            return _maximum(request)
        body = json.loads(_valid(request))
        for target in body["chunks"]:
            evidence = next(
                entry
                for entry in payload["evidence_catalog"]
                if entry["source_chunk_id"] == target["chunk_id"]
                and entry["scope"] == "body"
            )
            target["fields"] = [
                _fact("section_title", "Invented title", evidence["evidence_id"])
            ]
        return _envelope(request, "STOP", json.dumps(body))

    gateway = _CachedGateway(respond)
    result = _run(gateway, sources)
    assert len(result) == 8 and len(gateway.new_requests) == 36
    per_node: dict[tuple[str, ...], int] = {}
    for request in gateway.new_requests:
        node = tuple(json.loads(request.prompt)["target_chunk_ids"])
        per_node[node] = per_node.get(node, 0) + 1
    assert max(per_node.values()) == 4 and len(gateway.new_requests) <= 60
    before = dict(gateway.lines)
    assert _run(gateway, sources) == result
    assert len(gateway.new_requests) == 36 and gateway.lines == before


@pytest.mark.parametrize("cached_children", [1, 2])
@pytest.mark.parametrize("finish", ["MAX_TOKENS", "STOP"])
def test_existing_child_acks_preserve_child_only_facts_before_parent_reserve(
    cached_children: int,
    finish: str,
) -> None:
    sources, base, invalid, primary = _fixture()
    children = [
        metadata.build_metadata_request(
            sources, [source.chunk_id for source in sources[offset : offset + 4]]
        )
        for offset in (0, 4)
    ]
    gateway = _CachedGateway(
        lambda request: _envelope(request, "STOP", _valid(request))
    )
    gateway.seed(base, invalid)
    if finish == "MAX_TOKENS":
        _seed_max(gateway, primary)
    else:
        _seed(gateway, primary, '{"chunks":[],"warnings":[]}', _USAGE)
    for child in children[:cached_children]:
        payload = json.loads(child.prompt)
        rows = []
        for target in payload["target_chunk_ids"]:
            evidence = next(
                entry
                for entry in payload["evidence_catalog"]
                if entry["source_chunk_id"] == target and entry["scope"] == "heading"
            )
            rows.append(
                {
                    "chunk_id": target,
                    "fields": [
                        _fact(
                            "article_no",
                            evidence["text"].removeprefix("MADDE "),
                            evidence["evidence_id"],
                        )
                    ],
                }
            )
        gateway.seed(child, json.dumps({"chunks": rows, "warnings": []}))
    before = dict(gateway.lines)
    result = _run(gateway, sources)
    assert [request_hash for request_hash, _leaf in result] == [
        child.request_hash for child in children
    ]
    assert gateway.new_requests == children[cached_children:]
    for _request_hash, leaf in result[:cached_children]:
        for target in leaf.chunks:
            facts = {fact.field: fact for fact in target.fields}
            assert facts["article_no"].value == target.chunk_id.removeprefix("target-")
            assert facts["article_no"].retained_request_hash is None
            assert (
                facts["publication_number"].retained_request_hash == base.request_hash
            )
    assert all(gateway.lines[key] == value for key, value in before.items())
    assert (
        sum(
            key == metadata_job._metadata_submission_key("sgm4-frozen", child)
            for key in gateway.probes
            for child in children
        )
        <= 2
    )
    after = dict(gateway.lines)
    assert _run(gateway, sources) == result and gateway.lines == after
    assert gateway.new_requests == children[cached_children:]


@pytest.mark.parametrize("corruption", ["provider", "correlation"])
def test_invalid_child_envelope_is_not_accepted_by_cache_presence(
    corruption: str,
) -> None:
    sources, base, invalid, primary = _fixture()
    child = metadata.build_metadata_request(
        sources, [source.chunk_id for source in sources[:4]]
    )
    gateway = _CachedGateway(
        lambda _request: pytest.fail("Invalid child ACK cannot authorize inference")
    )
    gateway.seed(base, invalid)
    _seed_max(gateway, primary)
    envelope = _envelope(child, "STOP", _valid(child))
    if corruption == "provider":
        envelope["status"] = "provider failed"
    else:
        envelope["request"]["generationConfig"]["maxOutputTokens"] = 65536
    gateway.lines[metadata_job._metadata_submission_key("sgm4-frozen", child)] = (
        json.dumps(envelope)
    )
    with pytest.raises(ValueError):
        _run(gateway, sources)
    assert gateway.new_requests == []


def test_cached_valid_parent_reserve_precedes_existing_child_tree() -> None:
    sources, base, invalid, primary = _fixture()
    reserved = metadata.build_metadata_output_reserve_request(primary)
    assert reserved is not None
    child = metadata.build_metadata_request(
        sources, [source.chunk_id for source in sources[:4]]
    )
    gateway = _CachedGateway(
        lambda _request: pytest.fail("The valid parent ACK is cached")
    )
    gateway.seed(base, invalid)
    _seed_max(gateway, primary)
    gateway.seed(reserved, _valid(reserved))
    gateway.seed(child, _valid(child))
    assert _run(gateway, sources)[0][0] == reserved.request_hash
    assert gateway.new_requests == []


def test_repeated_split_retains_valid_partial_facts_from_cached_parent_reserve() -> (
    None
):
    sources, base, invalid, primary = _fixture()
    reserved = metadata.build_metadata_output_reserve_request(primary)
    assert reserved is not None

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        payload = json.loads(request.prompt)
        if len(payload["target_chunk_ids"]) == 4:
            return _envelope(request, "STOP", _valid(request))
        if request == reserved:
            body = json.loads(_valid(request))
            for target in body["chunks"]:
                evidence = next(
                    entry
                    for entry in payload["evidence_catalog"]
                    if entry["source_chunk_id"] == target["chunk_id"]
                    and entry["scope"] == "body"
                    and "Kural." in entry["text"]
                )
                target["fields"] = [
                    _fact("article_title", "Kural.", evidence["evidence_id"]),
                    _fact("section_title", "Invented", evidence["evidence_id"]),
                ]
            return _envelope(request, "STOP", json.dumps(body))
        return _envelope(request, "STOP", '{"chunks":[],"warnings":[]}')

    gateway = _CachedGateway(respond)
    gateway.seed(base, invalid)
    _seed_max(gateway, primary)
    first = _run(gateway, sources)
    assert len(first) == 2 and len(gateway.new_requests) == 4
    assert all(
        next(
            fact for fact in target.fields if fact.field == "article_title"
        ).retained_request_hash
        == reserved.request_hash
        for _hash, leaf in first
        for target in leaf.chunks
    )
    before = dict(gateway.lines)
    assert _run(gateway, sources) == first
    assert len(gateway.new_requests) == 4 and gateway.lines == before
