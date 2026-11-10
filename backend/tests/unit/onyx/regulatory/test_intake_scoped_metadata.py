"""Sparse document extraction preserves target-specific legal metadata checks."""

import json
from typing import Any

import pytest

from onyx.regulatory.document_intake import metadata
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import (
    _response,
    _sources,
)


def test_scoped_checkpoint_freezes_mode_without_changing_legacy_serialization() -> None:
    from onyx.regulatory.document_intake.metadata_checkpoint import MetadataCheckpoint

    legacy = MetadataCheckpoint.build(
        recipe_sha256="a" * 64,
        provider_binding={},
        source_snapshot={},
        request_hashes={},
    )
    assert "extraction_mode" not in legacy.model_dump(mode="json")
    scoped = MetadataCheckpoint.build(
        recipe_sha256="a" * 64,
        provider_binding={},
        source_snapshot={},
        request_hashes={},
        extraction_mode="scoped_v1",
    )
    assert scoped.extraction_mode == "scoped_v1"
    assert scoped.checkpoint_sha256 != legacy.checkpoint_sha256
    scoped.validate_checksum()


def test_invalid_neighbor_does_not_discard_valid_prior_date() -> None:
    from onyx.regulatory.document_intake.metadata_scopes import (
        expand_scoped_metadata_response,
    )

    chunks = [
        metadata.MetadataChunk(
            chunk_id="rule",
            text="Bu Yönetmelik 1/1/2025 tarihinde yürürlüğe girer.",
            heading_path=[],
            chunk_type="paragraph",
        )
    ]
    catalog = metadata.build_metadata_evidence_catalog(chunks)
    evidence = next(f"e{i}" for i, item in enumerate(catalog) if item.scope == "body")
    prior = {
        "covered_targets": [0],
        "shared_fields": [],
        "assignments": [
            {
                "targets": [0],
                "fields": [
                    {
                        "field": "validity_start_date",
                        "value": "2025-01-01",
                        "evidence_id": evidence,
                        "basis": "explicit",
                        "reference_evidence_id": "",
                    },
                    {
                        "field": "document_type",
                        "value": "regulation",
                        "evidence_id": "missing",
                        "basis": "explicit",
                        "reference_evidence_id": "",
                    },
                ],
            }
        ],
        "warnings": [],
    }
    current = {
        "covered_targets": [0],
        "shared_fields": [],
        "assignments": [],
        "warnings": [],
    }
    results = expand_scoped_metadata_response(
        json.dumps(current),
        chunks=chunks,
        previous_attempts=[("a" * 64, json.dumps(prior))],
    )
    fact = results[0].chunks[0].fields[0]
    assert (fact.field, fact.value, fact.retained_request_hash) == (
        "validity_start_date",
        "2025-01-01",
        "a" * 64,
    )


def test_prior_validated_title_precedes_a_shorter_local_hint() -> None:
    from onyx.regulatory.document_intake.metadata_scopes import (
        expand_scoped_metadata_response,
    )

    chunks = [
        metadata.MetadataChunk(
            chunk_id="rule",
            text="Hüküm uygulanır.",
            heading_path=["Amaç ve kapsam"],
            structural_metadata={"article_title": "Amaç"},
            chunk_type="paragraph",
        )
    ]
    catalog = metadata.build_metadata_evidence_catalog(chunks)
    evidence = next(
        f"e{i}" for i, item in enumerate(catalog) if item.scope == "heading"
    )
    prior = {
        "covered_targets": [0],
        "shared_fields": [],
        "assignments": [
            {
                "targets": [0],
                "fields": [
                    {
                        "field": "article_title",
                        "value": "Amaç ve kapsam",
                        "evidence_id": evidence,
                        "basis": "explicit",
                        "reference_evidence_id": "",
                    }
                ],
            }
        ],
        "warnings": [],
    }
    current = {
        "covered_targets": [0],
        "shared_fields": [],
        "assignments": [],
        "warnings": [],
    }
    results = expand_scoped_metadata_response(
        json.dumps(current),
        chunks=chunks,
        previous_attempts=[("a" * 64, json.dumps(prior))],
    )
    fact = results[0].chunks[0].fields[0]
    assert (fact.value, fact.retained_request_hash) == ("Amaç ve kapsam", "a" * 64)


def test_explicit_own_commencement_cannot_be_silently_omitted() -> None:
    from onyx.regulatory.document_intake.metadata_scopes import (
        expand_scoped_metadata_response,
    )

    chunks = [
        metadata.MetadataChunk(
            chunk_id="rule",
            text="Bu Yönetmelik 1/1/2025 tarihinde yürürlüğe girer.",
            heading_path=[],
            chunk_type="paragraph",
        )
    ]
    with pytest.raises(ValueError, match="commencement"):
        expand_scoped_metadata_response(
            json.dumps(
                {
                    "covered_targets": [0],
                    "shared_fields": [],
                    "assignments": [],
                    "warnings": [],
                }
            ),
            chunks=chunks,
        )


def _scoped_response() -> tuple[list[metadata.MetadataChunk], dict[str, Any]]:
    sources = _sources()
    indices = {source.chunk_id: i for i, source in enumerate(sources)}
    aliases = {
        entry.evidence_id: f"e{i}"
        for i, entry in enumerate(metadata.build_metadata_evidence_catalog(sources))
    }
    validated = json.loads(_response(sources, valid=True))
    assignments = []
    for target in validated["chunks"]:
        fields = []
        for fact in target["fields"]:
            fields.append(
                {
                    **fact,
                    "evidence_id": aliases[fact["evidence_id"]],
                    "reference_evidence_id": aliases.get(
                        fact["reference_evidence_id"], ""
                    ),
                }
            )
        assignments.append({"targets": [indices[target["chunk_id"]]], "fields": fields})
    return sources, {
        "covered_targets": list(range(len(sources))),
        "shared_fields": [],
        "assignments": assignments,
        "warnings": [],
    }


@pytest.mark.parametrize("incorrect_general", [False, True])
def test_special_start_overrides_general_and_split_targets_are_independent(
    incorrect_general: bool,
) -> None:
    from onyx.regulatory.document_intake.metadata_scopes import (
        expand_scoped_metadata_response,
    )

    chunks = [
        metadata.MetadataChunk(
            chunk_id="special",
            text="MADDE 1-\n(1) Bu madde 1/3/2025 tarihinde yürürlüğe girer.",
            heading_path=["MADDE 1"],
            chunk_type="paragraph",
        ),
        metadata.MetadataChunk(
            chunk_id="general",
            text="Bu Yönetmelik 1/7/2025 tarihinde yürürlüğe girer.",
            heading_path=[],
            chunk_type="paragraph",
        ),
    ]
    aliases = {
        entry.source_chunk_id: f"e{i}"
        for i, entry in enumerate(metadata.build_metadata_evidence_catalog(chunks))
        if entry.scope == "body"
    }
    response = json.dumps(
        {
            "covered_targets": [0],
            "shared_fields": [],
            "assignments": [
                {
                    "targets": [0],
                    "fields": [
                        {
                            "field": "validity_start_date",
                            "value": "2025-07-01"
                            if incorrect_general
                            else "2025-03-01",
                            "basis": "explicit",
                            "evidence_id": aliases[
                                "general" if incorrect_general else "special"
                            ],
                            "reference_evidence_id": "",
                        }
                    ],
                }
            ],
            "warnings": [],
        }
    )
    if incorrect_general:
        with pytest.raises(ValueError, match="target.*commencement"):
            expand_scoped_metadata_response(response, chunks=chunks, target_indices=[0])
    else:
        assert (
            expand_scoped_metadata_response(
                response, chunks=chunks, target_indices=[0]
            )[0]
            .chunks[0]
            .fields[0]
            .value
            == "2025-03-01"
        )


def test_scoped_extraction_keeps_distinct_dates_and_publication_relative_rules() -> (
    None
):
    from onyx.regulatory.document_intake import metadata_scopes

    sources, response = _scoped_response()
    request = metadata_scopes.build_scoped_metadata_request(sources)
    results = metadata_scopes.expand_scoped_metadata_response(
        json.dumps(response), chunks=sources
    )
    facts = {
        target.chunk_id: {fact.field: fact for fact in target.fields}
        for result in results
        for target in result.chunks
    }
    expected = json.loads(_response(sources, valid=True))
    for target in expected["chunks"]:
        for fact in target["fields"]:
            assert facts[target["chunk_id"]][fact["field"]].value == fact["value"]
    assert {fact["validity_start_date"].value for fact in facts.values() if fact} == {
        "2025-01-05",
        "2025-07-01",
    }
    assert sum(len(result.chunks) for result in results) == len(sources)
    payload = json.loads(request.prompt)
    assert payload["input_sha256"] == metadata.metadata_input_hash(sources)
    assert len(payload["complete_document"]) == len(sources)
    assert len(request.prompt.encode()) < len(
        metadata.build_metadata_request(sources, [sources[0].chunk_id]).prompt.encode()
    )


def test_scoped_dates_can_differ_between_articles_and_from_general_commencement() -> (
    None
):
    from onyx.regulatory.document_intake import metadata_scopes

    sources, response = _scoped_response()
    sources.extend(
        [
            metadata.MetadataChunk(
                chunk_id="article_one",
                text="MADDE 1-\n(1) Amaç ve kapsam.",
                heading_path=[sources[0].heading_path[0], "MADDE 1"],
                chunk_type="paragraph",
            ),
            metadata.MetadataChunk(
                chunk_id="article_three",
                text="MADDE 3-\n(1) Ürünlerin kapsamı.",
                heading_path=[sources[0].heading_path[0], "MADDE 3"],
                chunk_type="paragraph",
            ),
        ]
    )
    entries = metadata.build_metadata_evidence_catalog(sources)

    def evidence(prefix: str) -> str:
        return next(
            f"e{i}"
            for i, entry in enumerate(entries)
            if entry.scope == "body" and entry.text.startswith(prefix)
        )

    response["covered_targets"] = list(range(len(sources)))
    response["assignments"].extend(
        [
            {
                "targets": [len(sources) - 2],
                "fields": [
                    {
                        "field": "validity_start_date",
                        "value": "2025-01-05",
                        "basis": "relative_publication",
                        "evidence_id": evidence("a) 1 inci,"),
                        "reference_evidence_id": evidence("5 Ocak"),
                    }
                ],
            },
            {
                "targets": [len(sources) - 1],
                "fields": [
                    {
                        "field": "validity_start_date",
                        "value": "2025-03-01",
                        "basis": "explicit",
                        "evidence_id": evidence("b) 3 üncü"),
                        "reference_evidence_id": "",
                    }
                ],
            },
        ]
    )
    # Appending sources preserves the existing frozen catalog aliases.
    results = metadata_scopes.expand_scoped_metadata_response(
        json.dumps(response), chunks=sources
    )
    starts = {
        target.chunk_id: fact.value
        for result in results
        for target in result.chunks
        for fact in target.fields
        if fact.field == "validity_start_date"
    }
    assert starts["article_one"] == "2025-01-05"
    assert starts["article_three"] == "2025-03-01"
    assert "2025-07-01" in starts.values()


def test_scoped_gateway_replays_its_valid_receipt_without_new_provider_work() -> None:
    from typing import cast

    from onyx.regulatory.document_intake.metadata_scopes import (
        build_scoped_metadata_request,
    )
    from onyx.regulatory.document_intake.metadata_scopes_job import (
        extract_scoped_metadata_document,
    )
    from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway
    from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
    from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import (
        _CachedGateway,
    )

    sources, response = _scoped_response()
    gateway = _CachedGateway(
        lambda request: _envelope(request, "STOP", json.dumps(response))
    )

    def extract() -> list[tuple[str, metadata.LegalMetadataResult]]:
        return extract_scoped_metadata_document(
            gateway=cast(VertexBatchGateway, gateway),
            chunks=sources,
            identity_prefix="sgm4-frozen",
            heartbeat=lambda: None,
        )

    result = extract()
    assert len(gateway.new_requests) == 1
    assert result == extract()
    assert len(gateway.new_requests) == 1
    assert {request_hash for request_hash, _result in result} == {
        build_scoped_metadata_request(sources).request_hash
    }


def test_split_fallback_retains_valid_parent_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.document_intake import metadata_scopes_job as job

    chunks = [
        metadata.MetadataChunk(
            chunk_id=f"c{i}",
            text="Bu Yönetmelik 1/1/2025 tarihinde yürürlüğe girer."
            if i == 0
            else "Hüküm uygulanır.",
            heading_path=[],
            chunk_type="paragraph",
        )
        for i in range(10)
    ]
    evidence = next(
        f"e{i}"
        for i, entry in enumerate(metadata.build_metadata_evidence_catalog(chunks))
        if entry.scope == "body" and entry.source_chunk_id == "c0"
    )
    parent_hashes = []

    def respond(**kwargs: Any) -> tuple[str, bool]:
        request = kwargs["request"]
        payload = json.loads(request.prompt)
        targets = payload["target_indices"]
        parent = len(targets) == 10
        if parent:
            parent_hashes.append(request.request_hash)
        return json.dumps(
            {
                "covered_targets": targets,
                "shared_fields": [],
                "assignments": [
                    {
                        "targets": targets
                        if parent
                        else [i for i in targets if i != 0],
                        "fields": [
                            {
                                "field": "validity_start_date",
                                "value": "2025-01-01",
                                "evidence_id": evidence,
                                "basis": "explicit",
                                "reference_evidence_id": "",
                            }
                        ],
                    }
                ],
                "warnings": [],
                **({"invalid_neighbor": True} if parent else {}),
            }
        ), False

    # Parent has valid facts but a rejected envelope. Children must preserve them.
    monkeypatch.setattr(job, "_run_metadata_request", respond)
    from typing import cast

    from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway

    results = job.extract_scoped_metadata_document(
        gateway=cast(VertexBatchGateway, object()),
        chunks=chunks,
        identity_prefix="split-test",
        heartbeat=lambda: None,
    )
    facts = {
        target.chunk_id: target.fields
        for _, result in results
        for target in result.chunks
    }
    assert len(parent_hashes) == 3
    assert len(facts) == 10
    assert facts["c0"][0].retained_request_hash in parent_hashes


@pytest.mark.parametrize(
    "defect", ["missing", "foreign", "duplicate", "conflict", "body_scope"]
)
def test_scoped_extraction_rejects_incomplete_conflicting_or_wrong_scope_facts(
    defect: str,
) -> None:
    from onyx.regulatory.document_intake import metadata_scopes

    sources, response = _scoped_response()
    if defect == "missing":
        response["covered_targets"].pop()
    elif defect == "foreign":
        response["assignments"][0]["targets"] = [len(sources)]
    elif defect == "duplicate":
        response["covered_targets"].append(0)
    elif defect == "conflict":
        import copy

        other = copy.deepcopy(response["assignments"][0])
        other["fields"][0]["value"] = "2030-01-01"
        response["assignments"].append(other)
    else:
        entries = metadata.build_metadata_evidence_catalog(sources)
        selected = next(
            i
            for i, entry in enumerate(entries)
            if entry.scope == "heading" and "1/7/2025" in entry.text
        )
        response["assignments"][0]["fields"][0]["evidence_id"] = f"e{selected}"
    with pytest.raises(ValueError):
        metadata_scopes.expand_scoped_metadata_response(
            json.dumps(response), chunks=sources
        )
