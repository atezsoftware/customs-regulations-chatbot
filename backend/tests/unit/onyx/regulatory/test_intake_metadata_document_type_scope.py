"""Real Markdown intake evidence-scope failure, including durable invalid repairs."""

import json
from pathlib import Path

import pytest

from onyx.regulatory.document_intake import metadata
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
from tests.unit.onyx.regulatory.test_intake_metadata_cached_repair import _run
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import _CachedGateway


@pytest.mark.parametrize("cached_generic", [False, True])
@pytest.mark.parametrize("valid_repair", [False, True])
def test_document_type_scope_repair_keeps_strict_evidence_and_prior_facts(
    cached_generic: bool, valid_repair: bool
) -> None:
    frozen = json.loads(
        (
            Path(__file__).parent / "fixtures/md_annex_navigation/metadata_scope.json"
        ).read_text()
    )
    sources = [metadata.MetadataChunk.model_validate(row) for row in frozen["chunks"]]
    target = sources[0].chunk_id
    invalid = json.dumps(frozen["response"], ensure_ascii=False)
    base = metadata.build_metadata_request(sources, [target])
    generic = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )
    with pytest.raises(ValueError, match="requires body evidence"):
        metadata.validate_metadata_transport_response(
            invalid, chunks=sources, target_ids=[target]
        )
    body = next(
        entry
        for entry in metadata.build_metadata_evidence_catalog(sources)
        if entry.scope == "body" and entry.text == sources[0].text
    )
    valid_result = json.loads(invalid)
    for fact in valid_result["chunks"][0]["fields"]:
        if fact["field"] == "document_type":
            fact["evidence_id"] = body.evidence_id
    valid = json.dumps(valid_result, ensure_ascii=False)
    metadata.validate_metadata_transport_response(
        valid, chunks=sources, target_ids=[target]
    )

    def respond(request: VertexBatchRequest) -> dict[str, object]:
        feedback = json.loads(request.prompt)["validation_feedback"]
        assert "body_scope_feedback" in feedback
        scope = feedback["body_scope_feedback"]
        assert scope["invalid_scope_total"] == 1
        assert scope["errors"][0]["field"] == "document_type"
        return _envelope(request, "STOP", valid if valid_repair else invalid)

    gateway = _CachedGateway(respond)
    gateway.seed(base, invalid)
    if cached_generic:
        gateway.seed(generic, invalid)
    before = dict(gateway.lines)
    if valid_repair:
        result = _run(gateway, sources)
        facts = {fact.field: fact for fact in result[0][1].chunks[0].fields}
        assert facts["document_type"].evidence_quote == sources[0].text
        assert facts["article_no"].value == "1"
        assert facts["paragraph_no"].value == "1"
        assert facts["article_title"].value == "Amaç"
        after = dict(gateway.lines)
        assert _run(gateway, sources) == result
        assert gateway.lines == after
    else:
        with pytest.raises(ValueError, match="after bounded repair"):
            _run(gateway, sources)
    assert len(gateway.new_requests) == 1
    assert all(gateway.lines[key] == value for key, value in before.items())
    payload = json.loads(gateway.new_requests[0].prompt)
    original = json.loads(base.prompt)
    for key in (
        "complete_document",
        "evidence_catalog",
        "input_sha256",
        "target_chunk_ids",
    ):
        assert payload[key] == original[key]
