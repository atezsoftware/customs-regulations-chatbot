"""A deadline for one regime cannot expire a continuing rule in the same target."""

import json
from pathlib import Path

import pytest

from onyx.regulatory.document_intake import metadata


def cached_case() -> tuple[list[metadata.MetadataChunk], str]:
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/kayin_cached_metadata_target_end.json"
        ).read_text()
    )
    return [
        metadata.MetadataChunk.model_validate(c) for c in fixture["chunks"]
    ], fixture["response"]


def test_cached_mixed_regime_has_field_specific_feedback_and_preserves_other_facts() -> (
    None
):
    chunks, response = cached_case()
    targets = [c["chunk_id"] for c in json.loads(response)["chunks"]]
    with pytest.raises(ValueError):
        metadata.validate_metadata_transport_response(
            response, chunks=chunks, target_ids=targets
        )
    request = metadata.build_metadata_request(chunks, targets)
    repair = metadata.build_metadata_repair_request(
        request, response, validation_category="invalid_coverage_or_evidence"
    )
    original = json.loads(request.prompt)
    payload = json.loads(repair.prompt)
    for key in (
        "complete_document",
        "target_chunk_ids",
        "evidence_catalog",
        "input_sha256",
    ):
        assert payload[key] == original[key]
    feedback = payload["validation_feedback"]["target_end_feedback"]
    assert feedback["errors"][0]["error"] == "target_end_not_proven"
    assert feedback["errors"][0]["field"] == "validity_end_date"
    corrected = json.loads(response)
    for target in corrected["chunks"]:
        target["fields"] = [
            f for f in target["fields"] if f["field"] != "validity_end_date"
        ]
    corrected["warnings"] = []
    result = metadata.validate_metadata_transport_response(
        json.dumps(corrected), chunks=chunks, target_ids=targets
    )
    result = metadata.retain_validated_metadata_facts(
        result, response, request.request_hash, chunks, targets
    )
    assert sum(len(c.fields) for c in result.chunks) == 9
    assert all(f.field != "validity_end_date" for c in result.chunks for f in c.fields)
    assert any("whole target" in w for w in result.warnings)
    assert "01.01.2027" in chunks[-1].text


def test_field_feedback_does_not_accept_the_same_invalid_expiry() -> None:
    chunks, response = cached_case()
    targets = [c["chunk_id"] for c in json.loads(response)["chunks"]]
    request = metadata.build_metadata_request(chunks, targets)
    metadata.build_metadata_repair_request(
        request, response, validation_category="invalid_coverage_or_evidence"
    )
    with pytest.raises(ValueError):
        metadata.validate_metadata_transport_response(
            response, chunks=chunks, target_ids=targets
        )


def test_document_job_repairs_one_field_and_retains_nine_independent_facts() -> None:
    from tests.unit.onyx.regulatory.test_intake_metadata_budget import (
        _envelope,
        _Gateway,
        _run,
    )

    chunks, response = cached_case()

    def respond(request: metadata.VertexBatchRequest) -> dict:
        payload = json.loads(request.prompt)
        feedback = payload.get("validation_feedback")
        if feedback is None:
            return _envelope(request, "STOP", response)
        assert (
            feedback["target_end_feedback"]["errors"][0]["error"]
            == "target_end_not_proven"
        )
        corrected = {
            "chunks": [
                {"chunk_id": key, "fields": []} for key in payload["target_chunk_ids"]
            ],
            "warnings": [],
        }
        return _envelope(request, "STOP", json.dumps(corrected))

    gateway = _Gateway(respond)
    results = _run(gateway, chunks)
    assert len(gateway.requests) == 2 and len(results) == 1
    result = results[0][1]
    assert sum(len(c.fields) for c in result.chunks) == 9
    assert all(f.field != "validity_end_date" for c in result.chunks for f in c.fields)
    assert any("whole target" in warning for warning in result.warnings)


def test_omission_warning_preserves_all_32_existing_model_warnings() -> None:
    chunks, response = cached_case()
    original = json.loads(response)
    targets = [c["chunk_id"] for c in original["chunks"]]
    corrected = json.loads(response)
    for target in corrected["chunks"]:
        target["fields"] = [
            f for f in target["fields"] if f["field"] != "validity_end_date"
        ]
    model_warnings = [f"Existing diagnostic {index}" for index in range(32)]
    corrected["warnings"] = model_warnings
    result = metadata.validate_metadata_transport_response(
        json.dumps(corrected), chunks=chunks, target_ids=targets
    )
    request = metadata.build_metadata_request(chunks, targets)
    retained = metadata.retain_validated_metadata_facts(
        result, response, request.request_hash, chunks, targets
    )
    assert len(retained.warnings) == 32
    assert all(any(w in stored for stored in retained.warnings) for w in model_warnings)
    assert any("whole target" in w for w in retained.warnings)
    assert sum(len(c.fields) for c in retained.chunks) == 9
