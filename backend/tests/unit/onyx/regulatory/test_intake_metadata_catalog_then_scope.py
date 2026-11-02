"""Known heading dates remain repairable after unknown catalog IDs are corrected."""

import hashlib
import json

import pytest

from onyx.regulatory.document_intake import metadata
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
from tests.unit.onyx.regulatory.test_intake_metadata_cached_repair import (
    _entry,
    _fact,
    _response,
    _run,
    _sources,
)
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import _CachedGateway


@pytest.mark.parametrize("cached_repairs", [False, True])
@pytest.mark.parametrize("valid_scope_response", [False, True])
def test_catalog_then_body_scope_is_bounded_and_preserves_prior_facts(
    cached_repairs: bool, valid_scope_response: bool
) -> None:
    sources = _sources("scoped")
    target = sources[0].chunk_id
    targets = [target]
    base = metadata.build_metadata_request(sources, targets)
    article_id = _entry(
        sources, "MADDE 5 - Taşımacılık faaliyetlerinde genel kurallar", "heading"
    )
    original = json.loads(
        _response(
            target,
            [
                _fact("article_no", "5", article_id),
                _fact("validity_start_date", "2022-01-01", "0" * 64),
            ],
        )
    )
    original["warnings"] = ["Preserve the earlier source warning."]
    invalid_base = json.dumps(original)
    primary = metadata.build_metadata_repair_request(
        base, invalid_base, validation_category="invalid_coverage_or_evidence"
    )
    unknown_date = _response(
        target, [_fact("validity_start_date", "2022-01-01", "1" * 64)]
    )
    catalog_recovery = metadata.build_metadata_catalog_recovery_request(
        primary, unknown_date
    )
    assert catalog_recovery is not None
    frozen_requests = (base.model_dump_json(), primary.model_dump_json())
    clause = "(1) Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer."
    heading_date = _response(
        target,
        [
            _fact(
                "validity_start_date", "2022-01-01", _entry(sources, clause, "heading")
            )
        ],
    )
    body_date = _response(
        target,
        [_fact("validity_start_date", "2022-01-01", _entry(sources, clause, "body"))],
    )
    for response in (invalid_base, unknown_date, heading_date):
        with pytest.raises(ValueError):
            metadata.validate_metadata_transport_response(
                response, chunks=sources, target_ids=targets
            )

    def respond(request: VertexBatchRequest) -> dict[str, object]:
        feedback = json.loads(request.prompt)["validation_feedback"]
        response = (
            (body_date if valid_scope_response else heading_date)
            if "body_scope_feedback" in feedback
            else (
                heading_date if "catalog_recovery_version" in feedback else unknown_date
            )
        )
        return _envelope(request, "STOP", response)

    gateway = _CachedGateway(respond)
    gateway.seed(base, invalid_base)
    if cached_repairs:
        gateway.seed(primary, unknown_date)
        gateway.seed(catalog_recovery, heading_date)
    before = dict(gateway.lines)
    result = None
    if valid_scope_response:
        result = _run(gateway, sources)
        facts = {fact.field: fact for fact in result[0][1].chunks[0].fields}
        assert facts["validity_start_date"].value == "2022-01-01"
        assert facts["validity_start_date"].source_chunk_id == sources[1].chunk_id
        assert facts["validity_start_date"].evidence_quote == clause
        assert facts["article_no"].value == "5"
        assert facts["article_no"].retained_request_hash == base.request_hash
        assert any(
            "earlier source warning" in warning for warning in result[0][1].warnings
        )
    else:
        with pytest.raises(ValueError, match="after bounded repair"):
            _run(gateway, sources)
    assert len(gateway.new_requests) == (1 if cached_repairs else 3)
    scope_request = gateway.new_requests[-1]
    payload = json.loads(scope_request.prompt)
    feedback = payload["validation_feedback"]
    assert feedback["catalog_recovery_version"] == 1
    assert feedback["body_scope_feedback"]["invalid_scope_total"] == 1
    assert feedback["cached_repair_request_hash"] == catalog_recovery.request_hash
    assert (
        feedback["cached_repair_response_sha256"]
        == hashlib.sha256(heading_date.encode()).hexdigest()
    )
    original_payload = json.loads(base.prompt)
    for key in (
        "complete_document",
        "evidence_catalog",
        "target_chunk_ids",
        "input_sha256",
    ):
        assert payload[key] == original_payload[key]
    assert scope_request.generation_config == base.generation_config
    assert frozen_requests == (base.model_dump_json(), primary.model_dump_json())
    assert all(gateway.lines[key] == value for key, value in before.items())
    after = dict(gateway.lines)
    if valid_scope_response:
        assert _run(gateway, sources) == result
    else:
        with pytest.raises(ValueError, match="after bounded repair"):
            _run(gateway, sources)
    assert len(gateway.new_requests) == (1 if cached_repairs else 3)
    assert gateway.lines == after
