"""Fresh and cached semantic feedback retain exact failed-request provenance."""

import json
from typing import Any

import pytest

from onyx.regulatory.document_intake import metadata
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
from tests.unit.onyx.regulatory.test_intake_metadata_cached_repair import _case, _run
from tests.unit.onyx.regulatory.test_intake_metadata_output_pressure import (
    _seed,
    _with_usage,
)
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import _CachedGateway
from tests.unit.onyx.regulatory.test_intake_metadata_value_recovery import (
    _entry,
    _fact,
    _response,
)


def test_body_scope_feedback_defaults_keep_legacy_hash_and_actual_64k_origin() -> None:
    _sources, base, response, _repair, _invalid, _valid = _case("scoped")
    generic = metadata.build_metadata_repair_request(
        base, response, validation_category="invalid_coverage_or_evidence"
    )
    reserved = metadata.build_metadata_output_reserve_request(generic)
    assert reserved is not None
    legacy = metadata.build_metadata_body_scope_repair_request(
        base, response, cached_repair_response=response
    )
    assert legacy is not None
    assert (
        legacy.request_hash
        == "fca31f378e8e0e3e09bed838734e110a73c4cdc47b0b543c8e539b59ef46c9d2"
    )
    explicit = metadata.build_metadata_body_scope_repair_request(
        base,
        response,
        cached_repair_response=response,
        cached_repair_request_hash=generic.request_hash,
    )
    assert explicit == legacy
    actual = metadata.build_metadata_body_scope_repair_request(
        base,
        response,
        cached_repair_response=response,
        cached_repair_request_hash=reserved.request_hash,
    )
    assert actual is not None
    assert (
        json.loads(actual.prompt)["validation_feedback"]["cached_repair_request_hash"]
        == reserved.request_hash
    )
    assert actual.request_hash != legacy.request_hash
    assert actual.generation_config == legacy.generation_config


def test_body_scope_origin_is_ignored_without_a_previous_repair_response() -> None:
    _sources, base, response, _repair, _invalid, _valid = _case("scoped")
    assert metadata.build_metadata_body_scope_repair_request(
        base, response, cached_repair_request_hash="a" * 64
    ) == metadata.build_metadata_body_scope_repair_request(base, response)


@pytest.mark.parametrize("invalid_hash", ["forged", "g" * 64])
def test_body_scope_repair_rejects_invalid_explicit_provenance(
    invalid_hash: str,
) -> None:
    _sources, base, response, _repair, _invalid, _valid = _case("scoped")
    with pytest.raises(ValueError, match="actual request hash"):
        metadata.build_metadata_body_scope_repair_request(
            base,
            response,
            cached_repair_response=response,
            cached_repair_request_hash=invalid_hash,
        )


@pytest.mark.parametrize("cached_primary", [False, True])
def test_fresh_and_cached_generic_failures_use_identical_narrow_feedback(
    cached_primary: bool,
) -> None:
    sources, base, invalid_base, primary, invalid_primary, valid = _case("generic")
    targeted = metadata.build_metadata_catalog_recovery_request(
        primary, invalid_primary
    )
    assert targeted is not None
    gateway = _CachedGateway(
        lambda request: _envelope(
            request,
            "STOP",
            invalid_primary if request.request_hash == primary.request_hash else valid,
        )
    )
    gateway.seed(base, invalid_base)
    if cached_primary:
        gateway.seed(primary, invalid_primary)
    before = dict(gateway.lines)
    result = _run(gateway, sources)
    assert result[0][0] == targeted.request_hash
    assert len(gateway.new_requests) == (1 if cached_primary else 2)
    assert gateway.new_requests[-1] == targeted
    assert (
        targeted.system_instruction == base.system_instruction
        and targeted.generation_config == base.generation_config
    )
    assert (
        json.loads(targeted.prompt)["complete_document"]
        == json.loads(base.prompt)["complete_document"]
    )
    assert all(gateway.lines[key] == value for key, value in before.items())
    after = dict(gateway.lines)
    assert _run(gateway, sources) == result and gateway.lines == after
    assert len(gateway.new_requests) == (1 if cached_primary else 2)


def test_one_narrow_feedback_failure_is_terminal_and_cached_retry_cannot_recurse() -> (
    None
):
    sources, base, invalid_base, _primary, invalid_primary, _valid = _case("generic")
    gateway = _CachedGateway(
        lambda request: _envelope(request, "STOP", invalid_primary)
    )
    gateway.seed(base, invalid_base)
    with pytest.raises(ValueError, match="after bounded repair attempts"):
        _run(gateway, sources)
    assert len(gateway.new_requests) == 2
    before = dict(gateway.lines)
    with pytest.raises(ValueError, match="after bounded repair attempts"):
        _run(gateway, sources)
    assert len(gateway.new_requests) == 2 and gateway.lines == before


def test_valid_cached_primary_ack_never_sends_narrow_feedback() -> None:
    sources, base, invalid_base, primary, _invalid_primary, valid = _case("generic")
    gateway = _CachedGateway(
        lambda _request: pytest.fail("Valid cached primary must be reused")
    )
    gateway.seed(base, invalid_base)
    gateway.seed(primary, valid)
    assert _run(gateway, sources)[0][0] == primary.request_hash
    assert gateway.new_requests == []


@pytest.mark.parametrize(
    "targeted_shape_error,legacy_state",
    [(False, None), (True, None), (False, "invalid"), (False, "valid")],
)
def test_reserved_64k_semantic_failure_uses_actual_hash_for_body_feedback(
    targeted_shape_error: bool,
    legacy_state: str | None,
) -> None:
    sources, base, heading_date, _scoped, _invalid, valid = _case("scoped")
    target = sources[0].chunk_id
    title = sources[0].heading_path[0]
    invalid_title = _response(
        [
            _fact(
                "document_title",
                title.replace("KULLANILACAK", "KULLANILACAL"),
                _entry(sources, title, "heading").evidence_id,
            )
        ],
        target,
    )
    primary = metadata.build_metadata_repair_request(
        base, invalid_title, validation_category="value_not_covered_by_evidence"
    )
    reserved = metadata.build_metadata_output_reserve_request(primary)
    assert reserved is not None
    legacy = metadata.build_metadata_body_scope_repair_request(
        base, invalid_title, cached_repair_response=heading_date
    )
    assert legacy is not None

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        feedback = json.loads(request.prompt)["validation_feedback"]
        if "body_scope_feedback" in feedback:
            assert feedback["cached_repair_request_hash"] == reserved.request_hash
            if targeted_shape_error:
                return _with_usage(
                    _envelope(request, "STOP", '{"chunks":[],"warnings":[]}')
                )
            return _envelope(request, "STOP", valid)
        assert request.request_hash == reserved.request_hash
        return _with_usage(_envelope(request, "STOP", heading_date))

    gateway = _CachedGateway(respond)
    gateway.seed(base, invalid_title)
    _seed(
        gateway,
        primary,
        '{"chunks":[],"warnings":[]}',
        {"thoughtsTokenCount": 31455, "candidatesTokenCount": 8},
    )
    if legacy_state is not None:
        gateway.seed(reserved, heading_date)
        gateway.seed(legacy, valid if legacy_state == "valid" else heading_date)
    seeded = dict(gateway.lines)
    result = None
    if targeted_shape_error:
        with pytest.raises(
            ValueError, match="invalid structured output for one target"
        ):
            _run(gateway, sources)
    else:
        result = _run(gateway, sources)
    expected_new = 2 if legacy_state is None else 0 if legacy_state == "valid" else 1
    assert len(gateway.new_requests) == expected_new
    assert all(
        request.generation_config == reserved.generation_config
        for request in gateway.new_requests
    )
    assert all(gateway.lines[key] == value for key, value in seeded.items())
    if legacy_state == "valid":
        assert result is not None and result[0][0] == legacy.request_hash
    else:
        targeted = gateway.new_requests[-1]
        assert targeted.generation_config == reserved.generation_config
        assert targeted.system_instruction == base.system_instruction
        assert (
            json.loads(targeted.prompt)["complete_document"]
            == json.loads(base.prompt)["complete_document"]
        )
    before = dict(gateway.lines)
    if targeted_shape_error:
        with pytest.raises(
            ValueError, match="invalid structured output for one target"
        ):
            _run(gateway, sources)
    else:
        assert _run(gateway, sources) == result
    assert len(gateway.new_requests) == expected_new and gateway.lines == before


def test_unsupported_explicit_date_error_is_terminal_without_semantic_retry_loop() -> (
    None
):
    sources, base, invalid_base, primary, _invalid, _valid = _case("scoped")
    quote = "(1) Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer."
    invalid_date = _response(
        [
            _fact(
                "validity_start_date",
                "1900-01-01",
                _entry(sources, quote, "body").evidence_id,
            )
        ],
        sources[0].chunk_id,
    )
    gateway = _CachedGateway(lambda request: _envelope(request, "STOP", invalid_date))
    gateway.seed(base, invalid_base)
    with pytest.raises(ValueError, match="after bounded repair attempts"):
        _run(gateway, sources)
    assert len(gateway.new_requests) == 1
    assert gateway.new_requests[0].request_hash == primary.request_hash
