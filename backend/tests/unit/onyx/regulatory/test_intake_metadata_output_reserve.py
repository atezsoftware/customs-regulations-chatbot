"""Output reservation preserves the exact legacy request and reasoning settings."""

import pytest

from onyx.regulatory.document_intake import metadata
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from tests.unit.onyx.regulatory.test_intake_metadata_repair_feedback import (
    _continuation,
)


def test_reserve_changes_only_output_cap_and_keeps_legacy_cache_identity() -> None:
    request = metadata.build_metadata_request([_continuation()], ["continuation"])
    assert request.generation_config is not None
    original = request.model_dump_json()
    assert (
        request.request_hash
        == "6c50514e8071afbadbc68241cd3a06107138a4107e38f34963477ed81739235e"
    )
    reserved = metadata.build_metadata_output_reserve_request(request)
    assert reserved is not None and reserved.generation_config is not None
    assert reserved.prompt == request.prompt
    assert reserved.system_instruction == request.system_instruction
    assert reserved.generation_config == {
        **request.generation_config,
        "maxOutputTokens": 65536,
    }
    assert reserved.generation_config["thinkingConfig"] == {"thinkingLevel": "medium"}
    assert request.model_dump_json() == original
    assert reserved.request_hash != request.request_hash
    assert metadata.build_metadata_output_reserve_request(request) == reserved


def test_reserved_request_cannot_chain_and_missing_configuration_is_unsupported() -> (
    None
):
    request = metadata.build_metadata_request([_continuation()], ["continuation"])
    reserved = metadata.build_metadata_output_reserve_request(request)
    assert reserved is not None
    assert metadata.build_metadata_output_reserve_request(reserved) is None
    assert (
        metadata.build_metadata_output_reserve_request(
            VertexBatchRequest(prompt="source")
        )
        is None
    )


@pytest.mark.parametrize("cap", [16384, 65536, "32768", 32768.0, None])
def test_only_an_integer_legacy_cap_can_reserve_output(
    cap: int | str | float | None,
) -> None:
    request = VertexBatchRequest(
        prompt="source", generation_config={"maxOutputTokens": cap}
    )
    assert metadata.build_metadata_output_reserve_request(request) is None
