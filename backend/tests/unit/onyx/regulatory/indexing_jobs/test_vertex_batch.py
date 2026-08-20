import json
import logging
from uuid import UUID

import pytest

from onyx.regulatory.indexing_jobs.legacy_vertex_batch import _legacy_jsonl_line
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchRequest,
    VertexBatchResultError,
    build_vertex_jsonl,
    parse_vertex_jsonl_output,
    vertex_batch_submission_key,
)

_FIRST_HASH = "27948fe650396b332c6e0b7073fbc4adf9cda51e33c0fc013fcd5b0be01a6f5f"
_SECOND_HASH = "5fa17eb7621a1e36adb1f59543cab32abd36396b57eac8f2482ba99d1b230e2f"


def _request_payload(prompt: str) -> dict[str, object]:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 256},
    }


def _output_line(
    prompt: str,
    *,
    text: str | None = "context",
    finish_reason: str = "STOP",
    status: str = "",
) -> str:
    response: dict[str, object]
    if text is None:
        response = {"candidates": []}
    else:
        response = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text}]},
                    "finishReason": finish_reason,
                }
            ]
        }
    return json.dumps(
        {
            "key": VertexBatchRequest(prompt=prompt).request_hash,
            "status": status,
            "response": response,
        }
    )


def test_build_vertex_jsonl_has_stable_hash_and_exact_request_shape() -> None:
    first = VertexBatchRequest(prompt="first prompt")
    repeated = VertexBatchRequest(prompt="first prompt")

    encoded = build_vertex_jsonl([first])

    assert first.request_hash == _FIRST_HASH
    assert repeated.request_hash == _FIRST_HASH
    assert json.loads(encoded) == {
        "key": _FIRST_HASH,
        "request": _request_payload("first prompt"),
    }
    assert encoded.endswith("\n")


def test_legacy_vertex_jsonl_keeps_gcs_batch_request_shape() -> None:
    encoded = _legacy_jsonl_line(VertexBatchRequest(prompt="first prompt"))

    assert json.loads(encoded) == {"request": _request_payload("first prompt")}


def test_build_vertex_jsonl_rejects_duplicate_request_hashes() -> None:
    with pytest.raises(VertexBatchContractError, match="duplicate request hash"):
        build_vertex_jsonl(
            [
                VertexBatchRequest(prompt="first prompt"),
                VertexBatchRequest(prompt="first prompt"),
            ]
        )


def test_submission_key_is_stable_across_request_order() -> None:
    first = VertexBatchRequest(prompt="first prompt")
    second = VertexBatchRequest(prompt="second prompt")

    assert vertex_batch_submission_key([first, second]) == (
        vertex_batch_submission_key([second, first])
    )
    assert vertex_batch_submission_key([first, second]).startswith(
        "regulatory-context-"
    )


def test_partial_parse_returns_only_correlated_available_results() -> None:
    results = parse_vertex_jsonl_output(
        _output_line("first prompt", text="first context"),
        {_FIRST_HASH, _SECOND_HASH},
        require_complete=False,
    )

    assert set(results) == {_FIRST_HASH}
    assert results[_FIRST_HASH].context == "first context"


def test_parse_correlates_shuffled_output_by_canonical_request_hash() -> None:
    output = "\n".join(
        [
            _output_line("second prompt", text="second context"),
            _output_line("first prompt", text="first context"),
        ]
    )

    results = parse_vertex_jsonl_output(output, {_FIRST_HASH, _SECOND_HASH})

    assert results[_FIRST_HASH].context == "first context"
    assert results[_SECOND_HASH].context == "second context"
    assert all(result.error is None for result in results.values())


@pytest.mark.parametrize(
    ("output", "expected_error"),
    [
        (_output_line("first prompt", text=None), VertexBatchResultError.EMPTY),
        (
            _output_line("first prompt", text="", finish_reason="SAFETY"),
            VertexBatchResultError.SAFETY,
        ),
        (
            _output_line("first prompt", status="Bad Request: hidden detail"),
            VertexBatchResultError.REMOTE_ERROR,
        ),
        (
            json.dumps(
                {
                    "key": _FIRST_HASH,
                    "error": {"code": 429, "message": "hidden detail"},
                }
            ),
            VertexBatchResultError.REMOTE_ERROR,
        ),
        (
            json.dumps(
                {
                    "key": _FIRST_HASH,
                    "status": "",
                    "response": {"candidates": [{"content": {"parts": [{}]}}]},
                }
            ),
            VertexBatchResultError.MALFORMED,
        ),
    ],
)
def test_parse_classifies_non_successful_outputs(
    output: str, expected_error: VertexBatchResultError
) -> None:
    result = parse_vertex_jsonl_output(output, {_FIRST_HASH})[_FIRST_HASH]

    assert result.context is None
    assert result.error is expected_error


@pytest.mark.parametrize(
    "output",
    [
        "\n".join([_output_line("first prompt"), _output_line("first prompt")]),
        _output_line("first prompt"),
        _output_line("second prompt"),
    ],
)
def test_parse_rejects_duplicate_missing_or_unexpected_hashes(output: str) -> None:
    with pytest.raises(VertexBatchContractError):
        parse_vertex_jsonl_output(output, {_FIRST_HASH, _SECOND_HASH})


def test_parse_rejects_json_without_a_correlatable_request() -> None:
    with pytest.raises(VertexBatchContractError, match="correlatable request"):
        parse_vertex_jsonl_output('{"response":{}}', {_FIRST_HASH})


def test_malformed_jsonl_failure_does_not_retain_the_full_line() -> None:
    with pytest.raises(VertexBatchContractError) as raised:
        parse_vertex_jsonl_output("LEGAL_JSONL_SENTINEL{", {_FIRST_HASH})

    assert "LEGAL_JSONL_SENTINEL" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_pure_contract_does_not_log_sensitive_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt = "PROMPT_SENTINEL credential-json VECTOR_SENTINEL"
    request = VertexBatchRequest(prompt=prompt)

    with caplog.at_level(logging.DEBUG):
        build_vertex_jsonl([request])
        parse_vertex_jsonl_output(
            _output_line(prompt, text="LEGAL_OUTPUT_SENTINEL"),
            {request.request_hash},
        )

    assert "PROMPT_SENTINEL" not in caplog.text
    assert "credential-json" not in caplog.text
    assert "VECTOR_SENTINEL" not in caplog.text
    assert "LEGAL_OUTPUT_SENTINEL" not in caplog.text


def test_submission_identity_is_scoped_to_tenant_job_prefix_and_attempt() -> None:
    request = VertexBatchRequest(prompt="same canonical request")
    job_a = UUID("00000000-0000-0000-0000-000000000001")
    job_b = UUID("00000000-0000-0000-0000-000000000002")

    identities = {
        vertex_batch_submission_key(
            [request],
            tenant_id="tenant-a",
            job_id=job_a,
            output_prefix="tenants/a/jobs/1",
            submission_attempt=1,
        ),
        vertex_batch_submission_key(
            [request],
            tenant_id="tenant-b",
            job_id=job_a,
            output_prefix="tenants/b/jobs/1",
            submission_attempt=1,
        ),
        vertex_batch_submission_key(
            [request],
            tenant_id="tenant-a",
            job_id=job_b,
            output_prefix="tenants/a/jobs/2",
            submission_attempt=1,
        ),
        vertex_batch_submission_key(
            [request],
            tenant_id="tenant-a",
            job_id=job_a,
            output_prefix="tenants/a/jobs/1",
            submission_attempt=2,
        ),
    }

    assert len(identities) == 4


def test_build_vertex_jsonl_enforces_utf8_byte_limit() -> None:
    requests = [
        VertexBatchRequest(prompt="ilk çğıöşü"),
        VertexBatchRequest(prompt="ikinci istek"),
    ]
    first_line_size = len(build_vertex_jsonl(requests[:1]).encode("utf-8"))

    assert build_vertex_jsonl(
        requests, max_bytes=first_line_size
    ) == build_vertex_jsonl(requests[:1])
