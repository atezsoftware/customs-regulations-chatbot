from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import JsonValue

from onyx.configs import app_configs
from onyx.legal_review import jev
from onyx.legal_review.jev import JevReviewer
from onyx.legal_review.models import ReviewCheck


def _checks() -> list[ReviewCheck]:
    return [
        ReviewCheck(
            id="I1.evidence",
            instructions="Does I1 materially lack original evidence for refund conditions?",
            issue_id="I1",
            dimension="exceptions_exemptions",
        ),
        ReviewCheck(
            id="global.coverage",
            instructions="Does the answer omit a material outcome in the full request?",
        ),
    ]


def _response() -> dict[str, JsonValue]:
    return {
        "model": "jev-2026-09-15",
        "answers": {
            "global.coverage": {"type": "noul", "noul": 0.04},
            "I1.evidence": {"type": "noul", "noul": 0.5},
        },
        "usage": {"input_tokens": 120, "output_tokens": 5},
    }


@pytest.mark.parametrize("stage", ["early_evidence", "draft"])
def test_wire_contract_preserves_full_packet_and_code_owned_flags(stage: str) -> None:
    checks = _checks()
    state: dict[str, JsonValue] = {
        "stage": stage,
        "request": "İlk ithalat vergilerinin iadesi ile tamirden dönüşü ayrı değerlendir.",
        "original_evidence": [
            {
                "citation": 1,
                "text": "A ve B birlikte gerekir. " * 500,
                "source_id": "original-law",
                "metadata": {"valid_from": "2024-01-01", "legal_status": "in_force"},
            }
        ],
    }
    before_request = MagicMock()
    check_active = MagicMock()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == httpx.URL("https://api.typesafe.ai/v1/systemone")
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer test-typesafe-key"
        payload = json.loads(request.content)
        assert payload["model"] == "jev-latest"
        assert payload["state"] == state
        assert set(payload["questions"]) == {check.id for check in checks}
        for check in checks:
            question = payload["questions"][check.id]
            assert question["type"] == "noul"
            assert question["instructions"].endswith(check.instructions)
        return httpx.Response(200, json=_response())

    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(handler),
        before_request=before_request,
        check_active=check_active,
    ).review(state, checks, timeout_seconds=5.0)

    assert result.completed is True
    assert result.failure_reason is None
    assert result.scores == {"I1.evidence": 0.5, "global.coverage": 0.04}
    assert result.flags == [checks[0]]
    assert result.flags[0] is checks[0]
    assert result.input_tokens == 120
    assert result.output_tokens == 5
    assert len(requests) == 1
    before_request.assert_called_once_with()
    assert check_active.call_count >= 2


@pytest.mark.parametrize(
    "replacement",
    [
        {"wrong": {"type": "noul", "noul": 0.0}},
        {"I1.evidence": {"type": "noul", "noul": 0.0}},
        {
            "I1.evidence": {"type": "noul", "noul": True},
            "global.coverage": {"type": "noul", "noul": 0.0},
        },
        {
            "I1.evidence": {"type": "noul", "noul": 1.01},
            "global.coverage": {"type": "noul", "noul": 0.0},
        },
        {
            "I1.evidence": {"type": "boolean", "noul": 0.0},
            "global.coverage": {"type": "noul", "noul": 0.0},
        },
        {
            "I1.evidence": {"type": "noul", "noul": float("nan")},
            "global.coverage": {"type": "noul", "noul": 0.0},
        },
    ],
)
def test_invalid_answer_contract_never_becomes_a_pass(
    replacement: dict[str, JsonValue],
) -> None:
    response = _response()
    response["answers"] = replacement
    body = json.dumps(response).encode("utf-8")
    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=body)
        ),
    ).review({}, _checks(), 5.0)

    assert result.completed is False
    assert result.failure_reason == "jev_invalid_review_response"
    assert not result.scores
    assert not result.flags


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", ""),
        ("model", None),
        ("model", "gpt-5-mini"),
        ("usage", {}),
        ("usage", {"input_tokens": -1, "output_tokens": 0}),
        ("usage", {"input_tokens": True, "output_tokens": 0}),
    ],
)
def test_invalid_metadata_never_becomes_a_pass(field: str, value: JsonValue) -> None:
    response = _response()
    response[field] = value
    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response)
        ),
    ).review({}, _checks(), 5.0)

    assert result.completed is False
    assert result.failure_reason == "jev_invalid_review_response"


def test_resolved_jev_identity_and_optional_output_usage_follow_protocol() -> None:
    response = _response()
    response["model"] = "jev1.13"
    response["usage"] = {"input_tokens": 120}
    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response)
        ),
    ).review({}, _checks(), 5.0)

    assert result.completed is True
    assert result.input_tokens == 120
    assert result.output_tokens == 0


def test_unknown_check_keeps_actual_usage_but_review_remains_incomplete() -> None:
    response = _response()
    response["answers"] = {"unknown": {"type": "noul", "noul": 0.0}}
    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response)
        ),
    ).review({}, _checks(), 5.0)

    assert result.completed is False
    assert result.input_tokens == 120
    assert result.output_tokens == 5
    assert result.scores == {}


def test_duplicate_response_ids_cannot_be_silently_overwritten() -> None:
    body = (
        b'{"model":"jev-latest","answers":{'
        b'"I1.evidence":{"type":"noul","noul":1.0},'
        b'"I1.evidence":{"type":"noul","noul":0.0},'
        b'"global.coverage":{"type":"noul","noul":0.0}},'
        b'"usage":{"input_tokens":120}}'
    )
    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=body)
        ),
    ).review({}, _checks(), 5.0)

    assert result.completed is False
    assert result.failure_reason == "jev_invalid_review_response"


def test_missing_credential_skips_http_and_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_configs, "TYPESAFE_API_KEY", None)
    handler = MagicMock()
    before_request = MagicMock()
    result = JevReviewer(
        transport=httpx.MockTransport(handler), before_request=before_request
    ).review({}, _checks(), 5.0)

    assert result.completed is False
    assert result.failure_reason == "jev_credential_unavailable"
    handler.assert_not_called()
    before_request.assert_not_called()


def test_oversized_packet_is_rejected_without_clipping_evidence() -> None:
    handler = MagicMock()
    before_request = MagicMock()
    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(handler),
        before_request=before_request,
    ).review({"original_evidence": "a" * 256_001}, _checks(), 5.0)

    assert result.completed is False
    assert result.failure_reason == "jev_review_packet_too_large"
    handler.assert_not_called()
    before_request.assert_not_called()


def test_duplicate_check_ids_are_rejected_before_http() -> None:
    handler = MagicMock()
    check = _checks()[0]
    result = JevReviewer(
        api_key="test-typesafe-key", transport=httpx.MockTransport(handler)
    ).review({}, [check, check], 5.0)

    assert result.completed is False
    assert result.failure_reason == "jev_invalid_checks"
    handler.assert_not_called()


@pytest.mark.parametrize("mode", ["redirect", "timeout", "oversized_response"])
def test_transport_failure_is_explicit_and_never_retried(mode: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if mode == "timeout":
            raise httpx.ReadTimeout("slow provider")
        if mode == "redirect":
            return httpx.Response(307, headers={"location": "https://other.invalid"})
        return httpx.Response(200, content=b" " * 256_001)

    result = JevReviewer(
        api_key="test-typesafe-key", transport=httpx.MockTransport(handler)
    ).review({}, _checks(), 5.0)

    assert result.completed is False
    assert result.failure_reason is not None
    assert len(requests) == 1


def test_request_budget_or_stop_callback_prevents_http() -> None:
    handler = MagicMock()
    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(handler),
        before_request=MagicMock(side_effect=RuntimeError("budget exhausted")),
    ).review({}, _checks(), 5.0)

    assert result.completed is False
    assert result.failure_reason == "jev_request_preflight_failed"
    handler.assert_not_called()


def test_response_chunks_obey_overall_review_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(jev, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class DelayedStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b" "
            now[0] = 6.0
            yield json.dumps(_response()).encode("utf-8")

    result = JevReviewer(
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=DelayedStream())
        ),
    ).review({}, _checks(), 5.0)

    assert result.completed is False
    assert result.failure_reason == "jev_review_timeout"
    assert not result.scores


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timeout_never_starts_http(timeout: float) -> None:
    handler = MagicMock()
    result = JevReviewer(
        api_key="test-typesafe-key", transport=httpx.MockTransport(handler)
    ).review({}, _checks(), timeout)

    assert result.completed is False
    assert result.failure_reason == "jev_invalid_timeout"
    handler.assert_not_called()
