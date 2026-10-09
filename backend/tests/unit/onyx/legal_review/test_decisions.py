from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import JsonValue

from onyx.legal_review import decisions
from onyx.legal_review.decisions import DecisionsReviewer
from onyx.legal_review.models import ReviewCheck


def _checks() -> list[ReviewCheck]:
    return [
        ReviewCheck(
            id="I1:kanıt/⟦1⟧",
            instructions="Does I1 materially lack original evidence for refund conditions?",
            issue_id="I1",
            dimension="exceptions_and_exemptions",
        ),
        ReviewCheck(
            id="global.coverage",
            instructions="Does the answer omit a material outcome in the full request?",
        ),
    ]


def _response() -> dict[str, JsonValue]:
    return {
        "model": "gpt-6-luna",
        "answers": [
            {"type": "predicate", "name": "q000002", "probability": 0.04},
            {"type": "predicate", "name": "q000001", "probability": 0.5},
        ],
        "usage": {
            "input_tokens": 120,
            "output_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 120,
        },
    }


def _review(response: dict[str, JsonValue]) -> decisions.ReviewResult:
    return DecisionsReviewer(
        api_key="test-openai-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=response)
        ),
    ).review({}, _checks(), 5.0)


@pytest.mark.parametrize("stage", ["early_evidence", "draft"])
def test_complete_state_is_shared_once_and_wire_names_bind_code_owned_checks(
    stage: str,
) -> None:
    checks = _checks()
    state: dict[str, JsonValue] = {
        "stage": stage,
        "request": "İlk ithalat vergilerinin iadesi ile tamirden dönüşü ayrı değerlendir.",
        "original_evidence": [
            {
                "citation": 1,
                "passages": [
                    {"span_number": 1, "text": "A ve B birlikte gerekir. " * 250},
                    {"span_number": 2, "text": "İstisna birlikte uygulanır. " * 250},
                ],
                "source_id": "original-law",
                "metadata": {"legal_status": "in_force"},
            }
        ],
        "requirements": [{"supports": [{"citation": 1, "span_number": 2}]}],
    }
    before_request = MagicMock()
    check_active = MagicMock()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == httpx.URL("https://api.openai.com/v1/decisions")
        assert request.method == "POST"
        assert request.headers["authorization"] == "Bearer test-openai-key"
        payload = json.loads(request.content)
        assert set(payload) == {"model", "input", "questions"}
        assert payload["model"] == "gpt-6-luna"
        assert isinstance(payload["input"], str)
        assert json.loads(payload["input"]) == state
        assert request.content.count(b"original-law") == 1
        assert [question["name"] for question in payload["questions"]] == [
            "q000001",
            "q000002",
        ]
        for check, question in zip(checks, payload["questions"], strict=True):
            assert question["type"] == "predicate"
            assert question["instructions"].endswith(check.instructions)
            assert "citation and span_number" in question["instructions"]
            assert "complete canonical original" in question["instructions"]
        return httpx.Response(
            200, json=_response(), headers={"x-request-id": "req_test"}
        )

    result = DecisionsReviewer(
        api_key="test-openai-key",
        transport=httpx.MockTransport(handler),
        before_request=before_request,
        check_active=check_active,
    ).review(state, checks, 5.0)

    assert result.completed is True
    assert result.scores == {checks[0].id: 0.5, checks[1].id: 0.04}
    assert result.flags == [checks[0]]
    assert result.flags[0] is checks[0]
    assert result.input_tokens == 120
    assert result.output_tokens == 0
    assert result.http_status == 200
    assert result.request_id == "req_test"
    assert len(requests) == 1
    before_request.assert_called_once_with()
    assert check_active.call_count >= 3


def test_all_checks_are_sent_in_one_request_without_question_clipping() -> None:
    checks = [
        ReviewCheck(
            id=f"I{index}.dimension", instructions="Does evidence lack support?"
        )
        for index in range(41)
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        assert len(payload["questions"]) == len(checks)
        return httpx.Response(
            200,
            json={
                "model": "gpt-6-luna",
                "answers": [
                    {"type": "predicate", "name": question["name"], "probability": 0.0}
                    for question in payload["questions"]
                ],
                "usage": {"input_tokens": 900, "output_tokens": 0},
            },
        )

    result = DecisionsReviewer(
        api_key="test-key", transport=httpx.MockTransport(handler)
    ).review({}, checks, 5.0)
    assert result.completed
    assert len(result.scores) == 41
    assert len(requests) == 1


@pytest.mark.parametrize(
    "probability", [True, "0.1", None, -0.01, 1.01, float("nan"), float("inf")]
)
def test_probabilities_are_strict_finite_and_bounded(probability: JsonValue) -> None:
    response = _response()
    response["answers"] = [
        {"type": "predicate", "name": "q000001", "probability": probability},
        {"type": "predicate", "name": "q000002", "probability": 0.0},
    ]
    body = json.dumps(response).encode()
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=body)
        ),
    ).review({}, _checks(), 5.0)
    assert not result.completed
    assert result.failure_reason == "openai_decision_invalid_review_response"
    assert result.scores == {} and result.flags == []
    assert result.input_tokens == 120


@pytest.mark.parametrize(
    "answers",
    [
        [],
        [{"type": "predicate", "name": "q000001", "probability": 0.0}],
        [
            {"type": "predicate", "name": "unknown", "probability": 0.0},
            {"type": "predicate", "name": "q000002", "probability": 0.0},
        ],
        [
            {"type": "predicate", "name": "q000001", "probability": 0.0},
            {"type": "predicate", "name": "q000001", "probability": 0.0},
        ],
        [
            {"type": "predicate", "name": None, "probability": 0.0},
            {"type": "predicate", "name": "q000002", "probability": 0.0},
        ],
        [
            {"type": "choice", "name": "q000001", "probability": 0.0},
            {"type": "predicate", "name": "q000002", "probability": 0.0},
        ],
    ],
)
def test_answer_inventory_must_exactly_match_unique_host_wire_names(
    answers: JsonValue,
) -> None:
    response = _response()
    response["answers"] = answers
    result = _review(response)
    assert not result.completed
    assert result.failure_reason == "openai_decision_invalid_review_response"
    assert result.input_tokens == 120
    assert result.scores == {} and result.flags == []


def test_refusal_preserves_usage_but_never_publishes_partial_scores() -> None:
    response = _response()
    response["answers"] = [
        {"type": "refusal", "name": "q000001"},
        {"type": "predicate", "name": "q000002", "probability": 0.0},
    ]
    result = _review(response)
    assert not result.completed
    assert result.failure_reason == "openai_decision_refusal"
    assert result.input_tokens == 120
    assert result.scores == {} and result.flags == []


@pytest.mark.parametrize(
    "model", ["gpt-5-mini", "gpt-6-luna-2026-10-06", "openai/gpt-6-luna", None]
)
def test_only_the_pinned_model_response_is_accepted(model: JsonValue) -> None:
    response = _response()
    response["model"] = model
    result = _review(response)
    assert not result.completed
    assert result.input_tokens == 120


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": True, "output_tokens": 0},
        {"input_tokens": 120, "output_tokens": -1},
        {"input_tokens": 120},
    ],
)
def test_invalid_usage_prevents_success_and_retains_independently_usable_counters(
    usage: JsonValue,
) -> None:
    response = _response()
    response["usage"] = usage
    result = _review(response)
    assert not result.completed
    assert result.failure_reason == "openai_decision_invalid_review_response"
    assert result.input_tokens == (
        120 if isinstance(usage, dict) and usage.get("input_tokens") == 120 else 0
    )
    assert result.output_tokens == 0


def test_duplicate_json_keys_do_not_overwrite_answer_fields() -> None:
    body = b'{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"q000001","probability":1,"probability":0}],"usage":{"input_tokens":120,"output_tokens":0}}'
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=body)
        ),
    ).review({}, _checks(), 5.0)
    assert not result.completed
    assert result.failure_reason == "openai_decision_invalid_review_response"


@pytest.mark.parametrize("credential", [None, "", " \t"])
def test_no_credential_means_no_http_or_quota_admission(credential: str | None) -> None:
    handler, before_request = MagicMock(), MagicMock()
    result = DecisionsReviewer(
        api_key=credential,
        transport=httpx.MockTransport(handler),
        before_request=before_request,
    ).review({}, _checks(), 5.0)
    assert result.failure_reason == "openai_decision_credential_unavailable"
    handler.assert_not_called()
    before_request.assert_not_called()


@pytest.mark.parametrize(
    "state",
    [
        {"original_evidence": "a" * 256_001},
        {"original_evidence": "ğ" * 150_000},
        {"invalid": float("nan")},
    ],
)
def test_packet_guard_precedes_quota_and_does_not_clip_sources(
    state: dict[str, JsonValue],
) -> None:
    handler, before_request = MagicMock(), MagicMock()
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        before_request=before_request,
    ).review(state, _checks(), 5.0)
    assert not result.completed
    handler.assert_not_called()
    before_request.assert_not_called()


@pytest.mark.parametrize(
    "checks",
    [[], [_checks()[0], _checks()[0]], [ReviewCheck(id=" ", instructions="Check.")]],
)
def test_invalid_check_inventory_precedes_http_and_quota(
    checks: list[ReviewCheck],
) -> None:
    handler, before_request = MagicMock(), MagicMock()
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        before_request=before_request,
    ).review({}, checks, 5.0)
    assert result.failure_reason == "openai_decision_invalid_checks"
    handler.assert_not_called()
    before_request.assert_not_called()


def test_admission_or_stop_failure_prevents_http() -> None:
    handler = MagicMock()
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        before_request=MagicMock(side_effect=RuntimeError("budget exhausted")),
    ).review({}, _checks(), 5.0)
    assert result.failure_reason == "openai_decision_request_preflight_failed"
    handler.assert_not_called()


def test_initial_cancellation_prevents_quota_admission() -> None:
    handler, before_request = MagicMock(), MagicMock()
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        before_request=before_request,
        check_active=MagicMock(side_effect=RuntimeError("stopped")),
    ).review({}, _checks(), 5.0)
    assert result.failure_reason == "openai_decision_request_preflight_failed"
    handler.assert_not_called()
    before_request.assert_not_called()


@pytest.mark.parametrize(
    "mode", ["redirect", "timeout", "oversized_response", "malformed_json"]
)
def test_failed_transport_never_retries_or_redirects(mode: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if mode == "timeout":
            raise httpx.ReadTimeout("slow provider")
        if mode == "redirect":
            return httpx.Response(307, headers={"location": "https://other.invalid"})
        if mode == "oversized_response":
            return httpx.Response(200, content=b" " * 256_001)
        return httpx.Response(200, content=b"not JSON")

    result = DecisionsReviewer(
        api_key="test-key", transport=httpx.MockTransport(handler)
    ).review({}, _checks(), 5.0)
    assert not result.completed
    assert result.failure_reason is not None
    assert len(requests) == 1


def test_http_error_exposes_only_bounded_safe_identifiers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-test-sensitive-key"
    result = DecisionsReviewer(
        api_key=secret,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={
                    "error": {
                        "message": f"{secret}: PRIVATE SOURCE TEXT",
                        "code": "invalid_request_error",
                        "param": "questions[0].name",
                    },
                    "usage": {"input_tokens": 42, "output_tokens": 0},
                },
                headers={"x-request-id": "req_example"},
            )
        ),
    ).review({"request": "PRIVATE SOURCE TEXT"}, _checks(), 5.0)
    assert not result.completed
    assert result.failure_reason == "openai_decision_review_http_error"
    assert result.http_status == 400
    assert result.error_code == "invalid_request_error"
    assert result.error_param == "questions[0].name"
    assert result.request_id == "req_example"
    assert result.input_tokens == 42
    assert secret not in caplog.text
    assert "PRIVATE SOURCE TEXT" not in caplog.text
    assert secret not in result.model_dump_json()
    assert "PRIVATE SOURCE TEXT" not in result.model_dump_json()


def test_arbitrary_error_strings_and_request_headers_are_not_diagnostics() -> None:
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                401,
                json={
                    "error": {
                        "code": "PRIVATE SOURCE TEXT",
                        "param": "sk-sensitive-key",
                    }
                },
                headers={"x-request-id": "sk-sensitive-key"},
            )
        ),
    ).review({}, _checks(), 5.0)
    assert result.http_status == 401
    assert (
        result.error_code is None
        and result.error_param is None
        and result.request_id is None
    )


def test_explicit_error_in_success_status_never_overrides_failure() -> None:
    response = _response()
    response["error"] = {"code": "invalid_request_error", "param": "model"}
    result = _review(response)
    assert not result.completed
    assert result.failure_reason == "openai_decision_invalid_review_response"
    assert result.input_tokens == 120
    assert result.error_code == "invalid_request_error"
    assert result.scores == {}


def test_response_stream_checks_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(decisions, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class DelayedStream(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b" "
            now[0] = 6.0
            yield json.dumps(_response()).encode()

    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=DelayedStream())
        ),
    ).review({}, _checks(), 5.0)
    assert result.failure_reason == "openai_decision_review_timeout"
    assert result.scores == {}


def test_cancellation_after_response_retains_usage_without_success() -> None:
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=_response())
        ),
        check_active=MagicMock(side_effect=[None, None, None, RuntimeError("stopped")]),
    ).review({}, _checks(), 5.0)
    assert not result.completed
    assert result.input_tokens == 120
    assert result.scores == {}


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timeout_never_admits_a_request(timeout: float) -> None:
    handler, before_request = MagicMock(), MagicMock()
    result = DecisionsReviewer(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        before_request=before_request,
    ).review({}, _checks(), timeout)
    assert result.failure_reason == "openai_decision_invalid_timeout"
    handler.assert_not_called()
    before_request.assert_not_called()
