import json
from unittest.mock import MagicMock

import pytest

from onyx.regulatory.amendments.annexes import acceptance_calibration as calibration


def test_scope_failure_does_not_start_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_DB", "customs-regulations-test")
    process = MagicMock()
    monkeypatch.setattr(calibration.subprocess, "Popen", process)
    report = calibration.run_native_calibration()
    assert report["status"] == "failed"
    assert report["failure"] == "dev_scope_required"
    assert report["attempt_count"] == 0
    process.assert_not_called()


def test_archived_native_fixtures_are_exact() -> None:
    fixtures = calibration.load_native_fixtures()
    assert {name: len(content) for name, content in fixtures.items()} == {
        "docx": 36624,
        "xlsx": 4837,
    }


@pytest.mark.parametrize(
    "outcome",
    [
        "correct",
        "wrong_positive",
        "transport_failure",
        "rate_limit",
        "rate_limit_exhausted",
        "rate_limit_budget",
    ],
)
def test_four_reconciliation_paths_keep_verdicts_with_bounded_retries(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import httpx

    from onyx.llm.factory import get_llm
    from onyx.regulatory import structured_llm

    monkeypatch.setattr(structured_llm.time, "sleep", lambda _seconds: None)
    calls: list[dict[str, object]] = []

    def send(
        _client: httpx.Client, request: httpx.Request, **_kwargs: object
    ) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert body.get("stream", False) is False
        messages = str(body["messages"])
        assert "Derived transcription under correction" in messages
        assert "Original native source" in messages
        if outcome in {"rate_limit", "rate_limit_exhausted", "rate_limit_budget"} and (
            outcome != "rate_limit" or report.cases[-1].attempt_count == 1
        ):
            return httpx.Response(
                429,
                request=request,
                headers={
                    "Retry-After": "90" if outcome == "rate_limit_budget" else "0"
                },
                json={"error": {"message": "fixture capacity"}},
            )
        if outcome == "transport_failure":
            return httpx.Response(
                500,
                request=request,
                json={"error": {"message": "SENSITIVE_PROVIDER_ERROR"}},
            )
        supported = report.cases[-1].expected_supported and outcome != "wrong_positive"
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "fixture-response",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/fictional-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "supported": supported,
                                    "rationale": "Fixed fixture verdict",
                                }
                            ),
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    llm = get_llm(
        provider="openrouter",
        model="openai/fictional-model",
        max_input_tokens=32000,
        deployment_name=None,
        api_key="fixture-only",
        temperature=0,
        timeout=60,
    )
    reports: list[calibration.CalibrationReport] = []
    report = calibration.CalibrationReport()
    calibration.run_cases(
        llm, report, lambda value: reports.append(value.model_copy(deep=True))
    )
    per_case = (
        2 if outcome == "rate_limit" else 3 if outcome == "rate_limit_exhausted" else 1
    )
    assert len(calls) == report.attempt_count == 4 * per_case
    assert all(
        case.http_request_count == case.attempt_count == per_case
        for case in report.cases
    )
    assert [case.proposed_value for case in report.cases] == ["7%", "9%", "7%", "9%"]
    assert [case.original_value for case in report.cases] == ["7%"] * 4
    assert report.status == (
        "passed" if outcome in {"correct", "rate_limit"} else "failed"
    ), [
        (case.supported, case.status, case.failure, case.rationale)
        for case in report.cases
    ]
    if outcome in {"transport_failure", "rate_limit_exhausted", "rate_limit_budget"}:
        assert all(case.failure for case in report.cases)
        for case in report.cases:
            assert case.failure_detail is not None
            assert json.loads(case.failure_detail)["stage"] == "calibration"
        assert "SENSITIVE_PROVIDER_ERROR" not in report.model_dump_json()
        return
    assert len([case for case in report.cases if case.rationale]) == 4
    assert all(case.input_sha256 for case in report.cases)
    assert reports[-1].cases[-1].rationale == "Fixed fixture verdict"


def test_partial_report_survives_process_timeout() -> None:
    report = calibration.CalibrationReport(attempt_count=2)
    report.cases = [
        calibration.CalibrationCase(
            format="docx",
            proposed_value="7%",
            expected_supported=True,
            supported=True,
            rationale="First response retained",
            status="passed",
        )
    ]
    output = (report.model_dump_json() + "\n").encode()
    result = calibration.report_from_output(output, failure="process_timeout")
    assert result["status"] == "failed"
    assert result["attempt_count"] == 2
    parsed = calibration.CalibrationReport.model_validate(result)
    assert parsed.cases[0].rationale == "First response retained"
    assert "process_timeout" == result["failure"]


@pytest.fixture
def completed_calibration_report() -> calibration.CalibrationReport:
    return calibration.CalibrationReport(
        status="passed",
        attempt_count=4,
        database_read_only=True,
        fixture_verified=True,
        model_snapshot={
            "model_provider": "openrouter",
            "model_name": "openai/fictional-model",
        },
        module_sha256={name: "a" * 64 for name in calibration.MODULES},
        cases=[
            calibration.CalibrationCase(
                format=format,
                proposed_value=proposed,
                expected_supported=expected,
                supported=expected,
                rationale="Retained verdict",
                status="passed",
                attempt_count=1,
                http_request_count=1,
                input_sha256="b" * 64,
            )
            for format in ("docx", "xlsx")
            for proposed, expected in (("7%", True), ("9%", False))
        ],
    )


@pytest.mark.parametrize(
    ("character", "ensure_ascii"),
    [("ğ", False), ("😀", False), ("\u0000", False), ("😀", True)],
)
@pytest.mark.parametrize("failure", [None, "process_timeout"])
def test_maximum_unicode_rationales_retain_all_completed_verdicts(
    completed_calibration_report: calibration.CalibrationReport,
    character: str,
    ensure_ascii: bool,
    failure: str | None,
) -> None:
    report = completed_calibration_report
    for case in report.cases:
        case.rationale = character * 4000
    output = (
        json.dumps(report.model_dump(mode="json"), ensure_ascii=ensure_ascii) + "\n"
    ).encode()
    result = calibration.CalibrationReport.model_validate(
        calibration.report_from_output(output, failure=failure)
    )
    assert result.status == ("failed" if failure else "passed")
    assert result.failure == failure
    assert result.attempt_count_complete is (failure is None)
    assert result.attempt_count == 4
    assert [case.supported for case in result.cases] == [True, False, True, False]
    assert [case.status for case in result.cases] == ["passed"] * 4
    assert [case.rationale for case in result.cases] == [character * 4000] * 4
    assert [case.input_sha256 for case in result.cases] == ["b" * 64] * 4


@pytest.mark.parametrize("invalid_tail", ["malformed", "oversized", "unterminated"])
@pytest.mark.parametrize("failure", [None, "process_timeout"])
def test_invalid_newest_report_retains_prior_evidence_but_fails_incomplete(
    completed_calibration_report: calibration.CalibrationReport,
    invalid_tail: str,
    failure: str | None,
) -> None:
    valid = completed_calibration_report.model_dump_json().encode()
    tail = {
        "malformed": b"{\n",
        "oversized": b" " * 300_000 + valid + b"\n",
        "unterminated": valid,
    }[invalid_tail]
    result = calibration.CalibrationReport.model_validate(
        calibration.report_from_output(valid + b"\n" + tail, failure=failure)
    )
    assert result.status == "failed"
    assert result.failure == (failure or "child_report_invalid")
    assert result.attempt_count_complete is False
    assert result.attempt_count == 4
    assert [case.supported for case in result.cases] == [True, False, True, False]
    assert [case.rationale for case in result.cases] == ["Retained verdict"] * 4


def test_subprocess_forces_readonly_and_retains_timeout_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_DB", "customs-regulations-dev")
    monkeypatch.setenv("REGULATORY_ANNEX_ENVIRONMENT", "dev")
    monkeypatch.setenv("PGOPTIONS", "-c default_transaction_read_only=off")
    process = MagicMock()
    partial_report = (
        calibration.CalibrationReport(attempt_count=1).model_dump_json().encode()
        + b"\n"
    )
    process.communicate.side_effect = [
        calibration.subprocess.TimeoutExpired("fixed-child", 330),
        (partial_report, None),
    ]
    process.returncode = -9
    start = MagicMock(return_value=process)
    monkeypatch.setattr(calibration.subprocess, "Popen", start)
    result = calibration.run_native_calibration()
    assert result["failure"] == "process_timeout"
    assert result["attempt_count"] == 1
    assert result["attempt_count_complete"] is False
    assert (
        start.call_args.kwargs["env"]["PGOPTIONS"]
        == "-c default_transaction_read_only=on"
    )
    process.kill.assert_called_once()
    assert calibration.os.environ["PGOPTIONS"] == "-c default_transaction_read_only=off"


def test_native_fixture_tampering_refuses_before_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration.Path, "read_bytes", lambda _path: b"changed ZIP bytes"
    )
    with pytest.raises(ValueError, match="fixed_fixture_hash_mismatch"):
        calibration.load_native_fixtures()


def test_child_setup_failure_has_safe_detail() -> None:
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from onyx.regulatory.amendments.annexes.acceptance_calibration import _child; _child()",
        ],
        env={**os.environ, "POSTGRES_DB": "not-dev"},
        capture_output=True,
        timeout=20,
        check=True,
    )
    report = calibration.report_from_output(result.stdout, failure=None)
    assert report["failure"] == "calibration_setup_failed"
    assert isinstance(report["failure_detail"], str)
    detail = json.loads(report["failure_detail"])
    assert detail["exceptions"][0]["type"] == "ValueError"
    assert detail["exceptions"][0]["frames"][-1]["function"] == "_child"
    assert "not-dev" not in result.stdout.decode()


def test_case_and_setup_failure_details_survive_runner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import regulatory_annex_dev_cutover as cutover

    from onyx.regulatory.amendments.annexes.dev_acceptance import safe_failure_detail

    detail = safe_failure_detail("calibration", ValueError("DO_NOT_LOG"))
    calibration_report = calibration.CalibrationReport(
        failure_detail=detail,
        cases=[
            calibration.CalibrationCase(
                format="docx",
                proposed_value="9%",
                expected_supported=False,
                status="failed",
                failure_detail=detail,
            )
        ],
    )
    report = {
        "phase": "preflight",
        "release_sha_metadata": "a" * 40,
        "status": "failed",
        "calibration": calibration_report.model_dump(mode="json"),
    }
    with pytest.raises(cutover.CutoverRefusal):
        cutover.emit_acceptance_report(json.dumps(report), "preflight", "a" * 40)
    retained = json.loads(capsys.readouterr().out)
    assert retained == report
    assert "DO_NOT_LOG" not in json.dumps(retained)
