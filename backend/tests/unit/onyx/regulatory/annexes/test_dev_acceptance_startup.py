"""The fixed CLI initializes encrypted reads and retains safe failed-stage evidence."""

import json
import sys
from contextlib import nullcontext
from unittest.mock import Mock

import pytest


@pytest.fixture
def isolated_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from onyx.db import regulatory_annex_acceptance
    from onyx.db.engine.sql_engine import SqlEngine
    from onyx.regulatory.amendments.annexes import (
        acceptance_calibration,
        acceptance_canary,
        acceptance_pdf_vision,
        dev_acceptance,
    )

    monkeypatch.setenv("POSTGRES_DB", "customs-regulations-dev")
    monkeypatch.setenv("REGULATORY_ANNEX_ENVIRONMENT", "dev")
    monkeypatch.setenv("ANNEX_ACCEPTANCE_RELEASE_SHA", "a" * 40)
    monkeypatch.setattr(dev_acceptance.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        regulatory_annex_acceptance,
        "verify_dev_configuration",
        lambda: {"database": "customs-regulations-dev"},
    )
    monkeypatch.setattr(dev_acceptance, "native_parser_probe", lambda: {"pdf_pages": 4})
    monkeypatch.setattr(
        acceptance_calibration, "run_native_calibration", lambda: {"status": "passed"}
    )
    monkeypatch.setattr(
        acceptance_pdf_vision, "run_pdf_vision_probe", lambda: {"status": "passed"}
    )
    monkeypatch.setattr(
        acceptance_canary, "run_canary", lambda _sha: {"status": "passed"}
    )
    monkeypatch.setattr(SqlEngine, "scoped_engine", Mock(return_value=nullcontext()))


@pytest.mark.usefixtures("isolated_probe")
@pytest.mark.parametrize("phase", ["preflight", "canary"])
def test_cli_initializes_enterprise_before_encrypted_configuration_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], phase: str
) -> None:
    from ee.onyx.utils import encryption as ee_encryption
    from onyx.db import regulatory_annex_acceptance
    from onyx.regulatory.amendments.annexes import dev_acceptance
    from onyx.utils import encryption, variable_functionality

    key = "fictional-test-only-key-1234567890"
    value = "fictional-provider-credential"
    ciphertext = ee_encryption._encrypt_string(value, key=key)
    monkeypatch.setattr(encryption, "ENCRYPTION_KEY_SECRET", key)
    monkeypatch.setattr(ee_encryption, "ENCRYPTION_KEY_SECRET", key)
    monkeypatch.setattr(variable_functionality, "_LICENSE_ENFORCEMENT_ENABLED", True)
    monkeypatch.setattr(variable_functionality.global_version, "_is_ee", False)
    variable_functionality.fetch_versioned_implementation.cache_clear()

    def read_encrypted_configuration() -> dict[str, object]:
        assert encryption.decrypt_bytes_to_string(ciphertext) == value
        return {"database": "customs-regulations-dev"}

    monkeypatch.setattr(
        regulatory_annex_acceptance,
        "verify_dev_configuration",
        read_encrypted_configuration,
    )
    monkeypatch.setattr(sys, "argv", ["dev_acceptance", phase])
    try:
        dev_acceptance.main()
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "passed"
    finally:
        variable_functionality.fetch_versioned_implementation.cache_clear()


@pytest.mark.usefixtures("isolated_probe")
@pytest.mark.parametrize(
    "stage",
    [
        "scope",
        "startup",
        "configuration",
        "native",
        "calibration",
        "pdf_vision",
        "canary",
    ],
)
def test_failed_stage_is_reported_without_exception_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stage: str
) -> None:
    from onyx.db import regulatory_annex_acceptance
    from onyx.regulatory.amendments.annexes import (
        acceptance_calibration,
        acceptance_canary,
        acceptance_pdf_vision,
        dev_acceptance,
    )
    from onyx.utils import variable_functionality

    targets = {
        "scope": (dev_acceptance, "validate_scope"),
        "startup": (variable_functionality, "set_is_ee_based_on_env_variable"),
        "configuration": (regulatory_annex_acceptance, "verify_dev_configuration"),
        "native": (dev_acceptance, "native_parser_probe"),
        "calibration": (acceptance_calibration, "run_native_calibration"),
        "pdf_vision": (acceptance_pdf_vision, "run_pdf_vision_probe"),
        "canary": (acceptance_canary, "run_canary"),
    }
    target, attribute = targets[stage]
    monkeypatch.setattr(
        target, attribute, Mock(side_effect=RuntimeError("DO_NOT_LOG_secret_or_source"))
    )
    phase = "canary" if stage == "canary" else "preflight"
    monkeypatch.setattr(sys, "argv", ["dev_acceptance", phase])
    with pytest.raises(SystemExit) as stopped:
        dev_acceptance.main()
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "failed"
    assert report["failure_stage"] == stage
    assert report["exception_type"] == "RuntimeError"
    detail = json.loads(report["failure"])
    assert detail["stage"] == stage
    assert detail["exceptions"][0]["frames"][-1]["function"] == "main"
    assert "DO_NOT_LOG" not in captured.out + captured.err


def test_safe_failure_retains_cause_and_bounds_untrusted_details() -> None:
    from onyx.regulatory.amendments.annexes import dev_acceptance

    cause = ValueError("DO_NOT_LOG_credential")
    error = RuntimeError("DO_NOT_LOG_document" * 1000)
    error.__cause__ = cause
    cause.__cause__ = error
    detail = dev_acceptance.safe_failure_detail("baseline", error)
    parsed = json.loads(detail)
    assert [item["type"] for item in parsed["exceptions"]] == [
        "RuntimeError",
        "ValueError",
    ]
    assert all(item["frames"] == [] for item in parsed["exceptions"])
    assert "DO_NOT_LOG" not in detail
    assert len(detail) <= 4000
    unknown = type("DO_NOT_LOG_secret", (Exception,), {})("secret")
    assert json.loads(dev_acceptance.safe_failure_detail("secret", unknown)) == {
        "stage": "unknown",
        "exceptions": [{"type": "Exception", "frames": []}],
    }


def test_failure_detail_limits_cause_depth_and_excludes_external_frames() -> None:
    from onyx.regulatory.amendments.annexes import dev_acceptance

    previous: BaseException | None = None
    for _ in range(10):
        try:
            dev_acceptance.validate_scope(
                database="secret", environment="secret", machine="secret"
            )
        except ValueError as error:
            error.__cause__ = previous
            previous = error
    assert previous is not None
    detail = dev_acceptance.safe_failure_detail("baseline", previous)
    parsed = json.loads(detail)
    assert len(parsed["exceptions"]) == 3
    for error in parsed["exceptions"]:
        assert len(error["frames"]) == 1
        assert error["frames"][0]["module"] == dev_acceptance.__name__
        assert error["frames"][0]["function"] == "validate_scope"
        assert isinstance(error["frames"][0]["line"], int)
    assert "secret" not in detail
    assert len(detail) <= 4000


@pytest.mark.usefixtures("isolated_probe")
def test_failed_calibration_is_not_overwritten_or_followed_by_pdf_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from onyx.regulatory.amendments.annexes import (
        acceptance_calibration,
        acceptance_pdf_vision,
        dev_acceptance,
    )

    monkeypatch.setattr(
        acceptance_calibration,
        "run_native_calibration",
        lambda: {"status": "failed", "attempt_count": 4},
    )
    pdf = Mock()
    monkeypatch.setattr(acceptance_pdf_vision, "run_pdf_vision_probe", pdf)
    monkeypatch.setattr(sys, "argv", ["dev_acceptance", "preflight"])
    with pytest.raises(SystemExit):
        dev_acceptance.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == report["calibration"]["status"] == "failed"
    pdf.assert_not_called()


@pytest.mark.usefixtures("isolated_probe")
def test_pdf_failed_report_keeps_safe_phase_and_passed_calibration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from onyx.regulatory.amendments.annexes import acceptance_pdf_vision, dev_acceptance

    monkeypatch.setattr(
        acceptance_pdf_vision,
        "run_pdf_vision_probe",
        lambda: {
            "status": "failed",
            "probe_stage": "grounding",
            "attempt_count": 3,
            "http_request_count": 3,
            "exception_type": "ValidationError",
        },
    )
    monkeypatch.setattr(sys, "argv", ["dev_acceptance", "preflight"])
    with pytest.raises(SystemExit):
        dev_acceptance.main()
    report = json.loads(capsys.readouterr().out)
    assert report["calibration"]["status"] == "passed"
    assert report["status"] == "failed" and report["failure_stage"] == "pdf_vision"
    assert report["exception_type"] == "ValidationError"
