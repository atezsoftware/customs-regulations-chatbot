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
    "stage", ["scope", "startup", "configuration", "native", "calibration", "canary"]
)
def test_failed_stage_is_reported_without_exception_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], stage: str
) -> None:
    from onyx.db import regulatory_annex_acceptance
    from onyx.regulatory.amendments.annexes import (
        acceptance_calibration,
        acceptance_canary,
        dev_acceptance,
    )
    from onyx.utils import variable_functionality

    targets = {
        "scope": (dev_acceptance, "validate_scope"),
        "startup": (variable_functionality, "set_is_ee_based_on_env_variable"),
        "configuration": (regulatory_annex_acceptance, "verify_dev_configuration"),
        "native": (dev_acceptance, "native_parser_probe"),
        "calibration": (acceptance_calibration, "run_native_calibration"),
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
    assert "DO_NOT_LOG" not in captured.out + captured.err
