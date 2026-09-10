"""Fixed release probes refuse the wrong environment before touching services."""

import importlib.util
import shutil
from pathlib import Path

import pytest


def test_fixed_dev_entrypoint_is_packaged() -> None:
    assert (
        importlib.util.find_spec("onyx.regulatory.amendments.annexes.dev_acceptance")
        is not None
    )


@pytest.mark.parametrize(
    "database,environment,machine",
    [
        ("annex_local", "dev", "x86_64"),
        ("customs-regulations-dev", "test", "x86_64"),
        ("customs-regulations-dev", "dev", "arm64"),
    ],
)
def test_scope_refuses_non_dev_or_non_amd64_runtime(
    database: str, environment: str, machine: str
) -> None:
    from onyx.regulatory.amendments.annexes.dev_acceptance import validate_scope

    with pytest.raises(ValueError):
        validate_scope(database=database, environment=environment, machine=machine)


def test_fixture_manifest_verifies_every_packaged_byte() -> None:
    from onyx.regulatory.amendments.annexes.dev_acceptance import load_fixtures

    fixtures = load_fixtures()
    assert set(fixtures) == {"old.pdf", "new.pdf", "page.png"}
    assert fixtures["old.pdf"].startswith(b"%PDF-")
    assert fixtures["page.png"].startswith(b"\x89PNG\r\n\x1a\n")


def test_packaged_fixture_corruption_refuses_before_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.regulatory.amendments.annexes import dev_acceptance

    original = Path(dev_acceptance.__file__).with_name("acceptance_fixtures")
    copied = tmp_path / "acceptance_fixtures"
    shutil.copytree(original, copied)
    (copied / "old.pdf").write_bytes(b"corrupted")
    monkeypatch.setattr(dev_acceptance, "__file__", str(tmp_path / "dev_acceptance.py"))
    with pytest.raises(ValueError, match="fixed_fixture_hash_mismatch"):
        dev_acceptance.load_fixtures()


def test_disabled_canary_refuses_before_mutating_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments.annexes import acceptance_canary, config

    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", False)
    with pytest.raises(ValueError, match="explicit_creation_activation"):
        acceptance_canary.run_canary("a" * 40)


def test_failed_batch_refuses_without_polling_until_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from unittest.mock import Mock
    from uuid import uuid4

    import httpx

    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.regulatory.amendments.annexes import acceptance_canary

    run = CanaryRun(
        release_sha="a" * 40, user_id=uuid4(), document_set_id=3, batch_id=4
    )
    request = Mock(side_effect=[[], [{"id": 4, "status": "failed"}]])
    monkeypatch.setattr(acceptance_canary, "request_json", request)
    sleep = Mock()
    monkeypatch.setattr(acceptance_canary.time, "sleep", sleep)
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="fictional_amendment_batch_failed"):
            acceptance_canary.wait_review(client, run, time.monotonic() + 600)
    sleep.assert_not_called()


def test_failed_canary_cleans_files_revokes_token_and_returns_owned_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock
    from uuid import uuid4

    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.regulatory.amendments.annexes import acceptance_canary, config

    run = CanaryRun(release_sha="a" * 40, user_id=uuid4())
    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    monkeypatch.setattr(acceptance_canary, "reserve_canary", Mock(return_value=run))
    monkeypatch.setattr(
        acceptance_canary, "issue_canary_token", Mock(return_value="memory-only")
    )
    monkeypatch.setattr(acceptance_canary, "request_json", Mock())
    monkeypatch.setattr(acceptance_canary, "save_canary", Mock())
    monkeypatch.setattr(
        acceptance_canary,
        "bootstrap_original",
        Mock(side_effect=RuntimeError("private detail")),
    )
    cleanup = Mock()
    revoke = Mock()
    monkeypatch.setattr(acceptance_canary, "cleanup_canary", cleanup)
    monkeypatch.setattr(acceptance_canary, "revoke_canary_token", revoke)
    report = acceptance_canary.run_canary(run.release_sha)
    assert report["status"] == "failed"
    assert report["canary"]["run_id"] == str(run.run_id)
    assert report["canary"]["evidence"]["acceptance_passed"] is False
    assert "private detail" not in str(report)
    assert "memory-only" not in str(report)
    cleanup.assert_called_once_with(run)
    revoke.assert_called_once_with(run)


def test_preflight_prints_failed_calibration_once_before_gate_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    import sys
    from unittest.mock import Mock

    from onyx.db import regulatory_annex_acceptance
    from onyx.regulatory.amendments.annexes import (
        acceptance_calibration,
        dev_acceptance,
    )

    monkeypatch.setenv("POSTGRES_DB", "customs-regulations-dev")
    monkeypatch.setenv("REGULATORY_ANNEX_ENVIRONMENT", "dev")
    monkeypatch.setenv("ANNEX_ACCEPTANCE_RELEASE_SHA", "a" * 40)
    monkeypatch.setattr(sys, "argv", ["dev_acceptance", "preflight"])
    monkeypatch.setattr(dev_acceptance.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        regulatory_annex_acceptance,
        "verify_dev_configuration",
        Mock(return_value={"database": "customs-regulations-dev"}),
    )
    monkeypatch.setattr(
        dev_acceptance, "native_parser_probe", Mock(return_value={"pdf_pages": 4})
    )
    calibration = Mock(
        return_value={
            "status": "failed",
            "cases": [{"supported": False}],
            "attempt_count": 1,
        }
    )
    monkeypatch.setattr(acceptance_calibration, "run_native_calibration", calibration)
    with pytest.raises(SystemExit) as result:
        dev_acceptance.main()
    assert result.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["calibration"]["cases"] == [{"supported": False}]
    calibration.assert_called_once_with()
