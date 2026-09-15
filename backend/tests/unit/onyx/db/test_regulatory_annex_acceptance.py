"""A stuck canary reservation must only reopen for genuinely transient failures."""

import json
from uuid import uuid4

from onyx.db.regulatory_annex_acceptance import CanaryRun, _canary_failure_is_transient


def _run(**evidence: object) -> CanaryRun:
    return CanaryRun(release_sha="a" * 40, user_id=uuid4(), evidence=evidence)


def test_read_timeout_is_transient() -> None:
    failure = json.dumps(
        {
            "exceptions": [
                {"frames": [], "type": "ReadTimeout"},
                {"frames": [], "type": "ReadTimeout"},
            ],
            "stage": "current_chat",
        }
    )
    assert _canary_failure_is_transient(_run(failure=failure)) is True


def test_value_error_is_never_transient() -> None:
    failure = json.dumps(
        {"exceptions": [{"frames": [], "type": "ValueError"}], "stage": "canary"}
    )
    assert _canary_failure_is_transient(_run(failure=failure)) is False


def test_mixed_transient_and_real_failure_is_not_transient() -> None:
    """One genuine exception in the chain must still require owned recovery,
    even alongside a transient one — never partial credit."""
    failure = json.dumps(
        {
            "exceptions": [
                {"frames": [], "type": "ReadTimeout"},
                {"frames": [], "type": "AssertionError"},
            ],
            "stage": "source_review",
        }
    )
    assert _canary_failure_is_transient(_run(failure=failure)) is False


def test_missing_failure_evidence_is_not_transient() -> None:
    assert _canary_failure_is_transient(_run()) is False


def test_malformed_failure_json_is_not_transient() -> None:
    assert _canary_failure_is_transient(_run(failure="not json")) is False


def test_empty_exception_list_is_not_transient() -> None:
    failure = json.dumps({"exceptions": [], "stage": "canary"})
    assert _canary_failure_is_transient(_run(failure=failure)) is False
