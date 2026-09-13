import json
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
from scripts.regulatory_annex_dev_cutover import CutoverRefusal, emit_acceptance_report

from onyx.db.regulatory_annex_acceptance import CanaryRun
from onyx.regulatory.amendments.annexes import acceptance_canary as canary


@pytest.mark.parametrize("failure", ["deadline", "status", "post"])
def test_markdown_failure_retains_last_progress_before_cleanup(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    run = CanaryRun(release_sha="a" * 40, user_id=uuid4(), document_set_id=42)
    identifier = str(uuid4())
    clock = [0.0]
    saved: list[dict[str, Any]] = []
    posts: list[str] = []
    monkeypatch.setattr(canary.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        canary.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(
        canary, "upload_canary_markdown", lambda *_args: {"id": identifier}
    )
    monkeypatch.setattr(
        canary, "save_canary", lambda value: saved.append(dict(value.evidence))
    )

    def request(_client: object, method: str, path: str, **_kwargs: Any) -> Any:
        if method == "POST":
            posts.append(path)
            if failure == "post":
                raise RuntimeError("original request failure")
            return {}
        return [
            {
                "id": identifier,
                "status": "FAILED" if failure == "status" else "CHUNKED",
                "regulatory_indexing_progress": {
                    "status": "RETRY_WAIT",
                    "stage": "EMBEDDING",
                    "attempt_count": 2,
                    "error_summary": "secret must not be saved",
                    "total_items": 1,
                },
            }
        ]

    monkeypatch.setattr(canary, "request_json", request)
    expected = {"deadline": TimeoutError, "status": ValueError, "post": RuntimeError}[
        failure
    ]
    with pytest.raises(expected):
        canary.markdown_canary(Mock(), run, 3.0)
    assert saved
    evidence = saved[-1]
    assert evidence["markdown_upload_accepted"] is True
    assert evidence["markdown_last_status"] == (
        "FAILED" if failure == "status" else "CHUNKED"
    )
    assert evidence["markdown_index_post_attempted"] is (failure != "status")
    assert evidence["markdown_index_post_accepted"] is (failure == "deadline")
    assert len(posts) == (0 if failure == "status" else 1)
    assert evidence["markdown_job_status"] == "RETRY_WAIT"
    assert evidence["markdown_job_attempt_count"] == 2
    assert "secret" not in json.dumps(evidence)
    assert evidence["stage_markdown_remaining_start_ms"] == 3000
    assert evidence["stage_markdown_elapsed_ms"] == (
        3000 if failure == "deadline" else 0
    )


def test_progress_save_failure_preserves_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = CanaryRun(release_sha="a" * 40, user_id=uuid4())
    monkeypatch.setattr(
        canary, "save_canary", Mock(side_effect=ValueError("save failed"))
    )
    with pytest.raises(RuntimeError, match="original operation"):
        with canary.record_canary_stage(run, "baseline"):
            raise RuntimeError("original operation")
    assert run.evidence["progress_save_failed"] is True


def test_stage_capture_keeps_one_shared_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = CanaryRun(release_sha="a" * 40, user_id=uuid4())
    clock = [0.0]
    save = Mock()
    monkeypatch.setattr(canary.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(canary, "save_canary", save)
    with canary.record_canary_stage(run, "source_review", 10.0):
        clock[0] = 6.0
    with canary.record_canary_stage(run, "approval", 10.0):
        clock[0] = 8.0
    assert run.evidence["stage_source_review_elapsed_ms"] == 6000
    assert run.evidence["stage_approval_remaining_start_ms"] == 4000
    assert run.evidence["stage_approval_remaining_end_ms"] == 2000
    assert save.call_count == 2


@pytest.mark.parametrize("value", [True, -1, "private", 86400001])
def test_runner_refuses_invalid_progress_numbers(value: object) -> None:
    report = {
        "phase": "canary",
        "status": "passed",
        "release_sha_metadata": "a" * 40,
        "canary": {"evidence": {"stage_markdown_elapsed_ms": value}},
    }
    with pytest.raises(CutoverRefusal):
        emit_acceptance_report(json.dumps(report), "canary", "a" * 40)


def test_runner_retains_progress(capsys: pytest.CaptureFixture[str]) -> None:
    evidence = {
        "stage_markdown_elapsed_ms": 3000,
        "markdown_index_post_accepted": True,
        "markdown_last_status": "CHUNKED",
        "markdown_job_stage": "EMBEDDING",
    }
    emit_acceptance_report(
        json.dumps(
            {
                "phase": "canary",
                "status": "passed",
                "release_sha_metadata": "a" * 40,
                "canary": {"evidence": evidence},
            }
        ),
        "canary",
        "a" * 40,
    )
    assert json.loads(capsys.readouterr().out)["canary"]["evidence"] == evidence
