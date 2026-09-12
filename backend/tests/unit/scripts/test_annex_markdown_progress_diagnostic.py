"""Retained Markdown progress is scoped and never exports source payloads."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.db import regulatory_markdown_progress_diagnostic as diagnostic
from onyx.db.models import (
    AmendmentBatch,
    ChatSession,
    DocumentSet,
    KVStore,
    RegulatoryFilePublication,
    UserFile,
)
from onyx.db.regulatory_annex_acceptance import CanaryRun, CreationIntent


def fixture() -> tuple[Mock, dict[type, Any]]:
    run = CanaryRun(
        release_sha=diagnostic.RUNTIME,
        run_id=UUID("e96ad70f-f49f-45c1-a414-cf02e02f6359"),
        user_id=UUID("7e0d56bc-6f9c-4cec-b29b-5a8c9ec2b844"),
        file_id=UUID("eca03e71-2c07-497b-aa1c-84504cddc12b"),
        document_set_id=27,
        batch_id=51,
        persona_id=3,
        package_id=UUID("961f595b-b903-4c10-ad1a-bcb7898fbd52"),
        markdown_file_ids=[diagnostic.MARKDOWN_FILE],
        phase="cleaned",
        created_at=datetime.fromisoformat("2026-09-12T20:20:10.448214+00:00"),
        evidence={"cleanup_complete": True},
    )
    run.creation_intents = [
        CreationIntent(
            kind="markdown",
            marker="ANNEXCANARY" + run.run_id.hex + ".md",
            artifact_id=diagnostic.MARKDOWN_FILE,
        )
    ]
    run.chat_ids = [
        UUID("0d6e8e7b-0c11-4c53-8669-aeb079c4daa2"),
        UUID("c94409a8-f6bd-4ef4-aaf5-3af0dc71968a"),
    ]
    run.creation_intents.extend(
        CreationIntent(
            kind="chat", marker=run.name + " / dated " + day, artifact_id=chat_id
        )
        for chat_id, day in zip(run.chat_ids, ("2026-09-09", "2026-09-10"))
    )
    rows: dict[type, Any] = {
        ChatSession: None,
        KVStore: SimpleNamespace(value=run.model_dump(mode="json")),
        DocumentSet: SimpleNamespace(
            is_public=False, user_id=run.user_id, name=run.name
        ),
        AmendmentBatch: SimpleNamespace(
            created_by=run.user_id,
            document_set_id=27,
            user_file_ids=[str(run.file_id)],
            source_package_id=run.package_id,
        ),
        RegulatoryFilePublication: SimpleNamespace(
            scope_key="scope",
            original_ingestion_receipt=None,
            writer_manifest=None,
            gate_closed=True,
        ),
        UserFile: None,
    }
    session = Mock()
    session.get.side_effect = lambda model, _key: rows[model]
    session.scalars.side_effect = [[], [], []]
    return session, rows


def test_absent_receipt_is_not_success() -> None:
    session, _ = fixture()
    result = diagnostic.load_markdown_progress(
        cast(Session, session), expected_scope_key="scope"
    )
    assert result["status"] == "read"
    assert result["receipt_present"] is False
    assert result["receipt_valid"] is False
    assert result["temporal_count"] == 0


def test_chunked_receipt_and_published_rows_are_distinct_and_redacted() -> None:
    session, rows = fixture()
    rows[RegulatoryFilePublication].original_ingestion_receipt = {
        "version": 1,
        "file_id": "PRIVATE_FILESTORE",
        **{
            key: "a" * 64
            for key in (
                "raw_sha256",
                "documents_sha256",
                "plaintext_sha256",
                "canonical_sha256",
                "generation_hash",
            )
        },
    }
    stamp = datetime(2026, 9, 12, 20, 25, tzinfo=timezone.utc)
    session.scalars.side_effect = [
        [SimpleNamespace(payload_sha256="b" * 64, created_at=stamp)],
        [
            SimpleNamespace(
                payload_sha256="c" * 64, published_at=stamp, retired_at=stamp
            )
        ],
        [],
    ]
    result = diagnostic.load_markdown_progress(
        cast(Session, session), expected_scope_key="scope"
    )
    assert result["receipt_valid"] is True
    assert result["canonical_count"] == result["temporal_count"] == 1
    assert result["temporal_retired_count"] == 1
    assert result["first_published_at"] == stamp.isoformat()
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("release_sha", "0" * 40),
        ("phase", "running"),
        ("document_set_id", 28),
        ("created_at", "2026-09-12T20:20:11Z"),
        ("markdown_file_ids", []),
        ("creation_intents", []),
    ],
)
def test_foreign_or_unowned_run_refused(field: str, value: Any) -> None:
    session, rows = fixture()
    rows[KVStore].value[field] = value
    result = diagnostic.load_markdown_progress(
        cast(Session, session), expected_scope_key="scope"
    )
    assert result["status"] == "scope_refused"
    session.scalars.assert_not_called()


def test_owner_scope_refused() -> None:
    session, _ = fixture()
    assert (
        diagnostic.load_markdown_progress(
            cast(Session, session), expected_scope_key="foreign"
        )["scope_failure"]
        == "publication_scope"
    )
    session.scalars.assert_not_called()


def test_sequential_real_configuration_and_read_engine_scopes() -> None:
    from contextlib import ExitStack
    from unittest.mock import patch

    from sqlalchemy.engine import Engine

    from onyx.db import regulatory_annex_dev_cutover
    from onyx.db.engine import sql_engine

    engines = [Mock(spec=Engine), Mock(spec=Engine)]
    config_session = Mock()
    config_session.execute.return_value.scalar_one.return_value = (
        "customs-regulations-dev"
    )
    config_session.execute.return_value.scalars.return_value.all.return_value = [
        "fixture_dev_index"
    ]

    def read(_session: Session, *, expected_scope_key: str) -> dict[str, object]:
        assert len(expected_scope_key) == 64
        assert sql_engine.SqlEngine.get_engine() is engines[1]
        return {"status": "read"}

    with ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                "os.environ",
                {
                    "POSTGRES_DB": "customs-regulations-dev",
                    "REGULATORY_ANNEX_ENVIRONMENT": "dev",
                    "PGOPTIONS": "-c default_transaction_read_only=on",
                },
            )
        )
        stack.enter_context(patch.object(sql_engine.SqlEngine, "_engine", None))
        stack.enter_context(patch.object(sql_engine.SqlEngine, "_engine_profile", None))
        stack.enter_context(patch.object(sql_engine.ShardRegistry, "reset"))
        stack.enter_context(
            patch.object(sql_engine, "create_engine", side_effect=engines)
        )
        stack.enter_context(
            patch.object(
                sql_engine,
                "build_connection_string",
                return_value="postgresql://fixture.invalid/readonly",
            )
        )
        session = stack.enter_context(
            patch.object(regulatory_annex_dev_cutover, "get_session_with_tenant")
        )
        session.return_value.__enter__.return_value = config_session
        stack.enter_context(
            patch.object(regulatory_annex_dev_cutover, "MULTI_TENANT", False)
        )
        stack.enter_context(patch.object(sql_engine, "get_session_with_current_tenant"))
        stack.enter_context(
            patch.object(diagnostic, "load_markdown_progress", side_effect=read)
        )
        result = diagnostic.read_markdown_progress()
        assert result["status"] == "read"
        assert sql_engine.SqlEngine._engine is None
    for engine in engines:
        engine.dispose.assert_called_once()


def test_runner_report_rejects_unknown_payload_and_preserves_progress() -> None:
    from scripts import regulatory_annex_dev_cutover as runner

    session, _ = fixture()
    report = {
        "stage": "markdown_a8",
        "database_read_only": True,
        **diagnostic.load_markdown_progress(
            cast(Session, session), expected_scope_key="scope"
        ),
    }
    runner.validate_markdown_progress_report(report)
    with pytest.raises(runner.CutoverRefusal):
        runner.validate_markdown_progress_report({**report, "raw_error": "PRIVATE"})


def test_worker_health_only_returns_fixed_process_states() -> None:
    from unittest.mock import patch

    from scripts import regulatory_annex_dev_cutover as runner

    with patch(
        "subprocess.run",
        return_value=SimpleNamespace(
            stdout="celery_worker_user_file_processing RUNNING pid 123, uptime 1:00:00\ncelery_worker_regulatory_indexing FATAL PRIVATE STDERR\ncelery_beat_regulatory_indexing RUNNING pid 456, uptime 1:00:00",
            returncode=3,
        ),
    ) as run:
        result = runner.markdown_worker_health()
    assert "PRIVATE" not in json.dumps(result)
    assert result["worker_health_status"] == "read"
    assert (
        cast(list[dict[str, object]], result["workers"])[1]["process_state"] == "FATAL"
    )
    assert run.call_args.kwargs["timeout"] == 5


@pytest.mark.parametrize(
    "receipt", [{}, {"version": 1, "file_id": "PRIVATE", "raw_sha256": "not-a-hash"}]
)
def test_malformed_receipt_is_present_but_not_valid(receipt: dict[str, object]) -> None:
    session, rows = fixture()
    rows[RegulatoryFilePublication].original_ingestion_receipt = receipt
    result = diagnostic.load_markdown_progress(
        cast(Session, session), expected_scope_key="scope"
    )
    assert result["receipt_present"] is True
    assert result["receipt_valid"] is False
    assert result["receipt_raw_sha256"] is None


def test_exact_embedded_program_refuses_wrong_environment_without_db() -> None:
    import contextlib
    import io
    import signal
    from unittest.mock import patch

    from scripts import regulatory_annex_dev_cutover as runner

    def command(arguments: list[str], *, timeout: int) -> str:
        assert timeout == 80
        program = arguments[-1]
        compile(program, "markdown-a8-diagnostic", "exec")
        output = io.StringIO()
        old = signal.getsignal(signal.SIGALRM)
        try:
            with (
                patch.dict("os.environ", {"POSTGRES_DB": "not-dev"}),
                contextlib.redirect_stdout(output),
            ):
                exec(program, {})
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        return output.getvalue()

    driver = Mock(sha=diagnostic.RUNTIME)
    driver.command.side_effect = command
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runner.diagnose_markdown_progress(driver, "fixed-pod", "fixed-container")
    result = json.loads(output.getvalue())
    assert result["status"] == "scope_refused"
    assert result["scope_failure"] == "env"


def test_owned_chat_timestamps_only() -> None:
    session, rows = fixture()
    run = CanaryRun.model_validate(rows[KVStore].value)
    stamp = datetime(2026, 9, 12, 20, 28, tzinfo=timezone.utc)
    original = session.get.side_effect

    def get(model: type, key: object) -> Any:
        if model is ChatSession:
            day = "2026-09-09" if key == run.chat_ids[0] else "2026-09-10"
            return SimpleNamespace(
                user_id=run.user_id,
                persona_id=3,
                description=run.name + " / dated " + day,
                time_created=stamp,
                time_updated=stamp,
            )
        return original(model, key)

    session.get.side_effect = get
    result = diagnostic.load_markdown_progress(
        cast(Session, session), expected_scope_key="scope"
    )
    assert (
        result["old_chat_created_at"]
        == result["new_chat_updated_at"]
        == stamp.isoformat()
    )
    assert "description" not in result
