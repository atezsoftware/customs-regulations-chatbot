import datetime
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
)
from onyx.db.regulatory_indexing_jobs import (
    RegulatoryIndexingConfigSnapshot,
    claim_regulatory_indexing_job,
    create_or_get_regulatory_indexing_job,
    schedule_regulatory_indexing_retry,
)


def test_regulatory_indexing_enums_expose_the_durable_wire_values() -> None:
    assert [status.value for status in RegulatoryIndexingJobStatus] == [
        "QUEUED",
        "RUNNING",
        "RETRY_WAIT",
        "SUCCEEDED",
        "FAILED",
        "CANCELLING",
        "CANCELLED",
    ]
    assert [stage.value for stage in RegulatoryIndexingStage] == [
        "PREPARING",
        "CONTEXT_SUBMIT",
        "CONTEXT_WAIT",
        "CONTEXT_APPLY",
        "EMBEDDING",
        "INDEX_WRITE",
        "VERIFY",
        "PUBLISH",
    ]
    assert [status.value for status in RegulatoryIndexingItemStatus] == [
        "PENDING",
        "CONTEXT_READY",
        "EMBEDDED",
        "FAILED",
        "SKIPPED",
    ]


def test_claim_returns_false_when_the_atomic_update_matches_no_job() -> None:
    db_session = MagicMock(spec=Session)
    db_session.scalar.return_value = None

    claimed = claim_regulatory_indexing_job(
        cast(Session, db_session),
        job_id=uuid4(),
        expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        expected_generation=3,
        now=datetime.datetime(2026, 8, 19, 10, 0, tzinfo=datetime.timezone.utc),
    )

    assert claimed is False


def test_retry_error_text_is_bounded_before_persistence() -> None:
    db_session = MagicMock(spec=Session)
    db_session.scalar.return_value = uuid4()

    scheduled = schedule_regulatory_indexing_retry(
        cast(Session, db_session),
        job_id=uuid4(),
        expected_stage=RegulatoryIndexingStage.EMBEDDING,
        expected_generation=2,
        next_retry_at=datetime.datetime(
            2026, 8, 19, 10, 1, tzinfo=datetime.timezone.utc
        ),
        error_code="provider_timeout",
        error_message="x" * 5000,
    )

    statement = db_session.scalar.call_args.args[0]
    assert scheduled is True
    assert len(statement.compile().params["error_message"]) == 4000


@pytest.mark.parametrize(
    ("config_snapshot", "error_pattern"),
    [
        (
            {"vertex": {"client_secret": "must-not-persist"}},
            "secret-like key",
        ),
        (
            {"embedding": {"dimension": 1536, "payload": b"not-json"}},
            "JSON-safe",
        ),
    ],
)
def test_job_creation_rejects_unsafe_snapshot_before_sql(
    config_snapshot: dict[str, object],
    error_pattern: str,
) -> None:
    db_session = MagicMock(spec=Session)

    with pytest.raises(ValueError, match=error_pattern):
        create_or_get_regulatory_indexing_job(
            cast(Session, db_session),
            user_file_id=uuid4(),
            content_hash="content-v1",
            search_settings_id=17,
            prompt_hash="prompt-v1",
            chunk_generation_hash="b" * 64,
            config_snapshot=cast(RegulatoryIndexingConfigSnapshot, config_snapshot),
            now=datetime.datetime(2026, 8, 19, tzinfo=datetime.timezone.utc),
        )

    db_session.scalar.assert_not_called()


def test_job_creation_rejects_unknown_input_hash_version_before_sql() -> None:
    db_session = MagicMock(spec=Session)

    with pytest.raises(ValueError, match="input hash version"):
        create_or_get_regulatory_indexing_job(
            cast(Session, db_session),
            user_file_id=uuid4(),
            content_hash="a" * 64,
            search_settings_id=17,
            prompt_hash="prompt-v1",
            chunk_generation_hash="b" * 64,
            config_snapshot={
                "input_content_hash": "a" * 64,
                "input_hash_version": "future-v3",
                "chunk_generation_hash": "b" * 64,
            },
            now=datetime.datetime(2026, 8, 19, tzinfo=datetime.timezone.utc),
        )

    db_session.scalar.assert_not_called()
