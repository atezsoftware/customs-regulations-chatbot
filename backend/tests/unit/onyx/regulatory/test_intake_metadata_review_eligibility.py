"""A terminal metadata failure can be reviewed without granting publication."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_review as review
from onyx.db import regulatory_labeling as labeling
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryLabelingRun,
    UserFile,
)


@pytest.mark.parametrize(
    "change,allowed",
    [
        ("none", True),
        ("index_stage", False),
        ("live_lease", False),
        ("active_run", False),
        ("nonterminal_run", False),
        ("wrong_cohort", False),
        ("published_file", False),
        ("index_attempt", False),
        ("pending_shard", False),
        ("receipt_chunking_frozen", True),
        ("receipt_chunking_missing", False),
        ("receipt_chunking_stale", False),
        ("receipt_chunking_wrong_hash", False),
    ],
)
def test_failed_metadata_review_keeps_all_existing_ownership_fences(
    monkeypatch: pytest.MonkeyPatch, change: str, allowed: bool
) -> None:
    fid, run_id = uuid4(), uuid4()
    batch = cast(
        RegulatoryDocumentIntake,
        SimpleNamespace(
            status="failed",
            stage="metadata",
            error="Saved locator evidence failed validation",
            labeling_run_id=run_id,
            document_set_id=15,
            lease_token=None,
            lease_expires_at=None,
        ),
    )
    run = SimpleNamespace(
        id=run_id,
        file_ids=[str(fid)],
        status="completed_with_errors",
        error="One unresolved label",
        lease_token=None,
        lease_expires_at=None,
    )
    member = SimpleNamespace(user_file_id=fid, approval_attempt_id=None)
    file = SimpleNamespace(id=fid, status=UserFileStatus.CHUNKED)
    session = MagicMock(spec=Session)
    session.get.side_effect = lambda model, _id: (
        run if model is RegulatoryLabelingRun else file if model is UserFile else None
    )
    session.scalars.return_value = []
    session.scalar.return_value = None
    monkeypatch.setattr(review, "effective_intake_files", lambda *_args: [member])
    monkeypatch.setattr(intake, "review_digest", lambda *_args: "canonical")
    monkeypatch.setattr(editing, "get_user_file_projection_repair", lambda *_args: None)
    monkeypatch.setattr(
        labeling,
        "get_active_run_id",
        lambda *_args: uuid4() if change == "active_run" else None,
    )
    if change == "index_stage":
        batch.stage = "indexing"
    elif change == "live_lease":
        batch.lease_token = uuid4()
    elif change == "nonterminal_run":
        run.status = "running"
    elif change == "wrong_cohort":
        run.file_ids = [str(uuid4())]
    elif change == "published_file":
        file.status = UserFileStatus.COMPLETED
    elif change == "index_attempt":
        member.approval_attempt_id = uuid4()
    elif change == "pending_shard":
        session.scalar.return_value = uuid4()
    if change.startswith("receipt_chunking"):
        from onyx.db.document_intake_metadata import metadata_chunks
        from onyx.regulatory.document_intake.metadata import (
            METADATA_AUDIT_KEY,
            METADATA_MODEL,
            METADATA_VERSION,
            metadata_input_hash,
        )

        batch.stage = "chunking"
        row = RegulatoryChunk(
            id="source",
            user_file_id=fid,
            text="(1) Kaynak hüküm.",
            heading_path=["MADDE 1"],
            chunk_type="paragraph",
            chunk_metadata={},
        )
        row.chunk_metadata = {
            METADATA_AUDIT_KEY: {
                "model": METADATA_MODEL,
                "version": METADATA_VERSION,
                "input_sha256": metadata_input_hash(metadata_chunks([row])),
                "input_stale": change == "receipt_chunking_stale",
                "request_hash": "a" * 64,
            }
        }
        if change == "receipt_chunking_wrong_hash":
            row.chunk_metadata[METADATA_AUDIT_KEY]["input_sha256"] = "b" * 64
        session.scalars.side_effect = [
            [] if change == "receipt_chunking_missing" else [row],
            [],
            [row],
            [],
        ]
    digest = editing.edit_review_digest(session, batch)
    assert bool(digest) is allowed
    if allowed:
        batch.error = "A different failure occurred"
        assert editing.edit_review_digest(session, batch) != digest
        assert batch.status == "failed" and batch.stage in {"metadata", "chunking"}
        assert batch.labeling_run_id == run_id
