"""Fresh label snapshots follow metadata completion, including prelabel reviews."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4, uuid5

import pytest

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_metadata as metadata
from onyx.db import document_intake_metadata_checkpoint as checkpoints
from onyx.db import document_intake_metadata_review as review
from onyx.db import document_intake_reprepare as reprepare
from onyx.db import document_intake_structural_readers as readers
from onyx.db import regulatory_labeling as labeling
from onyx.db.models import RegulatoryDocumentIntake
from onyx.regulatory.document_intake.metadata import (
    ChunkLegalMetadata,
    LegalMetadataResult,
    metadata_input_hash,
)
from tests.unit.onyx.regulatory.test_intake_reviewed_metadata_work import _work


@pytest.mark.parametrize("fresh", [False, True])
def test_reviewed_source_never_reuses_completed_old_label_snapshots(
    monkeypatch: pytest.MonkeyPatch, fresh: bool
) -> None:
    session, batch, members, rows, _unchanged = _work(monkeypatch, labeled=True)
    assert isinstance(batch.lease_token, UUID)
    old_id = batch.labeling_run_id
    old = SimpleNamespace(
        id=old_id,
        status="completed",
        file_ids=[str(m.user_file_id) for m in members],
        taxonomy={},
        uses_current_labels=True,
    )
    replacement = SimpleNamespace(id=uuid4())
    create = MagicMock(return_value=(replacement, True))
    binding = SimpleNamespace(model="gemini-3.5-flash", model_dump=lambda **_kw: {})
    batch.document_set_id, batch.created_by, batch.labeling_model_configuration_id = (
        15,
        uuid4(),
        1,
    )
    for member in members:
        member.status = "chunked"
    session.get.return_value = old
    monkeypatch.setattr(reprepare, "latest_reprepare", lambda *_args: None)
    monkeypatch.setattr(labeling, "get_active_run_id", lambda *_args: None)
    monkeypatch.setattr(labeling, "create_labeling_run", create)
    monkeypatch.setattr(
        intake,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kw: binding,
    )
    monkeypatch.setattr(
        readers, "load_intake_structural_reader_proofs", lambda *_args, **_kw: {}
    )
    monkeypatch.setattr(
        editing, "reuse_completed_intake_labels", lambda *_args, **_kw: None
    )
    monkeypatch.setattr(metadata, "refresh_intake_document_name", lambda *_args: None)
    monkeypatch.setattr(metadata, "heartbeat_metadata", lambda *_args: None)
    before = deepcopy(batch.metadata_checkpoint)
    if not fresh:
        with pytest.raises(
            intake.IntakeConflictError, match="must be current before labeling"
        ):
            intake._start_intake_labeling_for_batch(session, batch)
        create.assert_not_called()
        assert batch.labeling_run_id == old_id
        return
    metadata.apply_metadata_results(
        session,
        intake_id=batch.id,
        lease_token=batch.lease_token,
        document_id=members[0].id,
        expected_input_sha256=metadata_input_hash(metadata.metadata_chunks(rows)),
        results=[
            (
                f"{offset + 1:064x}",
                LegalMetadataResult(
                    chunks=[
                        ChunkLegalMetadata(chunk_id=r.id, fields=[])
                        for r in rows[offset : offset + 8]
                    ]
                ),
            )
            for offset in range(0, len(rows), 8)
        ],
    )
    assert intake._start_intake_labeling_for_batch(session, batch) == replacement.id
    create.assert_called_once()
    assert create.call_args.kwargs["idempotency_key"] != uuid5(batch.id, str(old_id))
    assert batch.labeling_run_id != old_id and old.status == "completed"
    assert batch.metadata_checkpoint == before


@pytest.mark.parametrize(
    "change,allowed",
    [
        ("none", True),
        ("active_set", False),
        ("lease", False),
        ("foreign_source", False),
        ("existing_run", False),
        ("indexing", False),
    ],
)
def test_prelabel_edit_digest_is_an_idle_frozen_source_review_only(
    monkeypatch: pytest.MonkeyPatch, change: str, allowed: bool
) -> None:
    batch = RegulatoryDocumentIntake(
        id=uuid4(),
        source_package_id=uuid4(),
        document_set_id=15,
        status="failed",
        stage="metadata",
    )
    if change == "lease":
        batch.lease_token = uuid4()
    elif change == "existing_run":
        batch.labeling_run_id = uuid4()
    elif change == "indexing":
        batch.stage = "indexing"
    monkeypatch.setattr(
        labeling,
        "get_active_run_id",
        lambda *_args: uuid4() if change == "active_set" else None,
    )
    monkeypatch.setattr(intake, "review_digest", lambda *_args: "a" * 64)
    source: dict[str, Any] = {
        "intake_id": str(batch.id),
        "documents": {"member": {"input_sha256": "b" * 64}},
    }

    def snapshot(
        *_args: object, for_review: bool
    ) -> tuple[dict[str, Any], dict[str, list[str]], set[str]]:
        assert for_review is True
        if change == "foreign_source":
            raise intake.IntakeConflictError("Source ownership changed")
        return source, {}, set()

    monkeypatch.setattr(checkpoints, "_source_snapshot", snapshot)
    digest = review.prelabel_metadata_edit_digest(MagicMock(), batch)
    assert bool(digest) is allowed
    assert batch.status == "failed"
    if allowed:
        source["documents"]["member"]["input_sha256"] = "c" * 64
        assert review.prelabel_metadata_edit_digest(MagicMock(), batch) != digest
