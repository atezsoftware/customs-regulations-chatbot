"""Reviewed text edits dispatch metadata before starting replacement labels."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.background.celery.tasks.regulatory_labeling import tasks as labeling_tasks
from onyx.db.models import RegulatoryDocumentIntake
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataCheckpoint
from onyx.regulatory.document_intake.metadata_review import (
    ReviewedMetadataDocument,
    ReviewedMetadataEdit,
)
from onyx.server.features.regulatory import document_intake_api as api


def reviewed_batch(*, prior_run: bool) -> RegulatoryDocumentIntake:
    identifier, package_id = uuid4(), uuid4()
    run_id = uuid4() if prior_run else None
    intent = ReviewedMetadataEdit(
        intake_id=identifier,
        package_id=package_id,
        previous_labeling_run_id=run_id,
        expected_review_sha256="a" * 64,
        documents={
            uuid4(): ReviewedMetadataDocument(
                user_file_id=uuid4(),
                previous_input_sha256="b" * 64,
                input_sha256="c" * 64,
                canonical_source_sha256="d" * 64,
                revisions={"rc_graph": uuid4()},
                refresh_metadata=True,
            )
        },
    )
    checkpoint = MetadataCheckpoint.build(
        recipe_sha256="e" * 64,
        provider_binding={"model": "frozen"},
        source_snapshot={"reviewed_edit": intent.model_dump(mode="json")},
        request_hashes={},
    )
    return cast(
        RegulatoryDocumentIntake,
        SimpleNamespace(
            id=identifier,
            source_package_id=package_id,
            labeling_run_id=run_id,
            metadata_checkpoint=checkpoint.model_dump(mode="json"),
            status="preparing",
            stage="chunking",
        ),
    )


@pytest.mark.parametrize("prior_run", [False, True])
def test_reviewed_source_edit_prepares_metadata_before_any_new_labels(
    monkeypatch: pytest.MonkeyPatch, prior_run: bool
) -> None:
    batch = reviewed_batch(prior_run=prior_run)
    session = MagicMock(spec=Session)
    prepare, labels = MagicMock(), MagicMock()
    monkeypatch.setattr(api, "_enqueue_batch", prepare)
    monkeypatch.setattr(labeling_tasks, "enqueue_labeling_run", labels)
    api._enqueue_edited_labels(session, batch, None, "owner-tenant")
    prepare.assert_called_once_with(session, batch, "owner-tenant")
    labels.assert_not_called()
    session.commit.assert_not_called()


@pytest.mark.parametrize("change", ["intake", "package", "prior_run", "checksum"])
def test_changed_reviewed_dispatch_authority_cannot_enqueue_work(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    batch = reviewed_batch(prior_run=True)
    prepare = MagicMock()
    monkeypatch.setattr(api, "_enqueue_batch", prepare)
    if change == "intake":
        batch.id = uuid4()
    elif change == "package":
        batch.source_package_id = uuid4()
    elif change == "prior_run":
        batch.labeling_run_id = uuid4()
    else:
        assert batch.metadata_checkpoint is not None
        batch.metadata_checkpoint["request_hashes"] = {"tampered": ["f" * 64]}
    with pytest.raises(OnyxError):
        api._enqueue_edited_labels(MagicMock(spec=Session), batch, None, "tenant")
    prepare.assert_not_called()


@pytest.mark.parametrize(
    "state", ["ready", "labeling", "missing_checkpoint", "unreviewed"]
)
def test_noop_edits_and_other_stages_do_not_start_metadata(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    batch = reviewed_batch(prior_run=False)
    prepare = MagicMock()
    monkeypatch.setattr(api, "_enqueue_batch", prepare)
    if state == "ready":
        batch.status = "ready"
    elif state == "labeling":
        batch.stage = "labeling"
    elif state == "missing_checkpoint":
        batch.metadata_checkpoint = None
    else:
        batch.metadata_checkpoint = MetadataCheckpoint.build(
            recipe_sha256="e" * 64,
            provider_binding={},
            source_snapshot={},
            request_hashes={},
        ).model_dump(mode="json")
    api._enqueue_edited_labels(MagicMock(spec=Session), batch, None, "tenant")
    prepare.assert_not_called()


def test_metadata_only_edit_keeps_normal_label_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = reviewed_batch(prior_run=True)
    run_id = uuid4()
    prepare, labels = MagicMock(), MagicMock()
    monkeypatch.setattr(api, "_enqueue_batch", prepare)
    monkeypatch.setattr(labeling_tasks, "enqueue_labeling_run", labels)
    api._enqueue_edited_labels(MagicMock(spec=Session), batch, run_id, "tenant")
    labels.assert_called_once_with(run_id=run_id, tenant_id="tenant")
    prepare.assert_not_called()
