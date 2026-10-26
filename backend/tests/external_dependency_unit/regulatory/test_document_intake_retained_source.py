from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    User,
    UserFile,
)
from onyx.file_store.file_store import FileStore
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import clone_intake_original
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
    _package,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


def test_deleted_import_retains_source_and_history_and_cannot_be_approved(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    owner, target, model = intake_owner
    package, assets, links = _package(intake_session, owner, target)
    batch, _created = intake.create_intake(
        intake_session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    store = PostgresBackedFileStore()
    bound_store = cast(
        FileStore,
        SimpleNamespace(
            read_file=lambda identifier: store.read_file(
                identifier, db_session=intake_session
            ),
            save_file=lambda content, **kwargs: store.save_file(
                content, **kwargs, db_session=intake_session
            ),
        ),
    )
    originals = {}
    for asset in assets:
        with store.read_file(asset.file_id, db_session=intake_session) as stream:
            originals[asset.file_id] = stream.read()
    copies = {
        asset.id: clone_intake_original(
            intake_id=batch.id,
            asset=IntakeSourceAsset.model_validate(asset),
            store=bound_store,
        )
        for asset in assets
    }
    members = intake.admit_intake_assets(
        intake_session,
        batch,
        assets,
        links,
        root_hash=assets[0].sha256,
        owned_file_ids=copies,
    )
    member = next(item for item in members if item.role == "annex")
    file = intake_session.get_one(UserFile, member.user_file_id)
    assert file.file_id == copies[member.source_asset_id]
    assert file.file_id not in originals
    batch.status, batch.stage = "ready", "review"
    batch.review_sha256 = intake.review_digest(intake_session, batch)
    reviewed = batch.review_sha256
    assert reviewed
    original_name = file.name
    stored_copy = file.file_id
    intake_session.execute(
        delete(DocumentSet__UserFile).where(
            DocumentSet__UserFile.user_file_id == file.id
        )
    )
    store.delete_file(stored_copy, db_session=intake_session)
    intake_session.delete(file)
    intake_session.flush()

    snapshot = intake.intake_snapshot(intake_session, batch)
    deleted = next(
        item for item in snapshot["documents"] if item["id"] == str(member.id)
    )
    assert deleted["status"] == "deleted"
    assert deleted["display_name"] == original_name
    assert deleted["parent_document_id"] is not None
    assert intake.intake_files(intake_session, batch)
    for source_id, original in originals.items():
        with store.read_file(source_id, db_session=intake_session) as stream:
            assert stream.read() == original
    with pytest.raises(intake.IntakeConflictError, match="membership changed"):
        intake.approve_intake(intake_session, batch, reviewed)
