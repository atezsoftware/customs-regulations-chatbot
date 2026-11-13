"""Only completed New document imports enter the legislation catalog."""

from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    DocumentSet,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
)
from onyx.db.regulatory_catalog import (
    catalog_search_rows,
    list_catalog_files,
    list_catalog_folders,
)
from tests.external_dependency_unit.regulatory.test_amendment_published_source_identity import (
    _intake,
)
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_amendment_structural_source_identity import (
    _file,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk


@pytest.mark.parametrize(
    "state,visible",
    [
        ("failed", False),
        ("preparing", False),
        ("ready", False),
        ("approving", False),
        ("source_evidence", False),
        ("completed", True),
        ("legacy_unindexed", True),
    ],
)
def test_intake_visibility_agrees_across_files_folders_and_search(
    source_session: Session, state: str, visible: bool
) -> None:
    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    token = uuid4().hex
    folder = f"catalog-{token}"
    file.name = f"{folder}/{token}.md"
    row = _chunk(session, file, 0, "Unchanged source content.")
    if state != "legacy_unindexed":
        _intake(session, group, file, "completed")
        member = session.scalar(
            select(RegulatoryDocumentIntakeFile).where(
                RegulatoryDocumentIntakeFile.user_file_id == file.id
            )
        )
        assert member is not None
        batch = session.get_one(RegulatoryDocumentIntake, member.intake_id)
        batch.status = "completed" if state == "source_evidence" else state
        member.status = (
            "source_evidence"
            if state == "source_evidence"
            else ("completed" if visible else "ready")
        )
    if state != "completed":
        file.status = UserFileStatus.CHUNKED
    session.flush()
    files, total = list_catalog_files(
        session, q=token, folder=folder, recursive=False, offset=0, limit=25
    )
    assert [item.id for item in files] == ([file.id] if visible else [])
    assert total == int(visible)
    folders, total_folders = list_catalog_folders(
        session, parent="", q=token, offset=0, limit=25
    )
    assert folders == ([(folder, 1)] if visible else [])
    assert total_folders == int(visible)
    hits = catalog_search_rows(session, [row.id])
    assert set(hits) == ({row.id} if visible else set())
