from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from onyx.auth.schemas import UserRole
from onyx.db.document_set import (
    _add_user_filters,
    link_user_file_to_document_set,
    mark_document_set_as_to_be_deleted,
    unlink_user_file_from_document_set,
)
from onyx.db.enums import UserFileStatus
from onyx.db.models import DocumentSet


def test_curator_creator_access_is_not_constrained_by_group_membership() -> None:
    user_id = uuid4()
    user = MagicMock(role=UserRole.CURATOR, is_anonymous=False, id=user_id)

    stmt = _add_user_filters(select(DocumentSet), user, get_editable=True)
    compiled_sql = str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert compiled_sql.rstrip().endswith(f"OR document_set.user_id = '{user_id}')")


@pytest.mark.parametrize(
    "status",
    [
        UserFileStatus.PROCESSING,
        UserFileStatus.INDEXING,
        UserFileStatus.COMPLETED,
        UserFileStatus.FAILED,
    ],
)
def test_link_and_unlink_mark_document_set_metadata_dirty(
    status: UserFileStatus,
) -> None:
    db_session = MagicMock()
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.status = status
    db_session.get.return_value = None

    linked = link_user_file_to_document_set(db_session, 7, user_file)

    assert linked is True
    assert user_file.needs_document_set_sync is True
    db_session.add.assert_called_once()
    db_session.commit.assert_called_once()

    db_session.reset_mock()
    association = MagicMock()
    db_session.get.return_value = association
    user_file.needs_document_set_sync = False

    unlinked = unlink_user_file_from_document_set(db_session, 7, user_file)

    assert unlinked is True
    assert user_file.needs_document_set_sync is True
    db_session.delete.assert_called_once_with(association)
    db_session.commit.assert_called_once()


@pytest.mark.parametrize(
    "status",
    [
        UserFileStatus.SKIPPED,
        UserFileStatus.CANCELED,
        UserFileStatus.DELETING,
    ],
)
def test_link_and_unlink_do_not_dirty_terminal_user_files(
    status: UserFileStatus,
) -> None:
    db_session = MagicMock()
    user_file = MagicMock(
        id=uuid4(),
        status=status,
        needs_document_set_sync=False,
    )
    db_session.get.return_value = None

    assert link_user_file_to_document_set(db_session, 7, user_file) is True
    assert user_file.needs_document_set_sync is False

    db_session.get.return_value = MagicMock()

    assert unlink_user_file_from_document_set(db_session, 7, user_file) is True
    assert user_file.needs_document_set_sync is False


def test_delete_preflight_rejects_benchmark_references_before_mutation() -> None:
    db_session = MagicMock()
    user = MagicMock()
    document_set = MagicMock(is_up_to_date=True)

    with (
        patch(
            "onyx.db.document_set.get_document_set_by_id_for_user",
            return_value=document_set,
        ),
        patch(
            "onyx.db.document_set.document_set_has_benchmark_questions",
            return_value=True,
        ),
        patch(
            "onyx.db.document_set._delete_document_set_cc_pairs__no_commit"
        ) as delete_connector_links,
        patch(
            "onyx.db.document_set._delete_document_set_user_files__no_commit"
        ) as delete_file_links,
    ):
        with pytest.raises(ValueError, match="benchmark questions"):
            mark_document_set_as_to_be_deleted(
                db_session=db_session,
                document_set_id=7,
                user=user,
            )

    delete_connector_links.assert_not_called()
    delete_file_links.assert_not_called()
    db_session.rollback.assert_called_once()
    db_session.commit.assert_not_called()
