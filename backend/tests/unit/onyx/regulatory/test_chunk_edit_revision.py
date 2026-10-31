"""Stale editor revisions must fail before publication or embedding work."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from threading import Event
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.regulatory_chunk_edits import (
    ChunkEditConflict,
    require_owned_chunk_revision,
)
from onyx.regulatory.writer_publication import correct_owned_chunk

DB_MODULE = "onyx.db.regulatory_chunk_edits"


@pytest.mark.parametrize("change", ["timestamp", "file", "deleted", "unchanged"])
def test_revision_is_checked_under_file_ownership(change: str) -> None:
    owner = MagicMock()
    owner.user_file_id = uuid4()
    expected = datetime(2026, 11, 1, tzinfo=timezone.utc)
    chunk = MagicMock()
    chunk.user_file_id = owner.user_file_id
    chunk.updated_at = expected
    if change == "timestamp":
        chunk.updated_at += timedelta(seconds=1)
    elif change == "file":
        chunk.user_file_id = uuid4()
    session = MagicMock()
    session.get.return_value = None if change == "deleted" else chunk
    session_factory = MagicMock()
    session_factory.return_value.__enter__.return_value = session
    with (
        patch(f"{DB_MODULE}.get_session_with_tenant", session_factory),
        patch(f"{DB_MODULE}.PublicationStore") as store,
    ):
        if change == "unchanged":
            require_owned_chunk_revision(owner, "chunk-id", expected)
        else:
            with pytest.raises(ChunkEditConflict, match="has changed"):
                require_owned_chunk_revision(owner, "chunk-id", expected)
        store.return_value.lock_owned_snapshot.assert_called_once_with(session, owner)
    session.commit.assert_not_called()


def test_stale_revision_stops_before_loading_or_publishing_correction() -> None:
    owner, client = MagicMock(), MagicMock()
    expected = datetime(2026, 11, 1, tzinfo=timezone.utc)

    @contextmanager
    def heartbeat(_owner: object):
        yield Event()

    with (
        patch(
            "onyx.regulatory.writer_publication.recover_owned_writer_before_next",
            return_value=owner,
        ) as recover,
        patch("onyx.regulatory.writer_publication.publication_heartbeat", heartbeat),
        patch(
            f"{DB_MODULE}.require_owned_chunk_revision",
            side_effect=ChunkEditConflict("has changed"),
        ) as check,
        patch("onyx.db.regulatory_writer_publication.load_owned_writer_inputs") as load,
        patch("onyx.regulatory.writer_projection.prepare_owned_correction") as prepare,
        patch(
            "onyx.regulatory.writer_publication.execute_writer_publication"
        ) as publish,
    ):
        with pytest.raises(ChunkEditConflict):
            correct_owned_chunk(
                owner,
                client,
                "chunk-id",
                expected_updated_at=expected,
                text="new text",
            )
        recover.assert_called_once_with(owner)
        check.assert_called_once_with(owner, "chunk-id", expected)
        load.assert_not_called()
        prepare.assert_not_called()
        publish.assert_not_called()
        client.assert_not_called()
