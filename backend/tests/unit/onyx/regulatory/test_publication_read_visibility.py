"""What a pinned regulatory read is allowed to hide.

Retrieval results pass through this check, so anything it hides disappears
from search with no error and no empty-result explanation.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from onyx.db import regulatory_publication
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import PublicationScope, ReadObservation
from onyx.regulatory.amendments.annexes.config import (
    ANNEX_DATABASE_IDENTITY,
    REGULATORY_ANNEX_ENVIRONMENT,
)


def _store() -> PublicationStore:
    return PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment=REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=ANNEX_DATABASE_IDENTITY,
        )
    )


def _unavailable(
    monkeypatch: pytest.MonkeyPatch, rows: list[Any], *, epoch: int = 5
) -> frozenset[UUID]:
    store = _store()

    @contextmanager
    def session(**_kwargs: object) -> Any:
        db_session = MagicMock()
        db_session.scalars.return_value = rows
        yield db_session

    monkeypatch.setattr(regulatory_publication, "get_session_with_tenant", session)
    monkeypatch.setattr(PublicationStore, "_check_session", lambda *_args: None)
    return store.unavailable(
        ReadObservation(scope=store.scope, committed_epoch=epoch),
        tuple(row.user_file_id for row in rows),
    )


def _row(*, scope_key: str, gate_closed: bool = False, epoch: int = 1) -> Any:
    return SimpleNamespace(
        user_file_id=uuid4(),
        scope_key=scope_key,
        gate_closed=gate_closed,
        epoch=epoch,
    )


def test_a_publication_written_under_another_scope_key_stays_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scope key folds in how a process reached the database.

    An API server and a worker on one deployment can derive different keys for
    the same publication, and hiding on that difference empties every result.
    """

    rows = [_row(scope_key="written-by-another-scope")]

    assert _unavailable(monkeypatch, rows) == frozenset()


def test_a_closed_gate_hides_the_file_whoever_closed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file being rewritten is unsafe to serve regardless of who owns it."""

    own = _row(scope_key=_store().scope_key, gate_closed=True)
    foreign = _row(scope_key="written-by-another-scope", gate_closed=True)
    rows = [own, foreign]

    assert _unavailable(monkeypatch, rows) == frozenset(
        {own.user_file_id, foreign.user_file_id}
    )


def test_an_epoch_past_the_snapshot_hides_only_this_scope_s_own_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Epochs are issued per scope, so a foreign epoch is not comparable."""

    own = _row(scope_key=_store().scope_key, epoch=9)
    foreign = _row(scope_key="written-by-another-scope", epoch=9)
    rows = [own, foreign]

    assert _unavailable(monkeypatch, rows, epoch=5) == frozenset({own.user_file_id})


def test_a_current_publication_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_row(scope_key=_store().scope_key, epoch=3)]

    assert _unavailable(monkeypatch, rows, epoch=5) == frozenset()
