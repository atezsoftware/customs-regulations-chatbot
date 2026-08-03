import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import RegulatoryChunkSource, RegulatoryChunkStatus
from onyx.db.models import RegulatoryChunk
from onyx.db.regulatory_chunks import (
    RegulatoryChunkValidityState,
    RegulatoryFileValidityWindow,
    apply_file_validity_window,
    common_reindex_validity_window,
    replace_indexed_chunks_for_file,
)
from onyx.regulatory.chunker import RegulatoryChunker


def _chunk(
    *,
    source: str = RegulatoryChunkSource.INDEXED.value,
    status: str = RegulatoryChunkStatus.ACTIVE.value,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    supersedes: str | None = None,
    superseded_by: str | None = None,
) -> RegulatoryChunk:
    return cast(
        RegulatoryChunk,
        SimpleNamespace(
            source=source,
            status=status,
            validity_start_date=start,
            validity_end_date=end,
            supersedes_chunk_id=supersedes,
            superseded_by_chunk_id=superseded_by,
        ),
    )


def _validity_state(
    *,
    source: str = RegulatoryChunkSource.INDEXED.value,
    status: str = RegulatoryChunkStatus.ACTIVE.value,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    supersedes: str | None = None,
    superseded_by: str | None = None,
) -> RegulatoryChunkValidityState:
    return RegulatoryChunkValidityState(
        source=source,
        status=status,
        validity_start_date=start,
        validity_end_date=end,
        supersedes_chunk_id=supersedes,
        superseded_by_chunk_id=superseded_by,
    )


def test_file_window_updates_only_unversioned_indexed_chunks() -> None:
    source_chunk = _chunk()
    superseded_chunk = _chunk(
        status=RegulatoryChunkStatus.SUPERSEDED.value,
        end=datetime.date(2026, 3, 15),
        superseded_by="new",
    )
    amendment_chunk = _chunk(
        source=RegulatoryChunkSource.AMENDMENT.value,
        start=datetime.date(2026, 3, 15),
        supersedes="old",
    )

    result = apply_file_validity_window(
        [source_chunk, superseded_chunk, amendment_chunk],
        validity_start_date=datetime.date(2025, 1, 1),
    )

    assert result.updated_chunk_count == 1
    assert result.skipped_versioned_chunk_count == 2
    assert result.previous_window == RegulatoryFileValidityWindow(None, None)
    assert result.updated_window == RegulatoryFileValidityWindow(
        datetime.date(2025, 1, 1), None
    )
    assert source_chunk.validity_start_date == datetime.date(2025, 1, 1)
    assert superseded_chunk.validity_start_date is None
    assert superseded_chunk.validity_end_date == datetime.date(2026, 3, 15)
    assert amendment_chunk.validity_start_date == datetime.date(2026, 3, 15)


def test_invalid_file_window_does_not_partially_mutate_chunks() -> None:
    first = _chunk(start=datetime.date(2020, 1, 1))
    second = _chunk(start=datetime.date(2021, 1, 1))

    with pytest.raises(ValueError, match="earlier"):
        apply_file_validity_window(
            [first, second],
            validity_start_date=datetime.date(2026, 1, 1),
            validity_end_date=datetime.date(2026, 1, 1),
        )

    assert first.validity_start_date == datetime.date(2020, 1, 1)
    assert first.validity_end_date is None
    assert second.validity_start_date == datetime.date(2021, 1, 1)
    assert second.validity_end_date is None


def test_file_window_can_clear_one_boundary_without_touching_the_other() -> None:
    chunk = _chunk(
        start=datetime.date(2025, 1, 1),
        end=datetime.date(2026, 1, 1),
    )

    apply_file_validity_window([chunk], validity_end_date=None)

    assert chunk.validity_start_date == datetime.date(2025, 1, 1)
    assert chunk.validity_end_date is None


def test_reindex_window_requires_one_uniform_unversioned_source() -> None:
    window = RegulatoryFileValidityWindow(
        start=datetime.date(2025, 1, 1),
        end=None,
    )
    assert (
        common_reindex_validity_window(
            [
                _validity_state(start=window.start, end=window.end),
                _validity_state(start=window.start, end=window.end),
            ]
        )
        == window
    )

    assert common_reindex_validity_window([]) is None
    assert (
        common_reindex_validity_window(
            [
                _validity_state(start=window.start),
                _validity_state(start=datetime.date(2026, 1, 1)),
            ]
        )
        is None
    )
    assert (
        common_reindex_validity_window(
            [
                _validity_state(
                    start=datetime.date(2026, 1, 1),
                    end=datetime.date(2026, 1, 1),
                )
            ]
        )
        is None
    )


@pytest.mark.parametrize(
    "versioned_chunk",
    [
        _validity_state(source=RegulatoryChunkSource.AMENDMENT.value),
        _validity_state(status=RegulatoryChunkStatus.SUPERSEDED.value),
        _validity_state(supersedes="old"),
        _validity_state(superseded_by="new"),
    ],
)
def test_reindex_window_never_spreads_versioned_history(
    versioned_chunk: RegulatoryChunkValidityState,
) -> None:
    assert (
        common_reindex_validity_window(
            [
                _validity_state(start=datetime.date(2025, 1, 1)),
                versioned_chunk,
            ]
        )
        is None
    )


def test_replace_indexed_chunks_preserves_uniform_file_window() -> None:
    user_file_id = uuid4()
    existing = [
        _validity_state(start=datetime.date(2025, 1, 1)),
        _validity_state(start=datetime.date(2025, 1, 1)),
    ]
    db_session = MagicMock(spec=Session)
    db_session.execute.return_value.all.return_value = existing
    parsed = RegulatoryChunker().chunk_text(
        "MADDE 1 - (1) Birinci hüküm.\n\nMADDE 2 - (1) İkinci hüküm.",
        source_file="snapshot.md",
    )

    rows = replace_indexed_chunks_for_file(
        db_session,
        user_file_id,
        parsed.chunks,
    )

    assert rows
    assert all(row.validity_start_date == datetime.date(2025, 1, 1) for row in rows)
    assert all(row.validity_end_date is None for row in rows)
    assert db_session.add.call_count == len(rows)
    assert db_session.execute.call_count == 2


def test_first_index_and_mixed_reindex_leave_new_rows_unbounded() -> None:
    parsed = RegulatoryChunker().chunk_text(
        "MADDE 1 - (1) Hüküm.", source_file="source.md"
    )
    for existing in (
        [],
        [
            _validity_state(start=datetime.date(2025, 1, 1)),
            _validity_state(start=datetime.date(2026, 1, 1)),
        ],
    ):
        db_session = MagicMock(spec=Session)
        db_session.execute.return_value.all.return_value = existing

        rows = replace_indexed_chunks_for_file(
            db_session,
            uuid4(),
            parsed.chunks,
        )

        assert rows
        assert all(row.validity_start_date is None for row in rows)
        assert all(row.validity_end_date is None for row in rows)
