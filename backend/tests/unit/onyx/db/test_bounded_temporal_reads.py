from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4

from onyx.db.regulatory_annex_publication import load_file_temporal_bindings


def test_temporal_payload_query_is_bounded_before_hydration() -> None:
    session = MagicMock()
    session.scalars.return_value = []
    assert (
        load_file_temporal_bindings(
            session,
            uuid4(),
            canonical_chunk_ids=("wanted",),
            index_uuid="live-index",
            as_of_date=date(2026, 1, 1),
        )
        == []
    )
    statement = session.scalars.call_args.args[0]
    query = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "canonical_chunk_id IN ('wanted')" in query
    assert "index_uuid = 'live-index'" in query
    assert "retired_at IS NULL" in query
    assert (
        "effective_start IS NULL" in query
        and "effective_start <= '2026-01-01'" in query
    )
    assert "effective_end IS NULL" in query and "effective_end > '2026-01-01'" in query


def test_empty_identity_scope_never_expands_to_whole_file() -> None:
    session = MagicMock()
    assert load_file_temporal_bindings(session, uuid4(), canonical_chunk_ids=()) == []
    session.scalars.assert_not_called()
