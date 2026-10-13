"""Exact search-hit batching preserves the ordinary public temporal reader."""

import json
from contextlib import nullcontext
from datetime import date
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.asv3.models import RunContext
from onyx.asv3.search_adapter import build_search_adapter
from onyx.db import regulatory_public_reads as reads
from onyx.db.asv3_candidate_inventory import (
    asv3_source_inventory_scope,
    current_asv3_source_inventory_scope,
)
from onyx.db.models import RegulatoryCanonicalRevision, RegulatoryChunk
from onyx.document_index.publication_models import (
    PublicationScope,
    ReadObservation,
    publication_digest,
)
from onyx.regulatory import publication_reads
from tests.unit.onyx.asv3.test_search_adapter import (
    search_boundaries,
    tool_and_broker,
    user_message,
)
from tests.unit.onyx.db.test_public_temporal_read_batches import (
    AS_OF,
    FILE_ID,
    INDEX,
    READ_BATCH_SIZE,
    _add_binding,
    _TrackingConnection,
)
from tests.unit.onyx.db.test_public_temporal_read_batches import session as session
from tests.unit.onyx.legal_composite.test_native_trace import client

SESSION_FACTORY = "onyx.db.engine.sql_engine.get_session_with_current_tenant"
SECOND_FILE = UUID("00000000-0000-0000-0000-000000000002")


def test_batch_matches_full_legacy_validation_with_one_bounded_cursor(
    session: Session,
) -> None:
    file_ids = (SECOND_FILE, FILE_ID, uuid4())
    slots = {file_id: tuple(range(12)) for file_id in file_ids[:2]}
    for file_id in file_ids[:2]:
        _add_binding(session, 0, file_id=file_id)
        _add_binding(session, 1, file_id=file_id, start=AS_OF)
        _add_binding(session, 2, file_id=file_id, end=AS_OF)
        _add_binding(session, 3, file_id=file_id, start=date(2027, 1, 1))
        _add_binding(session, 4, file_id=file_id, retired=True)
        _add_binding(session, 5, file_id=file_id, canonical_end=AS_OF)
        _add_binding(
            session,
            6,
            file_id=file_id,
            index=INDEX.model_copy(update={"model_name": "other"}),
        )
        _add_binding(
            session,
            7,
            file_id=file_id,
            index=INDEX.model_copy(update={"index_uuid": "other"}),
        )
        missing = _add_binding(session, 8, file_id=file_id)
        wrong_source = _add_binding(session, 9, file_id=file_id)
        _add_binding(session, 10, file_id=file_id)
        legacy_revision = _add_binding(session, 11, file_id=file_id)
        legacy_revision.canonical_revision_id = None
        session.flush()
        canonical = session.get(RegulatoryChunk, missing.canonical_chunk_id)
        assert canonical is not None
        session.delete(canonical)
        canonical = session.get(RegulatoryChunk, wrong_source.canonical_chunk_id)
        assert canonical is not None
        canonical.user_file_id = uuid4()
    session.flush()
    session.expunge_all()
    driver = cast(_TrackingConnection, session.info["driver"])
    with patch(SESSION_FACTORY, side_effect=lambda: nullcontext(session)):
        driver.cursors.clear()
        expected = reads.query_temporal_bindings(
            file_ids, index=INDEX, as_of_date=AS_OF, projection_ordinals=slots
        )
        legacy_queries = len(driver.cursors)
        driver.cursors.clear()
        driver.fetch_sizes.clear()
        driver.fetchall_count = 0
        with asv3_source_inventory_scope(
            scope_key="cropped", batch_temporal_reads=True
        ):
            actual = reads.query_temporal_bindings(
                file_ids, index=INDEX, as_of_date=AS_OF, projection_ordinals=slots
            )
    assert actual == expected
    assert list(actual) == list(file_ids)
    assert actual[file_ids[2]] == []
    assert [b.projection.ordinal for b in actual[FILE_ID]] == [11, 10, 1, 0]
    assert len(driver.cursors) == 1 < legacy_queries
    assert driver.fetchall_count == 0
    assert max(driver.fetch_sizes) <= READ_BATCH_SIZE
    assert all(cursor.closed for cursor in driver.cursors)


def test_exact_source_ordinal_pairs_never_cross_match_or_widen_empty_scope(
    session: Session,
) -> None:
    other = uuid4()
    for file_id in (FILE_ID, SECOND_FILE, other):
        for ordinal in range(3):
            row = _add_binding(session, ordinal, file_id=file_id)
            if (file_id, ordinal) not in ((FILE_ID, 1), (SECOND_FILE, 2)):
                # Accidentally reading a Cartesian pair would fail validation.
                row.payload_sha256 = "0" * 64
    session.flush()
    with (
        patch(SESSION_FACTORY, side_effect=lambda: nullcontext(session)),
        asv3_source_inventory_scope(scope_key="cropped", batch_temporal_reads=True),
    ):
        actual = reads.query_temporal_bindings(
            (FILE_ID, SECOND_FILE),
            index=INDEX,
            as_of_date=AS_OF,
            projection_ordinals={FILE_ID: (1, 1), SECOND_FILE: (2,), other: (0,)},
        )
        assert {
            key: [b.projection.ordinal for b in value] for key, value in actual.items()
        } == {
            FILE_ID: [1],
            SECOND_FILE: [2],
        }
        driver = cast(_TrackingConnection, session.info["driver"])
        driver.cursors.clear()
        empty = reads.query_temporal_bindings(
            (FILE_ID, SECOND_FILE),
            index=INDEX,
            as_of_date=AS_OF,
            projection_ordinals={FILE_ID: (), other: (0,)},
        )
        assert empty == {FILE_ID: [], SECOND_FILE: []}
        assert not driver.cursors


@pytest.mark.parametrize(
    "corruption,error",
    [
        ("payload", "temporal binding payload changed"),
        ("encoder", "encoder receipt is not accepted"),
        ("canonical", "canonical revision payload changed"),
        ("missing_revision", "canonical revision does not exist"),
        ("revision_scope", "canonical revision scope mismatch"),
        ("revision_authority", "temporal canonical revision authority mismatch"),
        ("lookup_identity", "temporal binding lookup identity mismatch"),
        ("source_identity", "temporal canonical revision authority mismatch"),
    ],
)
def test_mutated_selected_source_fails_closed_in_both_paths(
    session: Session, corruption: str, error: str
) -> None:
    _add_binding(session, 0)
    row = _add_binding(session, 0, file_id=SECOND_FILE)
    session.flush()
    if corruption == "payload":
        row.payload_sha256 = "0" * 64
    elif corruption == "encoder":
        row.payload = {
            **row.payload,
            "projection": {
                **row.payload["projection"],
                "embedding_config_json": '{"provider":"unaccepted"}',
            },
        }
        row.payload_sha256 = publication_digest(row.payload)
    elif corruption == "lookup_identity":
        row.canonical_chunk_id = "wrong-canonical"
        row.canonical_revision_id = None
    elif corruption == "source_identity":
        row.user_file_id = uuid4()
    else:
        revision = session.get(RegulatoryCanonicalRevision, row.canonical_revision_id)
        assert revision is not None
        if corruption == "missing_revision":
            session.delete(revision)
        elif corruption == "revision_scope":
            revision.canonical_chunk_id = "wrong-canonical"
        elif corruption == "revision_authority":
            row.payload = {**row.payload, "canonical_base_sha256": "0" * 64}
            row.payload_sha256 = publication_digest(row.payload)
        else:
            revision.payload_sha256 = "0" * 64
    file_ids = (FILE_ID, row.user_file_id)
    session.flush()
    driver = cast(_TrackingConnection, session.info["driver"])
    for enabled in (False, True):
        driver.cursors.clear()
        with (
            patch(SESSION_FACTORY, side_effect=lambda: nullcontext(session)),
            asv3_source_inventory_scope(scope_key="test", batch_temporal_reads=enabled),
            pytest.raises(ValueError, match=error),
        ):
            reads.query_temporal_bindings(
                file_ids,
                index=INDEX,
                as_of_date=AS_OF,
                projection_ordinals={file_id: (0,) for file_id in file_ids},
            )
        assert all(cursor.closed for cursor in driver.cursors)


@pytest.mark.parametrize("stop_check", [1, 3, 4])
def test_cancellation_returns_no_partial_group_and_closes_cursor(
    session: Session, stop_check: int
) -> None:
    for file_id in (FILE_ID, SECOND_FILE):
        _add_binding(session, 0, file_id=file_id)
    session.flush()
    driver = cast(_TrackingConnection, session.info["driver"])
    driver.cursors.clear()
    count = 0

    def check_active() -> None:
        nonlocal count
        count += 1
        if count == stop_check:
            raise RuntimeError("cancelled")

    with (
        patch(SESSION_FACTORY, side_effect=lambda: nullcontext(session)),
        asv3_source_inventory_scope(
            scope_key="cropped", batch_temporal_reads=True, check_active=check_active
        ),
        pytest.raises(RuntimeError, match="cancelled"),
    ):
        reads.query_temporal_bindings(
            (FILE_ID, SECOND_FILE),
            index=INDEX,
            as_of_date=AS_OF,
            projection_ordinals={FILE_ID: (0,), SECOND_FILE: (0,)},
        )
    assert all(cursor.closed for cursor in driver.cursors)
    assert len(driver.cursors) == (0 if stop_check == 1 else 1)


@pytest.mark.parametrize(
    "scoped,enabled,ordinals",
    [(False, False, (0,)), (True, False, (0,)), (True, True, None)],
)
def test_original_paths_never_enter_batch_reader(
    session: Session, scoped: bool, enabled: bool, ordinals: tuple[int, ...] | None
) -> None:
    _add_binding(session, 0)
    session.flush()
    with (
        patch(SESSION_FACTORY, side_effect=lambda: nullcontext(session)),
        patch.object(
            reads,
            "_query_temporal_binding_pairs",
            side_effect=AssertionError("must stay legacy"),
        ),
        asv3_source_inventory_scope(scope_key="original", batch_temporal_reads=enabled)
        if scoped
        else nullcontext(),
    ):
        actual = reads.query_temporal_bindings(
            (FILE_ID,),
            index=INDEX,
            as_of_date=AS_OF,
            projection_ordinals={FILE_ID: ordinals} if ordinals is not None else None,
        )
    assert len(actual[FILE_ID]) == 1


@pytest.mark.parametrize("cropped", [None, False, "true", True])
def test_search_adapter_enables_batch_only_for_literal_cropped_true(
    cropped: object,
) -> None:
    tool, broker, _ = tool_and_broker()
    adapter = build_search_adapter(
        tool, "question", broker, message_history=lambda _: [user_message("question")]
    )
    seen: list[bool] = []

    def capture_scope(**_kwargs: object) -> list[object]:
        scope = current_asv3_source_inventory_scope()
        assert scope is not None
        seen.append(scope.batch_temporal_reads)
        return []

    with search_boundaries() as (pipeline, _scope, _time):
        pipeline.side_effect = capture_scope
        adapter(
            {"query": "question", "mode": "keyword", "expand_query": False},
            RunContext(services={"asv3_exp_cropped": cropped}),
        )
    assert seen and all(value == (cropped is True) for value in seen)
    assert current_asv3_source_inventory_scope() is None


def test_nested_inventory_scope_does_not_leak_temporal_opt_in() -> None:
    with asv3_source_inventory_scope(
        scope_key="cropped", batch_temporal_reads=True
    ) as outer:
        with asv3_source_inventory_scope(scope_key="ordinary") as inner:
            assert not inner.batch_temporal_reads
        assert current_asv3_source_inventory_scope() is outer
    assert current_asv3_source_inventory_scope() is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("publication_changed", [False, True])
def test_real_index_reader_retains_final_publication_fence_after_temporal_read(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    publication_changed: bool,
) -> None:
    row = _add_binding(session, 0)
    session.flush()
    sdk = MagicMock()
    sdk.search.return_value = {
        "took": 1,
        "hits": {
            "hits": [
                {
                    "_id": "authorized-hit",
                    "_score": 1.0,
                    "_source": json.loads(row.payload["projection"]["source_json"]),
                }
            ]
        },
    }
    sdk.indices.get.return_value = {
        INDEX.index_name: {"settings": {"index": {"uuid": INDEX.index_uuid}}}
    }
    index_client = client(sdk)
    index_client._index_name = INDEX.index_name
    observation = ReadObservation(
        scope=PublicationScope(
            tenant_id="fixture", environment="test", database_identity="sqlite"
        ),
        committed_epoch=12,
    )
    store = MagicMock()
    store.observe.return_value = observation
    store.unavailable.side_effect = [
        frozenset(),
        frozenset({FILE_ID}) if publication_changed else frozenset(),
    ]
    monkeypatch.setattr(publication_reads, "public_read_store", lambda: store)
    monkeypatch.setattr(
        reads, "current_qualified_file_ids", lambda _files: frozenset({FILE_ID})
    )
    monkeypatch.setattr(
        reads, "current_canonical_positions", lambda _ids: {row.canonical_chunk_id: 200}
    )
    with (
        patch(SESSION_FACTORY, side_effect=lambda: nullcontext(session)),
        asv3_source_inventory_scope(
            scope_key="index-search", batch_temporal_reads=enabled
        ),
    ):
        actual = index_client.search(
            {"query": {"term": {"access_control_list": "authorized-user"}}},
            None,
            as_of_date=AS_OF,
            publication_index=INDEX,
        )
    assert len(actual) == (0 if publication_changed else 1)
    assert store.unavailable.call_count == 2
    assert all(
        call.args[0] == observation and call.args[1] == (FILE_ID,)
        for call in store.unavailable.call_args_list
    )


def test_repeat_batch_refreshes_mutated_canonical_validity(session: Session) -> None:
    row = _add_binding(session, 0)
    session.flush()
    with (
        patch(SESSION_FACTORY, side_effect=lambda: nullcontext(session)),
        asv3_source_inventory_scope(scope_key="cropped", batch_temporal_reads=True),
    ):
        assert (
            len(
                reads.query_temporal_bindings(
                    (FILE_ID,),
                    index=INDEX,
                    as_of_date=AS_OF,
                    projection_ordinals={FILE_ID: (0,)},
                )[FILE_ID]
            )
            == 1
        )
        canonical = session.get(RegulatoryChunk, row.canonical_chunk_id)
        assert canonical is not None
        canonical.validity_end_date = AS_OF
        session.flush()
        assert reads.query_temporal_bindings(
            (FILE_ID,),
            index=INDEX,
            as_of_date=AS_OF,
            projection_ordinals={FILE_ID: (0,)},
        ) == {FILE_ID: []}
