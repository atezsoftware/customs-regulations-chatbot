"""Multi-source planning reads retain per-source authority and group publication fences."""

from collections.abc import Iterator
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from onyx.db import asv3_candidate_inventory as inventory
from onyx.db import regulatory_chunks as chunks
from onyx.db import regulatory_public_reads as public_reads
from onyx.document_index.publication_models import PublicationScope, ReadObservation
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory import publication_reads
from tests.unit.onyx.db.test_public_temporal_read_batches import INDEX

FIRST = UUID("00000000-0000-0000-0000-000000000001")
SECOND = UUID("00000000-0000-0000-0000-000000000002")
AS_OF = date(2026, 10, 13)
OBSERVATION = ReadObservation(
    scope=PublicationScope(
        tenant_id="owner-tenant", environment="test", database_identity="fixture"
    ),
    committed_epoch=12,
)


def original_row(source: UUID, ordinal: int, **updates: Any) -> dict[str, Any]:
    row = {
        "requested_source_id": source,
        "id": uuid4(),
        "payload_sha256": "a" * 64,
        "canonical_chunk_id": f"{source}:{ordinal}",
        "projection_ordinal": ordinal,
        "effective_start": None,
        "effective_end": None,
        "semantic_position": 100 - ordinal,
        "representation_text": f"Original paragraph {source}:{ordinal}",
        "derived_role": "canonical",
        "binding_index": INDEX.model_dump(mode="json"),
        "encoder_configuration": "{}",
        "observed_index": None,
        "source_chunk_id": f"{source}:{ordinal}",
        "source_document_id": str(source),
        "heading_path": ["MADDE 1"],
        "image_file_id": None,
        "live_chunk_type": "paragraph",
        "canonical_heading_path": ["MADDE 1"],
        **{
            prefix + key: (False if prefix == "has_" else None)
            for prefix in ("canonical_", "source_", "representation_", "has_")
            for key in inventory._STRUCTURAL_KEYS
        },
    }
    return {**row, **updates}


class Rows:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.closed = False

    def __iter__(self) -> Iterator[SimpleNamespace]:
        return iter(SimpleNamespace(_mapping=row) for row in self.rows)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    store = MagicMock()
    store.observe.return_value = OBSERVATION
    store.unavailable.return_value = frozenset()
    monkeypatch.setattr(publication_reads, "public_read_store", lambda: store)
    session = MagicMock(spec=Session)
    return SimpleNamespace(session=session, store=store)


def requests() -> dict[UUID, inventory.ASv3CandidateInventoryRequest]:
    return {
        SECOND: inventory.ASv3CandidateInventoryRequest(INDEX, (f"{SECOND}:2",)),
        FIRST: inventory.ASv3CandidateInventoryRequest(
            INDEX, (f"{FIRST}:0", f"{FIRST}:1")
        ),
    }


def read(
    boundary: SimpleNamespace,
    wanted: dict[UUID, inventory.ASv3CandidateInventoryRequest] | None = None,
) -> dict[UUID, tuple[inventory.ASv3CandidateInventoryRow, ...]]:
    return inventory.read_asv3_candidate_inventories(
        boundary.session,
        requests=wanted if wanted is not None else requests(),
        as_of_date=AS_OF,
        observation=OBSERVATION,
    )


def test_batch_matches_serial_rows_order_hash_and_pair_sql(
    boundary: SimpleNamespace,
) -> None:
    rows = {
        FIRST: [original_row(FIRST, 0), original_row(FIRST, 1)],
        SECOND: [original_row(SECOND, 2)],
    }
    expected = {}
    with inventory.asv3_source_inventory_scope(scope_key="same-owner"):
        for source, request in requests().items():
            boundary.session.execute.return_value = Rows(rows[source])
            expected[source] = inventory.read_asv3_candidate_inventory(
                boundary.session,
                source_id=source,
                index=request.index,
                canonical_chunk_ids=request.canonical_chunk_ids,
                as_of_date=AS_OF,
                observation=OBSERVATION,
            )
    boundary.session.reset_mock()
    boundary.store.reset_mock()
    result = Rows(rows[FIRST] + rows[SECOND])
    boundary.session.execute.return_value = result
    with inventory.asv3_source_inventory_scope(
        scope_key="same-owner", batch_candidate_inventory=True
    ):
        actual = read(boundary)
    assert actual == expected
    assert list(actual) == [SECOND, FIRST]
    assert [row.ordinal for row in actual[FIRST]] == [0, 1]
    assert result.closed
    assert boundary.session.execute.call_count == 1
    assert boundary.store.unavailable.call_count == 2
    assert all(
        call.args == (OBSERVATION, (SECOND, FIRST))
        for call in boundary.store.unavailable.call_args_list
    )
    statement = boundary.session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert (
        "regulatory_chunk.user_file_id = regulatory_temporal_projection.user_file_id"
        in sql
    )
    assert "regulatory_temporal_projection.retired_at IS NULL" in sql
    for owner in ("regulatory_temporal_projection", "regulatory_chunk"):
        for suffix in (
            (".effective_start", ".effective_end")
            if owner.endswith("projection")
            else (".validity_start_date", ".validity_end_date")
        ):
            assert owner + suffix in sql
    params = compiled.params
    assert params["user_file_id_1"] == SECOND and params["canonical_chunk_id_1"] == [
        f"{SECOND}:2"
    ]
    assert params["user_file_id_2"] == FIRST and params["canonical_chunk_id_2"] == [
        f"{FIRST}:0",
        f"{FIRST}:1",
    ]
    assert statement.get_execution_options()["yield_per"] == 128


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"source_document_id": str(FIRST)}, "planning source identity"),
        ({"source_chunk_id": f"{FIRST}:2"}, "planning source identity"),
        ({"encoder_configuration": '{"provider":"different"}'}, "encoder receipt"),
        (
            {
                "encoder_configuration": None,
                "observed_index": INDEX.model_copy(
                    update={"model_name": "other"}
                ).model_dump(mode="json"),
            },
            "observation index",
        ),
    ],
)
def test_bad_member_rejects_whole_batch_without_cache_fill(
    boundary: SimpleNamespace, mutation: dict[str, Any], error: str
) -> None:
    result = Rows([original_row(FIRST, 0), original_row(SECOND, 2, **mutation)])
    boundary.session.execute.return_value = result
    with inventory.asv3_source_inventory_scope(
        scope_key="same-owner", batch_candidate_inventory=True
    ) as scope:
        with pytest.raises(ValueError, match=error):
            read(boundary)
        assert not scope.entries and scope.retained_bytes == 0
    assert result.closed


def test_mixed_indexes_use_each_sources_own_authority(
    boundary: SimpleNamespace,
) -> None:
    other = INDEX.model_copy(
        update={"index_uuid": "other-physical", "model_name": "other-model"}
    )
    wanted = requests()
    wanted[SECOND] = inventory.ASv3CandidateInventoryRequest(other, (f"{SECOND}:2",))
    boundary.session.execute.return_value = Rows(
        [
            original_row(FIRST, 0),
            original_row(SECOND, 2, binding_index=other.model_dump(mode="json")),
        ]
    )
    with inventory.asv3_source_inventory_scope(
        scope_key="same-owner", batch_candidate_inventory=True
    ):
        actual = read(boundary, wanted)
        assert len(actual[FIRST]) == len(actual[SECOND]) == 1
        with pytest.raises(ValueError, match="physical index changed"):
            read(boundary)
    compiled = boundary.session.execute.call_args.args[0].compile(
        dialect=postgresql.dialect()
    )
    assert compiled.params["index_uuid_1"] == "other-physical"
    assert compiled.params["index_uuid_2"] == INDEX.index_uuid


def test_wrong_binding_index_is_not_blessed_by_other_sources_index(
    boundary: SimpleNamespace,
) -> None:
    other = INDEX.model_copy(update={"model_name": "other-model"})
    wanted = requests()
    wanted[SECOND] = inventory.ASv3CandidateInventoryRequest(other, (f"{SECOND}:2",))
    boundary.session.execute.return_value = Rows(
        [original_row(FIRST, 0), original_row(SECOND, 2)]
    )
    with inventory.asv3_source_inventory_scope(
        scope_key="same-owner", batch_candidate_inventory=True
    ):
        actual = read(boundary, wanted)
    assert len(actual[FIRST]) == 1 and actual[SECOND] == ()


@pytest.mark.parametrize("warm_first", [False, True])
def test_group_withdrawal_fences_cached_and_new_members(
    boundary: SimpleNamespace, warm_first: bool
) -> None:
    with inventory.asv3_source_inventory_scope(
        scope_key="same-owner", batch_candidate_inventory=True
    ) as scope:
        if warm_first:
            boundary.session.execute.return_value = Rows([original_row(FIRST, 0)])
            read(boundary, {FIRST: requests()[FIRST]})
        previous_keys = set(scope.entries)
        boundary.store.reset_mock()
        boundary.store.unavailable.side_effect = [frozenset(), frozenset({FIRST})]
        result = Rows(
            [original_row(SECOND, 2)]
            if warm_first
            else [original_row(FIRST, 0), original_row(SECOND, 2)]
        )
        boundary.session.execute.return_value = result
        with pytest.raises(OnyxError):
            read(boundary)
        assert set(scope.entries) == previous_keys
        assert boundary.store.unavailable.call_count == 2
        assert all(
            set(call.args[1]) == {FIRST, SECOND}
            for call in boundary.store.unavailable.call_args_list
        )
        assert result.closed


def test_empty_exact_ids_do_not_expand_and_all_cache_hits_are_refenced(
    boundary: SimpleNamespace,
) -> None:
    with inventory.asv3_source_inventory_scope(
        scope_key="same-owner", batch_candidate_inventory=True
    ):
        assert read(
            boundary, {FIRST: inventory.ASv3CandidateInventoryRequest(INDEX, ())}
        ) == {FIRST: ()}
        assert not boundary.session.execute.called
        boundary.session.execute.return_value = Rows([original_row(FIRST, 0)])
        wanted = {FIRST: requests()[FIRST]}
        expected = read(boundary, wanted)
        boundary.session.reset_mock()
        boundary.store.reset_mock()
        assert read(boundary, wanted) == expected
        assert not boundary.session.execute.called
        assert boundary.store.unavailable.call_count == 2
        with pytest.raises(ValueError, match="exact canonical IDs"):
            read(
                boundary, {FIRST: inventory.ASv3CandidateInventoryRequest(INDEX, None)}
            )


@pytest.mark.parametrize("cancel_at", [1, 3, 5])
def test_cancellation_cannot_publish_or_cache_partial_group(
    boundary: SimpleNamespace, cancel_at: int
) -> None:
    result = Rows([original_row(FIRST, 0), original_row(SECOND, 2)])
    boundary.session.execute.return_value = result
    checks = 0

    def check() -> None:
        nonlocal checks
        checks += 1
        if checks == cancel_at:
            raise RuntimeError("cancelled")

    with inventory.asv3_source_inventory_scope(
        scope_key="same-owner", batch_candidate_inventory=True, check_active=check
    ) as scope:
        with pytest.raises(RuntimeError, match="cancelled"):
            read(boundary)
        assert not scope.entries and scope.retained_bytes == 0
    assert result.closed if cancel_at > 1 else not boundary.session.execute.called


def test_cache_stays_bounded_scope_local_and_batch_requires_opt_in(
    boundary: SimpleNamespace,
) -> None:
    boundary.session.execute.return_value = Rows(
        [original_row(FIRST, 0), original_row(SECOND, 2)]
    )
    with inventory.asv3_source_inventory_scope(
        scope_key="owner", batch_candidate_inventory=True, max_cache_bytes=1
    ) as scope:
        actual = read(boundary)
        assert sum(map(len, actual.values())) == 2
        assert not scope.entries and scope.retained_bytes == 0
        with inventory.asv3_source_inventory_scope(scope_key="other-owner"):
            with pytest.raises(ValueError, match="explicit Cropped"):
                read(boundary)
    assert not scope.query_indexes
    with pytest.raises(ValueError, match="explicit Cropped"):
        read(boundary)


def test_visibility_hook_keeps_input_order_and_its_final_fence(
    boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = [
        chunks.RegulatoryChunkSiblingCandidate(
            regulatory_chunk_id=f"{source}:{ordinal}",
            user_file_id=source,
            position=ordinal,
            text="canonical",
            status="active",
            heading_path=("MADDE 1",),
        )
        for source, ordinal in [(SECOND, 2), (FIRST, 1), (FIRST, 0)]
    ]
    boundary.session.execute.return_value = Rows(
        [original_row(FIRST, 0), original_row(FIRST, 1), original_row(SECOND, 2)]
    )
    monkeypatch.setattr(
        public_reads, "qualified_file_ids", lambda *_: frozenset({FIRST, SECOND})
    )
    with inventory.asv3_source_inventory_scope(
        scope_key="owner", batch_candidate_inventory=True
    ):
        actual = chunks._public_sibling_candidates(
            boundary.session,
            candidates,
            as_of_date=AS_OF,
            query_indexes={FIRST: INDEX, SECOND: INDEX},
            observation=OBSERVATION,
        )
    assert [row.regulatory_chunk_id for row in actual] == [
        row.regulatory_chunk_id for row in candidates
    ]
    assert all(
        row.planning_inventory is not None and row.source_json is None for row in actual
    )
    assert boundary.store.unavailable.call_count == 3
    boundary.store.reset_mock()
    boundary.store.unavailable.side_effect = [
        frozenset(),
        frozenset(),
        frozenset({SECOND}),
    ]
    with inventory.asv3_source_inventory_scope(
        scope_key="owner", batch_candidate_inventory=True
    ):
        actual = chunks._public_sibling_candidates(
            boundary.session,
            candidates,
            as_of_date=AS_OF,
            query_indexes={FIRST: INDEX, SECOND: INDEX},
            observation=OBSERVATION,
        )
    assert [row.regulatory_chunk_id for row in actual] == [f"{FIRST}:1", f"{FIRST}:0"]
