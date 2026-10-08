import json
from contextlib import nullcontext
from datetime import date
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus, regulatory_public_reads
from onyx.db import legal_composite_sources as sources
from onyx.db.asv3_corpus import CorpusSource
from onyx.db.engine import sql_engine
from onyx.db.models import (
    RegulatoryCanonicalRevision,
    RegulatoryChunk,
    RegulatoryTemporalProjection,
    SearchSettings,
    User,
)
from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.publication_evidence import _encoder_receipt
from onyx.regulatory.publication_baseline import (
    observed_baseline_binding,
    observed_index_snapshot,
)
from shared_configs.enums import EmbeddingProvider
from tests.unit.onyx.regulatory.indexing_jobs.test_encoder_authority import (
    _batch,
    _sync,
)
from tests.unit.onyx.regulatory.test_publication_baseline import baseline_case


def settings() -> SearchSettings:
    return SearchSettings(
        id=11,
        index_name="physical-index",
        model_name="legacy",
        model_dim=2,
        provider_type=None,
        normalize=False,
        passage_prefix="",
        query_prefix="",
    )


def temporal_row(
    source_id: UUID, position: int, index: PublicationIndexSnapshot
) -> tuple[RegulatoryTemporalProjection, RegulatoryCanonicalRevision]:
    inputs, evidence = baseline_case()
    canonical = inputs.canonical[0].model_copy(
        update={
            "id": f"{source_id}-chunk-{position}",
            "user_file_id": str(source_id),
            "position": position,
            "projection_ordinal": position,
            "text": f"ÖRNEK KANUNU\nMADDE {position + 1}- Exact source {source_id}.",
        }
    )
    original = json.loads(evidence.source_json)
    original.update(
        document_id=str(source_id),
        regulatory_chunk_id=canonical.id,
        chunk_index=position,
        content=canonical.text,
    )
    evidence = evidence.model_copy(
        update={"index": index, "source_json": json.dumps(original)}
    )
    binding = observed_baseline_binding(evidence, [canonical])
    revision = RegulatoryCanonicalRevision(
        id=uuid4(),
        user_file_id=source_id,
        canonical_chunk_id=canonical.id,
        payload=canonical.model_dump(mode="json"),
        payload_sha256=publication_digest(canonical.model_dump(mode="json")),
    )
    payload = binding.model_dump(mode="json")
    row = RegulatoryTemporalProjection(
        id=binding.id,
        user_file_id=source_id,
        canonical_chunk_id=canonical.id,
        canonical_revision_id=revision.id,
        index_uuid=index.index_uuid,
        projection_ordinal=position,
        payload=payload,
        payload_sha256=publication_digest(payload),
    )
    return row, revision


def test_temporal_batch_matches_canonical_reads_with_constant_row_revision_queries() -> (
    None
):
    identifiers = (uuid4(), uuid4(), uuid4())
    index = observed_index_snapshot(settings(), "physical-uuid")
    pairs = [
        temporal_row(source_id, position, index)
        for source_id in identifiers
        for position in range(4)
    ]
    batch = MagicMock(spec=Session)
    batch.scalars.side_effect = [
        [row for row, _ in reversed(pairs)],
        [revision for _, revision in pairs],
    ]
    filters = IndexFilters(access_control_list=[], as_of_date=date(2020, 1, 1))
    actual = sources._opening_rows(
        cast(Session, batch), identifiers, filters, dict.fromkeys(identifiers, index)
    )
    expected: dict[UUID, list[str]] = {}
    for identifier in identifiers:
        own = [
            (row, revision) for row, revision in pairs if row.user_file_id == identifier
        ]
        legacy = MagicMock(spec=Session)
        legacy.scalars.side_effect = [
            [row for row, _ in own],
            [revision for _, revision in own],
        ]
        bindings = asv3_corpus._bounded_temporal_bindings(
            cast(Session, legacy), identifier, index, date(2020, 1, 1), 0, 4
        )
        expected[identifier] = [binding.representation_text for binding in bindings[:3]]
        assert legacy.scalars.call_count == 2
    assert actual == expected and batch.scalars.call_count == 2
    query = (
        batch.scalars.call_args_list[0].args[0].compile(dialect=postgresql.dialect())
    )
    sql = str(query)
    assert "row_number() OVER (PARTITION BY" in sql
    for fence in (
        "retired_at IS NULL",
        "canonical_chunk_id",
        "effective_start",
        "effective_end",
        "validity_start_date",
        "validity_end_date",
    ):
        assert fence in sql
    assert filters.as_of_date in query.params.values()
    assert 4 in query.params.values()
    assert set(query.params["param_1"]) == {
        (identifier, index.index_uuid) for identifier in identifiers
    }


@pytest.mark.parametrize(
    "corruption", ["binding_digest", "revision_digest", "revision_text", "index"]
)
def test_temporal_batch_rejects_changed_immutable_authority(corruption: str) -> None:
    identifier = uuid4()
    index = observed_index_snapshot(settings(), "physical-uuid")
    row, revision = temporal_row(identifier, 0, index)
    if corruption == "binding_digest":
        row.payload_sha256 = "0" * 64
    elif corruption == "revision_digest":
        revision.payload_sha256 = "0" * 64
    elif corruption == "revision_text":
        revision.payload = {**revision.payload, "text": "Changed governing original"}
        revision.payload_sha256 = publication_digest(revision.payload)
    else:
        row.payload = {
            **row.payload,
            "index": index.model_copy(update={"index_uuid": "foreign"}).model_dump(
                mode="json"
            ),
        }
        row.payload_sha256 = publication_digest(row.payload)
    session = MagicMock(spec=Session)
    session.scalars.side_effect = [[row], [revision]]
    with pytest.raises(ValueError):
        sources._opening_rows(
            cast(Session, session),
            (identifier,),
            IndexFilters(access_control_list=[]),
            {identifier: index},
        )


@pytest.mark.parametrize("as_of", [None, date(2020, 1, 1)])
def test_timeless_opening_batch_preserves_whole_order_and_date_scope(
    as_of: date | None,
) -> None:
    identifiers = (uuid4(), uuid4())
    rows = [
        RegulatoryChunk(
            id=f"{source_id}-{position}",
            user_file_id=source_id,
            position=position,
            text=f"whole-{position}",
        )
        for source_id in identifiers
        for position in range(4)
    ]
    session = MagicMock(spec=Session)
    session.scalars.return_value = list(reversed(rows))
    actual = sources._opening_rows(
        cast(Session, session),
        identifiers,
        IndexFilters(access_control_list=[], as_of_date=as_of),
        {},
    )
    assert actual == {
        identifier: ["whole-0", "whole-1", "whole-2"] for identifier in identifiers
    }
    assert session.scalars.call_count == 1
    query = session.scalars.call_args.args[0].compile(dialect=postgresql.dialect())
    assert "row_number() OVER (PARTITION BY" in str(query)
    assert "hierarchical_aggregate" in query.params.values()
    assert set(query.params["user_file_id_1"]) == set(identifiers)
    if as_of is None:
        assert "active" in query.params.values()
    else:
        assert as_of in query.params.values()
        assert "validity_start_date" in str(query) and "validity_end_date" in str(query)


def test_batched_index_authority_matches_exact_single_file_canonical_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = settings()
    identifiers = (uuid4(), uuid4())
    indexes = {
        identifier: observed_index_snapshot(current, f"physical-{identifier}")
        for identifier in identifiers
    }
    batch = MagicMock(spec=Session)
    batch.scalars.return_value.one_or_none.return_value = current
    batch.execute.return_value = [
        (identifier, index.index_uuid, index.model_dump(mode="json"))
        for identifier, index in indexes.items()
    ]
    monkeypatch.setattr(
        sources, "get_current_search_settings", lambda _session: current
    )
    actual = sources._opening_query_indexes(cast(Session, batch), identifiers)
    for identifier, index in indexes.items():
        canonical = MagicMock(spec=Session)
        setting_result = MagicMock()
        setting_result.one_or_none.return_value = current
        canonical.scalars.side_effect = [
            setting_result,
            [index.model_dump(mode="json")],
        ]
        monkeypatch.setattr(
            sql_engine,
            "get_session_with_current_tenant",
            lambda: nullcontext(canonical),
        )
        expected = regulatory_public_reads.resolve_public_query_index(
            current.index_name, index.index_uuid, file_ids=(identifier,)
        )
        assert actual[identifier] == expected
    assert batch.execute.call_count == batch.scalars.call_count == 1


@pytest.mark.parametrize("count", [1, 40])
def test_index_query_count_does_not_grow_with_source_count(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    current = settings()
    identifiers = tuple(uuid4() for _ in range(count))
    index = observed_index_snapshot(current, "physical-uuid")
    session = MagicMock(spec=Session)
    session.scalars.return_value.one_or_none.return_value = current
    session.execute.return_value = [
        (identifier, index.index_uuid, index.model_dump(mode="json"))
        for identifier in identifiers
    ]
    monkeypatch.setattr(
        sources, "get_current_search_settings", lambda _session: current
    )
    assert set(
        sources._opening_query_indexes(cast(Session, session), identifiers)
    ) == set(identifiers)
    assert session.execute.call_count == session.scalars.call_count == 1


def test_encoder_receipts_stay_with_their_own_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = settings()
    current.provider_type = EmbeddingProvider.OPENROUTER
    current.model_name = "openai/text-embedding-3-large"
    current.model_dim = 3
    current.normalize = True
    identifiers = (uuid4(), uuid4())
    indexes = {}
    for identifier, config in zip(identifiers, (_sync(), _batch()), strict=True):
        receipt = _encoder_receipt(config, resolution="1" * 64)
        indexes[identifier] = PublicationIndexSnapshot(
            index_name=current.index_name,
            index_uuid="physical-uuid",
            search_settings_id=current.id,
            model_provider=receipt.authority.provider or "",
            model_name=receipt.authority.model,
            vector_dimension=3,
            embedding_config_sha256=context_hash(config),
            multitenant=sources.MULTI_TENANT,
            encoder_authority=receipt.authority,
            encoder_receipts=(receipt,),
        )
    session = MagicMock(spec=Session)
    session.scalars.return_value.one_or_none.return_value = current
    session.execute.return_value = [
        (identifier, index.index_uuid, index.model_dump(mode="json"))
        for identifier, index in indexes.items()
    ]
    monkeypatch.setattr(
        sources, "get_current_search_settings", lambda _session: current
    )
    actual = sources._opening_query_indexes(cast(Session, session), identifiers)
    assert actual == indexes
    assert (
        actual[identifiers[0]].encoder_receipts
        != actual[identifiers[1]].encoder_receipts
    )


def test_missing_qualified_index_never_uses_timeless_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualified, ordinary = uuid4(), uuid4()
    monkeypatch.setattr(sources, "observe_publication_read", lambda: object())
    publication = MagicMock()
    monkeypatch.setattr(sources, "require_publication_files", publication)
    monkeypatch.setattr(
        sources, "qualified_file_ids", lambda *_args: frozenset({qualified})
    )
    monkeypatch.setattr(sources, "_opening_query_indexes", lambda *_args: {})
    read = MagicMock(return_value={ordinary: ["A" * 4000, "B" * 300]})
    monkeypatch.setattr(sources, "_opening_rows", read)
    actual = sources._opening_batch(
        cast(Session, object()),
        (qualified, ordinary),
        IndexFilters(access_control_list=[]),
        lambda: None,
    )
    assert actual[qualified] is None
    assert actual[ordinary] == ("A" * 4000, "B" * 96)
    assert read.call_args.args[1] == (ordinary,)
    assert publication.call_count == 2


def test_page_reauthorization_rejects_revoked_or_renamed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = [CorpusSource(uuid4(), "Original source.md", "file") for _ in range(3)]
    find = MagicMock(side_effect=[(page[:2], False), ([page[0]], False)])
    monkeypatch.setattr(sources, "find_sources", find)
    monkeypatch.setattr(
        sources,
        "get_session_with_current_tenant",
        lambda: nullcontext(cast(Session, object())),
    )
    read = MagicMock(
        side_effect=lambda _session, identifiers, _filters, _active: dict.fromkeys(
            identifiers, ("GÜMRÜK KANUNU",)
        )
    )
    monkeypatch.setattr(sources, "_opening_batch", read)
    actual = sources._source_openings(
        cast(Session, object()),
        page,
        user=cast(User, object()),
        filters=IndexFilters(access_control_list=[]),
        check_active=lambda: None,
        opening_workers=4,
    )
    assert actual == [("GÜMRÜK KANUNU",), None, None]
    assert {
        identifier for call in read.call_args_list for identifier in call.args[1]
    } == {page[0].id, page[1].id}
    assert all(
        call.kwargs["source_ids"] == tuple(source.id for source in page)
        for call in find.call_args_list
    )


def test_page_progress_reports_authorized_count_without_invented_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = (
        CorpusSource(uuid4(), "law.md", "one"),
        CorpusSource(uuid4(), "law.md", "two"),
    )
    monkeypatch.setattr(
        sources,
        "find_sources",
        MagicMock(side_effect=[([first], True), ([second], False)]),
    )
    monkeypatch.setattr(sources, "_document_types", lambda *_args: ({}, True))
    monkeypatch.setattr(
        sources, "_source_openings", lambda *_args, **_kwargs: [("GÜMRÜK KANUNU",)]
    )
    progress = MagicMock()
    sources.load_source_lane_catalogue(
        cast(Session, object()),
        user=cast(User, MagicMock(id=uuid4())),
        filters=IndexFilters(access_control_list=[]),
        check_active=lambda: None,
        on_progress=progress,
    )
    assert [call.args for call in progress.call_args_list] == [(1, True), (2, False)]


def test_default_inventory_exhausts_scoped_pages_beyond_ten_thousand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total = 10_001
    pages: list[int] = []

    def find(_session: Session, **kwargs: object) -> tuple[list[CorpusSource], bool]:
        offset, limit = cast(int, kwargs["offset"]), cast(int, kwargs["limit"])
        assert 1 <= limit <= 100
        pages.append(offset)
        count = min(limit, total - offset)
        return [
            CorpusSource(UUID(int=offset + n + 1), "source.md", "file")
            for n in range(count)
        ], offset + count < total

    monkeypatch.setattr(sources, "find_sources", find)
    monkeypatch.setattr(sources, "_document_types", lambda *_args: ({}, True))
    monkeypatch.setattr(
        sources,
        "_source_openings",
        lambda _session, page, **_kwargs: [() for _ in page],
    )
    catalogue = sources.load_source_lane_catalogue(
        cast(Session, object()),
        user=cast(User, MagicMock(id=uuid4())),
        filters=IndexFilters(access_control_list=[]),
        check_active=lambda: None,
    )
    assert len(catalogue.records) == total and catalogue.complete
    assert pages[-1] == 10_000 and len(pages) == 101
    assert catalogue.records[-1].source_id == UUID(int=total)


@pytest.mark.parametrize("bound", [True, 0, -1])
def test_invalid_inventory_bound_is_rejected_before_sql(bound: int) -> None:
    session = MagicMock(spec=Session)
    with pytest.raises(ValueError, match="positive bound"):
        sources.load_source_lane_catalogue(
            cast(Session, session),
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            check_active=lambda: None,
            max_sources=bound,
        )
    session.execute.assert_not_called()
