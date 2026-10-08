from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db import legal_composite_sources as sources
from onyx.db.asv3_corpus import CorpusScopeUnavailable, CorpusSource
from onyx.db.models import RegulatoryTemporalProjection, User
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.publication_baseline import observed_index_snapshot
from tests.unit.legal_composite.test_batched_source_openings import (
    settings,
    temporal_row,
)


def thin_row(row: RegulatoryTemporalProjection) -> SimpleNamespace:
    return SimpleNamespace(
        id=row.id,
        user_file_id=row.user_file_id,
        canonical_chunk_id=row.canonical_chunk_id,
        canonical_revision_id=row.canonical_revision_id,
        index_uuid=row.index_uuid,
        projection_ordinal=row.projection_ordinal,
        payload_sha256=row.payload_sha256,
        effective_start=row.effective_start,
        effective_end=row.effective_end,
        index_name=row.payload["index"]["index_name"],
        semantic_position=str(row.payload["semantic_position"]),
        canonical_base_sha256=row.payload["canonical_base_sha256"],
        representation_text=row.payload["representation_text"],
    )


def publication_scope(monkeypatch: pytest.MonkeyPatch, ids: tuple[UUID, ...]) -> None:
    monkeypatch.setattr(sources, "qualified_file_ids", lambda *_args: frozenset(ids))
    monkeypatch.setattr(sources, "observe_publication_read", lambda: object())
    monkeypatch.setattr(sources, "require_publication_files", lambda *_args: None)
    monkeypatch.setattr(
        sources, "get_current_search_settings", lambda *_args: settings()
    )


def test_thin_read_preserves_representation_not_changed_canonical_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = (uuid4(), uuid4())
    publication_scope(monkeypatch, ids)
    index = observed_index_snapshot(settings(), "physical-uuid")
    pairs = [
        temporal_row(identifier, position, index)
        for identifier in ids
        for position in range(4)
    ]
    # Search representation is allowed to differ from the retained base.
    pairs[0][0].payload["representation_text"] = (
        "GÜMRÜK YÖNETMELİĞİ\nMADDE 1- Representation."
    )
    pairs[0][0].payload_sha256 = publication_digest(pairs[0][0].payload)
    session = MagicMock(spec=Session)
    session.execute.side_effect = [
        [(identifier, index.index_uuid) for identifier in ids],
        MagicMock(all=lambda: [thin_row(row) for row, _ in reversed(pairs)]),
    ]
    session.scalars.return_value = [revision for _, revision in pairs]
    filters = IndexFilters(access_control_list=[], as_of_date=date(2020, 1, 1))
    actual = sources._routing_opening_batch(
        cast(Session, session), ids, filters, lambda: None
    )
    assert actual[ids[0]].identity_available
    assert actual[ids[0]].texts[0] == pairs[0][0].payload["representation_text"]
    assert actual[ids[0]].texts[0] != pairs[0][1].payload["text"]
    assert actual[ids[0]].witnesses[0].text_sha256 == context_hash(
        actual[ids[0]].texts[0]
    )
    assert actual[ids[0]].witnesses[0].revision_sha256 == pairs[0][1].payload_sha256
    full = MagicMock(spec=Session)
    full.scalars.side_effect = [
        [row for row, _ in pairs],
        [revision for _, revision in pairs],
        [revision for _, revision in pairs],
    ]
    witnesses: dict[UUID, tuple[sources.SourceOpeningWitness, ...]] = {}
    full_texts = sources._opening_rows(
        cast(Session, full),
        ids,
        filters,
        dict.fromkeys(ids, index),
        witnesses=witnesses,
    )
    for identifier in ids:
        assert (
            sources._bounded_opening_texts(full_texts[identifier])
            == actual[identifier].texts
        )
        assert witnesses[identifier] == actual[identifier].witnesses
    physical_sql = str(
        session.execute.call_args_list[0].args[0].compile(dialect=postgresql.dialect())
    )
    assert "payload" not in physical_sql
    thin_sql = str(
        session.execute.call_args_list[1].args[0].compile(dialect=postgresql.dialect())
    )
    assert (
        "representation_text"
        in session.execute.call_args_list[1]
        .args[0]
        .compile(dialect=postgresql.dialect())
        .params.values()
    )
    assert "regulatory_temporal_projection.payload," not in thin_sql
    for fence in (
        "retired_at IS NULL",
        "effective_start",
        "effective_end",
        "validity_start_date",
        "validity_end_date",
    ):
        assert fence in thin_sql
    session.add.assert_not_called()
    session.commit.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "legacy",
        "revision_digest",
        "revision_base",
        "index_name",
        "ambiguous",
        "missing",
    ],
)
def test_incomplete_thin_authority_remains_unknown_without_timeless_fallback(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    ids = (uuid4(),)
    publication_scope(monkeypatch, ids)
    index = observed_index_snapshot(settings(), "physical-uuid")
    row, revision = temporal_row(ids[0], 0, index)
    if failure == "legacy":
        row.canonical_revision_id = None
    elif failure == "revision_digest":
        revision.payload_sha256 = "0" * 64
    elif failure == "revision_base":
        row.payload["canonical_base_sha256"] = "0" * 64
    elif failure == "index_name":
        row.payload["index"]["index_name"] = "unconfigured"
    physical = [(ids[0], index.index_uuid)]
    if failure == "ambiguous":
        physical.append((ids[0], "other-physical"))
    elif failure == "missing":
        physical = []
    session = MagicMock(spec=Session)
    session.execute.side_effect = [physical, MagicMock(all=lambda: [thin_row(row)])]
    session.scalars.return_value = [revision]
    actual = sources._routing_opening_batch(
        cast(Session, session), ids, IndexFilters(access_control_list=[]), lambda: None
    )
    assert not actual[ids[0]].identity_available
    assert all(
        "regulatory_chunk.text" not in str(call.args[0])
        for call in session.execute.call_args_list
    )
    if failure in {"ambiguous", "missing"}:
        assert session.execute.call_count == 1


def revalidation_case(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CorpusSource, sources.SourceClassification, MagicMock]:
    source = CorpusSource(uuid4(), "wrong-yonetmelik.md", "file")
    text = "CUMHURBAŞKANLIĞI KARARNAMESİ\nMADDE 1- Amaç."
    witness = sources.SourceOpeningWitness(
        chunk_id="original", text_sha256=context_hash(text)
    )
    recorded = sources.classify_source(
        source, ("yonetmelik",), opening_texts=(text,)
    ).model_copy(update={"routing_only": True, "opening_witnesses": (witness,)})
    monkeypatch.setattr(sources, "require_source", lambda *_args, **_kwargs: source)
    publication_scope(monkeypatch, ())
    monkeypatch.setattr(sources, "_opening_query_indexes", lambda *_args: {})

    def full(
        _session: Session,
        _ids: tuple[UUID, ...],
        _filters: IndexFilters,
        _indexes: object,
        witnesses: dict[UUID, tuple[sources.SourceOpeningWitness, ...]],
    ) -> dict[UUID, list[str]]:
        witnesses[source.id] = (witness,)
        return {source.id: [text]}

    monkeypatch.setattr(sources, "_opening_rows", full)
    session = MagicMock(spec=Session)
    session.execute.return_value.all.return_value = [(source.id, "kanun")]
    return source, recorded, session


def test_provisional_type_also_retains_unknown_and_wrong_metadata_cannot_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, recorded, session = revalidation_case(monkeypatch)
    assert recorded.kind is sources.SourceKind.PRESIDENTIAL_DECREE
    assert recorded.admits(sources.SourceKind.UNKNOWN)
    assert not recorded.admits(sources.SourceKind.REGULATION)
    assert (
        sources.revalidate_source_classification(
            cast(Session, session),
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            recorded=recorded,
            check_active=lambda: None,
        )
        == source
    )


def test_legacy_unknown_can_transition_only_after_full_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, recorded, session = revalidation_case(monkeypatch)
    unknown = recorded.model_copy(
        update={
            "kind": sources.SourceKind.UNKNOWN,
            "opening_identity_sha256": None,
            "original_kind": None,
        }
    )
    assert (
        sources.revalidate_source_classification(
            cast(Session, session),
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            recorded=unknown,
            check_active=lambda: None,
        )
        == source
    )

    def rejected(*_args: object, **_kwargs: object) -> object:
        raise CorpusScopeUnavailable("Full publication rejected.")

    monkeypatch.setattr(sources, "_opening_rows", rejected)
    with pytest.raises(CorpusScopeUnavailable, match="Full publication"):
        sources.revalidate_source_classification(
            cast(Session, session),
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            recorded=unknown,
            check_active=lambda: None,
        )


def test_changed_witness_is_rejected_even_for_provisional_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, recorded, session = revalidation_case(monkeypatch)
    changed = recorded.model_copy(
        update={
            "kind": sources.SourceKind.UNKNOWN,
            "opening_witnesses": (
                recorded.opening_witnesses[0].model_copy(
                    update={"text_sha256": "0" * 64}
                ),
            ),
        }
    )
    with pytest.raises(CorpusScopeUnavailable, match="opening witness changed"):
        sources.revalidate_source_classification(
            cast(Session, session),
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            recorded=changed,
            check_active=lambda: None,
        )


def test_qualified_missing_full_index_never_uses_timeless_openings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, recorded, session = revalidation_case(monkeypatch)
    monkeypatch.setattr(
        sources, "qualified_file_ids", lambda *_args: frozenset((source.id,))
    )
    full = MagicMock()
    monkeypatch.setattr(sources, "_opening_rows", full)
    with pytest.raises(CorpusScopeUnavailable, match="verified query index"):
        sources.revalidate_source_classification(
            cast(Session, session),
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            recorded=recorded,
            check_active=lambda: None,
        )
    full.assert_not_called()


@pytest.mark.parametrize("corrupt_payload", [False, True])
def test_real_legacy_binding_unknown_requires_the_full_payload_digest(
    monkeypatch: pytest.MonkeyPatch, corrupt_payload: bool
) -> None:
    identifier = uuid4()
    publication_scope(monkeypatch, (identifier,))
    index = observed_index_snapshot(settings(), "physical-uuid")
    row, _revision = temporal_row(identifier, 0, index)
    row.canonical_revision_id = None
    if corrupt_payload:
        row.payload_sha256 = "0" * 64
    thin = MagicMock(spec=Session)
    thin.execute.side_effect = [
        [(identifier, index.index_uuid)],
        MagicMock(all=lambda: [thin_row(row)]),
    ]
    filters = IndexFilters(access_control_list=[])
    opening = sources._routing_opening_batch(
        cast(Session, thin), (identifier,), filters, lambda: None
    )[identifier]
    assert not opening.identity_available and opening.witnesses
    source = CorpusSource(identifier, "misleading-yonetmelik.md", "file")
    record = sources.classify_source(source, ("yonetmelik",)).model_copy(
        update={"routing_only": True, "opening_witnesses": opening.witnesses}
    )
    monkeypatch.setattr(sources, "require_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        sources, "_opening_query_indexes", lambda *_args: {identifier: index}
    )
    monkeypatch.setattr(
        sources, "_document_types", lambda *_args: ({identifier: ("yonetmelik",)}, True)
    )
    full = MagicMock(spec=Session)
    full.scalars.side_effect = [[row]]
    if corrupt_payload:
        with pytest.raises(CorpusScopeUnavailable, match="payload changed"):
            sources.revalidate_source_classification(
                cast(Session, full),
                user=cast(User, object()),
                filters=filters,
                recorded=record,
                check_active=lambda: None,
            )
    else:
        assert (
            sources.revalidate_source_classification(
                cast(Session, full),
                user=cast(User, object()),
                filters=filters,
                recorded=record,
                check_active=lambda: None,
            )
            == source
        )


def test_routing_catalogue_mode_carries_provisional_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CorpusSource(uuid4(), "kanun.md", "file")
    text = "GÜMRÜK KANUNU\nMADDE 1- Amaç."
    opening = sources.RoutingOpening(
        texts=(text,),
        witnesses=(
            sources.SourceOpeningWitness(
                chunk_id="chunk", text_sha256=context_hash(text)
            ),
        ),
        identity_available=True,
    )
    monkeypatch.setattr(
        sources,
        "find_source_inventory_page",
        lambda *_args, **_kwargs: ([source], False),
    )
    monkeypatch.setattr(
        sources, "_document_types", lambda *_args: ({source.id: ("kanun",)}, True)
    )
    reader = MagicMock(return_value=[opening])
    monkeypatch.setattr(sources, "_source_openings", reader)
    catalogue = sources.load_source_lane_catalogue(
        cast(Session, MagicMock()),
        user=cast(User, SimpleNamespace(id=uuid4())),
        filters=IndexFilters(access_control_list=[]),
        check_active=lambda: None,
        routing_only=True,
    )
    assert catalogue.records[0].kind is sources.SourceKind.STATUTE
    assert catalogue.records[0].routing_only
    assert catalogue.records[0].opening_witnesses == opening.witnesses
    assert catalogue.source_ids(sources.SourceKind.UNKNOWN) == (source.id,)
    assert catalogue.provenance()["provisional_routing_source_count"] == 1
    assert catalogue.provenance()["classification_is_full_source_proof"] is False
    assert reader.call_args.kwargs["routing_only"] is True
