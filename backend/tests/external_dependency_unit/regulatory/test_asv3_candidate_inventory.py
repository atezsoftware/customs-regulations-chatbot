"""Actual PostgreSQL planning/verified-packet equivalence; no provider calls."""

import json
from collections.abc import Generator
from datetime import date
from time import perf_counter
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import delete, event
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus, regulatory_public_reads
from onyx.db.asv3_candidate_inventory import (
    asv3_source_inventory_scope,
    iter_verified_asv3_inventory_members,
    read_asv3_candidate_inventory,
)
from onyx.db.asv3_corpus import read_search_source_closures, resolve_pc_corpus_scope
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryCanonicalRevision,
    RegulatoryChunk,
    RegulatoryTemporalProjection,
    User,
    UserFile,
)
from onyx.db.regulatory_chunks import (
    get_bounded_same_provision_siblings,
    get_regulatory_provision_heading_source,
)
from onyx.document_index.publication_models import (
    PublicationIndexSnapshot,
    publication_digest,
)
from onyx.regulatory import provision_retrieval
from onyx.regulatory.provision_retrieval import build_regulatory_rerank_packets
from onyx.regulatory.publication_reads import observe_publication_read
from tests.external_dependency_unit.conftest import create_test_user
from tests.external_dependency_unit.regulatory.test_asv3_corpus_fences import (
    pc_corpus as pc_corpus,
)
from tests.unit.onyx.db.test_public_temporal_read_batches import _add_binding
from tests.unit.onyx.regulatory.test_provision_retrieval import _chunk


@pytest.fixture
def inventory_source(
    db_session: Session,
    request: pytest.FixtureRequest,
) -> Generator[
    tuple[UserFile, PublicationIndexSnapshot, list[RegulatoryTemporalProjection]],
    None,
    None,
]:
    owner = create_test_user(db_session, "asv3_candidate_inventory")
    source = UserFile(
        id=uuid4(),
        user_id=owner.id,
        file_id=uuid4().hex,
        name="ASv3 inventory",
        file_type="text/plain",
    )
    db_session.add(source)
    db_session.flush()
    index = PublicationIndexSnapshot(
        index_name="asv3-inventory",
        index_uuid=uuid4().hex,
        search_settings_id=9,
        model_provider="fixture",
        model_name="fixture",
        vector_dimension=4096,
        embedding_config_sha256=publication_digest({}),
        multitenant=False,
    )
    rows = []
    for ordinal in range(getattr(request, "param", 48)):
        row = _add_binding(db_session, ordinal, file_id=source.id, index=index)
        payload = json.loads(json.dumps(row.payload))
        original = json.loads(payload["projection"]["source_json"])
        heading = [
            "Law",
            "Ek 1" if ordinal < 3 else "Ek 2",
            "MADDE 3" if ordinal < 6 else f"MADDE {ordinal}",
            f"({ordinal % 3 + 1})",
        ]
        text = f"Original operative paragraph {ordinal}"
        original.update(
            heading_path=heading,
            content="summary\n" + text + "\ncontext",
            doc_summary="summary\n",
            chunk_context="\ncontext",
            content_vector=[0.125] * 4096,
            source_links=json.dumps({"0": "https://example.test/law"}),
        )
        payload["projection"]["source_json"] = json.dumps(original)
        payload["representation_text"] = text
        payload["semantic_position"] = ordinal
        payload["representation_metadata"] = {
            "article_no": "3" if ordinal < 6 else str(ordinal),
            "content_vector": [0.125] * 4096,
        }
        if ordinal == 2:
            payload["derived_role"] = "image_companion"
        canonical = next(
            item
            for item in db_session.new
            if isinstance(item, RegulatoryChunk) and item.id == row.canonical_chunk_id
        )
        canonical.position = ordinal
        canonical.text = text
        canonical.heading_path = heading
        canonical.chunk_metadata = {
            "article_no": "3" if ordinal < 6 else str(ordinal),
            "embedding": [0.125] * 4096,
        }
        revision = next(
            item
            for item in db_session.new
            if isinstance(item, RegulatoryCanonicalRevision)
            and item.id == row.canonical_revision_id
        )
        revision.payload = {
            **revision.payload,
            "text": text,
            "metadata": {"embedding": [0.125] * 4096},
        }
        revision.payload_sha256 = publication_digest(revision.payload)
        from onyx.regulatory.amendments.annexes.context_dependencies import context_hash

        payload["canonical_base_sha256"] = context_hash(text)
        row.payload = payload
        row.payload_sha256 = publication_digest(payload)
        rows.append(row)
    db_session.commit()
    try:
        yield source, index, rows
    finally:
        db_session.rollback()
        for model in (
            RegulatoryTemporalProjection,
            RegulatoryCanonicalRevision,
            RegulatoryChunk,
        ):
            db_session.execute(delete(model).where(model.user_file_id == source.id))
        db_session.delete(source)
        db_session.delete(owner)
        db_session.commit()


@pytest.mark.usefixtures("tenant_context")
@pytest.mark.parametrize("inventory_source", [48, 2500], indirect=True)
def test_inventory_preserves_explicit_expansion_and_bounds_initial_packets(
    db_session: Session,
    inventory_source: tuple[
        UserFile, PublicationIndexSnapshot, list[RegulatoryTemporalProjection]
    ],
    record_property: Any,
) -> None:
    source, index, rows = inventory_source
    as_of = date(2026, 10, 2)
    ids = [rows[0].canonical_chunk_id, rows[4].canonical_chunk_id]
    started = perf_counter()
    old = get_bounded_same_provision_siblings(
        db_session,
        ids,
        query="operative paragraph",
        as_of_date=as_of,
        query_indexes={source.id: index},
    )
    old_seconds = perf_counter() - started
    seeds = [
        _chunk(
            row.projection_ordinal, row.canonical_chunk_id, document_id=str(source.id)
        ).model_copy(update={"publication_index": index})
        for row in (rows[0], rows[4])
    ]
    old_navigation = (
        get_regulatory_provision_heading_source(
            db_session, ids, as_of_date=as_of, query_indexes={source.id: index}
        )
        if len(rows) == 48
        else None
    )
    statements: list[str] = []

    def capture(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    with asv3_source_inventory_scope(scope_key="test-owner-and-pinned-corpus") as scope:
        observation = observe_publication_read()
        connection = db_session.connection()
        event.listen(connection, "before_cursor_execute", capture)
        started = perf_counter()
        try:
            inventory = read_asv3_candidate_inventory(
                db_session,
                source_id=source.id,
                index=index,
                as_of_date=as_of,
                observation=observation,
            )
        finally:
            event.remove(connection, "before_cursor_execute", capture)
        inventory_seconds = perf_counter() - started
        assert len(inventory) == len(rows)
        assert len(statements) >= 1
        inventory_sql = next(
            statement for statement in statements if "representation_text" in statement
        )
        selected_columns = inventory_sql.split("FROM", 1)[0]
        assert "regulatory_temporal_projection.payload," not in selected_columns
        assert "regulatory_canonical_revision" not in inventory_sql
        assert "content_vector" not in selected_columns
        assert (
            read_asv3_candidate_inventory(
                db_session,
                source_id=source.id,
                index=index,
                as_of_date=as_of,
                observation=observation,
            )
            is inventory
        )
        assert len(scope.entries) == 1
        assert scope.retained_bytes <= scope.max_cache_bytes
        started = perf_counter()
        new = get_bounded_same_provision_siblings(
            db_session,
            ids,
            query="operative paragraph",
            as_of_date=as_of,
            query_indexes={source.id: index},
        )
        new_seconds = perf_counter() - started
        assert new == old
        if len(rows) == 48:
            assert (
                get_regulatory_provision_heading_source(
                    db_session, ids, as_of_date=as_of, query_indexes={source.id: index}
                )
                == old_navigation
            )
        assert rows[2].canonical_chunk_id in {item.regulatory_chunk_id for item in new}
        assert not any("Ek 2" in item.heading_path for item in new[:3])
        new_packets = build_regulatory_rerank_packets(
            db_session, seeds, query="operative paragraph", as_of_date=as_of
        )
        assert [
            packet.primary_member.regulatory_chunk_id for packet in new_packets
        ] == ids
        assert all(len(packet.members) <= 5 for packet in new_packets)
        assert all(
            member.metadata["asv3_operative_unit_complete"] == "false"
            for packet in new_packets
            for member in packet.members
        )
        assert all(
            (member.heading_path or [])[:2]
            == (packet.primary_member.heading_path or [])[:2]
            for packet in new_packets
            for member in packet.members
        )
        record_property("old_selected_read_seconds", old_seconds)
        record_property("narrow_inventory_seconds", inventory_seconds)
        record_property("new_selected_read_seconds", new_seconds)
        record_property("inventory_rows", len(inventory))
    assert scope.entries == {}


@pytest.mark.usefixtures("tenant_context")
@pytest.mark.parametrize("inventory_source", [48, 2500], indirect=True)
@pytest.mark.parametrize("all_siblings", [False, True])
def test_initial_packets_select_all_frozen_parent_siblings_without_source_inventory(
    db_session: Session,
    inventory_source: tuple[
        UserFile, PublicationIndexSnapshot, list[RegulatoryTemporalProjection]
    ],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Any,
    all_siblings: bool,
) -> None:
    source, index, rows = inventory_source
    selected = rows if all_siblings else rows[:20]
    for row in selected:
        payload = dict(row.payload)
        original = json.loads(payload["projection"]["source_json"])
        original["heading_path"] = [
            "Law",
            "Ek 1",
            "MADDE 3",
            str(row.projection_ordinal),
        ]
        payload["projection"] = {
            **payload["projection"],
            "source_json": json.dumps(original),
        }
        row.payload = payload
        row.payload_sha256 = publication_digest(payload)
    # Current metadata is a navigation lead; parent authority comes from frozen originals.
    db_session.get_one(RegulatoryChunk, rows[1].canonical_chunk_id).heading_path = [
        "Other parent",
        "MADDE 999",
    ]
    db_session.commit()
    seeds = [
        _chunk(
            row.projection_ordinal, row.canonical_chunk_id, document_id=str(source.id)
        ).model_copy(
            update={"publication_index": index, "score": float(row.projection_ordinal)}
        )
        for row in rows[:20]
    ]
    converted_ids: list[str] = []
    original_converter = provision_retrieval._asv3_parent_scoring_chunk

    def convert(*args: Any, **kwargs: Any) -> Any:
        converted_ids.append(args[0].regulatory_chunk_id)
        return original_converter(*args, **kwargs)

    monkeypatch.setattr(provision_retrieval, "_asv3_parent_scoring_chunk", convert)
    statements: list[str] = []

    def capture(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    with asv3_source_inventory_scope(scope_key="bounded-initial-packets") as scope:
        started = perf_counter()
        connection = db_session.connection()
        event.listen(connection, "before_cursor_execute", capture)
        try:
            packets = build_regulatory_rerank_packets(
                db_session, seeds, query="paragraph", as_of_date=date(2026, 10, 2)
            )
        finally:
            event.remove(connection, "before_cursor_execute", capture)
        record_property("initial_packet_seconds", perf_counter() - started)
        record_property("source_rows", len(rows))
        assert not scope.entries
    assert len(packets) == len(seeds) == 20
    assert [packet.primary_member.regulatory_chunk_id for packet in packets] == [
        seed.regulatory_chunk_id for seed in seeds
    ]
    assert all(len(packet.members) == len(selected) for packet in packets)
    assert [member.regulatory_chunk_id for member in packets[0].members] == [
        row.canonical_chunk_id for row in selected
    ]
    assert all(
        (member.heading_path or [])[:3] == ["Law", "Ek 1", "MADDE 3"]
        for packet in packets
        for member in packet.members
    )
    assert set(converted_ids) == {row.canonical_chunk_id for row in selected}
    assert len(converted_ids) == len(selected)
    assert not any("regulatory_canonical_revision" in sql for sql in statements)
    assert all(
        "regulatory_temporal_projection.payload," not in sql.split("FROM", 1)[0]
        for sql in statements
    )
    assert all(
        packet.primary_member.score == seed.score
        for packet, seed in zip(packets, seeds)
    )
    packets[0].primary_member.metadata["private_test_value"] = "first packet"
    assert "private_test_value" not in packets[1].members[0].metadata
    assert all(
        member.content.startswith("Original operative paragraph")
        for packet in packets
        for member in packet.members
    )
    assert all(
        member.metadata["asv3_operative_unit_complete"] == "false"
        for packet in packets
        for member in packet.members
    )
    assert all(
        member.metadata["asv3_evidence_stage"] == "uncitable_parent_candidate"
        for packet in packets
        for member in packet.members
    )
    originals = list(
        regulatory_public_reads.iter_public_temporal_bindings(
            db_session,
            source.id,
            index=index,
            as_of_date=date(2026, 10, 2),
            canonical_chunk_ids=(rows[0].canonical_chunk_id,),
        )
    )
    assert len(originals) == 1
    assert originals[0].representation_text == packets[0].primary_member.content


@pytest.mark.usefixtures("tenant_context")
@pytest.mark.parametrize("corruption", ["retired", "wrong_index", "expired"])
def test_initial_packet_selected_seed_fails_closed(
    db_session: Session,
    inventory_source: tuple[
        UserFile, PublicationIndexSnapshot, list[RegulatoryTemporalProjection]
    ],
    corruption: str,
) -> None:
    source, index, rows = inventory_source
    row = rows[0]
    if corruption == "payload":
        row.payload = {**row.payload, "representation_text": "unverified text"}
    elif corruption == "retired":
        from datetime import datetime, timezone

        row.retired_at = datetime.now(timezone.utc)
    elif corruption == "expired":
        row.effective_end = date(2026, 10, 2)
        row.payload = {**row.payload, "effective_end": "2026-10-02"}
        row.payload_sha256 = publication_digest(row.payload)
    else:
        index = index.model_copy(update={"model_name": "wrong model"})
    db_session.commit()
    seed = _chunk(0, row.canonical_chunk_id, document_id=str(source.id)).model_copy(
        update={"publication_index": index}
    )
    with (
        asv3_source_inventory_scope(scope_key="bounded-initial-packets"),
        pytest.raises(ValueError),
    ):
        build_regulatory_rerank_packets(
            db_session, [seed], query="paragraph", as_of_date=date(2026, 10, 2)
        )


@pytest.mark.usefixtures("tenant_context")
@pytest.mark.parametrize("corruption", ["payload", "revision_authority"])
def test_parent_scoring_lead_cannot_bypass_original_validation(
    db_session: Session,
    pc_corpus: DocumentSet,
    inventory_source: tuple[
        UserFile, PublicationIndexSnapshot, list[RegulatoryTemporalProjection]
    ],
    corruption: str,
) -> None:
    source, index, rows = inventory_source
    owner = db_session.get_one(User, source.user_id)
    previous_owner = pc_corpus.user_id
    pc_corpus.user_id = owner.id
    link = DocumentSet__UserFile(document_set_id=pc_corpus.id, user_file_id=source.id)
    db_session.add(link)
    db_session.commit()
    try:
        filters = resolve_pc_corpus_scope(
            db_session,
            user=owner,
            filters=IndexFilters(access_control_list=[], as_of_date=date(2026, 10, 2)),
        )
        seed = _chunk(
            0, rows[0].canonical_chunk_id, document_id=str(source.id)
        ).model_copy(update={"publication_index": index})
        assert seed.regulatory_chunk_id is not None
        with asv3_source_inventory_scope(scope_key="owned-parent-candidates"):
            packets = build_regulatory_rerank_packets(
                db_session, [seed], query="paragraph", as_of_date=filters.as_of_date
            )
        assert (
            packets[0].primary_member.metadata["asv3_evidence_stage"]
            == "uncitable_parent_candidate"
        )
        assert (
            len(
                list(
                    asv3_corpus.iter_source_chunks_by_ids(
                        db_session,
                        user=owner,
                        filters=filters,
                        source_id=source.id,
                        chunk_ids=(seed.regulatory_chunk_id,),
                        index=index,
                        check_active=lambda: None,
                    )
                )
            )
            == 1
        )
        row = rows[0]
        if corruption == "payload":
            row.payload = {
                **row.payload,
                "representation_text": "unverified operative text",
            }
        else:
            row.payload = {**row.payload, "canonical_base_sha256": "0" * 64}
            row.payload_sha256 = publication_digest(row.payload)
        db_session.commit()
        with pytest.raises(ValueError):
            list(
                asv3_corpus.iter_source_chunks_by_ids(
                    db_session,
                    user=owner,
                    filters=filters,
                    source_id=source.id,
                    chunk_ids=(seed.regulatory_chunk_id,),
                    index=index,
                    check_active=lambda: None,
                )
            )
    finally:
        db_session.rollback()
        db_session.delete(link)
        pc_corpus.user_id = previous_owner
        db_session.commit()


@pytest.mark.usefixtures("tenant_context")
@pytest.mark.parametrize(
    "corruption",
    [
        "payload",
        "payload_and_digest",
        "revision_authority",
        "retired",
        "canonical_metadata",
        "canonical_heading",
    ],
)
def test_selected_inventory_originals_fail_closed_after_change(
    db_session: Session,
    inventory_source: tuple[
        UserFile, PublicationIndexSnapshot, list[RegulatoryTemporalProjection]
    ],
    corruption: str,
) -> None:
    source, index, rows = inventory_source
    as_of = date(2026, 10, 2)
    with asv3_source_inventory_scope(scope_key="test-owner-and-pinned-corpus"):
        observation = observe_publication_read()
        inventory = read_asv3_candidate_inventory(
            db_session,
            source_id=source.id,
            index=index,
            as_of_date=as_of,
            observation=observation,
        )
        selected = tuple(item for item in inventory if item.ordinal == 0)
        row = rows[0]
        if corruption in {"payload", "payload_and_digest"}:
            row.payload = {
                **row.payload,
                "representation_text": "tampered operative text",
            }
            if corruption == "payload_and_digest":
                row.payload_sha256 = publication_digest(row.payload)
        elif corruption == "revision_authority":
            row.payload = {**row.payload, "canonical_base_sha256": "0" * 64}
            row.payload_sha256 = publication_digest(row.payload)
        elif corruption in {"canonical_metadata", "canonical_heading"}:
            canonical = db_session.get_one(RegulatoryChunk, row.canonical_chunk_id)
            if corruption == "canonical_metadata":
                canonical.chunk_metadata = {
                    **canonical.chunk_metadata,
                    "article_no": "999",
                }
            else:
                canonical.heading_path = ["Different Law", "MADDE 999"]
        else:
            from datetime import datetime, timezone

            row.retired_at = datetime.now(timezone.utc)
        db_session.commit()
        with pytest.raises(ValueError):
            list(
                iter_verified_asv3_inventory_members(
                    db_session,
                    rows=selected,
                    index=index,
                    as_of_date=as_of,
                    observation=observation,
                )
            )


@pytest.mark.usefixtures("tenant_context")
def test_inventory_is_scoped_by_date_index_and_bounded_without_dropping_rows(
    db_session: Session,
    inventory_source: tuple[
        UserFile, PublicationIndexSnapshot, list[RegulatoryTemporalProjection]
    ],
) -> None:
    source, index, rows = inventory_source
    boundary = date(2026, 10, 3)
    rows[0].effective_start = boundary
    rows[0].payload = {**rows[0].payload, "effective_start": boundary.isoformat()}
    rows[0].payload_sha256 = publication_digest(rows[0].payload)
    db_session.commit()
    observation = observe_publication_read()
    with asv3_source_inventory_scope(
        scope_key="owner-and-filters", max_cache_bytes=1
    ) as scope:
        before = read_asv3_candidate_inventory(
            db_session,
            source_id=source.id,
            index=index,
            as_of_date=date(2026, 10, 2),
            observation=observation,
        )
        at = read_asv3_candidate_inventory(
            db_session,
            source_id=source.id,
            index=index,
            as_of_date=boundary,
            observation=observation,
        )
        assert len(before) == 47 and len(at) == 48
        assert scope.entries == {} and scope.retained_bytes == 0
    with asv3_source_inventory_scope(scope_key="owner-and-filters") as scope:
        before = read_asv3_candidate_inventory(
            db_session,
            source_id=source.id,
            index=index,
            as_of_date=date(2026, 10, 2),
            observation=observation,
        )
        at = read_asv3_candidate_inventory(
            db_session,
            source_id=source.id,
            index=index,
            as_of_date=boundary,
            observation=observation,
        )
        with pytest.raises(ValueError, match="physical index changed"):
            read_asv3_candidate_inventory(
                db_session,
                source_id=source.id,
                index=index.model_copy(update={"model_name": "different-model"}),
                as_of_date=boundary,
                observation=observation,
            )
        assert len(before) == 47 and len(at) == 48
        narrow = read_asv3_candidate_inventory(
            db_session,
            source_id=source.id,
            index=index,
            as_of_date=boundary,
            observation=observation,
            canonical_chunk_ids=(
                rows[0].canonical_chunk_id,
                rows[47].canonical_chunk_id,
            ),
        )
        assert {item.ordinal for item in narrow} == {0, 47}
        assert len(scope.entries) == 3
        assert scope.query_indexes == {source.id: index}
    assert scope.query_indexes == {}


@pytest.mark.usefixtures("tenant_context")
def test_closure_reuses_inventory_but_validates_exact_owned_originals(
    db_session: Session,
    pc_corpus: DocumentSet,
    inventory_source: tuple[
        UserFile, PublicationIndexSnapshot, list[RegulatoryTemporalProjection]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, index, rows = inventory_source
    owner = db_session.get_one(User, source.user_id)
    stranger = create_test_user(db_session, "asv3_inventory_stranger")
    pc_corpus.user_id = owner.id
    link = DocumentSet__UserFile(document_set_id=pc_corpus.id, user_file_id=source.id)
    db_session.add(link)
    db_session.commit()
    calls: list[dict[str, Any]] = []
    original_reader = regulatory_public_reads.iter_public_temporal_bindings

    def tracked(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(
        regulatory_public_reads, "iter_public_temporal_bindings", tracked
    )
    monkeypatch.setattr(asv3_corpus, "iter_public_temporal_bindings", tracked)
    try:
        filters = resolve_pc_corpus_scope(
            db_session,
            user=owner,
            filters=IndexFilters(access_control_list=[], as_of_date=date(2026, 10, 2)),
        )

        def read(user: User = owner) -> Any:
            return read_search_source_closures(
                db_session,
                user=user,
                filters=filters,
                source_id=source.id,
                center_ids=(rows[0].canonical_chunk_id,),
                index=index,
                check_active=lambda: None,
            )

        before = read()
        assert any(
            call.get("canonical_chunk_ids") is None
            and call.get("projection_ordinals") is None
            for call in calls
        )
        calls.clear()
        with asv3_source_inventory_scope(
            scope_key="actual-owner-and-pinned-PC"
        ) as scope:
            after = read()
            assert after == before
            assert calls and all(
                call.get("projection_ordinals") is not None for call in calls
            )
            assert all(call.get("expected_payload_sha256") for call in calls)
            assert len(scope.entries) == 1
            assert rows[2].canonical_chunk_id not in {item.id for item in after.chunks}
            with pytest.raises(PermissionError):
                read(stranger)
    finally:
        db_session.rollback()
        pc_corpus.user_id = None
        db_session.delete(link)
        db_session.delete(stranger)
        db_session.commit()
