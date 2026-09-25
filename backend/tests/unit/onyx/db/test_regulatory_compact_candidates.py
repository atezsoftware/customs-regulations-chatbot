import datetime
import hashlib
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.db import regulatory_chunks as chunks
from onyx.db import regulatory_public_reads as reads
from onyx.db.models import RegulatoryChunk
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.regulatory import publication_reads
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.heading_path import RegulatoryProvisionReference

FILE_ID = UUID("00000000-0000-0000-0000-000000000101")
AS_OF = datetime.date(2026, 9, 25)
START = datetime.date(2026, 1, 1)
END = datetime.date(2027, 1, 1)


@pytest.fixture
def inventory(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    rows: list[RegulatoryChunk] = []
    bindings: dict[str, SimpleNamespace] = {}
    eager: list[chunks.RegulatoryChunkSiblingCandidate] = []
    for identifier, article, paragraph, position, text, image in [
        ("previous", "7", "1", 10, "Önceki hükmün yayımlanan metni.", None),
        ("seed", "8", "1", 20, "Başvuru şartları ve gerekli belgeler.", None),
        ("sibling", "8", "2", 21, "Başvuruya ilişkin şekil aşağıdadır.", "image-8"),
        ("next", "9", "1", 30, "Sonraki hükmün yayımlanan metni.", None),
        ("reference", "12", "1", 60, "Teminat mektubu bankadan alınır.", None),
    ]:
        heading = ["Yayımlanmış Kanun", f"MADDE {article}"]
        rows.append(
            cast(
                RegulatoryChunk,
                SimpleNamespace(
                    id=identifier,
                    user_file_id=FILE_ID,
                    position=position + 100,
                    projection_ordinal=position + 900,
                    text=f"Unpublished canonical text for {identifier}",
                    heading_path=["Stale heading"],
                    chunk_metadata={"article_no": article, "paragraph_no": paragraph},
                    chunk_type="paragraph",
                    status="superseded",
                    validity_start_date=None,
                    validity_end_date=None,
                ),
            )
        )
        source = json.dumps(
            {
                "regulatory_chunk_id": identifier,
                "heading_path": heading,
                "image_file_id": image,
                "content": text,
                "embeddings": {"full_chunk": [0.125, -0.25, 0.375] * 64},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        bindings[identifier] = SimpleNamespace(
            projection=SimpleNamespace(source_json=source, ordinal=position + 500),
            semantic_position=position,
            representation_text=text,
            effective_start=START,
            effective_end=END,
        )
        eager.append(
            chunks.RegulatoryChunkSiblingCandidate(
                regulatory_chunk_id=identifier,
                user_file_id=FILE_ID,
                position=position,
                text=text,
                status="active",
                heading_path=tuple(heading),
                article_no=article,
                paragraph_no=paragraph,
                chunk_type="paragraph",
                validity_start_date=START,
                validity_end_date=END,
                projection_ordinal=position + 500,
                source_json=source,
                image_file_id=image,
            )
        )

    def filtered(
        _session: object, file_id: UUID, **kwargs: object
    ) -> list[SimpleNamespace]:
        assert file_id == FILE_ID
        identifiers = kwargs.get("canonical_chunk_ids")
        ordinals = kwargs.get("projection_ordinals")
        return [
            binding
            for binding in bindings.values()
            if (
                identifiers is None
                or json.loads(binding.projection.source_json)["regulatory_chunk_id"]
                in cast(tuple[str, ...], identifiers)
            )
            and (
                ordinals is None
                or binding.projection.ordinal in cast(tuple[int, ...], ordinals)
            )
        ]

    def load(session: object, file_id: UUID, **kwargs: object) -> list[SimpleNamespace]:
        return sorted(
            filtered(session, file_id, **kwargs),
            key=lambda binding: (binding.semantic_position, binding.projection.ordinal),
        )

    def iterate(
        session: object, file_id: UUID, **kwargs: object
    ) -> Iterator[SimpleNamespace]:
        yield from reversed(filtered(session, file_id, **kwargs))

    authority = MagicMock()
    authority.unavailable.return_value = frozenset()
    monkeypatch.setattr(publication_reads, "public_read_store", lambda: authority)
    monkeypatch.setattr(reads, "qualified_file_ids", lambda *_: frozenset({FILE_ID}))
    monkeypatch.setattr(reads, "load_public_temporal_bindings", load)
    monkeypatch.setattr(reads, "iter_public_temporal_bindings", iterate, raising=False)
    return SimpleNamespace(
        rows=rows,
        bindings=bindings,
        eager=eager,
        session=MagicMock(),
        index=MagicMock(spec=PublicationIndexSnapshot),
        observation=authority.observe(),
        authority=authority,
    )


def _compact_candidates(
    inventory: SimpleNamespace,
) -> list[chunks.RegulatoryChunkSiblingCandidate]:
    return chunks._public_sibling_candidates(
        inventory.session,
        inventory.rows,
        as_of_date=AS_OF,
        query_indexes={FILE_ID: inventory.index},
        observation=inventory.observation,
    )


def test_candidates_drop_vector_payload_but_keep_exact_publication_digest(
    inventory: SimpleNamespace,
) -> None:
    candidates = _compact_candidates(inventory)

    assert len(candidates) == 5
    assert all(candidate.source_json is None for candidate in candidates)
    for candidate in candidates:
        source = inventory.bindings[
            candidate.regulatory_chunk_id
        ].projection.source_json
        assert (
            candidate.publication_source_sha256
            == hashlib.sha256(source.encode()).hexdigest()
        )


def _select(
    lane: str,
    candidates: list[chunks.RegulatoryChunkSiblingCandidate],
    *,
    as_of_date: datetime.date = AS_OF,
) -> list[chunks.RegulatoryChunkProjection]:
    if lane == "sibling":
        return chunks.select_bounded_same_provision_siblings(
            candidates, ["seed"], query="başvuru", as_of_date=as_of_date
        )
    if lane == "adjacent":
        return chunks.select_bounded_adjacent_provisions(
            candidates, ["seed"], query="", as_of_date=as_of_date, max_provisions=2
        )
    if lane == "reference":
        return chunks.select_bounded_referenced_provisions(
            candidates,
            ["seed"],
            [RegulatoryProvisionReference("12", None)],
            as_of_date=as_of_date,
        )
    assert lane == "lexical"
    return chunks.select_bounded_source_lexical_matches(
        candidates,
        user_file_id=FILE_ID,
        query="teminat mektubu",
        as_of_date=as_of_date,
    )


def _semantic_result(row: chunks.RegulatoryChunkProjection) -> tuple[object, ...]:
    return (
        row.regulatory_chunk_id,
        row.user_file_id,
        row.projection_index,
        row.position,
        row.text,
        row.heading_path,
        row.article_no,
        row.status,
        row.validity_start_date,
        row.validity_end_date,
        row.chunk_type,
        row.paragraph_no,
        row.clause_label,
        row.expansion_priority,
        row.structural_order,
        row.image_file_id,
    )


@pytest.mark.parametrize(
    ("lane", "expected_ids"),
    [
        ("sibling", ["seed", "sibling"]),
        ("adjacent", ["previous", "next"]),
        ("reference", ["reference"]),
        ("lexical", ["reference"]),
    ],
)
def test_compact_selection_preserves_published_semantics(
    inventory: SimpleNamespace, lane: str, expected_ids: list[str]
) -> None:
    compact = _compact_candidates(inventory)
    selected = _select(lane, compact)
    eager_selected = _select(lane, inventory.eager)

    assert [row.regulatory_chunk_id for row in selected] == expected_ids
    assert [_semantic_result(row) for row in selected] == [
        _semantic_result(row) for row in eager_selected
    ]
    assert all(row.source_json is None for row in selected)
    assert _select(lane, compact, as_of_date=END) == []
    assert _select(lane, compact, as_of_date=START - datetime.timedelta(days=1)) == []


def test_duplicate_canonical_binding_keeps_highest_position_then_ordinal(
    inventory: SimpleNamespace,
) -> None:
    original = inventory.bindings["seed"]
    winner_source = json.loads(original.projection.source_json)
    winner_source["content"] = "Selected immutable published version"
    winner_source["image_file_id"] = "selected-image"
    winner_raw = json.dumps(winner_source, ensure_ascii=False)
    inventory.bindings["winner"] = SimpleNamespace(
        projection=SimpleNamespace(source_json=winner_raw, ordinal=521),
        semantic_position=20,
        representation_text="Selected immutable published version",
        effective_start=START,
        effective_end=END,
    )
    inventory.bindings["earlier-position"] = SimpleNamespace(
        projection=SimpleNamespace(
            source_json=original.projection.source_json, ordinal=999
        ),
        semantic_position=19,
        representation_text="Earlier semantic position must not win",
        effective_start=START,
        effective_end=END,
    )

    selected = next(
        candidate
        for candidate in _compact_candidates(inventory)
        if candidate.regulatory_chunk_id == "seed"
    )

    assert (selected.position, selected.projection_ordinal) == (20, 521)
    assert selected.text == "Selected immutable published version"
    assert selected.image_file_id == "selected-image"
    assert selected.source_json is None
    assert (
        selected.publication_source_sha256
        == hashlib.sha256(winner_raw.encode()).hexdigest()
    )


def test_hydration_reads_only_selected_ordinals_and_restores_exact_payloads(
    inventory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    compact = _compact_candidates(inventory)
    selected = _select("sibling", compact)
    original_iterator = reads.iter_public_temporal_bindings

    def only_selected(
        session: Session,
        file_id: UUID,
        *,
        index: PublicationIndexSnapshot,
        as_of_date: datetime.date,
        projection_ordinals: tuple[int, ...] | None = None,
        canonical_chunk_ids: tuple[str, ...] | None = None,
    ) -> Iterator[AnnexTemporalProjection]:
        assert projection_ordinals == (520, 521)
        assert as_of_date == AS_OF
        yield from original_iterator(
            session,
            file_id,
            index=index,
            as_of_date=as_of_date,
            projection_ordinals=projection_ordinals,
            canonical_chunk_ids=canonical_chunk_ids,
        )

    monkeypatch.setattr(reads, "iter_public_temporal_bindings", only_selected)
    hydrated = chunks._hydrate_selected_regulatory_sources(
        inventory.session,
        selected,
        as_of_date=AS_OF,
        query_indexes={FILE_ID: inventory.index},
    )

    assert [row.regulatory_chunk_id for row in hydrated] == ["seed", "sibling"]
    assert [_semantic_result(row) for row in hydrated] == [
        _semantic_result(row) for row in _select("sibling", inventory.eager)
    ]
    assert [row.source_json for row in hydrated] == [
        inventory.bindings[identifier].projection.source_json
        for identifier in ("seed", "sibling")
    ]
    assert all(row.source_json is None for row in compact)
    assert all(row.source_json is None for row in selected)


@pytest.mark.parametrize(
    "mutation",
    ["raw_source", "position", "text", "start", "end", "missing", "index"],
)
def test_hydration_rejects_changed_or_unavailable_publication(
    inventory: SimpleNamespace, mutation: str
) -> None:
    selected = _select("sibling", _compact_candidates(inventory))
    binding = inventory.bindings["seed"]
    indexes = {FILE_ID: inventory.index}
    if mutation == "raw_source":
        # Equal parsed content is insufficient: preserve the exact frozen payload.
        binding.projection.source_json += " "
    elif mutation == "position":
        binding.semantic_position += 1
    elif mutation == "text":
        binding.representation_text += " Changed after selection."
    elif mutation == "start":
        binding.effective_start = START + datetime.timedelta(days=1)
    elif mutation == "end":
        binding.effective_end = END + datetime.timedelta(days=1)
    elif mutation == "missing":
        del inventory.bindings["seed"]
    else:
        assert mutation == "index"
        indexes = {}

    with pytest.raises(ValueError):
        chunks._hydrate_selected_regulatory_sources(
            inventory.session,
            selected,
            as_of_date=AS_OF,
            query_indexes=indexes,
        )


def _lexical_wrapper(
    inventory: SimpleNamespace,
) -> list[chunks.RegulatoryChunkProjection]:
    inventory.session.scalars.return_value.all.return_value = inventory.rows
    return chunks.get_bounded_source_lexical_matches(
        inventory.session,
        user_file_id=FILE_ID,
        query="teminat mektubu",
        as_of_date=AS_OF,
        query_indexes={FILE_ID: inventory.index},
    )


def test_lexical_db_wrapper_returns_exact_selected_frozen_source(
    inventory: SimpleNamespace,
) -> None:
    result = _lexical_wrapper(inventory)

    assert [row.regulatory_chunk_id for row in result] == ["reference"]
    assert [_semantic_result(row) for row in result] == [
        _semantic_result(row) for row in _select("lexical", inventory.eager)
    ]
    assert (
        result[0].source_json == inventory.bindings["reference"].projection.source_json
    )
    source_json = result[0].source_json
    assert source_json is not None
    assert (
        json.loads(source_json)["embeddings"]["full_chunk"]
        == [
            0.125,
            -0.25,
            0.375,
        ]
        * 64
    )


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_lexical_db_wrapper_rejects_publication_drift_after_selection(
    inventory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    original_iterator = reads.iter_public_temporal_bindings

    def change_on_hydration(
        session: Session,
        file_id: UUID,
        *,
        index: PublicationIndexSnapshot,
        as_of_date: datetime.date,
        projection_ordinals: tuple[int, ...] | None = None,
        canonical_chunk_ids: tuple[str, ...] | None = None,
    ) -> Iterator[AnnexTemporalProjection]:
        if projection_ordinals is not None:
            assert projection_ordinals == (560,)
            if mutation == "missing":
                return
            inventory.bindings["reference"].projection.source_json += " "
        yield from original_iterator(
            session,
            file_id,
            index=index,
            as_of_date=as_of_date,
            projection_ordinals=projection_ordinals,
            canonical_chunk_ids=canonical_chunk_ids,
        )

    monkeypatch.setattr(reads, "iter_public_temporal_bindings", change_on_hydration)
    with pytest.raises(ValueError):
        _lexical_wrapper(inventory)


def test_lexical_db_wrapper_withholds_result_when_publication_gate_closes(
    inventory: SimpleNamespace,
) -> None:
    inventory.authority.unavailable.side_effect = [frozenset(), frozenset({FILE_ID})]

    assert _lexical_wrapper(inventory) == []
