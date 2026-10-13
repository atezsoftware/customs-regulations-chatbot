"""Captured missing metadata remains discoverable without weakening source identity."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql

from onyx.asv3.corpus_tools import article_identity, build_corpus_specs
from onyx.asv3.cropped_corpus_identity import (
    verified_inserted_article_opening,
    verified_own_article_identity,
)
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.registry import CapabilityRegistry
from onyx.db import asv3_corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.models import User
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker
from tests.unit.onyx.asv3.test_numbered_source_resolution import (
    catalogue,
    chunk,
    filters,
    source,
)


def captured_chunks() -> list[CorpusChunk]:
    data = json.loads(
        (
            Path(__file__).parent / "fixtures/cropped_article_binding_5172.json"
        ).read_text()
    )
    return [
        CorpusChunk(
            row["chunk_id"],
            UUID(row["source_id"]),
            row["text"],
            index,
            index,
            tuple(row["metadata"]["heading_path"]),
            row["metadata"],
            None,
            None,
            "active",
        )
        for index, row in enumerate(data["rows"], 30)
    ]


def captured_inserted_chunks(article: str) -> list[CorpusChunk]:
    capture = json.loads(
        (
            Path(__file__).parent / "fixtures/cropped_inserted_article_238A_11A.json"
        ).read_text()
    )
    source_id = (
        "601c6f1f-dc4e-4c26-89e4-8fef59784605"
        if article == "238/A"
        else "de32173d-cae2-4c2b-aa64-f7e3f51d1dd0"
    )
    chunks = []
    for row in capture["rows"]:
        if row["source_id"] != source_id:
            continue
        assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_hash"]
        chunks.append(
            CorpusChunk(
                row["id"],
                UUID(row["source_id"]),
                row["text"],
                row["position"],
                row["projection_ordinal"],
                tuple(row["heading_path"]),
                row["metadata"],
                None,
                None,
                row["status"],
            )
        )
    return chunks


def read_inserted(
    chunks: list[CorpusChunk],
    *,
    article: str = "238/A",
    paragraph: str | None = None,
    cropped: bool = True,
) -> ToolOutcome:
    broker = MemoryBroker(chunks)
    return CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_provision",
            arguments={
                "source_id": str(chunks[0].source_id),
                "article": article,
                **({"paragraph": paragraph} if paragraph is not None else {}),
            },
        ),
        RunContext(services={"asv3_exp_cropped": cropped}),
    )


@pytest.mark.parametrize(("article", "paragraph"), [("238/A", "2"), ("11/A", "5")])
def test_captured_inserted_articles_preserve_all_originals_and_next_boundary(
    article: str, paragraph: str
) -> None:
    rows = captured_inserted_chunks(article)
    result = read_inserted(rows, article=article, paragraph=paragraph)
    assert result.status == OutcomeStatus.FOUND
    assert result.data["subunit_verified"] is True
    assert result.data["scan_truncated"] is False
    assert result.data["next_position"] == rows[-1].position
    assert [item.text for item in result.evidence] == [row.text for row in rows[1:-1]]
    assert [item.text_hash for item in result.evidence] == [
        hashlib.sha256(row.text.encode()).hexdigest() for row in rows[1:-1]
    ]
    for item, row in zip(result.evidence, rows[1:-1]):
        canonical = item.metadata["canonical_metadata"]
        assert isinstance(canonical, dict)
        for field in (
            "article_no",
            "article_qualifier",
            "paragraph_no",
            "document_type",
        ):
            assert canonical.get(field) == row.metadata.get(field)


def test_original_experimental_still_uses_existing_inserted_metadata_behavior() -> None:
    result = read_inserted(captured_inserted_chunks("238/A"), cropped=False)
    assert result.status == OutcomeStatus.NOT_FOUND
    assert result.evidence == []


@pytest.mark.parametrize(
    "problem",
    [
        "missing_continuation",
        "missing_boundary",
        "wrong_source",
        "wrong_paragraph",
        "wrong_heading",
        "quoted_continuation",
        "derived",
        "untrusted",
        "truncated",
        "duplicate_opening",
    ],
)
def test_inserted_article_does_not_claim_closure_without_literal_ordered_proof(
    problem: str,
) -> None:
    previous, first, second, boundary = captured_inserted_chunks("238/A")
    if problem == "missing_continuation":
        rows = [previous, first, boundary]
    elif problem == "missing_boundary":
        rows = [previous, first, second]
    else:
        if problem == "wrong_source":
            second = replace(second, source_id=UUID(int=3))
        elif problem == "wrong_paragraph":
            second = replace(second, text=second.text.replace("(2)", "(3)"))
        elif problem == "wrong_heading":
            second = replace(
                second,
                heading_path=(
                    *second.heading_path[:-2],
                    "İlgisiz konu",
                    second.heading_path[-1],
                ),
            )
        elif problem == "quoted_continuation":
            second = replace(second, text="> " + second.text)
        elif problem == "duplicate_opening":
            second = replace(first, id="duplicate", position=second.position)
        else:
            second = replace(second, metadata={**second.metadata, problem: True})
        rows = [previous, first, second, boundary]
    result = read_inserted(rows, paragraph="2")
    assert result.status == OutcomeStatus.PARTIAL
    assert result.data["absence_proven"] is False
    assert result.data["subunit_verified"] is (problem == "missing_boundary")


@pytest.mark.parametrize(
    "text",
    [
        "> **MADDE 238/A -** (1) Alıntı.",
        "Başka hüküm: **MADDE 238/A -** (1) Alıntı.",
        "**MADDE 238/A -** (1) Aşağıdaki şekilde değiştirilmiştir.",
        "**MADDE 238/A -** (1) Metin.\n**MADDE 239 -** (1) Başka metin.",
    ],
)
def test_inserted_locator_rejects_crossrefs_amendments_and_quoted_headings(
    text: str,
) -> None:
    first = captured_inserted_chunks("238/A")[1]
    assert not verified_inserted_article_opening(
        replace(first, text=text), ("238/A", None)
    )


def test_inserted_literal_paragraph_never_uses_stale_first_paragraph_metadata() -> None:
    previous, first, second, boundary = captured_inserted_chunks("238/A")
    first = replace(first, metadata={**first.metadata, "paragraph_no": "2"})
    missing = read_inserted([previous, first, boundary], paragraph="2")
    assert missing.status == OutcomeStatus.PARTIAL
    assert missing.data["subunit_verified"] is False
    absent = read_inserted([previous, first, second, boundary], paragraph="3")
    assert absent.status == OutcomeStatus.PARTIAL
    assert absent.data["subunit_verified"] is False
    assert absent.data["scan_truncated"] is False
    assert len(absent.evidence) == 2


def test_literal_first_article_recovers_captured_missing_metadata() -> None:
    first, *_ = captured_chunks()
    assert article_identity(first) is None
    assert verified_own_article_identity(first) == ("15", None)
    assert article_identity(first, allow_canonical_text_fallback=True) == ("15", None)


@pytest.mark.parametrize(
    "text",
    [
        "Kanunda şu hüküm alıntılanmıştır:\n**MADDE 15** .- (1) Hüküm.",
        "> **MADDE 15** .- (1) Hüküm.",
        "“**MADDE 15** .- (1) Hüküm.”",
        "**MADDE 15** .- (1) Hüküm.\n**MADDE 16** .- (1) Diğer hüküm.",
        "**MADDE 15** .- (1) Aşağıdaki şekilde değiştirilmiştir.",
        "(2) MADDE 15 başka kanunun hükmüdür.",
    ],
)
def test_literal_witness_rejects_quotes_amendments_and_multiple_articles(
    text: str,
) -> None:
    first, *_ = captured_chunks()
    assert verified_own_article_identity(replace(first, text=text)) is None


@pytest.mark.parametrize("flag", ["untrusted", "truncated", "derived", "external"])
def test_literal_witness_rejects_noncanonical_chunks(flag: str) -> None:
    first, *_ = captured_chunks()
    assert (
        verified_own_article_identity(
            replace(first, metadata={**first.metadata, flag: True})
        )
        is None
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"provision_identifiers": []},
        {"provision_identifiers": ["madde 16"]},
        {"article_no": "14"},
        {"document_type": "genelge"},
        {"target_article_no": "15"},
        {"original_heading_path": ["MADDE 16"]},
    ],
)
def test_conflicting_identity_is_not_repaired_by_text(
    metadata: dict[str, object],
) -> None:
    first, *_ = captured_chunks()
    assert (
        verified_own_article_identity(
            replace(first, metadata={**first.metadata, **metadata})
        )
        is None
    )


@pytest.mark.parametrize("cropped", [False, True])
def test_read_provision_cropped_only_preserves_originals_and_boundary(
    cropped: bool,
) -> None:
    rows = captured_chunks()
    end = replace(
        rows[0],
        id="next-article",
        position=33,
        projection_ordinal=33,
        text="**MADDE 16** .- (1) Sonraki madde.",
        metadata={**rows[0].metadata, "provision_identifiers": ["madde 16"]},
    )
    broker = MemoryBroker([*rows, end])
    result = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_provision",
            arguments={
                "source_id": str(rows[0].source_id),
                "article": "15",
                "paragraph": "3",
            },
        ),
        RunContext(services={"asv3_exp_cropped": cropped}),
    )
    assert result.status == (OutcomeStatus.FOUND if cropped else OutcomeStatus.PARTIAL)
    assert [item.text for item in result.evidence] == (
        [row.text for row in rows] if cropped else []
    )
    if cropped:
        assert result.data["subunit_verified"] is True
        assert result.data["next_position"] == 33
        for item in result.evidence:
            canonical = item.metadata["canonical_metadata"]
            assert isinstance(canonical, dict) and canonical["article_no"] is None


def test_literal_witness_does_not_override_existing_navigation_identity() -> None:
    first, *_ = captured_chunks()
    inserted = replace(
        first,
        text="**MADDE 10/A** .- (1) İzin şartları.",
        heading_path=("KANUN", "MADDE 10"),
        metadata={
            **first.metadata,
            "article_no": "10",
            "provision_identifiers": ["madde 10", "madde 10/A"],
        },
    )
    assert article_identity(inserted) == ("10", None)
    assert article_identity(inserted, allow_canonical_text_fallback=True) == (
        "10",
        None,
    )
    assert verified_own_article_identity(inserted) == ("10/A", None)
    temporary = replace(
        first,
        text="**GEÇİCİ MADDE 15** .- (1) Geçiş şartları.",
        metadata={**first.metadata, "provision_identifiers": ["geçici madde 15"]},
    )
    assert verified_own_article_identity(temporary) == ("15", "gecici")


def test_captured_inserted_article_keeps_existing_continuations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = json.loads(
        (
            Path(__file__).parent / "fixtures/cropped_lookup_5077_inserted_article.json"
        ).read_text()
    )
    chunks = []
    for row in capture["rows"]:
        item = row["item"]
        meta = item["metadata"]
        chunks.append(
            CorpusChunk(
                item["chunk_id"],
                UUID(item["source_id"]),
                item["text"],
                meta["position"],
                item["search_doc"]["chunk_ind"],
                tuple(meta["heading_path"]),
                meta["canonical_metadata"],
                None,
                None,
                "active",
            )
        )
    assert all(article_identity(item) == ("10", None) for item in chunks)
    assert all(
        article_identity(item, allow_canonical_text_fallback=True) == ("10", None)
        for item in chunks
    )
    results = []
    for cropped in (False, True):
        broker = MemoryBroker(chunks)
        monkeypatch.setattr(
            broker, "related_sources_for_provision", MagicMock(return_value=None)
        )
        results.append(
            CapabilityRegistry(build_corpus_specs(broker)).dispatch(
                CapabilityCall(
                    name="read_provision",
                    arguments={"source_id": str(chunks[0].source_id), "article": "10"},
                ),
                RunContext(services={"asv3_exp_cropped": cropped}),
            )
        )
    assert results[0].status == results[1].status
    assert [item.chunk_id for item in results[1].evidence] == [
        item.id for item in chunks
    ]
    assert [item.text for item in results[0].evidence] == [
        item.text for item in results[1].evidence
    ]
    inserted = read_inserted(chunks, article="10/A", paragraph="4")
    assert inserted.status == OutcomeStatus.PARTIAL  # No next own heading captured.
    assert inserted.data["subunit_verified"] is True
    assert [item.text for item in inserted.evidence] == [item.text for item in chunks]
    first_only = read_inserted(chunks[:1], article="10/A", paragraph="2")
    assert first_only.status == OutcomeStatus.PARTIAL
    assert first_only.data["subunit_verified"] is False


def prepare_numeric_lookup(
    monkeypatch: pytest.MonkeyPatch,
    candidates: list[CorpusSource],
    *,
    root: str = "KABAHATLER KANUNU",
    kind: str = "kanun",
) -> tuple[MagicMock, MagicMock]:
    session = catalogue(monkeypatch, [[], [], candidates])
    monkeypatch.setattr(asv3_corpus, "resolve_source_query_index", lambda *_args: None)
    read = MagicMock(
        side_effect=lambda _session, **kwargs: (
            next(item for item in candidates if item.id == kwargs["source_id"]),
            [
                chunk(
                    next(item for item in candidates if item.id == kwargs["source_id"]),
                    root=root,
                    kind=kind,
                )
            ],
            False,
        )
    )
    monkeypatch.setattr(asv3_corpus, "read_source_chunks", read)
    return session, read


def test_numbered_filename_requires_canonical_own_formal_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = source("Diğer Kanunlar/5326_sayili_kanun.md")
    session, read = prepare_numeric_lookup(monkeypatch, [candidate])
    scope = filters()
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=scope,
        query="5326 sayılı Kabahatler Kanunu",
        allow_canonical_filename_fallback=True,
    ) == ([candidate], False)
    assert read.call_args.kwargs["filters"] is scope
    assert read.call_args.kwargs["limit"] == 2
    statement = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert (
        "^0*5326[ _.-]+sayili[ _.-]+kanunu?\\.(md|txt|pdf)$"
        in statement.params.values()
    )
    assert 73 in statement.params.values() and 9 in statement.params.values()


@pytest.mark.parametrize(
    ("filename", "root", "kind"),
    [
        ("5327_sayili_kanun.md", "KABAHATLER KANUNU", "kanun"),
        ("5326_sayili_kanun_degistiren.md", "KABAHATLER KANUNU", "kanun"),
        ("5326_sayili_kanun.md", "5327 SAYILI KABAHATLER KANUNU", "kanun"),
        ("5326_sayili_kanun.md", "GÜMRÜK KANUNU", "kanun"),
        ("5326_sayili_kanun.md", "Kabahatler Kanunu hakkında yazı", "kanun"),
        ("5326_sayili_kanun.md", "KABAHATLER KANUNU", "genelge"),
    ],
)
def test_number_filename_is_never_enough(
    monkeypatch: pytest.MonkeyPatch, filename: str, root: str, kind: str
) -> None:
    session, _ = prepare_numeric_lookup(
        monkeypatch, [source(filename)], root=root, kind=kind
    )
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query="5326 sayılı Kabahatler Kanunu",
        allow_canonical_filename_fallback=True,
    ) == ([], False)


def test_duplicate_numbered_laws_never_resolve_to_single_first_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [source("5326_sayili_kanun.md") for _ in range(2)]
    session, _ = prepare_numeric_lookup(monkeypatch, candidates)
    found, more = asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query="5326 sayılı Kabahatler Kanunu",
        allow_canonical_filename_fallback=True,
        limit=1,
    )
    assert found == candidates[:1] and more


def test_numbered_filename_fallback_is_disabled_for_original_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, read = prepare_numeric_lookup(
        monkeypatch, [source("5326_sayili_kanun.md")]
    )
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query="5326 sayılı Kabahatler Kanunu",
        allow_numbered_title_fallback=True,
    ) == ([], False)
    assert session.execute.call_count == 2
    read.assert_not_called()


def test_numeric_candidate_acl_and_publication_denial_precede_identity_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    denied, unpublished = (source("5326_sayili_kanun.md") for _ in range(2))
    session = catalogue(
        monkeypatch,
        [[], [], [denied, unpublished]],
        denied={denied.id},
        unpublished={unpublished.id},
    )
    read = MagicMock()
    monkeypatch.setattr(asv3_corpus, "read_source_chunks", read)
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query="5326 sayılı Kabahatler Kanunu",
        allow_canonical_filename_fallback=True,
    ) == ([], False)
    read.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_article_locator_hydrates_candidates_through_original_fences(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    rows = captured_chunks()
    session = MagicMock()
    session.execute.return_value = []
    session.scalars.return_value = [rows[0].id]
    require = MagicMock()
    final_fence = MagicMock()
    monkeypatch.setattr(asv3_corpus, "require_source", require)
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", lambda: "observation")
    monkeypatch.setattr(asv3_corpus, "require_publication_files", final_fence)
    monkeypatch.setattr(asv3_corpus, "qualified_file_ids", lambda *_args: set())
    hydrate = MagicMock(return_value=iter(rows[:1]))
    monkeypatch.setattr(asv3_corpus, "iter_source_chunks_by_ids", hydrate)
    scope = filters()
    assert asv3_corpus.source_provision_position(
        session,
        user=cast(User, object()),
        filters=scope,
        source_id=rows[0].source_id,
        article="15",
        qualifier=None,
        index=None,
        allow_canonical_text_fallback=enabled,
    ) == (30 if enabled else None)
    assert require.call_count == 2
    final_fence.assert_called_once_with("observation", (rows[0].source_id,))
    assert hydrate.call_count == int(enabled)
    if enabled:
        assert hydrate.call_args.kwargs["chunk_ids"] == (rows[0].id,)
        assert hydrate.call_args.kwargs["filters"] is scope
        compiled = session.scalars.call_args.args[0].compile(
            dialect=postgresql.dialect()
        )
        assert scope.as_of_date in compiled.params.values()
        assert "validity_end_date" in str(compiled)


@pytest.mark.parametrize("conflict", ["identity", "duplicate", "overflow", "cancel"])
def test_article_locator_does_not_publish_ambiguous_or_unverified_anchor(
    monkeypatch: pytest.MonkeyPatch, conflict: str
) -> None:
    first, *_ = captured_chunks()
    session = MagicMock()
    session.execute.return_value = []
    session.scalars.return_value = (
        [str(index) for index in range(65)] if conflict == "overflow" else [first.id]
    )
    monkeypatch.setattr(asv3_corpus, "require_source", MagicMock())
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", lambda: "observation")
    monkeypatch.setattr(asv3_corpus, "require_publication_files", MagicMock())
    monkeypatch.setattr(asv3_corpus, "qualified_file_ids", lambda *_args: set())
    rows = (
        [replace(first, text=first.text.replace("MADDE 15", "MADDE 16"))]
        if conflict == "identity"
        else [first, replace(first, id="duplicate", position=60)]
        if conflict == "duplicate"
        else [first]
    )
    hydrate = MagicMock(return_value=iter(rows))
    if conflict == "cancel":
        hydrate.side_effect = lambda *_args, **kwargs: kwargs["check_active"]()
    monkeypatch.setattr(asv3_corpus, "iter_source_chunks_by_ids", hydrate)

    def cancelled() -> None:
        raise TimeoutError("cancelled")

    def locate() -> int | None:
        return asv3_corpus.source_provision_position(
            session,
            user=cast(User, object()),
            filters=filters(),
            source_id=first.source_id,
            article="15",
            qualifier=None,
            index=None,
            allow_canonical_text_fallback=True,
            check_active=cancelled,
        )

    if conflict == "cancel":
        with pytest.raises(TimeoutError, match="cancelled"):
            locate()
    else:
        assert locate() is None
        assert hydrate.call_count == int(conflict != "overflow")


@pytest.mark.parametrize("enabled", [False, True])
def test_inserted_locator_uses_fenced_own_original_candidate_only_in_cropped(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    first = captured_inserted_chunks("238/A")[1]
    session = MagicMock()
    session.execute.return_value = []
    session.scalars.return_value = [first.id]
    require = MagicMock()
    monkeypatch.setattr(asv3_corpus, "require_source", require)
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", lambda: "observation")
    monkeypatch.setattr(asv3_corpus, "require_publication_files", MagicMock())
    monkeypatch.setattr(asv3_corpus, "qualified_file_ids", lambda *_args: set())
    hydrate = MagicMock(return_value=iter([first]))
    monkeypatch.setattr(asv3_corpus, "iter_source_chunks_by_ids", hydrate)
    result = asv3_corpus.source_provision_position(
        session,
        user=cast(User, object()),
        filters=filters(),
        source_id=first.source_id,
        article="238/A",
        qualifier=None,
        index=None,
        allow_canonical_text_fallback=enabled,
    )
    assert result == (1264 if enabled else None)
    assert require.call_count == (2 if enabled else 0)
    assert hydrate.call_count == int(enabled)
    if enabled:
        assert hydrate.call_args.kwargs["batch_publication_checks"] is True
        compiled = session.scalars.call_args.args[0].compile(
            dialect=postgresql.dialect()
        )
        assert "regulatory_chunk.text" in str(compiled)
        assert any(
            str(value).startswith("^[[:space:]]*") for value in compiled.params.values()
        )


def test_inserted_locator_ignores_later_corrected_structural_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, first, second, _ = captured_inserted_chunks("238/A")
    session = MagicMock()
    session.execute.return_value = [(second.position, ["MADDE 238/A"])]
    session.scalars.return_value = [first.id]
    monkeypatch.setattr(asv3_corpus, "require_source", MagicMock())
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", lambda: "observation")
    monkeypatch.setattr(asv3_corpus, "require_publication_files", MagicMock())
    monkeypatch.setattr(asv3_corpus, "qualified_file_ids", lambda *_args: set())
    monkeypatch.setattr(
        asv3_corpus, "iter_source_chunks_by_ids", MagicMock(return_value=iter([first]))
    )
    result = asv3_corpus.source_provision_position(
        session,
        user=cast(User, object()),
        filters=filters(),
        source_id=first.source_id,
        article="238/A",
        qualifier=None,
        index=None,
        allow_canonical_text_fallback=True,
    )
    assert result == first.position
    session.execute.assert_not_called()


def test_correct_inserted_metadata_keeps_existing_multi_paragraph_chunk_read() -> None:
    rows = captured_inserted_chunks("11/A")
    merged = replace(rows[1], text=rows[1].text + "\n\n" + rows[2].text)
    chunks = [
        rows[0],
        merged,
        *[replace(row, position=row.position - 1) for row in rows[3:]],
    ]
    original = read_inserted(chunks, article="11/A", paragraph="5", cropped=False)
    cropped = read_inserted(chunks, article="11/A", paragraph="5")
    assert original.status == cropped.status == OutcomeStatus.FOUND
    assert original.data["subunit_verified"] is cropped.data["subunit_verified"] is True
    assert [item.text for item in cropped.evidence] == [
        item.text for item in original.evidence
    ]
    assert len(cropped.evidence) == 4


@pytest.mark.parametrize("qualifier", ["gecici", "mukerrer", "ek"])
def test_inserted_witness_never_changes_qualifier(qualifier: str) -> None:
    first = captured_inserted_chunks("238/A")[1]
    assert not verified_inserted_article_opening(first, ("238/A", qualifier))
    assert not verified_inserted_article_opening(
        replace(first, metadata={**first.metadata, "article_qualifier": qualifier}),
        ("238/A", None),
    )
