import datetime
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.db.enums import RegulatoryChunkStatus
from onyx.db.regulatory_chunks import (
    DEFAULT_PROVISION_MAX_CHUNKS,
    RegulatoryChunkProjection,
    RegulatoryChunkSiblingCandidate,
    RegulatoryNavigationSeed,
    RegulatoryProvisionHeadingCandidate,
    get_regulatory_provision_heading_source,
    get_visible_regulatory_chunk_ids,
    is_regulatory_navigation_candidate_visible,
    select_bounded_adjacent_provisions,
    select_bounded_referenced_provisions,
    select_bounded_same_provision_siblings,
    select_bounded_source_lexical_matches,
    select_dominant_regulatory_navigation_seed_file,
)
from onyx.regulatory.heading_path import RegulatoryProvisionReference

FILE_A = UUID("00000000-0000-0000-0000-000000000001")
FILE_B = UUID("00000000-0000-0000-0000-000000000002")


def _candidate(
    regulatory_chunk_id: str,
    position: int,
    *,
    text: str | None = None,
    article_no: str | None = "61",
    article_title: str | None = None,
    heading_path: tuple[str, ...] = ("Belge", "MADDE 61"),
    chunk_type: str | None = None,
    paragraph_no: str | None = None,
    clause_label: str | None = None,
    user_file_id: UUID = FILE_A,
    validity_start_date: datetime.date | None = None,
    validity_end_date: datetime.date | None = None,
    status: str = RegulatoryChunkStatus.ACTIVE.value,
) -> RegulatoryChunkSiblingCandidate:
    return RegulatoryChunkSiblingCandidate(
        regulatory_chunk_id=regulatory_chunk_id,
        user_file_id=user_file_id,
        position=position,
        text=text if text is not None else regulatory_chunk_id,
        heading_path=heading_path,
        article_no=article_no,
        article_title=article_title,
        chunk_type=chunk_type,
        paragraph_no=paragraph_no,
        clause_label=clause_label,
        validity_start_date=validity_start_date,
        validity_end_date=validity_end_date,
        status=status,
    )


def _ids(rows: list[RegulatoryChunkProjection]) -> list[str]:
    return [row.regulatory_chunk_id for row in rows]


def test_visible_regulatory_chunk_ids_are_authoritative_and_deduplicated() -> None:
    db_session = MagicMock()
    db_session.scalars.return_value.all.return_value = ["active-one"]

    assert get_visible_regulatory_chunk_ids(
        db_session,
        ["active-one", "stale", "active-one"],
        as_of_date=None,
    ) == {"active-one"}
    db_session.scalars.assert_called_once()


def test_visible_regulatory_chunk_ids_skip_database_for_empty_input() -> None:
    db_session = MagicMock()

    assert (
        get_visible_regulatory_chunk_ids(
            db_session,
            [],
            as_of_date=None,
        )
        == set()
    )
    db_session.scalars.assert_not_called()


def test_projection_index_is_assigned_before_temporal_filtering() -> None:
    rows = [
        _candidate("a-first", 0),
        _candidate(
            "b-old",
            1,
            validity_end_date=datetime.date(2024, 1, 1),
        ),
        _candidate(
            "c-new",
            1,
            validity_start_date=datetime.date(2024, 1, 1),
        ),
        _candidate("d-last", 2),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["c-new"],
        query="teminat",
        as_of_date=datetime.date(2024, 1, 2),
    )

    assert _ids(selected) == ["a-first", "c-new", "d-last"]
    assert [row.projection_index for row in selected] == [0, 2, 3]


def test_default_same_provision_budget_retains_fragmented_clause_family() -> None:
    rows = [
        _candidate(
            f"row-{index}",
            index,
            text=f"Short independently operative clause {index}",
            chunk_type="clause",
            clause_label=str(index),
        )
        for index in range(30)
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["row-0"],
        query="independently operative clause",
        as_of_date=None,
    )

    assert DEFAULT_PROVISION_MAX_CHUNKS == 24
    assert len(selected) == DEFAULT_PROVISION_MAX_CHUNKS


def test_source_lexical_matches_prefer_rare_query_terms_within_one_file() -> None:
    rows = [
        _candidate(
            "common-only",
            0,
            text="The goods are handled under the applicable procedure.",
            article_no="1",
            heading_path=("Instrument", "ARTICLE 1"),
        ),
        _candidate(
            "rare-match",
            1,
            text="The goods are irretrievably destroyed by an unforeseen event.",
            article_no="2",
            heading_path=("Instrument", "ARTICLE 2"),
        ),
        _candidate(
            "other-file-match",
            0,
            text="The goods are irretrievably destroyed by an unforeseen event.",
            article_no="1",
            heading_path=("Other", "ARTICLE 1"),
            user_file_id=FILE_B,
        ),
    ]

    selected = select_bounded_source_lexical_matches(
        rows,
        user_file_id=FILE_A,
        query="goods irretrievably destroyed",
        as_of_date=None,
        max_matches=1,
    )

    assert _ids(selected) == ["rare-match"]


def test_source_lexical_matches_diversify_across_provisions() -> None:
    rows = [
        _candidate(
            "article-1-a",
            0,
            text="A narrow trigger applies to the retained amount.",
            article_no="1",
            heading_path=("Instrument", "ARTICLE 1", "(1)"),
        ),
        _candidate(
            "article-1-b",
            1,
            text="The same narrow trigger applies again.",
            article_no="1",
            heading_path=("Instrument", "ARTICLE 1", "(2)"),
        ),
        _candidate(
            "article-2",
            2,
            text="A separate narrow trigger determines the next step.",
            article_no="2",
            heading_path=("Instrument", "ARTICLE 2"),
        ),
    ]

    selected = select_bounded_source_lexical_matches(
        rows,
        user_file_id=FILE_A,
        query="narrow trigger",
        as_of_date=None,
        max_matches=2,
    )

    assert _ids(selected) == ["article-1-a", "article-2"]


def test_structural_companions_do_not_crowd_query_matched_sibling() -> None:
    article = ("Instrument", "ARTICLE 9 - Alpha process")
    nested_parent = (*article, "3) Review path")
    rows = [
        _candidate(
            "seed",
            0,
            text="d) Alpha trigger",
            article_no="9",
            heading_path=(*nested_parent, "d) Alpha trigger"),
            chunk_type="clause",
            clause_label="d",
        ),
        *[
            _candidate(
                f"peer-{index}",
                index,
                text=f"{index}) Structural context",
                article_no="9",
                heading_path=(*nested_parent, f"{index}) Structural context"),
                chunk_type="clause",
                clause_label=str(index),
            )
            for index in range(1, 6)
        ],
        _candidate(
            "query-match",
            7,
            text="Exact quux frobnication remedy",
            article_no="9",
            heading_path=(*article, "b) Exact quux frobnication remedy"),
            chunk_type="clause",
            clause_label="b",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="quux frobnication remedy",
        as_of_date=None,
        max_chunks_per_provision=4,
    )

    assert "query-match" in _ids(selected)


def test_temporal_end_boundary_selects_only_successor() -> None:
    boundary_date = datetime.date(2024, 1, 1)
    rows = [
        _candidate("old", 0, validity_end_date=boundary_date),
        _candidate("new", 0, validity_start_date=boundary_date),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["new"],
        query="",
        as_of_date=boundary_date,
    )

    assert _ids(selected) == ["new"]
    assert [row.projection_index for row in selected] == [0]


def test_same_provision_fails_closed_on_overlapping_visible_versions() -> None:
    rows = [
        _candidate(
            "conflicting-a",
            0,
            text="First allegedly current text.",
        ),
        _candidate(
            "conflicting-b",
            0,
            text="Second allegedly current text.",
        ),
        _candidate(
            "child",
            1,
            text="Shared child.",
        ),
    ]

    assert (
        select_bounded_same_provision_siblings(
            rows,
            ["conflicting-a"],
            query="current text",
            as_of_date=None,
        )
        == []
    )


def test_structural_companion_stays_with_current_or_historical_version() -> None:
    boundary_date = datetime.date(2024, 1, 1)
    prior_date = boundary_date - datetime.timedelta(days=1)
    rows = [
        _candidate(
            "active-parent",
            0,
            text="(3) Güncel üst norm.",
            chunk_type="paragraph",
            paragraph_no="3",
            validity_start_date=boundary_date,
        ),
        _candidate(
            "old-parent",
            0,
            text="(3) Önceki üst norm.",
            chunk_type="paragraph",
            paragraph_no="3",
            validity_end_date=boundary_date,
            status=RegulatoryChunkStatus.SUPERSEDED.value,
        ),
        _candidate(
            "active-seed",
            1,
            text="b) Güncel alt bent.",
            chunk_type="clause",
            clause_label="b",
            validity_start_date=boundary_date,
        ),
        _candidate(
            "old-seed",
            1,
            text="b) Önceki alt bent.",
            chunk_type="clause",
            clause_label="b",
            validity_end_date=boundary_date,
            status=RegulatoryChunkStatus.SUPERSEDED.value,
        ),
    ]

    current = select_bounded_same_provision_siblings(
        rows,
        ["active-seed"],
        query="",
        as_of_date=None,
        max_chunks_per_provision=2,
    )
    at_boundary = select_bounded_same_provision_siblings(
        rows,
        ["active-seed"],
        query="",
        as_of_date=boundary_date,
        max_chunks_per_provision=2,
    )
    historical = select_bounded_same_provision_siblings(
        rows,
        ["old-seed"],
        query="",
        as_of_date=prior_date,
        max_chunks_per_provision=2,
    )

    assert _ids(current) == ["active-parent", "active-seed"]
    assert _ids(at_boundary) == ["active-parent", "active-seed"]
    assert _ids(historical) == ["old-parent", "old-seed"]
    assert current[0].expansion_priority < current[1].expansion_priority
    assert historical[0].expansion_priority < historical[1].expansion_priority


def test_descendant_seed_reserves_article_parent_inside_tight_packet() -> None:
    article_path = ("Instrument", "MADDE 12")
    rows = [
        _candidate(
            "article-parent",
            0,
            text="The operative subject and triggering condition apply.",
            article_no="12",
            heading_path=article_path,
            chunk_type="article",
        ),
        _candidate(
            "ordinary-paragraph",
            1,
            text="A separate paragraph within the article.",
            article_no="12",
            heading_path=(*article_path, "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "selected-clause",
            2,
            text="(a) The narrow consequence.",
            article_no="12",
            heading_path=(*article_path, "(2)", "(a)"),
            chunk_type="clause",
            paragraph_no="2",
            clause_label="a",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["selected-clause"],
        query="narrow consequence",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["article-parent", "selected-clause"]
    priorities = {row.regulatory_chunk_id: row.expansion_priority for row in selected}
    assert priorities["article-parent"] == 0


def test_descendant_seed_keeps_split_article_lead_and_nearest_parent() -> None:
    article_path = ("Instrument", "MADDE 12")
    rows = [
        _candidate(
            "article-lead",
            0,
            text="The article's base scope applies.",
            article_no="12",
            heading_path=article_path,
            chunk_type="article",
        ),
        _candidate(
            "nearest-parent",
            1,
            text="The following branches complete that rule:",
            article_no="12",
            heading_path=article_path,
            chunk_type="article",
        ),
        _candidate(
            "lexical-distractor",
            2,
            text="A narrow consequence appears elsewhere in the article.",
            article_no="12",
            heading_path=(*article_path, "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "selected-clause",
            3,
            text="(a) The narrow consequence.",
            article_no="12",
            heading_path=(*article_path, "(2)", "(a)"),
            chunk_type="clause",
            paragraph_no="2",
            clause_label="a",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["selected-clause"],
        query="narrow consequence",
        as_of_date=None,
        max_chunks_per_provision=3,
    )

    assert _ids(selected) == [
        "article-lead",
        "nearest-parent",
        "selected-clause",
    ]


def test_clause_seed_prioritizes_complete_direct_sibling_family() -> None:
    article_path = ("Instrument", "Annex IV", "MADDE 14")
    clause_parent = (*article_path, "The authority is not obliged when:")
    rows = [
        _candidate(
            "article-parent",
            0,
            text="The authority is not obliged when:",
            article_no="14",
            heading_path=article_path,
            chunk_type="article",
        ),
        _candidate(
            "clause-a",
            1,
            text="(a) the first independent condition applies;",
            article_no="14",
            heading_path=(*clause_parent, "(a) First condition"),
            chunk_type="clause",
            clause_label="a",
        ),
        _candidate(
            "clause-b",
            2,
            text="(b) the second independent condition applies;",
            article_no="14",
            heading_path=(*clause_parent, "(b) Second condition"),
            chunk_type="clause",
            clause_label="b",
        ),
        _candidate(
            "clause-c",
            3,
            text="(c) the requested prerequisite is absent;",
            article_no="14",
            heading_path=(*clause_parent, "(c) Third condition"),
            chunk_type="clause",
            clause_label="c",
        ),
        _candidate(
            "seed-clause-d",
            4,
            text="(d) the threshold is not met.",
            article_no="14",
            heading_path=(*clause_parent, "(d) Fourth condition"),
            chunk_type="clause",
            clause_label="d",
        ),
        _candidate(
            "other-paragraph-clause-a",
            5,
            text="(a) a restarted list must remain separate.",
            article_no="14",
            heading_path=(*article_path, "A different parent:", "(a) Other"),
            chunk_type="clause",
            clause_label="a",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed-clause-d"],
        query="requested prerequisite threshold",
        as_of_date=None,
        max_chunks_per_provision=5,
    )

    assert _ids(selected) == [
        "article-parent",
        "clause-a",
        "clause-b",
        "clause-c",
        "seed-clause-d",
    ]
    priorities = {row.regulatory_chunk_id: row.expansion_priority for row in selected}
    assert priorities["clause-a"] == 0
    assert priorities["clause-b"] == 0
    assert priorities["clause-c"] == 0
    assert "other-paragraph-clause-a" not in priorities


def test_terse_structural_intro_prioritizes_same_article_child_packet() -> None:
    article_path = ("Instrument", "MADDE 3")
    intro_path = (*article_path, "1. In this instrument:")
    rows = [
        _candidate(
            "intro",
            0,
            text="1. In this instrument:",
            article_no="3",
            heading_path=intro_path,
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "child-a",
            1,
            text='(a) "alpha" means the first category;',
            article_no="3",
            heading_path=(*intro_path, '(a) "alpha"'),
            chunk_type="clause",
            clause_label="a",
        ),
        _candidate(
            "child-b",
            2,
            text='(b) "beta" means the second category;',
            article_no="3",
            heading_path=(*intro_path, '(b) "beta"'),
            chunk_type="clause",
            clause_label="b",
        ),
        _candidate(
            "next-paragraph",
            3,
            text="2. An independent rule applies.",
            article_no="3",
            heading_path=(*article_path, "2. An independent rule applies"),
            chunk_type="paragraph",
            paragraph_no="2",
        ),
        _candidate(
            "next-article",
            4,
            text="A rule from another article.",
            article_no="4",
            heading_path=("Instrument", "MADDE 4"),
            chunk_type="article",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["intro"],
        query="beta category",
        as_of_date=None,
        max_chunks_per_provision=4,
    )

    assert _ids(selected) == ["intro", "child-a", "child-b", "next-paragraph"]
    priorities = {row.regulatory_chunk_id: row.expansion_priority for row in selected}
    assert priorities["child-a"] == 0
    assert priorities["child-b"] == 0
    assert "next-article" not in priorities


def test_null_metadata_is_only_kept_as_an_internal_bridge() -> None:
    rows = [
        _candidate(
            "article-60",
            0,
            article_no="60",
            heading_path=("Belge", "MADDE 60"),
        ),
        _candidate("leading-unknown", 1, article_no=None, heading_path=("Belge",)),
        _candidate("seed", 2),
        _candidate("internal-bridge", 3, article_no=None, heading_path=("Belge",)),
        _candidate("same-article", 4, heading_path=("Belge",)),
        _candidate("trailing-unknown", 5, article_no=None, heading_path=("Belge",)),
        _candidate(
            "article-62",
            6,
            article_no="62",
            heading_path=("Belge", "MADDE 62"),
        ),
        _candidate("repeated-61", 7),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["seed", "internal-bridge", "same-article"]


def test_different_explicit_anchor_stops_same_numbered_article() -> None:
    rows = [
        _candidate(
            "appendix-one-seed",
            0,
            article_no="12",
            heading_path=("Belge", "EK I", "MADDE 12"),
        ),
        _candidate(
            "appendix-one-paragraph",
            1,
            article_no="12",
            heading_path=("Belge", "EK I", "Alt başlık"),
        ),
        _candidate(
            "appendix-four-article",
            2,
            article_no="12",
            heading_path=("Belge", "EK IV", "MADDE 12"),
        ),
        _candidate(
            "appendix-four-paragraph",
            3,
            article_no="12",
            heading_path=("Belge", "EK IV"),
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["appendix-one-seed"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["appendix-one-seed", "appendix-one-paragraph"]


def test_reverse_article_anchor_precedes_stale_article_metadata() -> None:
    rows = [
        _candidate(
            "article-four",
            0,
            article_no="4",
            heading_path=("Belge", "MADDE 4"),
        ),
        _candidate(
            "reverse-seed",
            1,
            article_no="4",
            heading_path=("Belge", "MADDE 4", "4A Maddesi:"),
        ),
        _candidate(
            "reverse-sibling",
            2,
            article_no="4",
            heading_path=("Belge", "MADDE 4", "4a MADDESİ:", "(1)"),
        ),
        _candidate(
            "article-five",
            3,
            article_no="5",
            heading_path=("Belge", "MADDE 4", "4A Maddesi:", "MADDE 5"),
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["reverse-seed"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["reverse-seed", "reverse-sibling"]


def test_unknown_seed_does_not_expand_without_a_provision_identity() -> None:
    rows = [
        _candidate("unknown-before", 0, article_no=None, heading_path=("Belge",)),
        _candidate("seed", 1, article_no=None, heading_path=("Belge",)),
        _candidate("unknown-after", 2, article_no=None, heading_path=("Belge",)),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["seed"]


def test_articleless_numbered_protocol_seed_adds_adjacent_peer_context() -> None:
    rows = [
        _candidate(
            "preamble",
            0,
            article_no=None,
            heading_path=("PROTOKOL",),
            chunk_type="free_text",
        ),
        _candidate(
            "scope-rule",
            1,
            text="1) Protokol yalnız ulusal transit hareketlerinde uygulanır.",
            article_no=None,
            heading_path=("PROTOKOL", "1) Uygulama kapsamı"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "seed-exclusion",
            2,
            text="2) Sistem kapsamında taşınamayacak eşya aşağıdadır:",
            article_no=None,
            heading_path=("PROTOKOL", "2) Eşya istisnaları:"),
            chunk_type="numbered_section",
        ),
        _candidate(
            "next-rule",
            3,
            text="3) Teminat tutarı belirlenir.",
            article_no=None,
            heading_path=("PROTOKOL", "3) Teminat tutarı"),
            chunk_type="paragraph",
            paragraph_no="3",
        ),
        _candidate(
            "nested-clause",
            4,
            article_no=None,
            heading_path=("PROTOKOL", "3) Teminat tutarı", "a) Alt bent"),
            chunk_type="clause",
            clause_label="a",
        ),
        _candidate(
            "restarted-list",
            5,
            article_no=None,
            heading_path=("PROTOKOL", "1. Yürürlük"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed-exclusion"],
        query="taşınamayacak eşya",
        as_of_date=None,
        max_chunks_per_provision=3,
    )

    assert _ids(selected) == ["scope-rule", "seed-exclusion", "next-rule"]
    assert selected[0].expansion_priority < selected[1].expansion_priority
    assert selected[2].expansion_priority < selected[1].expansion_priority


def test_articleless_numbered_peer_does_not_cross_parent_scope() -> None:
    rows = [
        _candidate(
            "other-scope",
            0,
            article_no=None,
            heading_path=("EK I", "1) Başka kapsam"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "seed",
            1,
            article_no=None,
            heading_path=("PROTOKOL", "2) Eşya istisnaları"),
            chunk_type="paragraph",
            paragraph_no="2",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="eşya istisnaları",
        as_of_date=None,
    )

    assert _ids(selected) == ["seed"]


def test_articleless_numbered_peer_requires_adjacent_ordinal() -> None:
    rows = [
        _candidate(
            "seed",
            0,
            article_no=None,
            heading_path=("PROTOKOL", "2) Eşya istisnaları"),
            chunk_type="paragraph",
            paragraph_no="2",
        ),
        _candidate(
            "sparse-seven",
            1,
            article_no=None,
            heading_path=("PROTOKOL", "7) Tahsilat usulü"),
            chunk_type="paragraph",
            paragraph_no="7",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="eşya istisnaları",
        as_of_date=None,
    )

    assert _ids(selected) == ["seed"]


def test_articleless_clause_seed_expands_exact_numbered_parent_family() -> None:
    rows = [
        _candidate(
            "previous-rule",
            0,
            article_no=None,
            heading_path=("PROTOKOL", "6) Belge sunumu"),
            chunk_type="paragraph",
            paragraph_no="6",
        ),
        _candidate(
            "parent-seven",
            1,
            article_no=None,
            heading_path=("PROTOKOL", "7) Tahsilat aşağıdaki gibi yapılır:"),
            chunk_type="numbered_section",
        ),
        _candidate(
            "clause-a",
            2,
            article_no=None,
            heading_path=(
                "PROTOKOL",
                "7) Tahsilat aşağıdaki gibi yapılır:",
                "a) İlk işlem",
            ),
            chunk_type="clause",
            clause_label="a",
        ),
        _candidate(
            "seed-clause-b",
            3,
            article_no=None,
            heading_path=(
                "PROTOKOL",
                "7) Tahsilat aşağıdaki gibi yapılır:",
                "b) İkinci işlem",
            ),
            chunk_type="clause",
            clause_label="b",
        ),
        _candidate(
            "clause-c",
            4,
            article_no=None,
            heading_path=(
                "PROTOKOL",
                "7) Tahsilat aşağıdaki gibi yapılır:",
                "c) Üçüncü işlem",
            ),
            chunk_type="clause",
            clause_label="c",
        ),
        _candidate(
            "restarted-one",
            5,
            article_no=None,
            heading_path=("PROTOKOL", "1. Yürürlük"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed-clause-b"],
        query="ikinci işlem",
        as_of_date=None,
    )

    assert _ids(selected) == [
        "parent-seven",
        "clause-a",
        "seed-clause-b",
        "clause-c",
    ]


def test_identityless_seed_expands_only_when_both_visible_neighbors_converge() -> None:
    rows = [
        _candidate("article-61", 0),
        _candidate("legacy-heading", 1, article_no=None, heading_path=("Belge",)),
        _candidate("article-61-paragraph", 2, heading_path=("Belge",)),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["legacy-heading"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == [
        "article-61",
        "legacy-heading",
        "article-61-paragraph",
    ]
    assert selected[1].article_no == "61"


def test_identityless_seed_between_different_provisions_stays_singleton() -> None:
    rows = [
        _candidate(
            "article-60",
            0,
            article_no="60",
            heading_path=("Belge", "MADDE 60"),
        ),
        _candidate("legacy-heading", 1, article_no=None, heading_path=("Belge",)),
        _candidate(
            "article-61",
            2,
            article_no="61",
            heading_path=("Belge", "MADDE 61"),
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["legacy-heading"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["legacy-heading"]


def test_one_sided_identityless_seed_stays_singleton() -> None:
    rows = [
        _candidate("legacy-heading", 0, article_no=None, heading_path=("Belge",)),
        _candidate("article-61", 1),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["legacy-heading"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["legacy-heading"]


def test_identityless_seed_between_same_number_in_different_scopes_stays_singleton() -> (
    None
):
    rows = [
        _candidate(
            "appendix-one",
            0,
            article_no="12",
            heading_path=("Belge", "EK I", "MADDE 12"),
        ),
        _candidate("legacy-heading", 1, article_no=None, heading_path=("Belge",)),
        _candidate(
            "appendix-four",
            2,
            article_no="12",
            heading_path=("Belge", "EK IV", "MADDE 12"),
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["legacy-heading"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["legacy-heading"]


def test_identityless_seed_ignores_neighbors_outside_temporal_snapshot() -> None:
    boundary = datetime.date(2025, 1, 1)
    rows = [
        _candidate("article-61", 0),
        _candidate("legacy-heading", 1, article_no=None, heading_path=("Belge",)),
        _candidate(
            "expired-61",
            2,
            validity_end_date=boundary,
            status=RegulatoryChunkStatus.SUPERSEDED.value,
        ),
        _candidate(
            "visible-62",
            3,
            article_no="62",
            heading_path=("Belge", "MADDE 62"),
            validity_start_date=boundary,
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["legacy-heading"],
        query="",
        as_of_date=boundary,
    )

    assert _ids(selected) == ["legacy-heading"]


def test_large_group_prefers_lexical_overlap_then_proximity() -> None:
    rows = [
        _candidate("seed", 0, text="seed"),
        _candidate("near", 1, text="yakın metin"),
        _candidate("far-match", 2, text="küresel teminat kuralları"),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="küresel teminat",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["seed", "far-match"]


def test_large_group_matches_inflected_query_terms() -> None:
    rows = [
        _candidate("seed", 0, text="opening rule"),
        _candidate("exact-distractor", 1, text="alkol raporu"),
        _candidate(
            "inflected-result",
            2,
            text="alkollü sürücü için yasal sınırlar uygulanır",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="alkol sınırı sürücülerinde",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["seed", "inflected-result"]


@pytest.mark.parametrize(
    "reference_text",
    [
        "1. 2 nci ve 4 üncü fıkralar birlikte uygulanır.",
        "1. 2 nci ve 4 üncü fıkraların hükümleri birlikte uygulanır.",
        "1. Paragraphs (2) and (4) apply together.",
    ],
)
def test_explicit_local_paragraph_references_are_structural_companions(
    reference_text: str,
) -> None:
    heading_path = ("Belge", "MADDE 117")
    rows = [
        _candidate(
            "seed",
            0,
            text=reference_text,
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "paragraph-two",
            1,
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="2",
        ),
        _candidate(
            "paragraph-three",
            2,
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="3",
        ),
        _candidate(
            "paragraph-four",
            3,
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="4",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="unrelated terms",
        as_of_date=None,
        max_chunks_per_provision=3,
    )

    assert _ids(selected) == ["seed", "paragraph-two", "paragraph-four"]
    assert [row.expansion_priority for row in selected[1:]] == [0, 0]


def test_local_paragraph_reference_beats_query_overlap_within_tight_budget() -> None:
    heading_path = ("Belge", "MADDE 117")
    rows = [
        _candidate(
            "seed",
            0,
            text="1. Paragraph 4 governs this exception.",
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "lexical-competitor",
            1,
            text="aranan ifade aranan ifade aranan ifade",
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="3",
        ),
        _candidate(
            "paragraph-four",
            2,
            text="4. The controlling exception.",
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="4",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="aranan ifade",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["seed", "paragraph-four"]
    assert selected[1].expansion_priority == 0


def test_local_paragraph_reference_does_not_cross_structural_scope() -> None:
    rows = [
        _candidate(
            "main-seed",
            0,
            text="1. 4 üncü fıkra hükümleri saklıdır.",
            article_no="10",
            heading_path=("Belge", "ANA METİN", "MADDE 10"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "annex-four",
            1,
            text="4. Ekteki farklı düzenleme.",
            article_no="10",
            heading_path=("Belge", "EK IV", "MADDE 10"),
            chunk_type="paragraph",
            paragraph_no="4",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["main-seed"],
        query="farklı düzenleme",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["main-seed"]


def test_other_article_paragraph_reference_is_not_local() -> None:
    heading_path = ("Belge", "MADDE 117")
    rows = [
        _candidate(
            "seed",
            0,
            text="Article 12, paragraph 4 governs the external rule.",
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "lexical-competitor",
            1,
            text="aranan ifade aranan ifade",
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="3",
        ),
        _candidate(
            "local-four",
            2,
            text="4. Same-article paragraph that must not be inferred.",
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="4",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="aranan ifade",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["seed", "lexical-competitor"]


def test_local_paragraph_reference_requires_explicit_current_article_anchor() -> None:
    rows = [
        _candidate(
            "anchorless-seed",
            0,
            text="1. 4 üncü fıkra hükümleri saklıdır.",
            article_no=None,
            heading_path=("Belge",),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "anchorless-four",
            1,
            text="4. Kimliği belirsiz düzenleme.",
            article_no=None,
            heading_path=("Belge",),
            chunk_type="paragraph",
            paragraph_no="4",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["anchorless-seed"],
        query="kimliği belirsiz",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["anchorless-seed"]


def test_excessive_local_paragraph_reference_list_fails_closed() -> None:
    heading_path = ("Belge", "MADDE 117")
    rows = [
        _candidate(
            "seed",
            0,
            text=("Paragraphs 2, 3, 4, 5, 6, 7, 8, 9 and 10 apply together."),
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "paragraph-two",
            1,
            text="2. Referenced but not lexically relevant.",
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="2",
        ),
        _candidate(
            "lexical-winner",
            2,
            text="exclusive lexical winner",
            article_no="117",
            heading_path=heading_path,
            chunk_type="paragraph",
            paragraph_no="99",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="exclusive lexical winner",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["seed", "lexical-winner"]
    assert selected[1].expansion_priority == 1


def test_clause_seed_reserves_nearest_paragraph_parent() -> None:
    rows = [
        _candidate(
            "parent",
            0,
            text="(3) Üçüncü türün kapsamı ve alt türleri şunlardır:",
            chunk_type="paragraph",
            paragraph_no="3",
        ),
        _candidate(
            "first-clause",
            1,
            text="a) Birinci alt tür.",
            chunk_type="clause",
            clause_label="a",
        ),
        _candidate(
            "seed",
            2,
            text="b) İkinci alt tür.",
            chunk_type="clause",
            clause_label="b",
        ),
        _candidate(
            "lexical-distractor",
            3,
            text="b) aranan aranan aranan",
            chunk_type="clause",
            clause_label="b",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="aranan",
        as_of_date=None,
        max_chunks_per_provision=2,
    )

    assert _ids(selected) == ["parent", "seed"]
    assert selected[0].expansion_priority < selected[1].expansion_priority


def test_terse_restarted_list_item_reserves_mapping_context() -> None:
    rows = [
        _candidate("first-intro", 0, text="İlk liste aşağıdadır."),
        _candidate(
            "first-one",
            1,
            text="1) Birinci uzun kategori açıklaması ve kapsamı",
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "first-two",
            2,
            text="2) İkinci uzun kategori açıklaması ve kapsamı",
            chunk_type="paragraph",
            paragraph_no="2",
        ),
        _candidate(
            "second-intro",
            3,
            text="Önceki kategorilere karşılık gelen ikinci liste aşağıdadır.",
            chunk_type="paragraph",
        ),
        _candidate(
            "second-one",
            4,
            text="1) 12 birim",
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "seed",
            5,
            text="2) 24 birim",
            chunk_type="paragraph",
            paragraph_no="2",
        ),
        _candidate(
            "lexical-distractor",
            6,
            text="3) aranan aranan aranan",
            chunk_type="paragraph",
            paragraph_no="3",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="aranan",
        as_of_date=None,
        max_chunks_per_provision=3,
    )

    assert _ids(selected) == ["first-two", "second-intro", "seed"]


def test_restarted_list_does_not_treat_previous_numbered_item_as_intro() -> None:
    rows = [
        _candidate(
            "category-one",
            0,
            text="1) Birinci kategori",
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "category-two",
            1,
            text="2) İkinci kategori",
            chunk_type="paragraph",
            paragraph_no="2",
        ),
        _candidate(
            "category-three",
            2,
            text="3) Üçüncü kategori",
            chunk_type="paragraph",
            paragraph_no="3",
        ),
        _candidate(
            "quantity-one",
            3,
            text="1) 300 birim",
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "seed",
            4,
            text="2) 500 birim",
            chunk_type="paragraph",
            paragraph_no="2",
        ),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="500 birim",
        as_of_date=None,
        max_chunks_per_provision=3,
    )

    assert _ids(selected) == ["category-two", "quantity-one", "seed"]
    assert "category-three" not in _ids(selected)


def test_character_budget_skips_oversized_candidate_and_keeps_seed() -> None:
    rows = [
        _candidate("seed", 0, text="seed"),
        _candidate("short", 1, text="kısa"),
        _candidate("long-match", 2, text="küresel teminat için çok uzun metin"),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["seed"],
        query="küresel teminat",
        as_of_date=None,
        max_chunks_per_provision=3,
        max_chars_per_provision=12,
    )

    assert _ids(selected) == ["seed", "short"]
    assert sum(len(row.text) for row in selected) <= 12


def test_default_chunk_limit_is_per_provision() -> None:
    rows = [_candidate(f"chunk-{position:02d}", position) for position in range(30)]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["chunk-10"],
        query="",
        as_of_date=None,
    )

    assert len(selected) == DEFAULT_PROVISION_MAX_CHUNKS
    assert "chunk-10" in _ids(selected)


def test_seed_file_does_not_pull_same_article_from_another_file() -> None:
    rows = [
        _candidate("file-a-seed", 0),
        _candidate("file-a-sibling", 1),
        _candidate("file-b-same-article", 0, user_file_id=FILE_B),
    ]

    selected = select_bounded_same_provision_siblings(
        rows,
        ["file-a-seed"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["file-a-seed", "file-a-sibling"]


def test_adjacent_provisions_select_immediate_same_scope_neighbors() -> None:
    rows = [
        _candidate(
            f"article-{article_no}",
            article_no - 110,
            article_no=str(article_no),
            heading_path=("Convention", "ANNEX IV", f"ARTICLE {article_no}"),
            text=f"Operative text {article_no}",
        )
        for article_no in range(111, 116)
    ]

    selected = select_bounded_adjacent_provisions(
        rows,
        ["article-113"],
        query="operative allocation",
        as_of_date=None,
    )

    assert _ids(selected) == ["article-112", "article-114"]


def test_active_amendment_keeps_logical_adjacent_provisions() -> None:
    rows = [
        _candidate(
            "article-89",
            89,
            article_no="89",
            heading_path=("Belge", "MADDE 89"),
        ),
        _candidate(
            "article-90-old",
            90,
            article_no="90",
            heading_path=("Belge", "MADDE 90"),
            status=RegulatoryChunkStatus.SUPERSEDED.value,
        ),
        _candidate(
            "article-90-amendment",
            90,
            article_no="90",
            heading_path=("Belge", "MADDE 90"),
        ),
        _candidate(
            "article-91",
            91,
            article_no="91",
            heading_path=("Belge", "MADDE 91"),
        ),
    ]

    selected = select_bounded_adjacent_provisions(
        rows,
        ["article-90-amendment"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["article-89", "article-91"]


def test_adjacent_provisions_do_not_cross_structural_scope() -> None:
    rows = [
        _candidate(
            "annex-three-last",
            1,
            article_no="9",
            heading_path=("Convention", "ANNEX III", "ARTICLE 9"),
        ),
        _candidate(
            "annex-four-first",
            2,
            article_no="1",
            heading_path=("Convention", "ANNEX IV", "ARTICLE 1"),
        ),
        _candidate(
            "annex-four-second",
            3,
            article_no="2",
            heading_path=("Convention", "ANNEX IV", "ARTICLE 2"),
        ),
    ]

    selected = select_bounded_adjacent_provisions(
        rows,
        ["annex-four-first"],
        query="",
        as_of_date=None,
    )

    assert _ids(selected) == ["annex-four-second"]


def test_returned_projection_is_immutable() -> None:
    selected = select_bounded_same_provision_siblings(
        [_candidate("seed", 0)],
        ["seed"],
        query="",
        as_of_date=None,
    )

    with pytest.raises(FrozenInstanceError):
        setattr(selected[0], "text", "changed")


@pytest.mark.parametrize(
    ("max_chunks", "max_chars"),
    [(0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_rejects_non_positive_budgets(max_chunks: int, max_chars: int) -> None:
    with pytest.raises(ValueError):
        select_bounded_same_provision_siblings(
            [_candidate("seed", 0)],
            ["seed"],
            query="",
            as_of_date=None,
            max_chunks_per_provision=max_chunks,
            max_chars_per_provision=max_chars,
        )


def test_explicit_reference_resolves_one_unique_multi_paragraph_provision() -> None:
    rows = [
        _candidate(
            "source-hit",
            20,
            article_no=None,
            heading_path=("Belge", "EK VII"),
        ),
        _candidate(
            "target-one",
            4,
            article_no="4A",
            heading_path=("Belge", "MADDE 4A", "(1)"),
        ),
        _candidate(
            "target-two",
            5,
            article_no="4A",
            heading_path=("Belge", "MADDE 4A", "(2)"),
        ),
        _candidate(
            "other-file-target",
            4,
            article_no="4A",
            heading_path=("Belge", "MADDE 4A"),
            user_file_id=FILE_B,
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="4A", qualifier=None)],
        as_of_date=None,
    )

    assert _ids(selected) == ["target-one", "target-two"]


def test_explicit_reference_resolves_legacy_target_from_authoritative_metadata() -> (
    None
):
    rows = [
        _candidate(
            "source-hit",
            20,
            article_no=None,
            heading_path=("Belge", "EK VII"),
        ),
        _candidate(
            "legacy-target",
            4,
            article_no="4A",
            heading_path=("Belge", "Export restrictions"),
            chunk_type="article",
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="4A", qualifier=None)],
        as_of_date=None,
    )

    assert _ids(selected) == ["legacy-target"]
    assert "MADDE 4A" in selected[0].heading_path


def test_explicit_reference_repairs_truncated_parent_before_article_anchor() -> None:
    rows = [
        _candidate(
            "target-paragraph-one",
            1,
            article_no="65",
            heading_path=(
                "Belge",
                "Authorization chapter",
                "6. A long paragraph from the prior provision was truncated...",
                "MADDE 65 - Current rule",
                "1. Current paragraph",
            ),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "target-paragraph-two",
            2,
            article_no=None,
            heading_path=(
                "Belge",
                "Authorization chapter",
                "2. Conditions:",
            ),
            chunk_type="numbered_section",
            paragraph_no="2",
        ),
        _candidate(
            "target-clause",
            3,
            article_no="65",
            heading_path=(
                "Belge",
                "Authorization chapter",
                "2. Conditions:",
                "(a) First condition",
            ),
            chunk_type="clause",
            clause_label="a",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 65 uyarınca işlem yapılır.",
            article_no="68",
            heading_path=("Belge", "Authorization chapter", "MADDE 68"),
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="65", qualifier=None)],
        as_of_date=None,
    )

    assert _ids(selected) == [
        "target-paragraph-one",
        "target-paragraph-two",
        "target-clause",
    ]
    assert selected[0].heading_path == (
        "Belge",
        "Authorization chapter",
        "MADDE 65 - Current rule",
        "1. Current paragraph",
    )


def test_explicit_reference_repairs_exact_boundary_parent_before_article_anchor() -> (
    None
):
    leaked_body = (
        "İzin vermeye yetkili gümrük makamı, yeniden değerlendirme sunucunu "
        "izin sahibine bildirir."
    )
    assert len(leaked_body) == 90
    rows = [
        _candidate(
            "target-article",
            1,
            article_no="67",
            heading_path=(
                "Belge",
                "Authorization chapter",
                f"2. {leaked_body}",
                "MADDE 67 - Suspension",
            ),
            chunk_type="article",
        ),
        _candidate(
            "target-intro",
            2,
            article_no=None,
            heading_path=(
                "Belge",
                "Authorization chapter",
                "1. The authority suspends the authorization when:",
            ),
            chunk_type="numbered_section",
        ),
        _candidate(
            "target-clause",
            3,
            article_no="67",
            heading_path=(
                "Belge",
                "Authorization chapter",
                "1. The authority suspends the authorization when:",
                "(a) The conditions are not met.",
            ),
            chunk_type="clause",
            clause_label="a",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 67 uyarınca işlem yapılır.",
            article_no="68",
            heading_path=("Belge", "Authorization chapter", "MADDE 68"),
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="67", qualifier=None)],
        as_of_date=None,
    )

    assert _ids(selected) == [
        "target-article",
        "target-intro",
        "target-clause",
    ]
    assert selected[0].heading_path == (
        "Belge",
        "Authorization chapter",
        "MADDE 67 - Suspension",
    )


def test_explicit_reference_inherits_contiguous_article_lineage() -> None:
    rows = [
        _candidate(
            "target-article",
            1,
            text="Madde 112",
            article_no="112",
            article_title="Debt and debtor",
            heading_path=(
                "Belge",
                "DEBT AND RECOVERY",
                "CHAPTER I",
                "MADDE 112 - Debt and debtor",
            ),
            chunk_type="article",
        ),
        _candidate(
            "target-intro",
            2,
            text="1. A debt is incurred:",
            article_no=None,
            heading_path=(
                "Belge",
                "DEBT AND RECOVERY",
                "Debt incurred",
                "1. A debt is incurred:",
            ),
            chunk_type="numbered_section",
        ),
        _candidate(
            "target-clause",
            3,
            text="(b) when a condition governing the procedure is not met.",
            article_no="112",
            article_title="Debt and debtor",
            heading_path=(
                "Belge",
                "DEBT AND RECOVERY",
                "Debt incurred",
                "1. A debt is incurred:",
                "(b) A condition is not met.",
            ),
            chunk_type="clause",
            clause_label="b",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 112(1)(b) uyarınca borçlu belirlenir.",
            article_no="113",
            heading_path=("Belge", "DEBT AND RECOVERY", "MADDE 113"),
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="112", qualifier=None)],
        as_of_date=None,
        query="rejimin kullanılmasını düzenleyen koşula uyulmaması",
    )

    assert _ids(selected) == [
        "target-article",
        "target-intro",
        "target-clause",
    ]
    assert selected[-1].heading_path == (
        "Belge",
        "DEBT AND RECOVERY",
        "CHAPTER I",
        "MADDE 112 - Debt and debtor",
        "Debt incurred",
        "1. A debt is incurred:",
        "(b) A condition is not met.",
    )


def test_contiguous_lineage_does_not_collapse_distinct_scope() -> None:
    rows = [
        _candidate(
            "main-article",
            1,
            article_no="10",
            heading_path=("Belge", "MAIN TEXT", "MADDE 10"),
            chunk_type="article",
        ),
        _candidate(
            "annex-paragraph",
            2,
            article_no="10",
            heading_path=("Belge", "ANNEX VI", "1. Annex rule"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 10 uyarınca işlem yapılır.",
            heading_path=("Belge", "OTHER"),
        ),
    ]

    assert (
        select_bounded_referenced_provisions(
            rows,
            ["source-hit"],
            [RegulatoryProvisionReference(article_no="10", qualifier=None)],
            as_of_date=None,
        )
        == []
    )


def test_contiguous_lineage_does_not_collapse_nested_annex_scope() -> None:
    rows = [
        _candidate(
            "main-article",
            1,
            article_no="10",
            article_title="Shared rule",
            heading_path=("Belge", "Instrument", "MAIN TEXT", "MADDE 10"),
            chunk_type="article",
        ),
        _candidate(
            "annex-paragraph",
            2,
            article_no="10",
            article_title="Shared rule",
            heading_path=(
                "Belge",
                "Instrument",
                "ANNEX VI",
                "1. Annex rule",
            ),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 10 uyarınca işlem yapılır.",
            article_no="20",
            heading_path=("Belge", "Instrument", "MADDE 20"),
        ),
    ]

    assert (
        select_bounded_referenced_provisions(
            rows,
            ["source-hit"],
            [RegulatoryProvisionReference(article_no="10", qualifier=None)],
            as_of_date=None,
        )
        == []
    )


def test_adverbial_boundary_word_does_not_break_valid_lineage() -> None:
    rows = [
        _candidate(
            "target-article",
            1,
            article_no="10",
            article_title="Controlling rule",
            heading_path=("Belge", "Instrument", "CHAPTER I", "MADDE 10"),
            chunk_type="article",
        ),
        _candidate(
            "target-child",
            2,
            article_no="10",
            article_title="Controlling rule",
            heading_path=(
                "Belge",
                "Instrument",
                "Ek olarak uygulanacak koşullar",
                "(1)",
            ),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 10 uyarınca işlem yapılır.",
            article_no="20",
            heading_path=("Belge", "Instrument", "MADDE 20"),
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="10", qualifier=None)],
        as_of_date=None,
    )

    assert _ids(selected) == ["target-article", "target-child"]
    assert "CHAPTER I" in selected[-1].heading_path


@pytest.mark.parametrize(
    ("old_article_id", "new_article_id"),
    [("a-old-article", "z-new-article"), ("z-old-article", "a-new-article")],
)
def test_contiguous_lineage_is_snapshot_local_and_id_order_independent(
    old_article_id: str,
    new_article_id: str,
) -> None:
    boundary = datetime.date(2025, 1, 1)
    rows = [
        _candidate(
            old_article_id,
            1,
            article_no="10",
            article_title="Historical rule",
            heading_path=("Belge", "Instrument", "OLD SCOPE", "MADDE 10"),
            chunk_type="article",
            status=RegulatoryChunkStatus.SUPERSEDED.value,
            validity_end_date=boundary,
        ),
        _candidate(
            new_article_id,
            1,
            article_no="10",
            article_title="Current rule",
            heading_path=("Belge", "Instrument", "NEW SCOPE", "MADDE 10"),
            chunk_type="article",
            validity_start_date=boundary,
        ),
        _candidate(
            "old-child",
            2,
            article_no="10",
            article_title="Historical rule",
            heading_path=("Belge", "Instrument", "Historical details", "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
            status=RegulatoryChunkStatus.SUPERSEDED.value,
            validity_end_date=boundary,
        ),
        _candidate(
            "new-child",
            2,
            article_no="10",
            article_title="Current rule",
            heading_path=("Belge", "Instrument", "Current details", "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
            validity_start_date=boundary,
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 10 uyarınca işlem yapılır.",
            article_no="20",
            heading_path=("Belge", "Instrument", "MADDE 20"),
        ),
    ]
    reference = RegulatoryProvisionReference(article_no="10", qualifier=None)

    historical = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [reference],
        as_of_date=boundary - datetime.timedelta(days=1),
    )
    current = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [reference],
        as_of_date=None,
    )

    assert _ids(historical) == [old_article_id, "old-child"]
    assert _ids(current) == [new_article_id, "new-child"]
    assert "OLD SCOPE" in historical[-1].heading_path
    assert "NEW SCOPE" in current[-1].heading_path


def test_metadata_free_structural_boundary_stops_article_lineage() -> None:
    rows = [
        _candidate(
            "chapter-one-article",
            1,
            article_no="10",
            article_title="Shared rule",
            heading_path=("Belge", "Instrument", "CHAPTER I", "MADDE 10"),
            chunk_type="article",
        ),
        _candidate(
            "chapter-two-boundary",
            2,
            article_no=None,
            heading_path=("Belge", "Instrument", "CHAPTER II"),
            chunk_type="heading",
        ),
        _candidate(
            "chapter-two-paragraph",
            3,
            article_no="10",
            article_title="Shared rule",
            heading_path=("Belge", "Instrument", "CHAPTER II", "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 10 uyarınca işlem yapılır.",
            article_no="20",
            heading_path=("Belge", "Instrument", "MADDE 20"),
        ),
    ]

    assert (
        select_bounded_referenced_provisions(
            rows,
            ["source-hit"],
            [RegulatoryProvisionReference(article_no="10", qualifier=None)],
            as_of_date=None,
        )
        == []
    )


def test_metadata_only_next_article_establishes_its_own_lineage() -> None:
    rows = [
        _candidate(
            "article-112",
            1,
            article_no="112",
            article_title="Earlier rule",
            heading_path=("Belge", "Instrument", "MADDE 112"),
            chunk_type="article",
        ),
        _candidate(
            "article-113",
            2,
            article_no="113",
            article_title="Later rule",
            heading_path=("Belge", "Instrument", "Later rule"),
            chunk_type="article",
        ),
        _candidate(
            "article-113-child",
            3,
            article_no="113",
            article_title="Later rule",
            heading_path=("Belge", "Instrument", "Later rule", "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 113 uyarınca işlem yapılır.",
            article_no="120",
            heading_path=("Belge", "Instrument", "MADDE 120"),
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="113", qualifier=None)],
        as_of_date=None,
    )

    assert _ids(selected) == ["article-113", "article-113-child"]
    assert all("MADDE 112" not in row.heading_path for row in selected)


@pytest.mark.parametrize(
    ("metadata_prefix", "heading_prefix", "qualifier"),
    [
        ("GEÇİCİ", "GEÇİCİ MADDE", "gecici"),
        ("MÜKERRER", "MÜKERRER MADDE", "mukerrer"),
    ],
)
def test_qualified_article_lineage_preserves_qualifier_identity(
    metadata_prefix: str,
    heading_prefix: str,
    qualifier: str,
) -> None:
    rows = [
        _candidate(
            "qualified-article",
            1,
            article_no=f"{metadata_prefix} 2",
            article_title="Qualified rule",
            heading_path=(
                "Belge",
                "Instrument",
                f"{heading_prefix} 2 - Qualified rule",
            ),
            chunk_type="article",
        ),
        _candidate(
            "qualified-child",
            2,
            article_no=f"{metadata_prefix} 2",
            article_title="Qualified rule",
            heading_path=("Belge", "Instrument", "Qualified details", "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "source-hit",
            10,
            text=f"{heading_prefix} 2 uyarınca işlem yapılır.",
            article_no="20",
            heading_path=("Belge", "Instrument", "MADDE 20"),
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="2", qualifier=qualifier)],
        as_of_date=None,
    )

    assert _ids(selected) == ["qualified-article", "qualified-child"]
    assert heading_prefix in selected[-1].heading_path[2]


def test_ordinary_article_metadata_does_not_inherit_qualified_lineage() -> None:
    rows = [
        _candidate(
            "temporary-article",
            1,
            article_no="GEÇİCİ 2",
            article_title="Temporary rule",
            heading_path=("Belge", "Instrument", "GEÇİCİ MADDE 2"),
            chunk_type="article",
        ),
        _candidate(
            "ordinary-child",
            2,
            article_no="2",
            article_title="Temporary rule",
            heading_path=("Belge", "Instrument", "Ordinary details", "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 2 uyarınca işlem yapılır.",
            article_no="20",
            heading_path=("Belge", "Instrument", "MADDE 20"),
        ),
    ]

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [RegulatoryProvisionReference(article_no="2", qualifier=None)],
        as_of_date=None,
    )

    assert _ids(selected) == ["ordinary-child"]
    assert all("GEÇİCİ MADDE 2" not in heading for heading in selected[0].heading_path)


def test_lineage_requires_matching_article_title_metadata() -> None:
    rows = [
        _candidate(
            "article",
            1,
            article_no="10",
            article_title="Controlling rule",
            heading_path=("Belge", "Instrument", "MAIN TEXT", "MADDE 10"),
            chunk_type="article",
        ),
        _candidate(
            "different-title-child",
            2,
            article_no="10",
            article_title="Different rule",
            heading_path=("Belge", "Instrument", "Details", "(1)"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 10 uyarınca işlem yapılır.",
            article_no="20",
            heading_path=("Belge", "Instrument", "MADDE 20"),
        ),
    ]

    assert (
        select_bounded_referenced_provisions(
            rows,
            ["source-hit"],
            [RegulatoryProvisionReference(article_no="10", qualifier=None)],
            as_of_date=None,
        )
        == []
    )


def test_disjoint_repeated_anchor_fails_closed() -> None:
    rows = [
        _candidate(
            "first-ten",
            1,
            article_no="10",
            heading_path=("Belge", "Instrument", "MADDE 10"),
        ),
        _candidate(
            "eleven",
            2,
            article_no="11",
            heading_path=("Belge", "Instrument", "MADDE 11"),
        ),
        _candidate(
            "second-ten",
            3,
            article_no="10",
            heading_path=("Belge", "Instrument", "MADDE 10"),
        ),
        _candidate(
            "source-hit",
            10,
            text="Madde 10 uyarınca işlem yapılır.",
            article_no="20",
            heading_path=("Belge", "Instrument", "MADDE 20"),
        ),
    ]

    assert (
        select_bounded_referenced_provisions(
            rows,
            ["source-hit"],
            [RegulatoryProvisionReference(article_no="10", qualifier=None)],
            as_of_date=None,
        )
        == []
    )


def test_explicit_reference_skips_same_number_in_multiple_scopes() -> None:
    rows = [
        _candidate(
            "source-hit",
            20,
            article_no=None,
            heading_path=("Belge", "EK VII"),
        ),
        _candidate(
            "main-ten",
            1,
            article_no="10",
            heading_path=("Belge", "MADDE 10"),
        ),
        _candidate(
            "annex-ten",
            2,
            article_no="10",
            heading_path=("Belge", "EK VI", "MADDE 10"),
        ),
    ]

    assert (
        select_bounded_referenced_provisions(
            rows,
            ["source-hit"],
            [RegulatoryProvisionReference(article_no="10", qualifier=None)],
            as_of_date=None,
        )
        == []
    )


def test_explicit_reference_preserves_snapshot_and_skips_overlap() -> None:
    boundary = datetime.date(2025, 1, 1)
    rows = [
        _candidate(
            "source-hit",
            20,
            article_no=None,
            heading_path=("Belge", "EK VII"),
        ),
        _candidate(
            "old-four",
            4,
            article_no="4",
            heading_path=("Belge", "MADDE 4"),
            validity_end_date=boundary,
            status=RegulatoryChunkStatus.SUPERSEDED.value,
        ),
        _candidate(
            "new-four",
            4,
            article_no="4",
            heading_path=("Belge", "MADDE 4"),
            validity_start_date=boundary,
        ),
    ]
    reference = RegulatoryProvisionReference(article_no="4", qualifier=None)

    current = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [reference],
        as_of_date=None,
    )
    historical = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [reference],
        as_of_date=boundary - datetime.timedelta(days=1),
    )

    assert _ids(current) == ["new-four"]
    assert _ids(historical) == ["old-four"]

    overlapping = [
        rows[0],
        _candidate(
            "overlap-a",
            4,
            article_no="4",
            heading_path=("Belge", "MADDE 4"),
            validity_start_date=datetime.date(2024, 1, 1),
        ),
        _candidate(
            "overlap-b",
            4,
            article_no="4",
            heading_path=("Belge", "MADDE 4"),
            validity_start_date=datetime.date(2024, 6, 1),
        ),
    ]
    assert (
        select_bounded_referenced_provisions(
            overlapping,
            ["source-hit"],
            [reference],
            as_of_date=datetime.date(2024, 7, 1),
        )
        == []
    )


def test_explicit_reference_lane_obeys_provision_and_chunk_caps() -> None:
    rows = [
        _candidate(
            "source-hit",
            20,
            article_no=None,
            heading_path=("Belge", "Konu"),
        )
    ]
    references: list[RegulatoryProvisionReference] = []
    for article_no in range(1, 6):
        references.append(
            RegulatoryProvisionReference(article_no=str(article_no), qualifier=None)
        )
        for paragraph in range(1, 4):
            rows.append(
                _candidate(
                    f"article-{article_no}-{paragraph}",
                    article_no * 10 + paragraph,
                    article_no=str(article_no),
                    heading_path=("Belge", f"MADDE {article_no}", f"({paragraph})"),
                )
            )

    selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        references,
        as_of_date=None,
        max_provisions=3,
        max_chunks_per_provision=2,
    )

    assert len(selected) == 6
    assert {row.article_no for row in selected} == {"1", "2", "3"}


def test_explicit_reference_uses_query_within_existing_chunk_budget() -> None:
    rows = [
        _candidate(
            "source-hit",
            20,
            text="Madde 65 uyarınca işlem yapılır.",
            article_no="68",
            heading_path=("Belge", "MADDE 68"),
        ),
        _candidate(
            "target-seed",
            1,
            text="1. İzin sahibi değişiklikleri bildirir.",
            article_no="65",
            heading_path=("Belge", "MADDE 65", "1. Bildirim"),
            chunk_type="paragraph",
            paragraph_no="1",
        ),
        _candidate(
            "nearby-one",
            2,
            text="2. Yanlış bilgi halinde izin iptal edilir.",
            article_no="65",
            heading_path=("Belge", "MADDE 65", "2. İptal"),
            chunk_type="paragraph",
            paragraph_no="2",
        ),
        _candidate(
            "nearby-two",
            3,
            text="3. Karar izin sahibine bildirilir.",
            article_no="65",
            heading_path=("Belge", "MADDE 65", "3. Bildirim"),
            chunk_type="paragraph",
            paragraph_no="3",
        ),
        _candidate(
            "material-condition",
            4,
            text="4. Mali yeterlilik koşulu artık karşılanmıyorsa izin değiştirilir.",
            article_no="65",
            heading_path=("Belge", "MADDE 65", "4. Koşullar"),
            chunk_type="paragraph",
            paragraph_no="4",
        ),
    ]
    reference = RegulatoryProvisionReference(article_no="65", qualifier=None)

    proximity_selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [reference],
        as_of_date=None,
        max_chunks_per_provision=3,
    )
    focused_selected = select_bounded_referenced_provisions(
        rows,
        ["source-hit"],
        [reference],
        as_of_date=None,
        query="mali yeterlilik koşulu artık karşılanmıyorsa",
        max_chunks_per_provision=3,
    )

    assert _ids(proximity_selected) == [
        "target-seed",
        "nearby-one",
        "nearby-two",
    ]
    assert _ids(focused_selected) == [
        "target-seed",
        "nearby-one",
        "material-condition",
    ]


def _navigation_seed(
    chunk_id: str,
    position: int,
    *,
    user_file_id: UUID = FILE_A,
) -> RegulatoryNavigationSeed:
    return RegulatoryNavigationSeed(
        regulatory_chunk_id=chunk_id,
        user_file_id=user_file_id,
        position=position,
    )


def _heading_candidate(
    *,
    status: str,
    validity_start_date: datetime.date | None = None,
    validity_end_date: datetime.date | None = None,
) -> RegulatoryProvisionHeadingCandidate:
    return RegulatoryProvisionHeadingCandidate(
        user_file_id=FILE_A,
        position=0,
        heading_path=("Belge", "MADDE 1"),
        status=status,
        validity_start_date=validity_start_date,
        validity_end_date=validity_end_date,
    )


def test_navigation_requires_two_distinct_regulatory_seeds_from_one_file() -> None:
    assert (
        select_dominant_regulatory_navigation_seed_file(
            [
                _navigation_seed("same", 1),
                _navigation_seed("same", 1),
                _navigation_seed("other", 2, user_file_id=FILE_B),
            ]
        )
        is None
    )


def test_navigation_chooses_highest_count_then_first_ranked_file() -> None:
    selected = select_dominant_regulatory_navigation_seed_file(
        [
            _navigation_seed("b-first", 8, user_file_id=FILE_B),
            _navigation_seed("a-first", 4),
            _navigation_seed("b-second", 7, user_file_id=FILE_B),
            _navigation_seed("a-second", 3),
        ]
    )

    assert selected == (FILE_B, (7, 8))


def _navigation_query_result(rows: list[SimpleNamespace]) -> MagicMock:
    result = MagicMock()
    result.all.return_value = rows
    return result


def _navigation_heading_record(
    user_file_id: UUID,
    position: int,
    heading_path: tuple[str, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"heading-{user_file_id}-{position}",
        user_file_id=user_file_id,
        position=position,
        heading_path=heading_path,
        chunk_metadata={},
        status=RegulatoryChunkStatus.ACTIVE.value,
        validity_start_date=None,
        validity_end_date=None,
    )


def test_heading_source_falls_back_when_dominant_file_is_not_structural() -> None:
    seed_rows = [
        SimpleNamespace(id="a-1", user_file_id=FILE_A, position=1),
        SimpleNamespace(id="a-2", user_file_id=FILE_A, position=2),
        SimpleNamespace(id="a-3", user_file_id=FILE_A, position=3),
        SimpleNamespace(id="b-1", user_file_id=FILE_B, position=40),
        SimpleNamespace(id="b-2", user_file_id=FILE_B, position=41),
    ]
    heading_rows = [
        _navigation_heading_record(FILE_A, 1, ("Üyelik tablosu", "EK VII")),
        _navigation_heading_record(FILE_A, 2, ("Üyelik tablosu", "Taraflar")),
        _navigation_heading_record(FILE_B, 40, ("Sözleşme", "MADDE 4")),
        _navigation_heading_record(FILE_B, 41, ("Sözleşme", "MADDE 4A")),
    ]
    db_session = MagicMock()
    db_session.execute.side_effect = [
        _navigation_query_result(seed_rows),
        _navigation_query_result(heading_rows),
    ]
    db_session.scalar.return_value = "Bazel Sözleşmesi"

    source = get_regulatory_provision_heading_source(
        db_session,
        [row.id for row in seed_rows],
        as_of_date=None,
    )

    assert source is not None
    assert source.user_file_id == FILE_B
    assert source.seed_positions == (40, 41)
    assert {candidate.user_file_id for candidate in source.candidates} == {FILE_B}


def test_heading_source_preserves_structural_dominant_file() -> None:
    seed_rows = [
        SimpleNamespace(id="a-1", user_file_id=FILE_A, position=1),
        SimpleNamespace(id="a-2", user_file_id=FILE_A, position=2),
        SimpleNamespace(id="a-3", user_file_id=FILE_A, position=3),
        SimpleNamespace(id="b-1", user_file_id=FILE_B, position=40),
        SimpleNamespace(id="b-2", user_file_id=FILE_B, position=41),
    ]
    heading_rows = [
        _navigation_heading_record(FILE_A, 1, ("Ana metin", "MADDE 1")),
        _navigation_heading_record(FILE_B, 40, ("Diğer metin", "MADDE 4A")),
    ]
    db_session = MagicMock()
    db_session.execute.side_effect = [
        _navigation_query_result(seed_rows),
        _navigation_query_result(heading_rows),
    ]
    db_session.scalar.return_value = "Ana metin"

    source = get_regulatory_provision_heading_source(
        db_session,
        [row.id for row in seed_rows],
        as_of_date=None,
    )

    assert source is not None
    assert source.user_file_id == FILE_A
    assert source.seed_positions == (1, 2, 3)


def test_heading_source_returns_none_without_two_seeds_from_one_file() -> None:
    seed_rows = [
        SimpleNamespace(id="a-1", user_file_id=FILE_A, position=1),
        SimpleNamespace(id="b-1", user_file_id=FILE_B, position=40),
    ]
    db_session = MagicMock()
    db_session.execute.return_value = _navigation_query_result(seed_rows)

    assert (
        get_regulatory_provision_heading_source(
            db_session,
            [row.id for row in seed_rows],
            as_of_date=None,
        )
        is None
    )
    assert db_session.execute.call_count == 1
    db_session.scalar.assert_not_called()


def test_heading_source_returns_none_when_no_eligible_file_has_articles() -> None:
    seed_rows = [
        SimpleNamespace(id="a-1", user_file_id=FILE_A, position=1),
        SimpleNamespace(id="a-2", user_file_id=FILE_A, position=2),
        SimpleNamespace(id="b-1", user_file_id=FILE_B, position=40),
        SimpleNamespace(id="b-2", user_file_id=FILE_B, position=41),
    ]
    heading_rows = [
        _navigation_heading_record(FILE_A, 1, ("Üyelik tablosu", "EK VII")),
        _navigation_heading_record(FILE_B, 40, ("Taraf listesi", "Bölüm 4")),
    ]
    db_session = MagicMock()
    db_session.execute.side_effect = [
        _navigation_query_result(seed_rows),
        _navigation_query_result(heading_rows),
    ]

    assert (
        get_regulatory_provision_heading_source(
            db_session,
            [row.id for row in seed_rows],
            as_of_date=None,
        )
        is None
    )
    db_session.scalar.assert_not_called()


def test_current_navigation_uses_active_status_only() -> None:
    future_active = _heading_candidate(
        status=RegulatoryChunkStatus.ACTIVE.value,
        validity_start_date=datetime.date(2030, 1, 1),
    )
    superseded = _heading_candidate(
        status=RegulatoryChunkStatus.SUPERSEDED.value,
        validity_end_date=datetime.date(2025, 1, 1),
    )

    assert is_regulatory_navigation_candidate_visible(
        future_active,
        as_of_date=None,
    )
    assert not is_regulatory_navigation_candidate_visible(
        superseded,
        as_of_date=None,
    )


def test_historical_navigation_uses_half_open_validity_window() -> None:
    boundary = datetime.date(2025, 1, 1)
    old = _heading_candidate(
        status=RegulatoryChunkStatus.SUPERSEDED.value,
        validity_start_date=datetime.date(2024, 1, 1),
        validity_end_date=boundary,
    )
    successor = _heading_candidate(
        status=RegulatoryChunkStatus.ACTIVE.value,
        validity_start_date=boundary,
    )

    assert not is_regulatory_navigation_candidate_visible(
        old,
        as_of_date=boundary,
    )
    assert is_regulatory_navigation_candidate_visible(
        successor,
        as_of_date=boundary,
    )
