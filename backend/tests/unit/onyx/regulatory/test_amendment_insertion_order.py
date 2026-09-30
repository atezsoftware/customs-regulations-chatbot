import pytest

from onyx.regulatory.amendments.insertion_order import (
    OrderMember,
    plan_insertion,
    reconcile_insertion_order,
)


def member(
    identifier: str,
    position: int,
    article: str,
    paragraph: str | None,
    clause: str | None = None,
) -> OrderMember:
    return OrderMember(
        id=identifier,
        position=position,
        article_no=article,
        paragraph_no=paragraph,
        clause_label=clause,
    )


def test_new_paragraph_stays_inside_its_article_before_the_following_article() -> None:
    rows = [
        member("p1", 10, "8", "1"),
        member("p2", 11, "8", "2"),
        member("p3", 12, "8", "3"),
        member("next", 13, "9", "1"),
    ]
    order = plan_insertion(rows, article_no="8", paragraph_no="4", clause_label=None)
    assert (order.after_chunk_id, order.before_chunk_id, order.position) == (
        "p3",
        "next",
        13,
    )


def test_new_clause_stays_in_its_parent_and_turkish_letters_are_distinct() -> None:
    rows = [
        member("g", 10, "8", "1", "g"),
        member("h", 11, "8", "1", "h"),
        member("next-parent", 12, "8", "2", "a"),
    ]
    order = plan_insertion(rows, article_no="8", paragraph_no="1", clause_label="ğ")
    assert (order.after_chunk_id, order.before_chunk_id, order.position) == (
        "g",
        "h",
        11,
    )


def test_existing_label_requires_a_reviewed_renumbering_instead_of_duplicate_identity() -> (
    None
):
    with pytest.raises(ValueError, match="already exists"):
        plan_insertion(
            [member("old", 4, "8", "1", "d")],
            article_no="8",
            paragraph_no="1",
            clause_label="d",
        )


def test_order_proof_detects_a_new_concurrent_sibling() -> None:
    rows = [member("a", 4, "8", "1", "a")]
    first = plan_insertion(rows, article_no="8", paragraph_no="1", clause_label="c")
    second = plan_insertion(
        [*rows, member("b", 5, "8", "1", "b")],
        article_no="8",
        paragraph_no="1",
        clause_label="c",
    )
    assert first.baseline_sha256 != second.baseline_sha256


def test_unrelated_change_does_not_invalidate_reviewed_insertion() -> None:
    original = [
        member("unrelated", 0, "7", "1"),
        member("parent", 1, "8", "3").model_copy(
            update={"source_sha256": "parent-source"}
        ),
        member("next", 2, "9", "1").model_copy(update={"source_sha256": "next-source"}),
    ]
    reviewed = plan_insertion(
        original, article_no="8", paragraph_no="4", clause_label=None
    )
    current = plan_insertion(
        [original[0].model_copy(update={"source_sha256": "changed"}), *original[1:]],
        article_no="8",
        paragraph_no="4",
        clause_label=None,
    )
    assert reviewed.baseline_sha256 != current.baseline_sha256
    assert reconcile_insertion_order(reviewed, current) == current
    legacy = reviewed.model_copy(
        update={"after_source_sha256": None, "before_source_sha256": None}
    )
    assert reconcile_insertion_order(legacy, current) == current


def test_prior_insertion_rebases_position_but_keeps_reviewed_boundary() -> None:
    original = [member("parent", 1, "8", "3"), member("next", 2, "9", "1")]
    reviewed = plan_insertion(
        original, article_no="8", paragraph_no="4", clause_label=None
    )
    current = plan_insertion(
        [
            member("earlier", 0, "7", "1"),
            *[
                row.model_copy(update={"position": row.position + 1})
                for row in original
            ],
        ],
        article_no="8",
        paragraph_no="4",
        clause_label=None,
    )
    assert current.position == reviewed.position + 1
    assert reconcile_insertion_order(reviewed, current) == current


def test_changed_reviewed_boundary_still_requires_new_review() -> None:
    original = [
        member("parent", 1, "8", "3").model_copy(
            update={"source_sha256": "old-source"}
        ),
        member("next", 2, "9", "1"),
    ]
    reviewed = plan_insertion(
        original, article_no="8", paragraph_no="4", clause_label=None
    )
    current = plan_insertion(
        [original[0].model_copy(update={"source_sha256": "new-source"}), original[1]],
        article_no="8",
        paragraph_no="4",
        clause_label=None,
    )
    with pytest.raises(ValueError, match="changed after review"):
        reconcile_insertion_order(reviewed, current)
    moved = plan_insertion(
        [original[0], member("different", 2, "9", "1")],
        article_no="8",
        paragraph_no="4",
        clause_label=None,
    )
    with pytest.raises(ValueError, match="changed after review"):
        reconcile_insertion_order(reviewed, moved)


def test_unknown_parent_is_not_placed_at_the_end_of_the_file() -> None:
    with pytest.raises(ValueError, match="parent"):
        plan_insertion(
            [member("other", 5, "9", "1")],
            article_no="8",
            paragraph_no="1",
            clause_label="a",
        )


def test_direct_article_clauses_do_not_invent_a_numbered_paragraph() -> None:
    rows = [
        member("parent", 0, "8", None).model_copy(
            update={"direct_clause_parent": True}
        ),
        member("a", 1, "8", None, "a"),
        member("b", 2, "8", None, "b"),
        member("next", 3, "9", "1"),
    ]
    order = plan_insertion(rows, article_no="8", paragraph_no=None, clause_label="c")
    assert order.paragraph_no is None and order.position == 3
    assert (order.after_chunk_id, order.before_chunk_id) == ("b", "next")


def test_missing_paragraph_metadata_alone_cannot_prove_a_direct_parent() -> None:
    with pytest.raises(ValueError, match="parent"):
        plan_insertion(
            [member("a", 0, "8", None, "a")],
            article_no="8",
            paragraph_no=None,
            clause_label="b",
        )


@pytest.mark.parametrize(
    ("identity", "after", "before", "position"),
    [
        ("6/A", "p2", "next", 3),
        ("GEÇİCİ 2", "temp1", "temp4", 5),
        ("GEÇİCİ 3", "temp1", "temp4", 5),
    ],
)
def test_new_articles_are_inserted_within_their_numbering_namespace(
    identity: str, after: str, before: str, position: int
) -> None:
    rows = [
        member("a6", 0, "6", None),
        member("p1", 1, "6", "1"),
        member("p2", 2, "6", "2"),
        member("next", 3, "7", None),
        member("temp1", 4, "GEÇİCİ 1", None),
        member("temp4", 5, "GEÇİCİ 4", None),
    ]
    order = plan_insertion(
        rows, article_no=identity, paragraph_no=None, clause_label=None
    )
    assert (order.after_chunk_id, order.before_chunk_id, order.position) == (
        after,
        before,
        position,
    )


def test_new_article_duplicate_and_disordered_source_are_rejected() -> None:
    with pytest.raises(ValueError, match="already exists"):
        plan_insertion(
            [member("existing", 0, "6/A", None)],
            article_no="6/A",
            paragraph_no=None,
            clause_label=None,
        )
    with pytest.raises(ValueError, match="order"):
        plan_insertion(
            [member("a7", 0, "7", None), member("a6", 1, "6", None)],
            article_no="6/A",
            paragraph_no=None,
            clause_label=None,
        )


def test_second_added_article_follows_first_without_replacing_it() -> None:
    rows = [
        member("a6", 0, "6", None),
        member("a6a", 1, "6/A", None),
        member("a7", 2, "7", None),
    ]
    order = plan_insertion(rows, article_no="6/B", paragraph_no=None, clause_label=None)
    assert (order.after_chunk_id, order.before_chunk_id) == ("a6a", "a7")


def test_first_temporary_article_precedes_document_annexes() -> None:
    rows = [member("a7", 0, "7", None), OrderMember(id="annex", position=1)]
    order = plan_insertion(
        rows, article_no="GEÇİCİ 1", paragraph_no=None, clause_label=None
    )
    assert (order.after_chunk_id, order.before_chunk_id, order.position) == (
        "a7",
        "annex",
        1,
    )


def test_explicit_article_neighbour_must_exist() -> None:
    with pytest.raises(ValueError, match="neighbour"):
        plan_insertion(
            [member("a7", 0, "7", None)],
            article_no="6/A",
            paragraph_no=None,
            clause_label=None,
            after_article_no="6",
        )


def test_new_article_cannot_cross_unknown_source_rows() -> None:
    rows = [
        member("a6", 0, "6", None),
        OrderMember(id="unrelated", position=1),
        member("a7", 2, "7", None),
    ]
    with pytest.raises(ValueError, match="boundary"):
        plan_insertion(rows, article_no="6/A", paragraph_no=None, clause_label=None)


def test_explicit_neighbour_preserves_interleaved_legal_namespaces() -> None:
    rows = [
        member("a6", 0, "6", None),
        member("temp1", 1, "GEÇİCİ 1", None),
        member("a7", 2, "7", None),
    ]
    order = plan_insertion(
        rows,
        article_no="6/A",
        paragraph_no=None,
        clause_label=None,
        after_article_no="6",
    )
    assert (order.after_chunk_id, order.before_chunk_id, order.position) == (
        "a6",
        "temp1",
        1,
    )
