import pytest

from onyx.regulatory.amendments.insertion_order import OrderMember, plan_insertion


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
