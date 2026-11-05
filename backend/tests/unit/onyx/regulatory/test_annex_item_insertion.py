import pytest

from onyx.regulatory.amendments.insertion_order import (
    InsertionOrderError,
    OrderMember,
    plan_insertion,
)
from onyx.regulatory.amendments.new_provision_policy import (
    annex_item_addition,
    explicitly_adds_top_level_provision,
)

TEXT = "MADDE 2- Aynı Tebliğin Ek I’ine aşağıdaki madde eklenmiştir:\n“23. Kontrol ünitesi: Motoru çalıştıran üniteyi ifade eder.”"


def test_numbered_annex_addition_has_its_own_namespace() -> None:
    addition = annex_item_addition(TEXT)
    assert (
        addition is not None
        and addition.item_no == "23"
        and addition.appendix_label == "EK-I"
    )
    assert not explicitly_adds_top_level_provision(TEXT)


def members() -> list[OrderMember]:
    return (
        [OrderMember(id="header", position=0, appendix_label="EK i")]
        + [
            OrderMember(
                id=f"i{x}",
                position=x,
                appendix_label="EK I",
                paragraph_no=str(x),
                annex_item_no=str(x),
                source_sha256=str(x) * 64,
            )
            for x in range(1, 23)
        ]
        + [OrderMember(id="next-annex", position=23, appendix_label="EK II")]
    )


def test_new_23_follows_22_and_precedes_the_next_annex() -> None:
    order = plan_insertion(
        members(),
        article_no=None,
        paragraph_no="23",
        clause_label=None,
        appendix_label="EK I",
    )
    assert (
        order.after_chunk_id == "i22"
        and order.before_chunk_id == "next-annex"
        and order.position == 23
    )
    assert order.article_no is None and order.appendix_label == "EK I"


def test_existing_item_identity_is_never_inserted_twice() -> None:
    with pytest.raises(InsertionOrderError, match="already exists"):
        plan_insertion(
            members(),
            article_no=None,
            paragraph_no="22",
            clause_label=None,
            appendix_label="EK I",
        )


def test_annex_item_cannot_borrow_a_different_annex_scope() -> None:
    with pytest.raises(InsertionOrderError):
        plan_insertion(
            members(),
            article_no=None,
            paragraph_no="23",
            clause_label=None,
            appendix_label="EK IV",
        )


def test_annex_order_rejects_number_not_proven_by_own_text() -> None:
    rows = members()
    rows[22] = rows[22].model_copy(update={"annex_item_no": "21"})
    with pytest.raises(InsertionOrderError, match="source"):
        plan_insertion(
            rows,
            article_no=None,
            paragraph_no="23",
            clause_label=None,
            appendix_label="EK I",
        )


def test_approval_rejects_header_prepended_to_new_annex_item() -> None:
    from onyx.regulatory.amendments.draft_integrity import (
        DraftIntegrityError,
        validate_added_article_draft,
    )

    order = plan_insertion(
        members(),
        article_no=None,
        paragraph_no="23",
        clause_label=None,
        appendix_label="EK I",
    )
    with pytest.raises(DraftIntegrityError, match="body"):
        validate_added_article_draft(
            [TEXT],
            metadata={"appendix_label": "EK I", "paragraph_no": "23"},
            heading_path=["EK I", "23. Kontrol ünitesi"],
            old_chunk_id=None,
            insertion_order=order.model_dump(mode="json"),
            draft_text="Tanımlar\n23. Kontrol ünitesi: Motoru çalıştıran üniteyi ifade eder.",
        )


def test_approval_rejects_a_heading_pointing_to_another_annex() -> None:
    from onyx.regulatory.amendments.draft_integrity import (
        DraftIntegrityError,
        validate_added_article_draft,
    )

    order = plan_insertion(
        members(),
        article_no=None,
        paragraph_no="23",
        clause_label=None,
        appendix_label="EK I",
    )
    with pytest.raises(DraftIntegrityError):
        validate_added_article_draft(
            [TEXT],
            metadata={"appendix_label": "EK I", "paragraph_no": "23"},
            heading_path=["Kaynak", "EK II", "23. Kontrol ünitesi"],
            old_chunk_id=None,
            insertion_order=order.model_dump(mode="json"),
            draft_text="23. Kontrol ünitesi: Motoru çalıştıran üniteyi ifade eder.",
        )
