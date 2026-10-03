import pytest

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc


def original(
    title: str,
    article: str,
    *,
    kind: str | None = "kanun",
    text: str = "İzin için başvuru gerekir.",
    external: bool = False,
) -> EvidenceItem:
    chunk = f"{title}-{article}"
    metadata = {"title": title, "heading_path": [title, f"Madde {article}"]}
    if kind is not None:
        metadata["document_type"] = kind
    return EvidenceItem(
        source_id=title,
        chunk_id=chunk,
        text=text,
        metadata={"canonical_metadata": metadata, "external": external},
        search_doc=SearchDoc(
            document_id=title,
            chunk_ind=0,
            semantic_identifier=title,
            link="https://example.test/provision",
            blurb=text,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": chunk},
            match_highlights=[],
        ),
    )


def ledger_with(*items: EvidenceItem) -> EvidenceLedger:
    ledger = EvidenceLedger()
    ledger.add(list(items), RunContext())
    return ledger


def test_lower_reference_is_not_the_named_statute_original() -> None:
    answer = "8917 sayılı Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir [1]."
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ", text=answer))
    gap = native_named_authority_gap(answer, ledger)
    assert gap is not None
    assert gap["named_authority_gaps"]


@pytest.mark.parametrize("kind", ["kanun", None])
def test_official_law_name_matches_without_a_number_in_canonical_metadata(
    kind: str | None,
) -> None:
    ledger = ledger_with(original("FAALİYET KANUNU", "27", kind=kind))
    assert (
        native_named_authority_gap(
            "8917 sayılı Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir [1].",
            ledger,
        )
        is None
    )
    # A bare number does not create a verified mapping to an unnamed source.
    assert native_named_authority_gap(
        "8917 sayılı Kanun gereğince izin gerekir [1].", ledger
    )


def test_explicit_lower_type_cannot_be_relabelled_from_its_title() -> None:
    ledger = ledger_with(original("8917 sayılı Faaliyet Kanunu", "27", kind="tebliğ"))
    assert native_named_authority_gap(
        "8917 sayılı Faaliyet Kanunu m. 27 uygulanır [1].", ledger
    )


def test_matching_original_elsewhere_cannot_supply_local_attribution() -> None:
    ledger = ledger_with(
        original("Uygulama Tebliği", "3", kind="tebliğ"),
        original("8917 sayılı Faaliyet Kanunu", "27"),
    )
    answer = (
        "8917 sayılı Faaliyet Kanunu m. 27 uyarınca izin gerekir [1].\n\n"
        "Başvuru ayrıca incelenir [2]."
    )
    assert native_named_authority_gap(answer, ledger)
    assert (
        native_named_authority_gap(
            answer.replace("gerekir [1]", "gerekir [1][2]"), ledger
        )
        is None
    )


def test_compound_statutes_keep_independent_local_article_matches() -> None:
    ledger = ledger_with(
        original("8917 sayılı Faaliyet Kanunu", "27"),
        original("7251 sayılı Veri Kanunu", "45"),
    )
    answer = (
        "8917 sayılı Faaliyet Kanunu m. 27 [1] ve 7251 sayılı Veri Kanunu m. 45 [2] "
        "uyarınca işlem yapılır."
    )
    assert native_named_authority_gap(answer, ledger) is None
    assert native_named_authority_gap(answer.replace("[2]", "[1]"), ledger)


def test_unnumbered_formal_name_is_recognized_from_a_delivered_numbered_reference() -> (
    None
):
    ledger = ledger_with(
        original(
            "Uygulama Tebliği",
            "3",
            kind="tebliğ",
            text="8917 sayılı Faaliyet Kanunu'nun 27. maddesi uygulanır.",
        )
    )
    answer = "Faaliyet Kanunu'nun 27. maddesi gereğince izin gerekir [1]."
    assert native_named_authority_gap(answer, ledger)
    ledger.add([original("8917 sayılı Faaliyet Kanunu", "27")], RunContext())
    assert native_named_authority_gap(answer.replace("[1]", "[2]"), ledger) is None


def test_unrelated_source_references_do_not_create_answer_obligations() -> None:
    ledger = ledger_with(
        original(
            "Uygulama Tebliği",
            "3",
            kind="tebliğ",
            text="8917 sayılı Faaliyet Kanunu'nun 27. maddesine bakınız.",
        )
    )
    assert (
        native_named_authority_gap(
            "Uygulama Tebliği'nin 3. maddesi kapsamında belge verilir [1].", ledger
        )
        is None
    )


def test_precise_uncited_original_gap_does_not_assert_a_statutory_result() -> None:
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ"))
    notice = "8917 sayılı Faaliyet Kanunu'nun 27. maddesinin özgün metni doğrulanamadı."
    assert native_named_authority_gap(notice, ledger) is None
    assert (
        native_named_authority_gap(
            notice.replace("doğrulanamadı", "henüz incelenmedi"), ledger
        )
        is None
    )
    assert native_named_authority_gap(notice + " [1]", ledger)
    assert native_named_authority_gap(
        "8917 sayılı Faaliyet Kanunu izin verir. " + notice, ledger
    )
    assert native_named_authority_gap("8917 sayılı Faaliyet Kanunu izin verir.", ledger)
    assert native_named_authority_gap(
        "8917 sayılı Faaliyet Kanunu izin verir ve özgün metni doğrulanamadı.", ledger
    )


def test_complete_parent_original_can_supply_a_named_child_attribution() -> None:
    ledger = ledger_with(original("8917 sayılı Faaliyet Kanunu", "27"))
    assert (
        native_named_authority_gap(
            "8917 sayılı Faaliyet Kanunu'nun 27/1-b maddesi uygulanır [1].", ledger
        )
        is None
    )
    assert native_named_authority_gap(
        "8917 sayılı Faaliyet Kanunu'nun 28. maddesi uygulanır [1].", ledger
    )


def test_external_or_misaligned_metadata_cannot_impersonate_a_canonical_original() -> (
    None
):
    external = original("8917 sayılı Faaliyet Kanunu", "27", external=True)
    misaligned = original("8917 sayılı Faaliyet Kanunu", "27")
    assert misaligned.search_doc is not None
    misaligned.search_doc.metadata["regulatory_chunk_id"] = "another-chunk"
    for item in (external, misaligned):
        assert native_named_authority_gap(
            "8917 sayılı Faaliyet Kanunu m. 27 uygulanır [1].", ledger_with(item)
        )


def test_a_known_different_number_cannot_be_closed_by_a_similar_formal_name() -> None:
    ledger = ledger_with(original("9923 sayılı Faaliyet Kanunu", "27"))
    assert native_named_authority_gap(
        "8917 sayılı Faaliyet Kanunu m. 27 uygulanır [1].", ledger
    )


def test_canonical_numeric_basename_requires_law_type_and_official_root() -> None:
    item = original("FAALİYET KANUNU", "27")
    metadata = item.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    metadata["title"] = "Kanunlar/8917_faaliyet_kanunu.md"
    answer = "8917 sayılı FAL Kanunu'nun 27/1-(b) maddesi uygulanır [1]."
    assert native_named_authority_gap(answer, ledger_with(item)) is None
    for title in (
        "Kanunlar/2024.10.03_faaliyet_kanunu.md",
        "Kanunlar/2024-10-03_faaliyet_kanunu.md",
    ):
        metadata["title"] = title
        assert native_named_authority_gap(answer, ledger_with(item))
    metadata["title"] = "Kanunlar/8917_faaliyet_kanunu.md"
    metadata["document_type"] = "tebliğ"
    assert native_named_authority_gap(answer, ledger_with(item))


def test_a_compilation_title_without_canonical_type_is_not_a_law_original() -> None:
    ledger = ledger_with(original("Faaliyet Kanunları", "27", kind=None))
    assert native_named_authority_gap(
        "8917 sayılı Faaliyet Kanunu m. 27 uygulanır [1].", ledger
    )


def test_only_literal_quotes_in_the_local_original_exempt_cross_references() -> None:
    quote = "8917 sayılı Faaliyet Kanunu'nun 27. maddesindeki haller uygulanır."
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ", text=quote))
    assert (
        native_named_authority_gap(f'Uygulama Tebliği şöyledir: "{quote}" [1].', ledger)
        is None
    )
    assert native_named_authority_gap(
        f'Uygulama Tebliği şöyledir: "{quote.replace("27.", "28.")}" [1].', ledger
    )
    assert native_named_authority_gap(
        f'8917 sayılı Faaliyet Kanunu m. 27 uygulanır: "{quote}" [1].', ledger
    )
