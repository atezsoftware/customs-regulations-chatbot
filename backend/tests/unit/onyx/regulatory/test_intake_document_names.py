from typing import Any

from onyx.regulatory.document_intake.document_names import intake_document_name
from onyx.regulatory.document_intake.metadata import METADATA_AUDIT_KEY


def _title(
    value: str | None,
    *,
    source: str = "llm",
    stale: bool = False,
    field: str = "document_title",
) -> dict[str, Any]:
    return {
        field: value,
        METADATA_AUDIT_KEY: {"fields": {field: {"source": source, "stale": stale}}},
    }


def test_source_title_replaces_technical_name_and_joins_line_wrapping() -> None:
    assert (
        intake_document_name(
            [_title("5000 sayılı Gümrük Kanunu"), _title("Gümrük Kanunu")],
            source_text="# 5000 sayılı Gümrük\nKanunu\nMADDE 1 — Kural.",
        )
        == "5000 sayılı Gümrük Kanunu"
    )


def test_filename_only_heading_and_stale_inference_cannot_name_document() -> None:
    assert (
        intake_document_name(
            [_title("20260101-1.pdf"), _title("Gümrük Kanunu", stale=True)],
            source_text="Gümrük Kanunu\nMADDE 1 — Kural.",
        )
        is None
    )


def test_conflicting_source_titles_require_review_instead_of_voting() -> None:
    assert (
        intake_document_name(
            [_title("Gümrük Kanunu"), _title("Vergi Kanunu")],
            source_text="Gümrük Kanunu, Vergi Kanunu hükümlerine dayanır.",
        )
        is None
    )


def test_manual_review_wins_and_clearing_title_is_authoritative() -> None:
    inferred = _title("Gümrük Kanunu")
    assert (
        intake_document_name(
            [inferred, _title("Gümrük Uygulama Kanunu", source="manual")],
            source_text="Gümrük Kanunu",
        )
        == "Gümrük Uygulama Kanunu"
    )
    assert (
        intake_document_name(
            [inferred, _title(None, source="manual")], source_text="Gümrük Kanunu"
        )
        is None
    )


def test_annex_uses_its_own_explicit_title() -> None:
    assert (
        intake_document_name(
            [_title("EK-1 — İbraz Edilecek Belgeler", field="annex_title")],
            source_text="EK-1 — İbraz Edilecek Belgeler\nBelge tablosu.",
        )
        == "EK-1 — İbraz Edilecek Belgeler"
    )


def test_turkish_title_comparison_matches_legal_metadata_validation() -> None:
    title = "Cihazların Enerji Tüketimi"
    assert (
        intake_document_name([_title(title)], source_text="CİHAZLARIN ENERJİ TÜKETİMİ")
        == title
    )


def test_manual_annex_title_wins_over_an_inferred_document_title() -> None:
    assert (
        intake_document_name(
            [
                _title("Atıf Yapılan Kanun"),
                _title("EK-1 — Belgeler", field="annex_title", source="manual"),
            ],
            source_text="Atıf Yapılan Kanun hükümlerine dayanır.",
        )
        == "EK-1 — Belgeler"
    )


def test_clearing_one_field_keeps_another_explicit_manual_title() -> None:
    assert (
        intake_document_name(
            [
                _title(None, source="manual"),
                _title("EK-1 — Belgeler", field="annex_title", source="manual"),
            ],
            source_text="EK-1 — Belgeler",
        )
        == "EK-1 — Belgeler"
    )


def test_conflicting_manual_values_require_review() -> None:
    assert (
        intake_document_name(
            [
                _title("Birinci Kanun", source="manual"),
                _title("İkinci Kanun", source="manual"),
            ],
            source_text="Birinci Kanun ve İkinci Kanun",
        )
        is None
    )


def test_verified_prepared_header_can_name_body_chunks_without_filename_fallback() -> (
    None
):
    assert (
        intake_document_name(
            [
                _title("upload-20260101-1.pdf"),
                _title("İthalatta Makine Uygunluk Tebliği"),
            ],
            source_text="İTHALATTA MAKİNE UYGUNLUK TEBLİĞİ\n\nMADDE 1 — Başvurular elektronik ortamda yapılır.",
        )
        == "İthalatta Makine Uygunluk Tebliği"
    )
