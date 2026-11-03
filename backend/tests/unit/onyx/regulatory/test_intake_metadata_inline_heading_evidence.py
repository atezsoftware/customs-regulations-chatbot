"""Flattened formal headings prove only their own uninterrupted paragraph scope."""

import pytest

from onyx.regulatory.document_intake import metadata

SOURCE7_PARENT = (
    "DÖRDÜNCÜ BÖLÜM Personel Nitelikleri ile İlgili Hükümler "
    "Personelde aranacak nitelikler MADDE 11 – (1) Muayene merkezlerinde "
    "çalıştırılacak personelde aşağıdaki şartlar aranır. a) Teknik yönetici/teknik "
    "yönetici yardımcısında, üniversitelerin mühendislik, teknoloji fakülteleri, "
    "4 yıllık teknik okul veya teknik eğitim fakültelerinden mezun olma şartı "
    "aranır. b) Onay Kuruluşu uzmanı/muayene uzmanında, üniversitelerin "
    "mühendislik, teknoloji fakülteleri, 4 yıllık teknik okul veya teknik eğitim "
    "fakültelerinden mezun olma şartı aranır. Üst yapı ile ilgili kaynak kontrolü "
    "gerektirecek istisnai muayenelerde görev alacak muayene uzmanı, muayene için "
    "gerekli olan metotlardan en az birinden tahribatsız muayene seviye II "
    "sertifikasına sahip olmak zorundadır."
)
SOURCE7_MIDDLE = (
    "c) Muayene teknisyeni/teknikerinde, meslek yüksek okullarının makine, "
    "otomotiv, motor, tesviye, elektrik, iklimlendirme ve soğutma, kontrol ve "
    "otomasyon, mekatronik, elektronik bölümü veya meslek liselerinin makine, "
    "motor, tesviye, otomotiv, kontrol ve enstrümantasyon, bilgisayarlı nümerik "
    "kontrol, makine ressamlığı, metal işleri, döküm, elektrik, proses kontrol, "
    "elektro-mekanik, elektro-mekanik taşıyıcılar, elektronik, soğutma ve "
    "iklimlendirme, hidrolik pnömatik, tesisat teknolojisi, iş makineleri veya "
    "mekatronik bölümlerinden birinden mezun olması ve her vardiyada en az "
    "birinin CE sınıfı sürücü belgesine sahip olması gerekir."
)
SOURCE7_CONTINUATION = (
    "ç) Teknik yönetici ve yardımcısı, onay kuruluşu uzmanı, muayene uzmanı, "
    "tekniker ve teknisyenlerin, ADR Genel Bilgilendirme, ADR Tank Muayeneleri "
    "ve ADR Araç Uygunluk Muayeneleri konularında, minimum 20 saat, Onay "
    "Kuruluşundan eğitim almaları gerekmektedir. Ayrıca söz konusu personelin "
    "her iki yılda bir ADR güncellemeleriyle ilgili sekiz saat yenileme eğitimi "
    "almaları zorunludur."
)


def _chunk(identifier: str, text: str, *headings: str) -> metadata.MetadataChunk:
    return metadata.MetadataChunk(
        chunk_id=identifier,
        text=text,
        heading_path=list(headings)
        or ["21 Nisan 2016 PERŞEMBE Resmî Gazete Sayı : 29691"],
        chunk_type="free_text",
    )


def _validate(
    chunks: list[metadata.MetadataChunk],
    source: metadata.MetadataChunk,
    target: metadata.MetadataChunk,
    *,
    value: str = "1",
    quote: str | None = None,
) -> metadata.LegalMetadataResult:
    result = metadata.LegalMetadataResult(
        chunks=[
            metadata.ChunkLegalMetadata(
                chunk_id=target.chunk_id,
                fields=[
                    metadata.LegalMetadataFact(
                        field="paragraph_no",
                        value=value,
                        source_chunk_id=source.chunk_id,
                        evidence_quote=source.text if quote is None else quote,
                    )
                ],
            )
        ]
    )
    return metadata.validate_metadata_response(
        result.model_dump_json(), chunks=chunks, target_ids=[target.chunk_id]
    )


@pytest.mark.parametrize("value", ["1", "(1)"])
def test_frozen_source7_heading_and_clause_continuation_keep_raw_evidence(
    value: str,
) -> None:
    parent = _chunk("parent", SOURCE7_PARENT)
    middle = _chunk("middle", SOURCE7_MIDDLE)
    continuation = _chunk("continuation", SOURCE7_CONTINUATION)
    result = _validate(
        [parent, middle, continuation], parent, continuation, value=value
    )
    fact = result.chunks[0].fields[0]
    assert fact.value == value
    assert fact.evidence_quote == SOURCE7_PARENT
    assert fact.source_chunk_id == parent.chunk_id


def test_frozen_heading_also_proves_its_own_single_paragraph() -> None:
    parent = _chunk("own", SOURCE7_PARENT)
    assert _validate([parent], parent, parent)


def test_formal_heading_can_contain_a_comma_separated_section_title() -> None:
    source = _chunk(
        "source",
        "İKİNCİ BÖLÜM Muayene Esasları, İşletme ve Genel Hükümler "
        "Personel nitelikleri MADDE 11 – (1) Muayene personeli görevlendirilir.",
    )
    target = _chunk("target", "a) Personel eğitilir.")
    assert _validate([source, target], source, target)


@pytest.mark.parametrize(
    "text",
    [
        "Bu hüküm MADDE 11 – (1) uyarınca uygulanır.",
        "DÖRDÜNCÜ BÖLÜM Önceki MADDE 11 – (1) uyarınca uygulanır.",
        "DÖRDÜNCÜ BÖLÜM Hüküm uygulanır MADDE 11 – (1) Personel görevlendirilir.",
        "DÖRDÜNCÜ BÖLÜM Personel MADDE 11 – (1) numaralı fıkra uygulanır.",
        "MADDE 10 – (4) Önceki hüküm. " + SOURCE7_PARENT,
        SOURCE7_PARENT + " (2) İkinci fıkranın şartı uygulanır.",
        SOURCE7_PARENT + " MADDE 12 – (1) Başka madde uygulanır.",
        "DÖRDÜNCÜ BÖLÜM 2025 yılı açıklamaları MADDE 11 – (1) Hüküm.",
        "DÖRDÜNCÜ BÖLÜM " + "Başlık " * 50 + "MADDE 11 – (1) Hüküm.",
    ],
)
def test_incidental_earlier_body_and_mixed_or_distant_scopes_are_rejected(
    text: str,
) -> None:
    source = _chunk("source", text)
    target = _chunk("target", SOURCE7_CONTINUATION)
    with pytest.raises(ValueError, match="Metadata paragraph"):
        _validate([source, target], source, target)


@pytest.mark.parametrize(
    "text,headings",
    [
        ("MADDE 12 – (1) Başka maddenin hükmü.", ()),
        ("(2) İkinci fıkra uygulanır.", ()),
        (SOURCE7_CONTINUATION, ("MADDE 12 – Başka madde",)),
    ],
)
def test_formal_parent_cannot_cross_target_article_or_paragraph(
    text: str, headings: tuple[str, ...]
) -> None:
    source = _chunk("source", SOURCE7_PARENT)
    target = _chunk("target", text, *headings)
    with pytest.raises(ValueError, match="Metadata paragraph"):
        _validate([source, target], source, target)


@pytest.mark.parametrize(
    "intervening",
    [
        "Çalışma günleri MADDE 12 – (1) İşletici çalışır.",
        "(2) Başka fıkra başlar.",
        "Atıf metninde (1) numaralı fıkra anılır.",
    ],
)
def test_parent_does_not_govern_after_an_intervening_operative_scope(
    intervening: str,
) -> None:
    source = _chunk("source", SOURCE7_PARENT)
    middle = _chunk("middle", intervening)
    target = _chunk("target", SOURCE7_CONTINUATION)
    with pytest.raises(ValueError, match="requires explicit source context"):
        _validate([source, middle, target], source, target)


def test_heading_does_not_supply_a_paragraph_to_an_earlier_target() -> None:
    source = _chunk("source", SOURCE7_PARENT)
    target = _chunk("target", SOURCE7_CONTINUATION)
    with pytest.raises(ValueError, match="requires explicit source context"):
        _validate([target, source], source, target)


def test_matching_raw_number_is_required() -> None:
    source = _chunk("source", SOURCE7_PARENT)
    target = _chunk("target", SOURCE7_CONTINUATION)
    with pytest.raises(ValueError, match="paragraph locator contradicts"):
        _validate([source, target], source, target, value="2")


def test_substring_quote_cannot_promote_an_inline_citation() -> None:
    source = _chunk("source", SOURCE7_PARENT)
    target = _chunk("target", SOURCE7_CONTINUATION)
    quote = SOURCE7_PARENT[SOURCE7_PARENT.index("MADDE 11") :]
    with pytest.raises(ValueError, match="requires explicit source context"):
        _validate([source, target], source, target, quote=quote)
