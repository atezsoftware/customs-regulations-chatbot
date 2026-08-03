"""Tests for the ported RegulatoryChunker (backend/onyx/regulatory/chunker.py).

Pure-logic tests: Turkish regulatory structure detection (madde / fıkra /
bent), heading_path construction, oversized-chunk splitting, and document
metadata inference. No services required.
"""

import pytest

from onyx.regulatory.chunker import RegulatoryChunker

SAMPLE_TEBLIG = """GÜMRÜK GENEL TEBLİĞİ

BİRİNCİ BÖLÜM

Amaç, Kapsam ve Dayanak

MADDE 1 - (1) Bu Tebliğin amacı, gümrük işlemlerinin basitleştirilmesine ilişkin usul ve esasları belirlemektir.

MADDE 2 - (1) Bu Tebliğ, aşağıdaki işlemleri kapsar:

a) İthalat işlemleri,

b) İhracat işlemleri.

(2) Transit işlemleri bu Tebliğ kapsamı dışındadır.

İKİNCİ BÖLÜM

Uygulama

MADDE 3 - (1) Başvurular elektronik ortamda yapılır.
"""


def test_article_paragraph_clause_structure() -> None:
    doc = RegulatoryChunker().chunk_text(SAMPLE_TEBLIG, source_file="teblig.md")

    articles = {
        chunk.metadata.article_no
        for chunk in doc.chunks
        if chunk.metadata.article_no is not None
    }
    assert articles == {"1", "2", "3"}

    clause_labels = {
        chunk.metadata.clause_label
        for chunk in doc.chunks
        if chunk.metadata.clause_label is not None
    }
    assert clause_labels == {"a", "b"}

    # The second paragraph of MADDE 2 must be its own chunk with fıkra no 2.
    para_2_2 = [
        chunk
        for chunk in doc.chunks
        if chunk.metadata.article_no == "2" and chunk.metadata.paragraph_no == "2"
    ]
    assert len(para_2_2) == 1
    assert "Transit" in para_2_2[0].text


def test_heading_path_contains_bolum_and_madde() -> None:
    doc = RegulatoryChunker().chunk_text(SAMPLE_TEBLIG, source_file="teblig.md")

    madde_1_chunks = [c for c in doc.chunks if c.metadata.article_no == "1"]
    assert madde_1_chunks
    heading_path = madde_1_chunks[0].metadata.heading_path
    assert any("BÖLÜM" in part for part in heading_path)
    assert any("MADDE 1" in part for part in heading_path)


def test_document_type_and_title_inference() -> None:
    doc = RegulatoryChunker().chunk_text(SAMPLE_TEBLIG, source_file="teblig.md")
    assert doc.metadata.document_type == "teblig"
    assert doc.metadata.title == "GÜMRÜK GENEL TEBLİĞİ"


@pytest.mark.parametrize(
    ("date_label", "expected_date"),
    [
        ("Resmî Gazete Tarihi: 18/6/2022", "2022-06-18"),
        ("Yayım Tarihi | 7.10.2009", "2009-10-07"),
        ("Kabul Tarihi: 17.02.2024", "2024-02-17"),
        ("Karar Tarihi\n2023-11-09", "2023-11-09"),
        ("Tarih: 3/4/2020", "2020-04-03"),
    ],
)
def test_labeled_header_document_date_inference(
    date_label: str, expected_date: str
) -> None:
    text = f"""GÜMRÜK YÖNETMELİĞİ

{date_label}

MADDE 1 - (1) Bu Yönetmelik uygulanır.
"""

    doc = RegulatoryChunker().chunk_text(text, source_file="upload.bin")

    assert doc.metadata.document_date == expected_date


def test_full_content_official_publication_table_date_is_inferred() -> None:
    body_lines = "\n".join(
        f"MADDE {article_no} - (1) Düzenleme metni." for article_no in range(1, 66)
    )
    text = f"""GÜMRÜK YÖNETMELİĞİ (7)(8)

Dayanak

MADDE 1 - (1) Bu Yönetmelik 10/7/2003 tarihli Kanuna dayanır.

{body_lines}

Yönetmeliğin Yayımlandığı Resmî Gazete’nin
Tarihi | Sayısı
18/6/2022 | 31870
Yönetmelikte Değişiklik Yapan Yönetmeliklerin Yayımlandığı Resmî Gazetelerin
Tarihi | Sayısı
24/4/2019 | 30754
"""

    doc = RegulatoryChunker().chunk_text(text, source_file="upload.bin")

    assert doc.metadata.document_date == "2022-06-18"


@pytest.mark.parametrize(
    ("source_file", "expected_date"),
    [
        ("yonetmelik_2024-05-17.pdf", "2024-05-17"),
        ("yonetmelik_17.05.2024.pdf", "2024-05-17"),
    ],
)
def test_explicit_filename_date_overrides_labeled_document_date(
    source_file: str, expected_date: str
) -> None:
    text = """GÜMRÜK YÖNETMELİĞİ

Resmî Gazete Tarihi: 18/6/2022

MADDE 1 - (1) Bu Yönetmelik uygulanır.
"""

    doc = RegulatoryChunker().chunk_text(text, source_file=source_file)

    assert doc.metadata.file_date == expected_date
    assert doc.metadata.document_date == expected_date


def test_body_enabling_amendment_and_cross_reference_dates_are_ignored() -> None:
    text = """GÜMRÜK YÖNETMELİĞİ

Dayanak

MADDE 1 - (1) Bu Yönetmelik 10/7/2003 tarihli ve 4925 sayılı Kanuna dayanır.

Değişiklik Tarihi: 24/4/2019

MADDE 2 - (1) 17/12/2022 tarihli ve 32046 sayılı Resmî Gazete'de yayımlanan Tebliğe atıf yapılır.

Bu değişiklik 1/1/2023 tarihinde yürürlüğe girer.
"""

    doc = RegulatoryChunker().chunk_text(text, source_file="upload.bin")

    assert doc.metadata.document_date is None


@pytest.mark.parametrize(
    "title",
    [
        "GÜMRÜK YÖNETMELİĞİ (7)(8)",
        "TEHLİKELİ MADDELERİN KARAYOLUYLA TAŞINMASI HAKKINDA YÖNETMELİK",
    ],
)
def test_early_title_document_type_wins_over_enabling_law(title: str) -> None:
    text = f"""{title}

Dayanak

MADDE 1 - (1) Bu Yönetmelik 4925 sayılı Karayolu Taşıma Kanununa dayanır.
"""

    doc = RegulatoryChunker().chunk_text(text, source_file="upload.bin")

    assert doc.metadata.document_type == "yonetmelik"


def test_filename_document_type_keeps_precedence_over_content_title() -> None:
    text = """GÜMRÜK YÖNETMELİĞİ

MADDE 1 - (1) Bu Yönetmelik uygulanır.
"""

    doc = RegulatoryChunker().chunk_text(text, source_file="gumruk-kanunu.pdf")

    assert doc.metadata.document_type == "kanun"


def test_oversized_article_is_split() -> None:
    sentence = "Bu fıkra uygulamaya ilişkin ayrıntılı hükümler içerir. "
    long_paragraphs = "\n\n".join(f"({i}) {sentence * 10}" for i in range(1, 11))
    text = f"MADDE 1 - Uygulama esasları\n\n{long_paragraphs}"

    chunker = RegulatoryChunker(max_chunk_chars=1000)
    doc = chunker.chunk_text(text, source_file="kanun.md")

    assert len(doc.chunks) > 1
    for chunk in doc.chunks:
        # Split parts must stay within the cap (single unsplittable lines are
        # the only exception; these paragraphs are all splittable).
        assert len(chunk.text) <= 1000


def test_empty_text_produces_no_chunks() -> None:
    doc = RegulatoryChunker().chunk_text("", source_file="empty.md")
    assert doc.chunks == []


def test_chunk_order_is_sequential() -> None:
    doc = RegulatoryChunker().chunk_text(SAMPLE_TEBLIG, source_file="teblig.md")
    orders = [chunk.metadata.chunk_order for chunk in doc.chunks]
    assert orders == list(range(len(doc.chunks)))


def test_bold_paragraph_is_retained_as_article_content() -> None:
    text = """GÜMRÜK KANUNU

MADDE 92

**(1) Transit sırasında bir olay meydana gelirse taşıyıcı en yakın gümrük idaresine bildirimde bulunur ve tutanak düzenlenmesini sağlar.**

(2) Gümrük idaresi gerekli önlemleri alır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(text, source_file="kanun.md")

    article_chunks = [
        chunk for chunk in doc.chunks if chunk.metadata.article_no == "92"
    ]
    assert [chunk.metadata.paragraph_no for chunk in article_chunks] == ["1", "2"]
    assert "tutanak düzenlenmesini" in article_chunks[0].text


def test_article_intro_is_not_dropped_when_child_paragraph_follows() -> None:
    text = """ULUSLARARASI SÖZLEŞME

MADDE 6

İhracatçı Devlet, sınırötesi taşınımı ilgili devletlere yazılı olarak bildirir ve bildirim gerekli bütün bilgileri içerir.

(2) İlgili devlet bildirime yazılı cevap verir.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="sozlesme.md"
    )

    article_chunks = [chunk for chunk in doc.chunks if chunk.metadata.article_no == "6"]
    assert any("sınırötesi taşınımı" in chunk.text for chunk in article_chunks)
    assert any(chunk.metadata.paragraph_no == "2" for chunk in article_chunks)
    assert not any(
        "sınırötesi taşınımı" in heading
        for chunk in article_chunks
        for heading in chunk.metadata.heading_path
    )


def test_ocr_paragraph_marker_does_not_cause_parent_text_loss() -> None:
    text = """KANUN

MADDE 10

l. Birinci fıkra OCR sırasında küçük L harfiyle işaretlenmiş olabilir.

(2) İkinci fıkra açık bir işaret taşır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(text, source_file="ocr.md")

    combined = "\n".join(
        chunk.text for chunk in doc.chunks if chunk.metadata.article_no == "10"
    )
    assert "küçük L harfiyle" in combined
    assert "İkinci fıkra" in combined


def test_article_punctuation_and_appendix_prose_classification() -> None:
    text = """TEBLİĞ

Ek bilgi sunulması gerekir.

MADDE: 1

(1) Birinci hüküm.

EK IV - BELGELER

MADDE. 2

(1) İkinci hüküm.

EK IIIa

MADDE 3

(1) Alfanümerik ek hükmü.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(text, source_file="teblig.md")

    by_article = {
        chunk.metadata.article_no: chunk
        for chunk in doc.chunks
        if chunk.metadata.article_no is not None
    }
    assert {"1", "2", "3"}.issubset(by_article)
    assert by_article["1"].metadata.appendix_label is None
    assert by_article["2"].metadata.appendix_label == "EK iv"
    assert by_article["3"].metadata.appendix_label == "EK iiia"


def test_inline_ocr_marker_and_following_article_body_are_retained() -> None:
    text = """GÜMRÜK KANUNU

**MADDE 92- l.** Transit eşya taşıyan araçta olay meydana gelirse durum gecikmeksizin gümrük idaresine bildirilir.

Eşyanın başka bir taşıta aktarılması gümrük idaresince tutanakla belgelendirilir.

**2.** Eşyanın telef veya kaybının kanıtlanması halinde vergiler aranmaz.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="gumruk-kanunu.md"
    )

    article_text = "\n".join(
        chunk.text for chunk in doc.chunks if chunk.metadata.article_no == "92"
    )
    assert "gecikmeksizin" in article_text
    assert "tutanakla belgelendirilir" in article_text
    assert "vergiler aranmaz" in article_text


def test_multiple_unnumbered_article_paragraphs_survive_later_numbered_unit() -> None:
    text = """BASEL SÖZLEŞMESİ

**Madde 6**

**Taraflar Arasında Sınırlarötesi Taşınım**

İhracatçı Devlet, ilgili devletlere yazılı olarak bildirecektir.

İthalatçı Devlet yazılı cevabını gönderecektir.

İhracatçı Devlet yazılı teyit almadıkça taşınıma izin vermeyecektir.

6. ve 7. fıkralarda belirtilen usuller ayrıca uygulanır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(text, source_file="basel.md")

    article_text = "\n".join(
        chunk.text for chunk in doc.chunks if chunk.metadata.article_no == "6"
    )
    assert "yazılı olarak bildirecektir" in article_text
    assert "yazılı cevabını" in article_text
    assert "yazılı teyit almadıkça" in article_text


def test_numbered_section_title_inside_article_keeps_article_context() -> None:
    text = """ORTAK TRANSİT SÖZLEŞMESİ

**Borç ve borçlu**

**Madde 112**

**Borcun doğması**

1. Madde 3(1)'de belirtilen borç:

(a) eşyanın ortak transit rejiminden çıkarılması;

2. Borç, aşağıdaki durumlarda ortadan kalkar:

(a) gerekli koşulların sonradan yerine getirilmesi.

**Madde 113**

1. Borçlu rejim hak sahibidir.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="ortak-transit.md"
    )

    numbered_sections = [
        chunk for chunk in doc.chunks if chunk.metadata.chunk_type == "numbered_section"
    ]
    assert [chunk.metadata.article_no for chunk in numbered_sections] == ["112", "112"]
    assert [chunk.metadata.paragraph_no for chunk in numbered_sections] == ["1", "2"]
    assert all(
        "MADDE 112 - Borç ve borçlu" in chunk.metadata.heading_path
        for chunk in numbered_sections
    )
    article_112_text = "\n".join(
        chunk.text for chunk in doc.chunks if chunk.metadata.article_no == "112"
    )
    assert "1. Madde 3(1)'de belirtilen borç:" in article_112_text
    assert "2. Borç, aşağıdaki durumlarda" in article_112_text
    assert all(
        "MADDE 112" not in heading
        for chunk in doc.chunks
        if chunk.metadata.article_no == "113"
        for heading in chunk.metadata.heading_path
    )


def test_article_clauses_keep_their_numbered_paragraph_parent() -> None:
    text = """YÖNETMELİK

MADDE 6 - Belge türleri

(3) Üçüncü belge türü: Faaliyetin şekline göre aşağıdaki türlere ayrılır:

a) Birinci alt tür hususi faaliyet yapacaklara verilir,

b) İkinci alt tür ticari faaliyet yapacaklara verilir.

(4) Dördüncü belge türü: Aşağıdaki türlere ayrılır:

a) Yeni bir alt tür belirlenir.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="belge-turleri.md"
    )

    second_subtype = next(
        chunk for chunk in doc.chunks if "İkinci alt tür" in chunk.text
    )
    next_subtype = next(
        chunk for chunk in doc.chunks if "Yeni bir alt tür" in chunk.text
    )

    assert any(
        heading.startswith("(3) Üçüncü belge türü")
        for heading in second_subtype.metadata.heading_path
    )
    assert not any(
        heading.startswith("a) Birinci alt tür")
        for heading in second_subtype.metadata.heading_path
    )
    assert any(
        heading.startswith("(4) Dördüncü belge türü")
        for heading in next_subtype.metadata.heading_path
    )
    assert not any(
        heading.startswith("(3) Üçüncü belge türü")
        for heading in next_subtype.metadata.heading_path
    )


def test_separate_article_title_does_not_create_title_only_chunk() -> None:
    text = """TEBLİĞ

**Madde 1**

**Amaç**

(1) Bu Tebliğin amacı uygulama esaslarını belirlemektir.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(text, source_file="teblig.md")

    article_chunks = [chunk for chunk in doc.chunks if chunk.metadata.article_no == "1"]
    assert len(article_chunks) == 1
    assert article_chunks[0].metadata.paragraph_no == "1"


@pytest.mark.parametrize("reverse_heading", ["4A Maddesi:", "4a MADDESİ."])
def test_reverse_article_heading_creates_article_sibling(
    reverse_heading: str,
) -> None:
    text = f"""TEBLİĞ

MADDE 4

(1) Önceki hüküm.

{reverse_heading}

(1) Eklenen hüküm.

MADDE 5

(1) Sonraki hüküm.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(text, source_file="teblig.md")

    article_four_a = [
        chunk for chunk in doc.chunks if chunk.metadata.article_no == "4A"
    ]
    article_five = [chunk for chunk in doc.chunks if chunk.metadata.article_no == "5"]

    assert article_four_a
    assert article_five
    assert all("MADDE 4A" in chunk.metadata.heading_path for chunk in article_four_a)
    assert all("MADDE 4" not in chunk.metadata.heading_path for chunk in article_four_a)
    assert all("MADDE 4A" not in chunk.metadata.heading_path for chunk in article_five)


def test_reverse_article_reference_prose_stays_in_current_article() -> None:
    text = """TEBLİĞ

MADDE 4

4A maddesi uyarınca işlem yapılır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(text, source_file="teblig.md")

    assert {chunk.metadata.article_no for chunk in doc.chunks} == {"4"}
    assert "4A maddesi uyarınca" in "\n".join(chunk.text for chunk in doc.chunks)


def test_leading_article_cross_references_stay_in_current_article() -> None:
    text = """ULUSLARARASI DÜZENLEME

**Madde 13**

**Bilgi aktarımı**

Taraflar gerekli bilgileri iletir.

Madde 5 uyarınca tayin edilen makam değişiklikleri bildirilir.

Madde 3 gereğince tanımlardaki değişiklikler ayrıca bildirilir.

Madde 4'ün hükümleri saklıdır.

**Madde 14**

**Mali konular**

Taraflar uygun mekanizmaları değerlendirir.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="duzenleme.md"
    )

    assert {chunk.metadata.article_no for chunk in doc.chunks} == {"13", "14"}
    article_thirteen_text = "\n".join(
        chunk.text for chunk in doc.chunks if chunk.metadata.article_no == "13"
    )
    assert "Madde 5 uyarınca" in article_thirteen_text
    assert "Madde 3 gereğince" in article_thirteen_text
    assert "Madde 4'ün hükümleri" in article_thirteen_text


def test_supplement_resets_stale_section_and_keeps_article_scope() -> None:
    text = """ULUSLARARASI SÖZLEŞME

**Ek I**

ON BİRİNCİ BÖLÜM

**Madde 118**

**Borçluya yönelik işlemler**

3. Önceki bölümdeki borç zorunlu süre içinde ödenir.

**İLAVE I**

**MADDE 77'NİN UYGULANMASI**

Kapsamlı teminat kullanımına ilişkin usuller

1. Geçici sınırlamanın uygulanabildiği durumlar

Madde 77(a)'da belirtilen özel koşullar yetkili makamca değerlendirilir.

2. Karar alma usulü ayrıca uygulanır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="sozlesme.md"
    )

    supplement_chunks = [
        chunk for chunk in doc.chunks if "İLAVE I" in chunk.metadata.heading_path
    ]
    assert supplement_chunks
    assert all("EK i" in chunk.metadata.heading_path for chunk in supplement_chunks)
    assert all(
        "ON BİRİNCİ BÖLÜM" not in chunk.metadata.heading_path
        and "Borçluya yönelik işlemler" not in chunk.metadata.heading_path
        for chunk in supplement_chunks
    )

    scoped_chunks = [
        chunk for chunk in supplement_chunks if chunk.metadata.article_no == "77"
    ]
    assert scoped_chunks
    assert all(
        "MADDE 77'NİN UYGULANMASI" in chunk.metadata.heading_path
        for chunk in scoped_chunks
    )
    assert "Madde 77(a)'da belirtilen özel koşullar" in "\n".join(
        chunk.text for chunk in scoped_chunks
    )


def test_parenthetical_article_reference_does_not_start_a_new_article() -> None:
    text = """ULUSLARARASI SÖZLEŞME

MADDE 12

(1) Yetkili makam gerekli incelemeyi yapar.

Madde 7(b)'de belirtilen koşullar ayrıca değerlendirilir.

Madde 8(a) uyarınca bildirim yapılır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="sozlesme.md"
    )

    assert {chunk.metadata.article_no for chunk in doc.chunks} == {"12"}
    article_text = "\n".join(chunk.text for chunk in doc.chunks)
    assert "Madde 7(b)'de belirtilen koşullar" in article_text
    assert "Madde 8(a) uyarınca bildirim yapılır" in article_text


def test_nested_dotted_provisions_create_hierarchical_heading_paths() -> None:
    text = """ULUSLARARASI TEKNİK DÜZENLEME

2.2.3 Sınıf 3 Alevlenebilir sıvılar

2.2.3.1 Kriterler

2.2.3.1.1 Birinci hüküm aynen uygulanır.

(a) İlk şart sağlanır.

2.2.3.1.2 İkinci hüküm aynen uygulanır.

2.2.3.2 Taşıma için kabul edilmeyen maddeler

2.2.3.2.1 Son hüküm aynen uygulanır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="technical-regulation.md"
    )

    first_provision = next(
        chunk for chunk in doc.chunks if chunk.text.startswith("2.2.3.1.1 ")
    )
    sibling_provision = next(
        chunk for chunk in doc.chunks if chunk.text.startswith("2.2.3.1.2 ")
    )
    next_branch = next(
        chunk for chunk in doc.chunks if chunk.text.startswith("2.2.3.2.1 ")
    )
    clause = next(chunk for chunk in doc.chunks if chunk.text.startswith("(a) "))

    assert first_provision.text == "2.2.3.1.1 Birinci hüküm aynen uygulanır."
    assert first_provision.metadata.heading_path[-3:] == [
        "2.2.3 Sınıf 3 Alevlenebilir sıvılar",
        "2.2.3.1 Kriterler",
        "2.2.3.1.1",
    ]
    assert "2.2.3.1.1" not in sibling_provision.metadata.heading_path
    assert sibling_provision.metadata.heading_path[-1] == "2.2.3.1.2"
    assert "2.2.3.1 Kriterler" not in next_branch.metadata.heading_path
    assert next_branch.metadata.heading_path[-2:] == [
        "2.2.3.2 Taşıma için kabul edilmeyen maddeler",
        "2.2.3.2.1",
    ]
    assert clause.metadata.clause_label == "a"
    assert "2.2.3.1.1" in clause.metadata.heading_path
    assert all(chunk.metadata.article_no is None for chunk in doc.chunks)


def test_bare_pdf_page_numbers_require_consecutive_page_evidence() -> None:
    page_filler = "Bu hüküm teknik koşulları ve uygulama esaslarını açıklar. " * 8
    text = f"""138
2.2.3 Sınıf 3 Alevlenebilir sıvılar

2.2.3.1 Kriterler

{page_filler}

139
2.2.3.1.1 Birinci hüküm uygulanır.

{page_filler}

140
2.2.3.1.2 İkinci hüküm uygulanır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="official-extract.txt"
    )

    assert doc.metadata.title == "2.2.3 Sınıf 3 Alevlenebilir sıvılar"
    assert "138" in doc.content
    assert not any(
        line.strip() in {"138", "139", "140"}
        for chunk in doc.chunks
        for line in chunk.text.splitlines()
    )
    assert not any(node.label in {"138", "139", "140"} for node in doc.structure)


def test_decorated_pdf_page_number_is_ignored_without_sequence() -> None:
    text = """- 587 -
BÖLÜM 8.2

8.2.1 Kapsam

8.2.1.1 Sürücü geçerli bir sertifika taşır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="official-extract.txt"
    )

    assert doc.metadata.title == "BÖLÜM 8.2"
    assert all("- 587 -" not in chunk.text for chunk in doc.chunks)
    provision = next(chunk for chunk in doc.chunks if chunk.text.startswith("8.2.1.1 "))
    assert provision.metadata.heading_path[-2:] == ["8.2.1 Kapsam", "8.2.1.1"]


def test_standalone_legal_numbers_are_retained_without_page_sequence() -> None:
    text = """TEKNİK DÜZENLEME

9.1.1.1 Kapsam

1
Birinci dipnotun açıklaması.

2
İkinci dipnotun açıklaması.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="technical-regulation.md"
    )
    combined_text = "\n".join(chunk.text for chunk in doc.chunks)

    assert "\n1\n" in combined_text
    assert "\n2\n" in combined_text


def test_section_reference_prose_does_not_replace_dotted_provision_path() -> None:
    text = """BÖLÜM 9.1

9.1.1 Kapsam ve tanımlar

9.1.1.2 Tanımlar

Kısım 9'un amaçları bakımından araç, taşıma aracıdır.

9.1.2 Araçların onaylanması

9.1.2.3 Yıllık teknik muayene
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="technical-regulation.md"
    )
    annual_inspection = next(
        chunk for chunk in doc.chunks if chunk.text.startswith("9.1.2.3 ")
    )

    assert "Kısım 9'un amaçları bakımından" in "\n".join(
        chunk.text for chunk in doc.chunks
    )
    assert all(
        "Kısım 9'un amaçları bakımından" not in heading
        for heading in annual_inspection.metadata.heading_path
    )


def test_dotted_reference_continuation_stays_in_current_clause() -> None:
    text = """BÖLÜM 8.2

8.2.2 Sürücülerin eğitimi

8.2.2.8 Sürücü eğitimi sertifikası

8.2.2.8.1 Sertifika aşağıdaki durumlarda düzenlenir:

(a) Sürücünün gerekli eğitimi tamamlaması,

(b) 8.2.2.7.1 uyarınca eğitimi tamamlaması ve

8.2.2.7.2 uyarınca sınavı geçmesi durumunda.

(c) Yetkili makamın gerekli kontrolleri tamamlaması.

8.2.2.8.2 Sertifikanın geçerlilik süresi beş yıldır.
"""

    doc = RegulatoryChunker(min_chunk_chars=0).chunk_text(
        text, source_file="technical-regulation.md"
    )
    clause_b = next(chunk for chunk in doc.chunks if chunk.metadata.clause_label == "b")
    clause_c = next(chunk for chunk in doc.chunks if chunk.metadata.clause_label == "c")

    assert "8.2.2.7.2 uyarınca sınavı geçmesi" in clause_b.text
    assert "8.2.2.8 Sürücü eğitimi sertifikası" in clause_b.metadata.heading_path
    assert "8.2.2.8.1" in clause_b.metadata.heading_path
    assert "8.2.2.8.1" in clause_c.metadata.heading_path
    assert all(
        heading != "8.2.2.7.2"
        for chunk in doc.chunks
        for heading in chunk.metadata.heading_path
    )
