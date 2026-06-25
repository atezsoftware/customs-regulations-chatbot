from fs_explorer.indexing.regulatory_chunker import RegulatoryChunker


def test_inline_article_paragraphs_keep_locator_metadata_without_parent_chunk() -> None:
    text = """
Gümrük Genel Tebliği (Transit Rejimi)

**Amaç**

**MADDE 1 -** (1) Bu Tebliğin amacı transit rejimine ilişkin usulleri belirlemektir.

**Kapsam**

**MADDE 2 -** (1) Bu Tebliğ transit işlemlerini kapsar.
"""

    result = RegulatoryChunker().chunk_text(
        text,
        source_file="gumruk_genel_tebligi_x1transit_rejimix2_seri_no_1.docx",
    )

    assert not any(chunk.metadata.chunk_type == "article" for chunk in result.chunks)

    paragraph_chunks = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_type == "paragraph"
    ]
    assert len(paragraph_chunks) == 2
    assert paragraph_chunks[0].metadata.article_no == "1"
    assert paragraph_chunks[0].metadata.article_title == "Amaç"
    assert paragraph_chunks[0].metadata.paragraph_no == "1"
    assert paragraph_chunks[0].metadata.parent_path[-2] == "MADDE 1 - Amaç"
    assert paragraph_chunks[1].metadata.article_no == "2"
    assert paragraph_chunks[1].metadata.article_title == "Kapsam"
    assert paragraph_chunks[1].metadata.paragraph_no == "1"


def test_header_chunks_are_not_dropped_before_numbered_sections() -> None:
    text = """
**T.C.**
**TİCARET BAKANLIĞI**
**Gümrükler Genel Müdürlüğü**
**GENELGE**
**2022/07**

**1. Transit süre aşımlarında uygulanan idari para cezası:**
Varış bildirimi süre aşımı halinde ilgili mevzuat uygulanır.
"""

    result = RegulatoryChunker().chunk_text(text, source_file="genelge_2022-07.docx")

    assert result.metadata.document_type == "genelge"
    assert result.metadata.document_number == "2022/07"
    assert not any(chunk.metadata.chunk_type == "heading_section" for chunk in result.chunks)
    numbered = [
        chunk.metadata.chunk_type == "numbered_section"
        and "Transit süre" in chunk.metadata.heading_path[-1]
        for chunk in result.chunks
    ]
    assert any(numbered)
    numbered_chunk = next(chunk for chunk in result.chunks if chunk.metadata.chunk_type == "numbered_section")
    assert numbered_chunk.metadata.parent_id is not None
    assert numbered_chunk.metadata.parent_path[0] == result.metadata.title


def test_article_clauses_are_chunks_and_roman_subclauses_stay_with_parent() -> None:
    text = """
TIR Sözleşmesi

**(a) TARİFLER**

**Madde 1**
Bu Sözleşmede:

**(a) "TIR taşıması"** deyiminden; taşıma anlaşılır.
**(b) "TIR işlemi"** deyiminden; işlem anlaşılır.
**(j) "Konteyner"** deyiminden; taşıma işlerinde kullanılan ve,
(i) içine eşya konmak üzere bir kompartman teşkil edecek şekilde,
(ii) devamlılık niteliğine sahip olup,
"Ayrılabilen karoseriler" konteyner olarak telaki edilir.
**(k) "Hareket Gümrük İdaresi"** deyiminden; idare anlaşılır.
"""

    result = RegulatoryChunker().chunk_text(text, source_file="tir_sozlesmesi.docx")

    assert not any(chunk.metadata.chunk_type == "article" for chunk in result.chunks)

    structural_heading = next(
        block for block in result.blocks if block.text == "**(a) TARİFLER**"
    )
    clause_source = next(
        block for block in result.blocks if '"TIR taşıması"' in block.text
    )
    assert structural_heading.kind == "heading"
    assert clause_source.kind == "paragraph"

    clause_chunks = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_type == "clause"
    ]
    assert [chunk.metadata.clause_label for chunk in clause_chunks] == [
        "a",
        "b",
        "j",
        "k",
    ]

    container = next(chunk for chunk in clause_chunks if chunk.metadata.clause_label == "j")
    assert "(i) içine eşya" in container.text
    assert "(ii) devamlılık" in container.text
    assert container.metadata.parent_path[-1] == '(j) "Konteyner"'
    assert container.metadata.parent_path[-2] == "Bu Sözleşmede"
    assert container.metadata.parent_path[-3] == "MADDE 1"
    assert container.metadata.parent_path[-4] == "(a) TARİFLER"

    assert not any(chunk.metadata.subclause_label == "i" for chunk in result.chunks)
    assert not any(
        chunk.text.startswith("(i) içine eşya") for chunk in result.chunks
    )


def test_treaty_preamble_becomes_parent_context_without_becoming_chunk() -> None:
    text = """
**TIR KARNELERİ HİMAYESİNDE ULUSLARARASI EŞYA TAŞINMASINA DAİR GÜMRÜK SÖZLEŞMESİ**
***Karar No.: 85/8993 R.G.: 31.03.1985***
14 Kasım 1975 tarihli ekli sözleşmenin onaylanması kararlaştırılmıştır.
**AKİT TARAFLAR,**
Karayolu taşıtları ile uluslararası eşya taşınmasını kolaylaştırmayı İSTEYEREK,
Aşağıdaki hususlarda ANLAŞMIŞLARDIR:
**Bölüm I**
**GENEL**
**(a) TARİFLER**
**Madde 1**
Bu Sözleşmede:
**(a) "TIR taşıması"** deyiminden; taşıma anlaşılır.
"""

    result = RegulatoryChunker().chunk_text(text, source_file="tir_sozlesmesi.docx")

    assert not any("ANLAŞMIŞLARDIR" in chunk.text for chunk in result.chunks)

    clause = next(chunk for chunk in result.chunks if chunk.metadata.clause_label == "a")
    assert clause.metadata.parent_path == [
        "TIR KARNELERİ HİMAYESİNDE ULUSLARARASI EŞYA TAŞINMASINA DAİR GÜMRÜK SÖZLEŞMESİ",
        "AKİT TARAFLAR, Aşağıdaki hususlarda ANLAŞMIŞLARDIR",
        "Bölüm I",
        "GENEL",
        "(a) TARİFLER",
        "MADDE 1",
        "Bu Sözleşmede",
        '(a) "TIR taşıması"',
    ]


def test_article_paragraph_marker_allows_missing_space_without_inline_splits() -> None:
    text = """
TIR Sözleşmesi

**Madde 6**
1. Bir Akit Tarafın gümrük makamları izin verebilir.
2.Bir ülkede kuruluşun kefaleti yetkilendirilmiş sayılır.
Bu cümlenin içinde 2. maddeye atıf var ama yeni chunk değildir.
Mükerrer-2, Bir uluslararası kuruluş ayrıca yetkilendirilmelidir.
3. Yetkilendirme şartları ayrıca değerlendirilir.
2.1. Bu alt numara ayrı bir üst fıkra değildir.
"""

    result = RegulatoryChunker().chunk_text(text, source_file="tir_sozlesmesi.docx")

    assert not any(chunk.metadata.chunk_type == "article" for chunk in result.chunks)

    paragraph_chunks = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_type == "paragraph"
    ]
    assert [chunk.metadata.paragraph_no for chunk in paragraph_chunks] == [
        "1",
        "2",
        "MÜKERRER 2",
        "3",
    ]
    assert paragraph_chunks[1].text.startswith(
        "2.Bir ülkede kuruluşun kefaleti"
    )
    assert "2. maddeye atıf" in paragraph_chunks[1].text
    assert paragraph_chunks[2].text.startswith(
        "Mükerrer-2, Bir uluslararası kuruluş"
    )
    assert "2.1. Bu alt numara" in paragraph_chunks[3].text


def test_file_date_and_temporary_article_labels_are_generic_locators() -> None:
    text = """
***Karar Sayısı : 2006/10784***
*04.08.2006 tarihli Resmî Gazete*

**MADDE 1** - (1) İstisna uygulanacak sınır kapıları belirlenmiştir.

**GEÇİCİ MADDE 1** - (1) Bu karar kapsamındaki geçiş hükümleri uygulanır.
"""

    result = RegulatoryChunker().chunk_text(
        text,
        source_file=(
            "11.09.2024_100701562_karayolu_tasimaciligi_alanindaki_"
            "gecis_belgesi_dagitim_esaslari_yonergesi.docx"
        ),
    )

    assert result.metadata.file_date == "2024-09-11"
    assert result.metadata.document_date == "2006-08-04"
    assert result.metadata.document_number == "2006/10784"

    temporary = [
        chunk for chunk in result.chunks if chunk.metadata.article_no == "GEÇİCİ 1"
    ]
    assert len(temporary) == 1
    assert temporary[0].metadata.chunk_type == "paragraph"
    assert temporary[0].metadata.paragraph_no == "1"
    assert temporary[0].metadata.parent_path[-2] == "GEÇİCİ MADDE 1"


def test_numbered_items_outside_madde_articles_split_into_their_own_chunks() -> None:
    """Protocols/agreements that number themselves `1)`, `2)`, ... `a)`
    instead of using MADDE have no article context at all — the inline
    paragraph/clause splitter must still apply to them, not just inside
    MADDE articles."""
    text = """
PROTOKOL
Gümrük Müsteşarlığı ile UND arasında aşağıdaki hususlarda mutabık kalmışlardır.
1) Bu protokol hükümleri konteyner ile taşınan eşyanın taşınmasını üstlenen firmalara uygulanacaktır.
2) Global Teminat Sistemi kapsamında taşınacak eşya tanımına:
-petrol ve petrol ürünleri,
dahil değildir.
3) Global Teminat Sistemine dahil eşyanın taşınması için teminat mektubu verilecektir.
4) Teminat mektubu tutarlarının artışı her yıl yeniden hesaplanacaktır.
5) Sistemden istifade edecek firmaların kabulüne Müsteşarlık yetkilidir.
6) Firmaların yetki belgeleri Müsteşarlığa gönderilecektir.
7) Anılan sistem kapsamında taşınan eşyanın süre aşımı halinde aşağıdaki uygulama yapılacaktır.
a) Hareket gümrük idaresince tayin edilen sürenin aşılması halinde usulsüzlük cezaları uygulanacaktır.
"""

    result = RegulatoryChunker().chunk_text(text, source_file="und_protokol.docx")

    assert not any(chunk.metadata.chunk_type == "article" for chunk in result.chunks)
    assert not any("Part" in entry for chunk in result.chunks for entry in chunk.metadata.heading_path)

    paragraph_chunks = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_type == "paragraph"
    ]
    assert [chunk.metadata.paragraph_no for chunk in paragraph_chunks] == [
        "1",
        "3",
        "4",
        "5",
        "6",
        "7",
    ]
    assert all(chunk.metadata.article_no is None for chunk in paragraph_chunks)

    seventh = next(chunk for chunk in paragraph_chunks if chunk.metadata.paragraph_no == "7")
    assert "süre aşımı" in seventh.text
    assert "a) Hareket" not in seventh.text

    # Items 3-7 are siblings of item 2 (and of each other) under the
    # document title — none of them should appear nested under a previous
    # numbered item just because it happened to become a structural node.
    third = next(chunk for chunk in paragraph_chunks if chunk.metadata.paragraph_no == "3")
    assert third.metadata.heading_path == [
        "PROTOKOL",
        "3) Global Teminat Sistemine dahil eşyanın taşınması için teminat mektubu verilecektir.",
    ]

    clause_chunks = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_type == "clause"
    ]
    assert [chunk.metadata.clause_label for chunk in clause_chunks] == ["a"]
    assert "usulsüzlük cezaları" in clause_chunks[0].text
    # The clause nests specifically under item 7 (its real parent), not
    # under some other sibling top-level item.
    assert clause_chunks[0].metadata.heading_path[-2].startswith("7) Anılan sistem")

    numbered_section_chunks = [
        chunk for chunk in result.chunks if chunk.metadata.chunk_type == "numbered_section"
    ]
    assert len(numbered_section_chunks) == 1
    assert "petrol ve petrol ürünleri" in numbered_section_chunks[0].text
