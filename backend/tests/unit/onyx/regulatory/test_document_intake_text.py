from uuid import uuid4

import pytest

from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.chunker import RegulatoryChunker
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.text import (
    attachment_reference_label,
    normalize_attachment_navigation,
    normalize_inline_articles,
    normalize_wrapped_annex_references,
    prepare_intake_text,
    require_chunk_source_coverage,
    require_resolved_attachment_references,
)


def test_wrapped_annex_reference_keeps_operative_sentence_and_following_article_scope() -> (
    None
):
    source = (
        "MAKİNE EMNİYETİ YÖNETMELİĞİ\nÜÇÜNCÜ BÖLÜM\nMADDE 15-\n"
        "(7) İthalatçılar, AB bütünleştirme beyanının bir kopyasını kısmen tamamlanmış makinenin piyasaya arz\n"
        "edildiği tarihten itibaren en az 10 yıl süreyle Bakanlığa sunmak üzere muhafaza eder ve Bakanlığın talebi üzerine EK-\n"
        "4 Kısım B'de belirtilen teknik dosyayı Bakanlığa sunar.\n"
        "(8) İthalatçılar, Bakanlığın gerekçeli talebi doğrultusunda bilgi ve belgeleri sunar.\n"
        "MADDE 16-\n(1) Dağıtıcılar özenle hareket eder."
    )
    ordinary = RegulatoryChunker().chunk_text(source)
    assert any(
        "4 Kısım B'de belirtilen teknik dosyayı Bakanlığa sunar."
        in chunk.metadata.heading_path
        for chunk in ordinary.chunks
    )
    normalized = prepare_intake_text(source)
    assert "".join(normalized.split()) == "".join(source.split())
    chunked = RegulatoryChunker().chunk_text(normalized)
    atomic = [
        chunk
        for chunk in chunked.chunks
        if chunk.metadata.chunk_variant != "hierarchical_aggregate"
    ]
    seventh = next(chunk for chunk in atomic if chunk.metadata.paragraph_no == "7")
    eighth = next(chunk for chunk in atomic if chunk.metadata.paragraph_no == "8")
    assert seventh.text.endswith(
        "EK-4 Kısım B'de belirtilen teknik dosyayı Bakanlığa sunar."
    )
    assert seventh.metadata.article_no == eighth.metadata.article_no == "15"
    assert all(
        "teknik dosyayı Bakanlığa sunar" not in heading
        for chunk in chunked.chunks
        for heading in chunk.metadata.heading_path
    )
    assert normalize_wrapped_annex_references(normalized) == normalized


@pytest.mark.parametrize(
    "source",
    [
        "EK-\n4 KISIM\nGenel hükümler",
        "Genel hükümler\n4 KISIM\nBaşvuru yapılır.",
        "# Açıklamada EK-\n4 Kısım B'de belirtilir.",
        "```text\nBaşvuruda EK-\n4 Kısım B'de belirtilir.\n```",
        "    Başvuruda EK-\n    4 Kısım B'de belirtilir.\n",
        "`Başvuruda EK-\n4 Kısım B'de belirtilir.`",
        "| Açıklama |\n| --- |\n| Başvuruda EK-\n4 Kısım B'de belirtilir. |",
        "Açıklama | Başvuruda EK-\n4 Kısım B'de belirtilir. |",
        "Başvuruda EK-\n4 Kısım B'de belirtilir. | diğer hücre",
    ],
)
def test_wrapped_reference_preserves_explicit_headings_code_and_table_boundaries(
    source: str,
) -> None:
    assert normalize_wrapped_annex_references(source) == source


def test_actual_tescil_prose_wrap_retains_the_same_numeric_annex_reference() -> None:
    source = (
        "c) Demiryolu aracının teknik verileri, bakım programını ve operasyonel özelliklerini içeren belgeler ile Ek-\n"
        "9'da belirtilen belge."
    )
    normalized = normalize_wrapped_annex_references(source)
    assert normalized == source.replace("Ek-\n9'da", "Ek-9'da")
    assert "".join(normalized.split()) == "".join(source.split())


@pytest.mark.parametrize("lead", ["MADDE 1 -", "GEÇİCİ MADDE 1/A –", "EK MADDE 2 —"])
def test_inline_article_body_survives_with_verbatim_nonwhitespace_source(
    lead: str,
) -> None:
    source = f"Yeni Belge\n{lead} ROOT_ONLY_RULE. Ekleri için tıklayınız"
    normalized = prepare_intake_text(source)
    assert "".join(normalized.split()) == "".join(source.split())
    assert f"{lead}\nROOT_ONLY_RULE" in normalized
    chunked = RegulatoryChunker().chunk_text(normalized)
    assert any("ROOT_ONLY_RULE" in chunk.text for chunk in chunked.chunks)


def test_coverage_detects_the_exact_inline_heading_content_loss() -> None:
    source = "Yeni Belge\nMADDE 1 - ROOT_ONLY_RULE. Ekleri için tıklayınız"
    existing = RegulatoryChunker().chunk_text(source)
    assert all("ROOT_ONLY_RULE" not in chunk.text for chunk in existing.chunks)
    with pytest.raises(ValueError, match="omit source content"):
        require_chunk_source_coverage(source)
    require_chunk_source_coverage(normalize_inline_articles(source))


def test_referenced_headings_and_table_cells_count_as_delivered_source() -> None:
    prepare_intake_text("# Regulation\n## MADDE 1 -\n(1) This rule applies.")
    prepare_intake_text("EK-1\n\tGTIP\tRate\n\t0101\t5%")


@pytest.mark.parametrize("prefix", ["#", "##", "###", "######"])
def test_actual_markdown_article_heading_keeps_its_title_attached(prefix: str) -> None:
    source = f"# Ana belge\n\n{prefix} MADDE 1 — Başvuru usulü\n\nBaşvuru elektronik ortamda verilir.\n"
    assert normalize_inline_articles(source, preserve_markdown_headings=True) == source
    assert prepare_intake_text(source, preserve_markdown_headings=True) == source
    chunks = RegulatoryChunker().chunk_text(source).chunks
    assert any("Başvuru elektronik ortamda verilir." in chunk.text for chunk in chunks)
    assert not any(chunk.text.strip() == f"{prefix} MADDE 1 —" for chunk in chunks)


def test_converted_nonmarkdown_inline_article_still_splits_body_without_wording_loss() -> (
    None
):
    source = "# Ana belge\n\n## MADDE 1 — Başvuru elektronik ortamda verilir."
    normalized = prepare_intake_text(source)
    assert "## MADDE 1 —\nBaşvuru elektronik ortamda verilir." in normalized
    assert "".join(source.split()) == "".join(normalized.split())


def test_attachment_placeholder_requires_an_outgoing_resolved_native_link() -> None:
    asset = uuid4().hex * 2
    parent = uuid4().hex * 2
    text = "MADDE 1 - Ekleri için tıklayınız."
    incoming = SourceLink(
        parent_asset_hash=parent,
        target_asset_hash=asset,
        source_field="html:0",
        label="Ekler",
        kind="url",
    )
    with pytest.raises(ValueError, match="no attachment target was resolved"):
        require_resolved_attachment_references(
            text, asset_sha256=asset, links=[incoming]
        )
    unresolved = incoming.model_copy(
        update={"parent_asset_hash": asset, "target_asset_hash": None}
    )
    with pytest.raises(ValueError, match="no attachment target was resolved"):
        require_resolved_attachment_references(
            text, asset_sha256=asset, links=[unresolved]
        )
    same_document = incoming.model_copy(update={"parent_asset_hash": asset})
    with pytest.raises(ValueError, match="no attachment target was resolved"):
        require_resolved_attachment_references(
            text, asset_sha256=asset, links=[same_document]
        )
    outgoing = incoming.model_copy(
        update={"parent_asset_hash": asset, "target_asset_hash": parent}
    )
    require_resolved_attachment_references(text, asset_sha256=asset, links=[outgoing])


def _navigation_context(
    title: str = "EK-1 — Belgeler",
) -> tuple[str, list[SourceLink], list[IntakeSourceAsset]]:
    parent, child = "a" * 64, "b" * 64
    asset = IntakeSourceAsset(
        id=uuid4(),
        package_id=uuid4(),
        sha256=child,
        file_id="source-child",
        text_file_id=None,
        text_sha256=None,
        mime_type="text/markdown",
        display_name=title,
        byte_count=1,
    )
    link = SourceLink(
        parent_asset_hash=parent,
        target_asset_hash=child,
        original_url="ekler/ek-1.md",
        requested_url="ekler/ek-1.md",
        source_field="markdown:0",
        label="Ekler için tıklayınız — EK-1",
        kind="embedded",
    )
    return parent, [link], [asset]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("[Ekler için tıklayınız — EK-1](ekler/ek-1.md)", "[EK-1](ekler/ek-1.md)"),
        ("[Ek için tıklayınız](ekler/ek-1.md)", "[EK-1](ekler/ek-1.md)"),
        ("[**Ekleri için tıklayınız**](ekler/ek-1.md)", "[EK-1](ekler/ek-1.md)"),
        (
            "[Ekler için tıklayınız][ref]\n\n[ref]: ekler/ek-1.md",
            "[EK-1][ref]\n\n[ref]: ekler/ek-1.md",
        ),
        (
            '<a href="ekler/ek-1.md"><strong>Ekler için tıklayınız</strong></a>',
            '<a href="ekler/ek-1.md">EK-1</a>',
        ),
        (
            "Beyanname zorunludur. Ekler için tıklayınız.",
            "Beyanname zorunludur. [EK-1](ekler/ek-1.md)",
        ),
        ("Ek için tıklayınız", "[EK-1](ekler/ek-1.md)"),
        ("EK-1 için tıklayınız", "[EK-1](ekler/ek-1.md)"),
        (
            "[EK-1 için tıklayınız (PDF)](ekler/ek-1.md)",
            "[EK-1](ekler/ek-1.md)",
        ),
    ],
)
def test_navigation_prompts_become_concise_verified_annex_references(
    text: str, expected: str
) -> None:
    parent, links, assets = _navigation_context()
    original_link = links[0].model_dump()
    assert (
        normalize_attachment_navigation(
            text, asset_sha256=parent, links=links, package_assets=assets
        )
        == expected
    )
    assert links[0].model_dump() == original_link


@pytest.mark.parametrize(
    "text",
    [
        '"Ekler için tıklayınız" ibaresinin kullanılması zorunludur.',
        "Ek için tıklayınız ifadesi başvuru formunun zorunlu alanıdır.",
        "Belgeler ekler için öngörülen usule uygun düzenlenir.",
        "`[Ekler için tıklayınız](ekler/ek-1.md)`",
        "```md\n[Ekler için tıklayınız](ekler/ek-1.md)\n```",
        "    [Ekler için tıklayınız](ekler/ek-1.md)\n",
        "[Ekler için tıklayınız ibaresi mevzuatta açıklanır](ekler/ek-1.md)",
        '"EK-1 için tıklayınız" ibaresinin kullanılması zorunludur.',
        "EK-1 için tıklayınız ifadesi başvuru formunun zorunlu alanıdır.",
        "Başvuru sırasında EK-1 ve EK-2 belgeleri sunulur.",
    ],
)
def test_operative_sentences_quoted_mentions_and_code_are_preserved(text: str) -> None:
    parent, links, assets = _navigation_context()
    assert (
        normalize_attachment_navigation(
            text, asset_sha256=parent, links=links, package_assets=assets
        )
        == text
    )


def test_unmatched_navigation_link_fails_instead_of_associating_a_different_annex() -> (
    None
):
    parent, links, assets = _navigation_context()
    with pytest.raises(ValueError, match="does not match"):
        normalize_attachment_navigation(
            "[Ekler için tıklayınız](unknown.md)",
            asset_sha256=parent,
            links=links,
            package_assets=assets,
        )


def test_unknown_annex_number_uses_actual_title_or_neutral_fallback() -> None:
    parent, links, assets = _navigation_context("Belge listesi.pdf")
    links = [
        links[0].model_copy(
            update={"original_url": "download", "requested_url": "download"}
        )
    ]
    assert (
        normalize_attachment_navigation(
            "[Ekler için tıklayınız](download)",
            asset_sha256=parent,
            links=links,
            package_assets=assets,
        )
        == "[Belge listesi](download)"
    )
    assert (
        normalize_attachment_navigation(
            "[Ekler için tıklayınız](download)", asset_sha256=parent, links=links
        )
        == "[Eki](download)"
    )


def test_ingestion_normalizes_before_canonical_chunking_without_losing_operative_body() -> (
    None
):
    parent, links, assets = _navigation_context()
    text = "# Ana belge\n\nMADDE 1 — Beyanname zorunludur.\n\n[Ekler için tıklayınız](ekler/ek-1.md)"
    normalized = prepare_intake_text(
        text, asset_sha256=parent, links=links, package_assets=assets
    )
    chunks = RegulatoryChunker().chunk_text(normalized).chunks
    assert "tıklayınız" not in normalized
    assert "[EK-1](ekler/ek-1.md)" in normalized
    assert any("Beyanname zorunludur." in chunk.text for chunk in chunks)


def test_display_label_is_concise_without_rewriting_source_relationships() -> None:
    _parent, links, assets = _navigation_context()
    original = links[0].model_dump()
    assert attachment_reference_label(links[0], package_assets=assets) == "EK-1"
    assert links[0].model_dump() == original
    citation = links[0].model_copy(
        update={"label": "Mevzuat kapsamındaki belge listesi"}
    )
    assert attachment_reference_label(citation, package_assets=assets) == citation.label


@pytest.mark.parametrize("representation", ["inspection", "canonical", "html"])
def test_actual_two_anchor_html_keeps_distinct_annex_labels_and_operative_text(
    representation: str,
) -> None:
    from onyx.file_processing.html_utils import parse_html_page_basic
    from onyx.regulatory.amendments.annexes.sources import inspect_source

    parent, links, assets = _navigation_context("ek-b.pdf")
    html = (
        "<main><h1>Başvuru Tebliği</h1><h2>MADDE 1 — Başvuru usulü</h2>"
        "<p>Başvuru sırasında EK-1 ve EK-2 belgeleri sunulur.</p>"
        "<p><a href='ekler/ek-b.pdf'>Ekler için tıklayınız: EK-1</a></p>"
        "<p><a href='ekler/a1.md'>EK-2 için tıklayınız</a></p>"
        "<h2>Yürürlük</h2>"
        "<p>MADDE 2 — Bu Tebliğ yayımı tarihinde yürürlüğe girer.</p></main>"
    )
    inspection = inspect_source(html.encode(), "text/html")
    children = [assets[0].sha256, "c" * 64]
    links = [
        SourceLink(
            parent_asset_hash=parent,
            target_asset_hash=child,
            original_url=discovered.target,
            requested_url=discovered.target,
            source_field=discovered.source_field,
            label=discovered.label,
            kind="url",
        )
        for discovered, child in zip(inspection.links, children, strict=True)
    ]
    assets.append(
        assets[0].model_copy(
            update={"id": uuid4(), "sha256": children[1], "display_name": "a1.md"}
        )
    )
    frozen_links = [link.model_dump() for link in links]
    frozen_assets = [asset.model_dump() for asset in assets]
    source = {
        "inspection": inspection.text,
        "canonical": parse_html_page_basic(html),
        "html": html,
    }[representation]
    normalized = normalize_attachment_navigation(
        source, asset_sha256=parent, links=links, package_assets=assets
    )
    expected = source.replace(
        "Ekler için tıklayınız: EK-1",
        "EK-1" if representation == "html" else "[EK-1](ekler/ek-b.pdf)",
    ).replace(
        "EK-2 için tıklayınız",
        "EK-2" if representation == "html" else "[EK-2](ekler/a1.md)",
    )
    assert normalized == expected
    assert "tıklayınız" not in normalized
    assert [
        attachment_reference_label(link, package_assets=assets) for link in links
    ] == [
        "EK-1",
        "EK-2",
    ]
    assert [link.model_dump() for link in links] == frozen_links
    assert [asset.model_dump() for asset in assets] == frozen_assets
    assert (
        normalize_attachment_navigation(
            "Ekler için tıklayınız",
            asset_sha256=parent,
            links=links,
            package_assets=assets,
        )
        == "[EK-1](ekler/ek-b.pdf), [EK-2](ekler/a1.md)"
    )


def test_explicit_plain_annex_number_does_not_fall_back_to_unrelated_targets() -> None:
    parent, links, assets = _navigation_context()
    with pytest.raises(ValueError, match="does not match"):
        normalize_attachment_navigation(
            "Ekler için tıklayınız: EK-2",
            asset_sha256=parent,
            links=links,
            package_assets=assets,
        )


def test_numbered_attachment_placeholder_requires_a_resolved_target() -> None:
    with pytest.raises(ValueError, match="no attachment target was resolved"):
        require_resolved_attachment_references(
            "EK-2 için tıklayınız", asset_sha256="a" * 64, links=[]
        )
