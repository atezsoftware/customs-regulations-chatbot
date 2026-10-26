from uuid import uuid4

import pytest

from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.chunker import RegulatoryChunker
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.text import (
    attachment_reference_label,
    normalize_attachment_navigation,
    normalize_inline_articles,
    prepare_intake_text,
    require_chunk_source_coverage,
    require_resolved_attachment_references,
)


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
