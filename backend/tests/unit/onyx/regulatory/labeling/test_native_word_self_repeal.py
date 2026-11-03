"""Exact5393 Word HTML keeps whole m72 proof distinct from its next section."""

import gzip
import json
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest
from bs4 import BeautifulSoup, Tag

from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.document_intake.structural_readers import _NativeStyles
from onyx.regulatory.labeling.self_repeal import (
    SelfRepealSourceProof,
    _own_header,
    build_self_repeal_source_proof,
)
from tests.unit.onyx.regulatory.labeling.test_source_only_self_repeal import (
    ASSET as ASSET_6001,
)
from tests.unit.onyx.regulatory.labeling.test_source_only_self_repeal import (
    CAPTURED_SHA as SHA_6001,
)
from tests.unit.onyx.regulatory.labeling.test_source_only_self_repeal import (
    FILE as FILE_6001,
)
from tests.unit.onyx.regulatory.labeling.test_source_only_self_repeal import (
    RAW as RAW_6001,
)
from tests.unit.onyx.regulatory.labeling.test_source_only_self_repeal import (
    rows as rows_6001,
)

FIXTURES = Path(__file__).parent / "fixtures"
RAW = (FIXTURES / "5393-uyap-original.html").read_bytes()
EXPECTED_SHA = "1ffacdf993592f18e70414af3a4ee6c9fbcc147dfbfdf8c5a7b5e9df7a1ae0cd"
FILE = UUID("6a6a7044-44fd-42d0-8a0d-99e3d04a5e3b")
ASSET = UUID("140ededf-a4e3-4a5d-a25d-8947437ad166")
TARGET = "rc_d7e6caa4d610f0fcb9d6f1bc9d559e4fe0384e41"
TRANSITION = "Çeşitli ve Son Hükümler\nÇeşitli Hükümler"


def rows() -> list[AnnexCanonicalSnapshot]:
    source = json.loads(
        gzip.decompress((FIXTURES / "5393-canonical-scope.json.gz").read_bytes())
    )
    return [AnnexCanonicalSnapshot.model_validate(row) for row in source["rows"]]


def proof(
    raw: bytes = RAW, scope: list[AnnexCanonicalSnapshot] | None = None
) -> SelfRepealSourceProof:
    return build_self_repeal_source_proof(
        raw_html=raw,
        source_sha256=sha256(raw).hexdigest(),
        source_asset_id=ASSET,
        user_file_id=FILE,
        chunk_id=TARGET,
        rows=rows() if scope is None else scope,
    )


def test_exact_official_whole_article72_with_all370_unchanged_canonical_rows() -> None:
    assert len(RAW) == 318898 and sha256(RAW).hexdigest() == EXPECTED_SHA
    scope = rows()
    before = [row.model_dump() for row in scope]
    target = next(row for row in scope if row.id == TARGET)
    assert len(scope) == 370 and target.metadata.get("document_title") is None
    assert target.text == "Madde 72-\n(Mülga: 24/7/2008-5793/47 md.)"
    result = proof(scope=scope)
    assert result.article_number == "72" and result.document_number == "5393"
    assert result.document_title == "BELEDİYE KANUNU" and result.source_paragraph == 547
    assert result.source_sha256 == EXPECTED_SHA
    assert [row.model_dump() for row in scope] == before
    soup = BeautifulSoup(RAW, "html.parser")
    styles = _NativeStyles(soup)
    paragraphs = soup.find_all("p")
    assert all(_own_header(paragraphs[index], styles) for index in (544, 547, 554))
    assert any(row.position == 271 and row.text == TRANSITION for row in scope)


@pytest.mark.parametrize("marker", ["MADDE", "Madde", "madde", "mAdDe"])
def test_native_marker_case_keeps_the_same_exact_role(marker: str) -> None:
    soup = BeautifulSoup(
        f"<html><body><p><b>{marker} 72-</b> body</p></body></html>", "html.parser"
    )
    assert soup.p is not None and _own_header(soup.p, _NativeStyles(soup))


def test_mixed_case_does_not_accept_a_nonarticle_prefix() -> None:
    soup = BeautifulSoup(
        "<html><body><p><b>MaddeDışı 72-</b></p></body></html>", "html.parser"
    )
    assert soup.p is not None and not _own_header(soup.p, _NativeStyles(soup))


@pytest.mark.parametrize(
    "selectors",
    [
        "p.3-NormalYaz",
        "p.valid, p.3-NormalYaz",
        "p.2-OrtaBaslk, li.2-OrtaBaslk, div.2-OrtaBaslk",
    ],
)
def test_invalid_selector_list_is_ignored_as_a_whole_css_rule(selectors: str) -> None:
    soup = BeautifulSoup(
        f'<html><head><style>{selectors}{{display:none}}</style></head><body><p class="valid 3-NormalYaz"><b>Madde 72-</b></p></body></html>',
        "html.parser",
    )
    assert soup.p is not None
    styles = _NativeStyles(soup)
    assert styles.computed(soup.p)["display"] != "none" and styles.visible_bold(soup.p)


def test_invalid_word_rule_never_suppresses_a_separate_valid_hiding_rule() -> None:
    soup = BeautifulSoup(
        '<html><head><style>p.3-NormalYaz{display:block}p.valid{display:none}</style></head><body><p class="valid"><b>Madde 72-</b></p></body></html>',
        "html.parser",
    )
    assert soup.p is not None
    styles = _NativeStyles(soup)
    assert styles.computed(soup.p)["display"] == "none" and not styles.visible_bold(
        soup.p
    )


@pytest.mark.parametrize(
    "selector",
    [
        "p[data-role]",
        "p:is(.foo,.bar)",
        "p:not(.bar,.baz)",
        ":where(p,.unused)",
        ":is(.foo,:unknown-pseudo)",
        ":is(p.3bad,p)",
    ],
)
def test_valid_but_unmodeled_selector_remains_fail_closed(selector: str) -> None:
    soup = BeautifulSoup(
        f'<html><head><style>{selector}{{display:none}}</style></head><body><p class="foo">body</p></body></html>',
        "html.parser",
    )
    with pytest.raises(ValueError, match="unsupported"):
        _NativeStyles(soup)


@pytest.mark.parametrize(
    "change",
    [
        "hidden_target",
        "hidden_own_caption",
        "hidden_part",
        "hidden_heading",
        "hidden_chapter",
        "hidden_next_caption",
        "hidden_next_marker",
        "quoted_target",
        "quoted_next_marker",
        "foreign_number",
        "foreign_next_article",
        "noncanonical_part",
        "missing_canonical_caption",
        "changed_canonical_caption",
        "extra_p_before",
        "extra_div_before",
        "extra_bare_before",
        "extra_heading_after",
        "duplicate_caption",
        "unmodeled_image",
        "truncated_root",
        "wrong_title_diacritic",
    ],
)
def test_own_source_and_next_section_proof_rejects_unsupported_content(
    change: str,
) -> None:
    scope = rows()
    soup = BeautifulSoup(RAW, "html.parser")
    paragraphs = soup.find_all("p")
    target_index = next(i for i, row in enumerate(scope) if row.id == TARGET)
    if change.startswith("hidden_"):
        hidden = {
            "hidden_target": 547,
            "hidden_own_caption": 546,
            "hidden_part": 549,
            "hidden_chapter": 551,
            "hidden_next_caption": 553,
            "hidden_next_marker": 554,
        }
        node = (
            soup.find(
                "h3",
                string=lambda text: bool(text and "Çeşitli ve Son Hükümler" in text),
            )
            if change == "hidden_heading"
            else paragraphs[hidden[change]]
        )
        assert isinstance(node, Tag)
        node["style"] = "display:none"
    elif change in {"quoted_target", "quoted_next_marker"}:
        paragraphs[547 if change == "quoted_target" else 554].wrap(
            soup.new_tag("blockquote")
        )
    elif change == "foreign_number":
        scope[target_index] = scope[target_index].model_copy(
            update={
                "metadata": {**scope[target_index].metadata, "document_number": "6001"}
            }
        )
    elif change == "foreign_next_article":
        scope = [
            row.model_copy(
                update={
                    "heading_path": [
                        *row.heading_path[:-1],
                        row.heading_path[-1].replace("MADDE 73", "MADDE 173"),
                    ]
                }
            )
            if str(row.metadata.get("article_no")) == "73"
            else row
            for row in scope
        ]
    elif change == "noncanonical_part":
        paragraphs[549].clear()
        paragraphs[549].append("YEDİNCİ KISIM")
    elif change in {"missing_canonical_caption", "changed_canonical_caption"}:
        if change == "missing_canonical_caption":
            scope = [row for row in scope if row.position != 271]
        else:
            scope = [
                row.model_copy(update={"text": "Unrelated canonical heading"})
                if row.position == 271
                else row
                for row in scope
            ]
    elif change in {
        "extra_p_before",
        "extra_div_before",
        "extra_bare_before",
        "extra_heading_after",
    }:
        if change == "extra_bare_before":
            paragraphs[547].insert_after("Bu madde kapsamında yeni yetki kullanılır.")
        else:
            node = soup.new_tag(
                {
                    "extra_p_before": "p",
                    "extra_div_before": "div",
                    "extra_heading_after": "h3",
                }[change]
            )
            node.string = "Bu madde kapsamında yeni yetki kullanılır."
            paragraphs[549 if change == "extra_heading_after" else 547].insert_after(
                node
            )
    elif change == "duplicate_caption":
        paragraphs[552].insert_after(paragraphs[552].__copy__())
    elif change == "unmodeled_image":
        paragraphs[549].insert_after(soup.new_tag("img"))
    elif change == "truncated_root":
        scope[target_index] = scope[target_index].model_copy(
            update={"heading_path": ["Kanunu", *scope[target_index].heading_path[1:]]}
        )
    elif change == "wrong_title_diacritic":
        scope[target_index] = scope[target_index].model_copy(
            update={
                "metadata": {
                    **scope[target_index].metadata,
                    "document_title": "Beledıye Kanunu",
                }
            }
        )
    with pytest.raises(ValueError):
        proof(raw=str(soup).encode(), scope=scope)


def test_next_paragraph_paths_keep_the_exact_own_article_caption() -> None:
    scope = [
        row.model_copy(
            update={"heading_path": [*row.heading_path, f"FIKRA {row.position}"]}
        )
        if str(row.metadata.get("article_no")) == "73"
        else row
        for row in rows()
    ]
    assert proof(scope=scope).article_number == "72"


def test_verified_mixed_case_title_matches_only_its_turkish_masthead() -> None:
    scope = rows()
    scope = [
        row.model_copy(
            update={"metadata": {**row.metadata, "document_title": "Belediye Kanunu"}}
        )
        if row.id == TARGET
        else row
        for row in scope
    ]
    assert proof(scope=scope).document_title == "Belediye Kanunu"


def test_truncated6001_root_still_cannot_replace_a_complete_own_masthead() -> None:
    scope = [
        row.model_copy(
            update={
                "heading_path": ["Hakkında Kanun [1]", *row.heading_path[1:]],
                "metadata": {
                    key: value
                    for key, value in row.metadata.items()
                    if key != "document_title"
                },
            }
        )
        for row in rows_6001()
    ]
    with pytest.raises(ValueError, match="formal title"):
        build_self_repeal_source_proof(
            raw_html=RAW_6001,
            source_sha256=SHA_6001,
            source_asset_id=ASSET_6001,
            user_file_id=FILE_6001,
            chunk_id="article-5",
            rows=scope,
        )


@pytest.mark.parametrize(
    "body",
    [
        "Geçici Madde 72- (Mülga: 24/7/2008-5793/47 md.)",
        "Madde 72- (İptal: Anayasa Mahkemesi kararı.)",
        "Madde 72- (Mülga ikinci fıkra: 24/7/2008-5793/47 md.)",
    ],
)
def test_native_source_compatibility_does_not_expand_self_repeal_semantics(
    body: str,
) -> None:
    scope = [
        row.model_copy(update={"text": body}) if row.id == TARGET else row
        for row in rows()
    ]
    with pytest.raises(ValueError):
        proof(scope=scope)


@pytest.mark.parametrize(
    "selector",
    [
        "p:is(.MsoNormal,.unused)",
        "p:not(.unused,.also-unused)",
        ":where(p,.unused)",
        ":is(p,:unknown-pseudo)",
        ":is(p.3bad,p)",
    ],
)
def test_full_source_complex_hiding_rules_never_become_visible_proof(
    selector: str,
) -> None:
    soup = BeautifulSoup(RAW, "html.parser")
    assert soup.head is not None
    hiding = soup.new_tag("style")
    hiding.string = selector + "{display:none}"
    soup.head.append(hiding)
    with pytest.raises(ValueError, match="unsupported"):
        proof(raw=str(soup).encode())
