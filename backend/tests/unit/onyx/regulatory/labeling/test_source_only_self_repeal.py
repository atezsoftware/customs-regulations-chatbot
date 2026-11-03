"""The captured 6001/5 abstention has source-only authority, never invented labels."""

from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.labeling.self_repeal import (
    SelfRepealSourceProof,
    build_self_repeal_source_proof,
)

FILE = UUID("634d428a-1675-4812-a902-229e9f3e3987")
ASSET = UUID("ca017e4f-a48c-4b6b-8550-cd4cc99d75e3")
TITLE = "Karayolları Genel Müdürlüğünün Hizmetleri Hakkında Kanun [1]"
TEXT = "MADDE 5 -\n(Mülga:2/7/2018-KHK-703/76 md.)"
RAW = (Path(__file__).parent / "fixtures/6001-uyap-original.html").read_bytes()
CAPTURED_SHA = "caf0ffd46102731922956af8ef318ca62004ff809a678aba6ced1e40a55819d2"


def rows() -> list[AnnexCanonicalSnapshot]:
    return [
        AnnexCanonicalSnapshot(
            id=f"article-{number}",
            user_file_id=str(FILE),
            chunk_type="article",
            status="active",
            projection_ordinal=number,
            supersedes_chunk_id=None,
            superseded_by_chunk_id=None,
            position=number,
            text=TEXT if number == 5 else f"MADDE {number} - body",
            heading_path=[TITLE, f"MADDE {number} - {caption}"],
            metadata={
                "chunk_variant": "atomic",
                "article_no": str(number),
                "document_number": "6001",
                "document_type": "kanun",
                "document_title": TITLE,
            },
            source="official-uyap",
            validity_start_date=None,
            validity_end_date=None,
        )
        for number, caption in [
            (4, "Görev ve yetkiler"),
            (5, "Teşkilat yapısı"),
            (6, "Genel Müdür ve Genel Müdür Yardımcıları"),
        ]
    ]


def proof(
    raw: bytes = RAW, scope: list[AnnexCanonicalSnapshot] | None = None
) -> SelfRepealSourceProof:
    return build_self_repeal_source_proof(
        raw_html=raw,
        source_sha256=sha256(raw).hexdigest(),
        source_asset_id=ASSET,
        user_file_id=FILE,
        chunk_id="article-5",
        rows=rows() if scope is None else scope,
    )


def test_actual_official_source_proves_complete_own_marker_without_changing_text() -> (
    None
):
    assert len(RAW) == 116682 and sha256(RAW).hexdigest() == CAPTURED_SHA
    scope = rows()
    before = [row.model_dump() for row in scope]
    receipt = proof(scope=scope)
    assert receipt.article_number == "5" and receipt.document_number == "6001"
    assert receipt.source_paragraph == 43 and receipt.source_sha256 == CAPTURED_SHA
    assert [row.model_dump() for row in scope] == before


@pytest.mark.parametrize(
    "body",
    [
        "(Mülga:2/7/2018-KHK-703/76 md.)",
        "MADDE 5 - (1) (Mülga:2/7/2018-KHK-703/76 md.)",
        "MADDE 5 - (Mülga ikinci fıkra:2/7/2018-KHK-703/76 md.)",
        "MADDE 5 - (Mülga:2/7/2018-KHK-703/76 md.) Şu şartla uygulanır.",
        "MADDE 5 - (Mülga:2/7/2018-KHK-703/76 md.) …",
        "MADDE 5 - (Mülga:31/2/2018-KHK-703/76 md.)",
        "MADDE 5 - (Mülga:2/7/2018-KHK-703/76 md.) (Mülga:2/7/2018-1/1 md.)",
        "MADDE 5 - Bu madde yürürlükten kaldırılmıştır.",
        "MADDE 5 - (Mülga ibare:2/7/2018-KHK-703/76 md.)",
        "MADDE 5 - (Mülga:2/7/2018-KHK-703/76 md.; şartları saklıdır.)",
    ],
)
def test_partial_conditional_mixed_invalid_or_unverified_text_is_not_source_only(
    body: str,
) -> None:
    scope = rows()
    scope[1] = scope[1].model_copy(update={"text": body})
    with pytest.raises(ValueError):
        proof(scope=scope)


@pytest.mark.parametrize(
    "change",
    [
        "number",
        "title",
        "article",
        "type",
        "split",
        "superseded",
        "scope_file",
        "quoted",
        "table",
        "unclosed_quote",
        "hidden",
        "extra_body",
        "duplicate",
        "truncated",
        "caption",
        "foreign_heading",
        "raw_hash",
    ],
)
def test_source_identity_ownership_completeness_and_bytes_are_mandatory(
    change: str,
) -> None:
    scope = rows()
    raw = RAW
    if change in {"number", "title", "article"}:
        key, value = {
            "number": ("document_number", "6455"),
            "title": ("document_title", "Other Kanun"),
            "article": ("article_no", "6"),
        }[change]
        scope[1] = scope[1].model_copy(
            update={"metadata": {**scope[1].metadata, key: value}}
        )
    elif change == "type":
        scope[1] = scope[1].model_copy(update={"chunk_type": "free_text"})
    elif change == "split":
        scope.append(
            scope[1].model_copy(update={"id": "extra-fragment", "text": "body"})
        )
    elif change == "superseded":
        scope[1] = scope[1].model_copy(update={"superseded_by_chunk_id": "new"})
    elif change == "scope_file":
        scope[0] = scope[0].model_copy(update={"user_file_id": str(ASSET)})
    else:
        target = b"<b>MADDE 5 - (M"
        assert target in raw
        if change == "quoted":
            raw = raw.replace(target, b"<q>" + target, 1).replace(
                b"<b>MADDE 6 -</b>", b"</q><b>MADDE 6 -</b>", 1
            )
        elif change == "table":
            raw = raw.replace(target, b"<table><tr><td>" + target, 1)
        elif change == "unclosed_quote":
            raw = raw.replace(target, "“".encode() + target, 1)
        elif change == "hidden":
            raw = raw.replace(target, b'<b style="display:none">MADDE 5 - (M', 1)
        elif change == "extra_body":
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(raw, "html.parser")
            node = soup.new_tag("p")
            node.string = "Operative residue."
            list(soup.find_all("p"))[43].insert_after(node)
            raw = str(soup).encode()
        elif change == "duplicate":
            raw += b"<p><b>MADDE 5 - (M\xc3\xbclga:2/7/2018-KHK-703/76 md.)</b></p>"
        elif change == "truncated":
            raw = raw[: raw.index(b"<b>MADDE 6 -</b>")]
        elif change == "caption":
            scope[1] = scope[1].model_copy(
                update={"heading_path": [TITLE, "MADDE 5 - Incorrect"]}
            )
        elif change == "foreign_heading":
            raw = raw.replace(b"<b>MADDE 4", b"<b>MADDE 231", 1)
        elif change == "raw_hash":
            with pytest.raises(ValueError, match="bytes"):
                build_self_repeal_source_proof(
                    raw_html=RAW + b" ",
                    source_sha256=CAPTURED_SHA,
                    source_asset_id=ASSET,
                    user_file_id=FILE,
                    chunk_id="article-5",
                    rows=scope,
                )
            return
    with pytest.raises(ValueError):
        proof(raw=raw, scope=scope)


@pytest.mark.parametrize(
    "change",
    [
        "hidden_caption",
        "hidden_boundary",
        "hidden_masthead",
        "opacity",
        "zero_font",
        "decimal_zero_font",
        "transparent",
        "reversed_quote",
        "div_body",
        "bare_body",
        "image",
        "head_container",
    ],
)
def test_native_display_and_full_dom_interval_are_required(change: str) -> None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(RAW, "html.parser")
    paragraphs = list(soup.find_all("p"))
    if change in {"hidden_caption", "hidden_boundary", "hidden_masthead"}:
        index = {"hidden_caption": 42, "hidden_boundary": 46, "hidden_masthead": 3}[
            change
        ]
        paragraphs[index]["style"] = "display:none"
    elif change in {"opacity", "zero_font", "decimal_zero_font", "transparent"}:
        paragraphs[43]["style"] = {
            "opacity": "opacity:0",
            "zero_font": "font-size:0px",
            "decimal_zero_font": "font-size:0.0px",
            "transparent": "color:transparent",
        }[change]
    elif change == "reversed_quote":
        paragraphs[42].insert_before("”")
        paragraphs[42].insert_after("“")
    elif change in {"div_body", "image"}:
        node = soup.new_tag("div" if change == "div_body" else "img")
        if change == "div_body":
            node.string = "Bu madde kapsamında yeni yetki kullanılır."
        paragraphs[43].insert_after(node)
    elif change == "bare_body":
        paragraphs[43].insert_after("Bu madde kapsamında yeni yetki kullanılır.")
    elif change == "head_container":
        assert soup.body is not None and soup.head is not None
        for node in list(soup.body.contents):
            soup.head.append(node.extract())
    with pytest.raises(ValueError):
        proof(raw=str(soup).encode())
