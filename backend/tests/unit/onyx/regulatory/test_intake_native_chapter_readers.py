"""Native structure proves reader scope without rewriting canonical source rows."""

import hashlib
from uuid import UUID

import pytest

from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.document_intake.structural_readers import (
    NativeChapterReaderProof,
    build_native_chapter_reader_proofs,
    validate_native_chapter_reader_proof,
)

FILE = UUID("10000000-0000-0000-0000-000000000001")
ASSET = UUID("20000000-0000-0000-0000-000000000001")
CHAPTER = "İKİNCİ BÖLÜM"
CSS = "<style>p.caption {text-align:center;font-weight:bold} p.body {text-align:justify}</style>"
HTML = (
    CSS
    + '<p class="caption">İKİNCİ BÖLÜM</p><p class="caption">Belgelerin sunulması</p><p class="body"><b>Belge süresi</b></p><p>MADDE 5- (1) Belge sunulur.</p><p>MADDE 6- (1) Belge saklanır.</p>'
)


def row(
    identifier: str, position: int, *, article: str | None = None, text: str = "Metin"
) -> AnnexCanonicalSnapshot:
    return AnnexCanonicalSnapshot(
        id=identifier,
        user_file_id=str(FILE),
        chunk_type="paragraph" if article else "free_text",
        status="active",
        projection_ordinal=0,
        supersedes_chunk_id=None,
        superseded_by_chunk_id=None,
        position=position,
        text=text,
        heading_path=["Belge", CHAPTER] + ([f"MADDE {article}"] if article else []),
        metadata={
            "chunk_variant": "atomic",
            "article_no": article,
            "unchanged_nullable_field": None,
        },
        source="user_file",
        validity_start_date=None,
        validity_end_date=None,
    )


def scope() -> list[AnnexCanonicalSnapshot]:
    return [
        row("heading", 0, text="Belgelerin sunulması"),
        row("a5", 1, article="5"),
        row("a6", 2, article="6"),
    ]


def build(
    html: str = HTML, rows: list[AnnexCanonicalSnapshot] | None = None
) -> dict[str, NativeChapterReaderProof]:
    raw = html.encode()
    return build_native_chapter_reader_proofs(
        raw_html=raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        user_file_id=FILE,
        source_asset_id=ASSET,
        rows=scope() if rows is None else rows,
    )


def test_generic_native_caption_binds_all_article_children_and_preserves_source() -> (
    None
):
    rows = scope()
    before = [r.model_dump(mode="json") for r in rows]
    proof = build(rows=rows)["heading"]
    assert proof.heading_text == "Belgelerin sunulması"
    assert proof.governed_chunk_ids == ("a5", "a6")
    assert proof.governed_article_numbers == ("5", "6")
    assert [r.model_dump(mode="json") for r in rows] == before
    assert (
        validate_native_chapter_reader_proof(
            proof,
            rows=rows,
            expected_source_sha256=proof.source_sha256,
            expected_source_asset_id=ASSET,
            user_file_id=FILE,
        )
        == proof
    )


@pytest.mark.parametrize(
    "html",
    [
        HTML.replace(
            '<p class="caption">İKİNCİ BÖLÜM</p>',
            '<p class="caption">Belge başlığı</p>',
        ),
        HTML.replace(
            '<p class="caption">Belgelerin sunulması</p>',
            '<p class="body"><b>Belgelerin sunulması</b></p>',
        ),
        HTML.replace(
            '<p class="caption">Belgelerin sunulması</p>',
            '<p class="caption"><span style="font-weight:normal">Belgelerin sunulması</span></p>',
        ),
        HTML.replace(
            '<p class="caption">Belgelerin sunulması</p>',
            '<p>Normatif ara metin</p><p class="caption">Belgelerin sunulması</p>',
        ),
        HTML.replace(
            '<p class="caption">Belgelerin sunulması</p>',
            '<div>Başka kapsam</div><p class="caption">Belgelerin sunulması</p>',
        ),
        HTML.replace("MADDE 6-", "MADDE 7-"),
        HTML.replace("MADDE 6-", "MADDE 5-"),
    ],
)
def test_unproven_caption_or_article_scope_is_not_a_reader(html: str) -> None:
    assert build(html) == {}


def test_native_next_chapter_limits_existing_canonical_heading_path_bleed() -> None:
    html = (
        HTML
        + '<p class="caption">ÜÇÜNCÜ BÖLÜM</p><p class="caption">Başka bölüm</p><p>MADDE 7- (1) Ayrı hüküm.</p>'
    )
    rows = [*scope(), row("a7", 3, article="7")]
    assert build(html, rows)["heading"].governed_chunk_ids == ("a5", "a6")


def test_duplicate_chapter_and_article_caption_selects_only_separate_chapter_row() -> (
    None
):
    html = HTML.replace("Belgelerin sunulması", "Cezalar").replace(
        "Belge süresi", "Cezalar"
    )
    rows = scope()
    rows[0] = rows[0].model_copy(update={"text": "Cezalar"})
    assert build(html, rows)["heading"].heading_text == "Cezalar"
    rows.append(rows[0].model_copy(update={"id": "ambiguous", "position": 3}))
    assert build(html, rows) == {}


@pytest.mark.parametrize("changed", ["text", "metadata", "position", "status", "file"])
def test_any_current_canonical_scope_drift_invalidates_frozen_proof(
    changed: str,
) -> None:
    proof = build()["heading"]
    rows = scope()
    update: dict[str, object] = {"text": "Kaynak değişti"}
    if changed == "metadata":
        update = {
            "metadata": {**rows[1].metadata, "unchanged_nullable_field": "changed"}
        }
    elif changed == "position":
        update = {"position": 9}
    elif changed == "status":
        update = {"status": "superseded"}
    elif changed == "file":
        update = {"user_file_id": str(ASSET)}
    rows[1] = rows[1].model_copy(update=update)
    with pytest.raises(ValueError):
        validate_native_chapter_reader_proof(
            proof,
            rows=rows,
            expected_source_sha256=proof.source_sha256,
            expected_source_asset_id=ASSET,
            user_file_id=FILE,
        )
