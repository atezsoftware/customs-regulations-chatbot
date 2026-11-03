"""Native paired captions require positive source and honest metadata witnesses."""

import json
import time
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.labeling.chapter_caption import (
    CHAPTER_CAPTION_KEY,
    SourceOnlyChapterCaptionReceipt,
    build_chapter_caption_source_proof,
    chapter_caption_candidate,
)
from onyx.regulatory.labeling.intake_abstention_repair import CaptionRepairSource
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import (
    REJECTED,
    audit,
    build,
    response,
)

FILE = UUID("90000000-0000-4000-8000-000000000001")
ASSET = UUID("90000000-0000-4000-8000-000000000002")
RAW = (Path(__file__).parent / "fixtures/chapter-caption-source.pdf").read_bytes()
WRAPPED_RAW = (
    Path(__file__).parent / "fixtures/chapter-caption-wrapped-source.pdf"
).read_bytes()
INFRASTRUCTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/chapter-caption-infrastructure-bodies.json"
    ).read_text()
)
REGULATORY_ACTS_RAW = (
    Path(__file__).parent / "fixtures/chapter-caption-regulatory-acts-source.pdf"
).read_bytes()
REGULATORY_ACTS = json.loads(
    (
        Path(__file__).parent / "fixtures/chapter-caption-regulatory-acts-bodies.json"
    ).read_text()
)


COMMITTEE_RAW = (
    Path(__file__).parent / "fixtures/chapter-caption-committee-source.pdf"
).read_bytes()
COMMITTEE = json.loads(
    (
        Path(__file__).parent / "fixtures/chapter-caption-committee-bodies.json"
    ).read_text()
)


def profile_raw(profile: int) -> bytes:
    if profile == 6:
        return COMMITTEE_RAW
    return REGULATORY_ACTS_RAW if profile == 5 else WRAPPED_RAW if profile > 1 else RAW


TITLE = "DEMİRYOLU ARAÇLARI TESCİL VE SİCİL YÖNETMELİĞİ"
CHAPTER = "ALTINCI BÖLÜM"
CAPTION = "Çeşitli ve Son Hükümler\nSorumlu hizmet birimi"
BODY = (
    "MADDE 20 –\n(1) Bu Yönetmelik kapsamında Bakanlıkça yerine getirilmesi gereken iş ve işlemler Demiryolu\n"
    "Düzenleme Genel Müdürlüğü tarafından yerine getirilir.\nAvrupa Birliği mevzuatına uyum"
)


def fact(quote: str, scope: str, source: str) -> dict:
    return {
        "source": "llm",
        "basis": "explicit",
        "stale": False,
        "request_hash": "a" * 64,
        "evidence_quote": quote,
        "evidence_scope": scope,
        "source_chunk_id": source,
    }


def sources(*, profile: int = 1) -> list[CaptionRepairSource]:
    config = (
        COMMITTEE
        if profile == 6
        else REGULATORY_ACTS
        if profile == 5
        else INFRASTRUCTURE[profile - 2]
        if profile > 1
        else dict(
            title=TITLE,
            chapter=CHAPTER,
            caption=CAPTION,
            body=BODY,
            position=114,
            article="20",
        )
    )
    title, chapter, caption, body, article = (
        str(config[k]) for k in ("title", "chapter", "caption", "body", "article")
    )
    position = int(str(config["position"]))
    section = "Çeşitli ve Son Hükümler" if profile in (4, 5, 6) else chapter
    section_scope = "body" if profile in (4, 5, 6) else "heading"
    rows = []
    entries = [
        ("root", 0, title, "free_text"),
        ("caption", position, caption, "free_text"),
        ("body", position + 1, body, "paragraph"),
    ]
    if profile == 4:
        entries.append(
            (
                "title_source",
                int(config["title_source_position"]),
                str(config["title_source_text"]),
                "paragraph",
            )
        )
    for identifier, position, text, kind in entries:
        headings = [title, chapter] + (
            [f"MADDE {article}"] if identifier == "body" else []
        )
        fields = {
            "section_title": fact(
                section,
                section_scope,
                "caption" if profile in (3, 4, 5, 6) else identifier,
            ),
            "document_title": fact(title, "body", "root"),
        }
        metadata = {
            "chunk_variant": "atomic",
            "intake_chunking_version": "2",
            "heading_path": headings,
            "section_title": section,
            "document_title": title,
            "intake_legal_metadata": {"input_stale": False, "fields": fields},
        }
        if identifier == "body":
            metadata.update(
                article_no=article,
                paragraph_no="1",
                article_title=None
                if profile in (2, 3, 6)
                else "Denetim"
                if profile == 4
                else "Sorumlu hizmet birimi",
            )
            fields.update(
                article_no=fact(f"MADDE {article}", "body", "body"),
            )
            if profile == 5:
                metadata["article_title"] = "Alt düzenleyici işlemler"
                metadata.pop("section_title")
                fields.pop("section_title")
            if profile in (1, 4, 5):
                fields["article_title"] = fact(
                    str(metadata["article_title"]), "body", "caption"
                )
        if profile == 4 and identifier == "caption":
            fields["document_title"] = fact(title, "heading", "title_source")
        snapshot = {
            "position": position,
            "chunk_type": kind,
            "heading_path": headings,
            "chunk_metadata": metadata,
            "validity_start_date": "2015-07-16",
            "validity_end_date": None,
            "context_member_ids": [row[0] for row in entries],
        }
        rows.append(
            CaptionRepairSource(
                identifier,
                str(FILE),
                text,
                context_hash(text),
                snapshot,
                text,
                deepcopy(snapshot),
                True,
            )
        )
    return rows


def proof(raw: bytes = RAW, rows: list[CaptionRepairSource] | None = None):
    return build_chapter_caption_source_proof(
        raw_pdf=raw,
        source_sha256=sha256(raw).hexdigest(),
        source_asset_id=ASSET,
        user_file_id=FILE,
        chunk_id="caption",
        sources=sources() if rows is None else rows,
    )


def rewrite_page(
    prefix: bytes = b"",
    suffix: bytes = b"",
    *,
    duplicate: bool = False,
    opacity: bool = False,
    root_feature: str | None = None,
    raw: bytes = RAW,
    page_index: int = 4,
    text_scale: float = 1,
) -> bytes:
    reader = PdfReader(BytesIO(raw))
    writer = PdfWriter()
    for index, page in enumerate(reader.pages):
        writer.add_page(page)
        if index == page_index:
            current = writer.pages[-1]
            stream = DecodedStreamObject()
            content = current.get_contents()
            assert content is not None
            if text_scale != 1:
                for args, operator in content.operations:
                    if operator == b"Tm":
                        args[0] = FloatObject(float(args[0]) * text_scale)
                        args[3] = FloatObject(float(args[3]) * text_scale)
            stream.set_data(prefix + content.get_data() + suffix)
            current[NameObject("/Contents")] = writer._add_object(stream)
            if opacity:
                resources = current["/Resources"]
                assert isinstance(resources, DictionaryObject)
                states = resources["/ExtGState"]
                assert isinstance(states, DictionaryObject)
                gs = states["/G3"].get_object()
                assert isinstance(gs, DictionaryObject)
                gs[NameObject("/ca")] = FloatObject(0.0)
    if duplicate:
        writer.add_page(reader.pages[page_index])
    if root_feature:
        writer._root_object[NameObject(root_feature)] = NumberObject(1)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def test_actual_native_pdf_pair_has_own_current_article_and_honest_heading_proof() -> (
    None
):
    result = proof()
    assert result.native_page == 5 and result.following_article_number == "20"
    assert result.source_byte_count == 227225
    assert (
        result.source_sha256
        == "2f4effff7a476c2614bdf8eded7b0dca5076ec928c868016a56787a22d972cf5"
    )
    assert result.chapter_marker == CHAPTER
    assert result.following_body_sha256 == context_hash(BODY)
    assert (
        sources()[1].snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"][
            "section_title"
        ]["evidence_scope"]
        == "heading"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Çeşitli ve Son Hükümler\nVergi uygulanmaz",
        "Çeşitli ve Son Hükümler\nÜcret ödemeden hizmet alınamaz",
        "Çeşitli ve Son Hükümler\nİmalatçı yedek parçaları hazır bulundurur",
        "Çeşitli ve Son Hükümler\nBaşvuru sahibi belgeleri ibraz eder",
        "Çeşitli ve Son Hükümler\nİthalatçı belgeleri saklar",
        "Çeşitli ve Son Hükümler\nBilinmeyen başlık",
        CAPTION + "\nEk başlık",
    ],
)
def test_nominal_positive_profile_never_buys_operative_or_unknown_free_text(
    text: str,
) -> None:
    assert not chapter_caption_candidate(text)
    rows = sources()
    rows[1] = replace(
        rows[1], text=text, current_text=text, canonical_text_sha256=context_hash(text)
    )
    with pytest.raises(ValueError):
        proof(rows=rows)


@pytest.mark.parametrize(
    "change",
    [
        "current_text",
        "current_metadata",
        "stale",
        "inactive",
        "foreign_file",
        "missing_following",
        "wrong_article",
        "wrong_title_source",
        "fake_body_chapter",
        "wrong_heading",
        "different_position",
        "duplicate_member",
        "wrong_root",
        "extra_locator",
        "changed_version",
    ],
)
def test_exact_frozen_current_context_and_metadata_provenance_cannot_be_substituted(
    change: str,
) -> None:
    rows = sources()
    if change == "current_text":
        rows[1] = replace(rows[1], current_text=CAPTION + " ")
    elif change == "current_metadata":
        rows[0].current_snapshot["chunk_metadata"]["extra"] = True
    elif change == "stale":
        rows[1].snapshot["chunk_metadata"]["intake_legal_metadata"]["input_stale"] = (
            True
        )
    elif change == "inactive":
        rows[1] = replace(rows[1], active=False)
    elif change == "foreign_file":
        rows[2] = replace(rows[2], file_id=str(ASSET))
    elif change == "missing_following":
        rows.pop()
    elif change == "wrong_article":
        rows[2].snapshot["chunk_metadata"]["article_no"] = "21"
    elif change == "wrong_title_source":
        rows[2].snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"][
            "article_title"
        ]["source_chunk_id"] = "body"
    elif change == "fake_body_chapter":
        rows[1].snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"][
            "section_title"
        ]["evidence_scope"] = "body"
    elif change == "wrong_heading":
        rows[2].snapshot["heading_path"] = [TITLE, "BEŞİNCİ BÖLÜM"]
    elif change == "different_position":
        rows[2].snapshot["position"] = 116
    elif change == "duplicate_member":
        rows[1].snapshot["context_member_ids"].append("body")
    elif change == "wrong_root":
        rows[0] = replace(
            rows[0],
            text="Başka belge",
            current_text="Başka belge",
            canonical_text_sha256=context_hash("Başka belge"),
        )
    elif change == "extra_locator":
        rows[1].snapshot["chunk_metadata"]["article_no"] = "20"
    elif change == "changed_version":
        rows[1].snapshot["chunk_metadata"]["intake_chunking_version"] = "4"
    with pytest.raises(ValueError):
        proof(rows=rows)


@pytest.mark.parametrize(
    "kind",
    [
        "hidden",
        "opacity",
        "duplicate",
        "overlay",
        "offclip",
        "optional",
        "form",
        "middle_overlay",
        "chapter_overlay",
        "marker_overlay",
        "article_overlay",
        "right_clip",
    ],
)
def test_native_visibility_ambiguity_is_not_authority(kind: str) -> None:
    overlays = {
        "overlay": b"0 100 595 150",
        "middle_overlay": b"310 180 10 20",
        "chapter_overlay": b"300 195 10 10",
        "marker_overlay": b"150 155 10 10",
        "article_overlay": b"170 168 10 10",
    }
    prefix, suffix = b"", b""
    if kind == "hidden":
        prefix = b"3 Tr\n"
    elif kind in ("offclip", "right_clip"):
        prefix = (
            b"q " + (b"0 0 1 1" if kind == "offclip" else b"0 0 300 842") + b" re W n\n"
        )
        suffix = b"Q\n"
    elif kind in overlays:
        prefix = b"q\n"
        suffix = b"Q\n1 1 1 rg " + overlays[kind] + b" re f\n"
    raw = rewrite_page(
        prefix=prefix,
        suffix=suffix,
        opacity=kind == "opacity",
        duplicate=kind == "duplicate",
        root_feature="/OCProperties"
        if kind == "optional"
        else "/AcroForm"
        if kind == "form"
        else None,
    )
    with pytest.raises(ValueError):
        proof(raw)


def test_native_proof_deadline_and_decoded_content_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.labeling import chapter_caption

    with monkeypatch.context() as current:
        times = iter([0.0, 9.0])
        current.setattr(chapter_caption.time, "monotonic", lambda: next(times))
        with pytest.raises(ValueError, match="budget"):
            proof()
    with monkeypatch.context() as current:
        current.setattr(chapter_caption, "_MAX_PAGE_CONTENT_BYTES", 1)
        with pytest.raises(ValueError, match="budget"):
            proof()


def test_native_retained_hash_and_following_body_are_bound() -> None:
    with pytest.raises(ValueError):
        build_chapter_caption_source_proof(
            raw_pdf=RAW,
            source_sha256="0" * 64,
            source_asset_id=ASSET,
            user_file_id=FILE,
            chunk_id="caption",
            sources=sources(),
        )
    rows = sources()
    new = BODY.replace("Bakanlıkça", "Başkanlıkça")
    rows[2] = replace(
        rows[2], text=new, current_text=new, canonical_text_sha256=context_hash(new)
    )
    with pytest.raises(ValueError):
        proof(rows=rows)


def test_receipt_requires_the_actual_empty_reconsideration_not_a_plain_failure() -> (
    None
):
    kwargs = dict(
        proof=proof(),
        frozen_target_sha256="a" * 64,
        taxonomy_sha256="b" * 64,
        provider_binding_sha256="c" * 64,
    )
    with pytest.raises(ValueError):
        SourceOnlyChapterCaptionReceipt.model_validate(kwargs)
    with pytest.raises(ValueError):
        SourceOnlyChapterCaptionReceipt.model_validate(
            {**kwargs, "provider_abstention": audit(build(), response())}
        )
    receipt = SourceOnlyChapterCaptionReceipt.model_validate(
        {**kwargs, "provider_abstention": audit(build(), REJECTED)}
    )
    assert (
        receipt.classification == "abstained"
        and receipt.provider_abstention.outcome.labels == []
    )


def test_a_stored_classification_receipt_does_not_change_the_source_only_source_proof() -> (
    None
):
    before = proof()
    rows = sources()
    rows[1].snapshot[CHAPTER_CAPTION_KEY] = {"some": "classification receipt"}
    assert proof(rows=rows) == before


@pytest.mark.parametrize(
    "profile,page,article,basis",
    [
        (2, 6, "13", "native_caption"),
        (3, 9, "23", "native_caption"),
        (4, 11, "33", "metadata_body_fact"),
    ],
)
def test_additional_actual_native_closed_profiles_preserve_honest_metadata(
    profile: int, page: int, article: str, basis: str
) -> None:
    rows = sources(profile=profile)
    result = proof(WRAPPED_RAW, rows)
    assert result.native_page == page and result.following_article_number == article
    assert result.article_title_evidence == basis
    assert result.section_title_evidence == (
        "body_chapter_caption" if profile == 4 else "heading_marker"
    )
    assert result.source_byte_count == 276133
    assert (
        result.source_sha256
        == "05eb63ba85d564044be213e05d30c71433529606533da735431b7b4078d3794c"
    )
    if profile in (2, 3):
        metadata = rows[2].snapshot["chunk_metadata"]
        assert metadata["article_title"] is None
        assert "article_title" not in metadata["intake_legal_metadata"]["fields"]


@pytest.mark.parametrize("profile", [2, 3, 4])
@pytest.mark.parametrize(
    "change",
    [
        "title_value",
        "title_fact",
        "section_basis",
        "section_source",
        "section_value",
        "stale",
        "manual",
        "article",
        "operative",
        "mixed",
    ],
)
def test_finite_profiles_reject_contradictory_facts_and_unproved_substitutions(
    profile: int, change: str
) -> None:
    rows = sources(profile=profile)
    target = rows[1].snapshot["chunk_metadata"]
    own = rows[2].snapshot["chunk_metadata"]
    fields = own["intake_legal_metadata"]["fields"]
    if change == "title_value":
        own["article_title"] = "Bilinmeyen başlık"
    elif change == "title_fact":
        fields["article_title"] = fact(str(own["article_title"]), "heading", "body")
    elif change == "section_basis":
        target["intake_legal_metadata"]["fields"]["section_title"]["evidence_scope"] = (
            "heading" if profile == 4 else "body"
        )
    elif change == "section_source":
        fields["section_title"]["source_chunk_id"] = (
            "body" if profile in (3, 4) else "caption"
        )
    elif change == "section_value":
        target["section_title"] = (
            "ON İKİNCİ BÖLÜM" if profile == 4 else "Çeşitli ve Son Hükümler"
        )
    elif change in ("stale", "manual"):
        section = fields["section_title"]
        section["stale" if change == "stale" else "source"] = (
            True if change == "stale" else "manual"
        )
    elif change == "article":
        own["article_no"] = "14"
    elif change in ("operative", "mixed"):
        text = (
            "Çeşitli ve Son Hükümler\nVergi uygulanmaz"
            if change == "operative"
            else "Çeşitli ve Son Hükümler\nPerformans değerlendirme"
        )
        rows[1] = replace(
            rows[1],
            text=text,
            current_text=text,
            canonical_text_sha256=context_hash(text),
        )
    for row in rows:
        row.current_snapshot.clear()
        row.current_snapshot.update(deepcopy(row.snapshot))
    with pytest.raises(ValueError):
        proof(WRAPPED_RAW, rows)


@pytest.mark.parametrize("profile,page", [(2, 5), (3, 8), (4, 10), (5, 6), (6, 15)])
@pytest.mark.parametrize(
    "kind", ["hidden", "opacity", "duplicate", "tail_overlay", "right_clip", "mirror"]
)
def test_each_additional_profile_requires_its_full_visible_native_sequence(
    profile: int, page: int, kind: str
) -> None:
    from onyx.regulatory.labeling.chapter_caption import _native_page

    original = profile_raw(profile)
    native_page = PdfReader(BytesIO(original)).pages[page]
    caption = sources(profile=profile)[1].text.splitlines()[0]
    span = next(
        s
        for s in _native_page(native_page, deadline=time.monotonic() + 8)
        if s.text == caption
    )
    prefix, suffix = b"", b""
    if kind == "hidden":
        prefix = b"3 Tr\n"
    elif kind == "tail_overlay":
        x, y = span.box[2] - 8, span.y
        prefix = b"q\n"
        suffix = f"Q\n1 1 1 rg {x} {y} 10 10 re f\n".encode()
    elif kind == "right_clip":
        prefix = f"q 0 0 {span.box[2] - 5} 842 re W n\n".encode()
        suffix = b"Q\n"
    elif kind == "mirror":
        prefix = b"q -1 0 0 1 595 0 cm\n"
        suffix = b"Q\n"
    raw = rewrite_page(
        prefix,
        suffix,
        raw=original,
        page_index=page,
        opacity=kind == "opacity",
        duplicate=kind == "duplicate",
    )
    with pytest.raises(ValueError):
        proof(raw, sources(profile=profile))


@pytest.mark.parametrize(
    "raw,title", [(RAW, TITLE), (WRAPPED_RAW, INFRASTRUCTURE[0]["title"])]
)
@pytest.mark.parametrize("change", ["quoted", "preface", "wrong_type"])
def test_own_native_opening_cannot_be_a_quoted_or_foreign_title(
    raw: bytes, title: str, change: str
) -> None:
    from onyx.regulatory.labeling.chapter_caption import _opening_masthead

    page = PdfReader(BytesIO(raw)).pages[0]
    text = page.extract_text()
    if change == "quoted":
        text = text.replace(title, "“" + title + "”", 1)
    elif change == "preface":
        text = "Başka belgenin açıklaması\n" + text
    else:
        text = text.replace("YÖNETMELİK", "TEBLİĞ", 1)
    with pytest.raises(ValueError):
        _opening_masthead(page, text, title, deadline=time.monotonic() + 8)


@pytest.mark.parametrize("profile", [1, 2, 3, 4, 5, 6])
def test_typed_native_proof_cannot_mix_section_or_article_witness_profiles(
    profile: int,
) -> None:
    from onyx.regulatory.labeling.chapter_caption import ChapterCaptionSourceProof

    original = proof(profile_raw(profile), sources(profile=profile)).model_dump(
        mode="json"
    )
    original["section_title_evidence"] = (
        "heading_marker" if profile in (4, 5, 6) else "body_chapter_caption"
    )
    with pytest.raises(ValueError):
        ChapterCaptionSourceProof.model_validate(original)


@pytest.mark.parametrize(
    "change",
    [
        "stale",
        "wrong_root",
        "cycle",
        "foreign_source",
        "scope",
        "quote",
        "reference_heading",
        "body_basis_substitution",
    ],
)
def test_context_heading_title_chain_requires_exact_depth_one_current_root_evidence(
    change: str,
) -> None:
    rows = sources(profile=4)
    target = rows[1].snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"][
        "document_title"
    ]
    reference = rows[3].snapshot["chunk_metadata"]
    reference_fact = reference["intake_legal_metadata"]["fields"]["document_title"]
    if change == "stale":
        reference_fact["stale"] = True
    elif change == "wrong_root":
        reference_fact["source_chunk_id"] = "body"
    elif change == "cycle":
        reference_fact["source_chunk_id"] = "caption"
    elif change == "foreign_source":
        target["source_chunk_id"] = "not-in-frozen-context"
    elif change == "scope":
        target["evidence_scope"] = "body"
    elif change == "quote":
        target["evidence_quote"] += " quote"
    elif change == "reference_heading":
        rows[3].snapshot["heading_path"][0] = "Başka Yönetmelik"
    elif change == "body_basis_substitution":
        target.update(evidence_scope="body", source_chunk_id="root")
    for row in rows:
        row.current_snapshot.clear()
        row.current_snapshot.update(deepcopy(row.snapshot))
    with pytest.raises(ValueError):
        proof(WRAPPED_RAW, rows)


@pytest.mark.parametrize(
    "profile,page", [(1, 4), (2, 5), (3, 8), (4, 10), (5, 6), (6, 15)]
)
@pytest.mark.parametrize("kind", ["prior_black_background", "nonunit_text_scale"])
def test_required_native_glyphs_need_contrast_and_supported_text_matrix(
    profile: int, page: int, kind: str
) -> None:
    from onyx.regulatory.labeling.chapter_caption import _native_page

    original = profile_raw(profile)
    caption = sources(profile=profile)[1].text.splitlines()[0]
    span = next(
        s
        for s in _native_page(
            PdfReader(BytesIO(original)).pages[page], deadline=time.monotonic() + 8
        )
        if s.text == caption
    )
    prefix = (
        f"q 0 g {span.box[0]} {span.box[1]} {span.box[2] - span.box[0]} {span.box[3] - span.box[1]} re f Q\n".encode()
        if kind == "prior_black_background"
        else b""
    )
    raw = rewrite_page(
        prefix,
        raw=original,
        page_index=page,
        text_scale=2 if kind == "nonunit_text_scale" else 1,
    )
    with pytest.raises(ValueError):
        proof(raw, sources(profile=profile))


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        (b"q\n", b"Q\n1 1 1 rg 300 180 10 20 re b\n"),
        (b"q\n", b"Q\n1 1 1 rg 300 180 10 20 re 0 0 1 1 re f\n"),
        (b"/DeviceGray cs 1 sc\n", b""),
        (b"1 Ts\n", b""),
        (b"/OC <<>> BDC\n", b"EMC\n"),
    ],
)
def test_unsupported_print_operators_or_compound_masks_cannot_supply_visibility(
    prefix: bytes, suffix: bytes
) -> None:
    with pytest.raises(ValueError):
        proof(rewrite_page(prefix, suffix))


def test_fifth_actual_pdf_profile_preserves_body_section_and_absent_following_section() -> (
    None
):
    rows = sources(profile=5)
    result = proof(REGULATORY_ACTS_RAW, rows)
    assert result.profile == "closing_chapter_regulatory_acts_caption_v1"
    assert result.native_page == 7 and result.following_article_number == "23"
    assert result.source_byte_count == 231398
    assert (
        result.source_sha256
        == "7759df850343e469337773c39193ea9ba257a10760bdf6294d640aaceed39399"
    )
    assert result.section_title_evidence == "body_chapter_caption"
    assert result.article_title_evidence == "metadata_body_fact"
    assert result.following_section_evidence_origin == "absent_native_section"
    metadata = rows[2].snapshot["chunk_metadata"]
    assert "section_title" not in metadata
    assert "section_title" not in metadata["intake_legal_metadata"]["fields"]


@pytest.mark.parametrize(
    "change",
    [
        "explicit_null",
        "present_null_fact",
        "fabricated_fact",
        "value",
        "stale",
        "target_scope",
        "target_source",
        "title_scope",
        "title_source",
        "changed_body",
        "operative_caption",
    ],
)
def test_fifth_profile_section_is_exact_absence_not_permission_to_rewrite_facts(
    change: str,
) -> None:
    rows = sources(profile=5)
    target = rows[1].snapshot["chunk_metadata"]
    own = rows[2].snapshot["chunk_metadata"]
    fields = own["intake_legal_metadata"]["fields"]
    if change == "explicit_null":
        own["section_title"] = None
    elif change == "present_null_fact":
        fields["section_title"] = None
    elif change == "fabricated_fact":
        fields["section_title"] = fact("Çeşitli ve Son Hükümler", "body", "caption")
    elif change == "value":
        own["section_title"] = "Çeşitli ve Son Hükümler"
    elif change == "stale":
        own["intake_legal_metadata"]["input_stale"] = True
    elif change == "target_scope":
        target["intake_legal_metadata"]["fields"]["section_title"]["evidence_scope"] = (
            "heading"
        )
    elif change == "target_source":
        target["intake_legal_metadata"]["fields"]["section_title"][
            "source_chunk_id"
        ] = "body"
    elif change == "title_scope":
        fields["article_title"]["evidence_scope"] = "heading"
    elif change == "title_source":
        fields["article_title"]["source_chunk_id"] = "body"
    elif change == "changed_body":
        changed = rows[2].text.replace("yapılabilir", "yapılamaz")
        rows[2] = replace(
            rows[2],
            text=changed,
            current_text=changed,
            canonical_text_sha256=context_hash(changed),
        )
    else:
        changed = "Çeşitli ve Son Hükümler\nAlt düzenleyici işlemler yapılabilir"
        rows[1] = replace(
            rows[1],
            text=changed,
            current_text=changed,
            canonical_text_sha256=context_hash(changed),
        )
    for row in rows:
        row.current_snapshot.clear()
        row.current_snapshot.update(deepcopy(row.snapshot))
    with pytest.raises(ValueError):
        proof(REGULATORY_ACTS_RAW, rows)


def test_sixth_actual_profile_keeps_nullable_title_and_exact_opening_qualifier() -> (
    None
):
    rows = sources(profile=6)
    result = proof(COMMITTEE_RAW, rows)
    assert result.profile == "closing_chapter_committee_caption_v1"
    assert result.native_page == 16 and result.following_article_number == "47"
    assert result.source_byte_count == 363766
    assert (
        result.source_sha256
        == "96a8a1602584d2be36ea30263106e9ded4339ff86e6932fa1db231afcc23c511"
    )
    assert result.article_title_evidence == "native_caption"
    assert result.section_title_evidence == "body_chapter_caption"
    assert result.following_section_evidence_origin == "chapter_caption"
    metadata = rows[2].snapshot["chunk_metadata"]
    assert metadata["article_title"] is None
    assert "article_title" not in metadata["intake_legal_metadata"]["fields"]


def rewrite_committee_qualifier(kind: str) -> bytes:
    reader = PdfReader(BytesIO(COMMITTEE_RAW))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    page = writer.pages[0]
    content = page.get_contents()
    assert content is not None
    ops = content.operations
    # These offsets address only the byte-pinned public regression fixture.
    assert ops[198][1] == b"Tj" and ops[197][1] == b"Tm"
    if kind == "missing":
        del ops[198]
    elif kind == "changed":
        ops[198] = (
            [TextStringObject(str(ops[198][0][0]).replace("\x16", "\x15", 1))],
            b"Tj",
        )
    elif kind == "duplicate":
        ops[200:200] = deepcopy(ops[195:200])
    elif kind == "hidden_then_reset":
        ops[199:199] = [([NumberObject(0)], b"Tr")]
        ops[198:198] = [([NumberObject(3)], b"Tr")]
    elif kind == "hidden_color_then_reset":
        ops[199:199] = [([NumberObject(0), NumberObject(0), NumberObject(0)], b"rg")]
        ops[198:198] = [([NumberObject(1), NumberObject(1), NumberObject(1)], b"rg")]
    elif kind == "inline":
        ops[197][0][5] = FloatObject(float(ops[188][0][5]))
    elif kind == "other_position":
        ops[197][0][4] = FloatObject(float(ops[197][0][4]) + 80)
    elif kind == "after_chapter":
        ops[197][0][5] = FloatObject(float(ops[197][0][5]) + 60)
    page.replace_contents(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "changed",
        "duplicate",
        "hidden_then_reset",
        "hidden_color_then_reset",
        "inline",
        "other_position",
        "after_chapter",
    ],
)
def test_sixth_qualifier_requires_its_unique_visible_centered_native_opening_position(
    kind: str,
) -> None:
    with pytest.raises(ValueError):
        proof(rewrite_committee_qualifier(kind), sources(profile=6))


@pytest.mark.parametrize(
    "change",
    [
        "quoted_title",
        "foreign_preface",
        "wrong_qualifier",
        "inline_qualifier",
        "missing_qualifier",
    ],
)
def test_sixth_qualified_opening_rejects_mixed_or_substituted_document_prefixes(
    change: str,
) -> None:
    from onyx.regulatory.labeling.chapter_caption import _opening_masthead

    page = PdfReader(BytesIO(COMMITTEE_RAW)).pages[0]
    text = page.extract_text()
    title = COMMITTEE["title"]
    if change == "quoted_title":
        text = text.replace(title, "“" + title + "”", 1)
    elif change == "foreign_preface":
        text = "Başka Yönetmeliğe ilişkin açıklama\n" + text
    elif change == "wrong_qualifier":
        text = text.replace("(2023/1230/AB)", "(2016/797/AB)", 1)
    elif change == "inline_qualifier":
        text = text.replace(title + "\n(2023/1230/AB)", title + " (2023/1230/AB)", 1)
    else:
        text = text.replace("(2023/1230/AB)\n", "", 1)
    with pytest.raises(ValueError):
        _opening_masthead(
            page,
            text,
            title,
            deadline=time.monotonic() + 8,
            qualifier=("(2023/1230/AB)",),
        )


@pytest.mark.parametrize(
    "change",
    [
        "title_value",
        "title_null_fact",
        "wrong_section_scope",
        "wrong_section_source",
        "wrong_document_root",
        "changed_body",
    ],
)
def test_sixth_profile_keeps_exact_honest_nullable_title_and_BODY_provenance(
    change: str,
) -> None:
    rows = sources(profile=6)
    own = rows[2].snapshot["chunk_metadata"]
    target = rows[1].snapshot["chunk_metadata"]
    if change == "title_value":
        own["article_title"] = "Komite işlemleri ve tecrübe değişimi"
    elif change == "title_null_fact":
        own["intake_legal_metadata"]["fields"]["article_title"] = None
    elif change == "wrong_section_scope":
        target["intake_legal_metadata"]["fields"]["section_title"]["evidence_scope"] = (
            "heading"
        )
    elif change == "wrong_section_source":
        own["intake_legal_metadata"]["fields"]["section_title"]["source_chunk_id"] = (
            "body"
        )
    elif change == "wrong_document_root":
        target["intake_legal_metadata"]["fields"]["document_title"][
            "source_chunk_id"
        ] = "body"
    else:
        new = rows[2].text.replace("katılım sağlar", "katılım sağlamaz")
        rows[2] = replace(
            rows[2], text=new, current_text=new, canonical_text_sha256=context_hash(new)
        )
    for row in rows:
        row.current_snapshot.clear()
        row.current_snapshot.update(deepcopy(row.snapshot))
    with pytest.raises(ValueError):
        proof(COMMITTEE_RAW, rows)
