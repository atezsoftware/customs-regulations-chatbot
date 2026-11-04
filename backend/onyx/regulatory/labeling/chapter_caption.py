"""Positive native evidence for a chapter caption followed by an article caption."""

import hashlib
import math
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.document_intake.structural_readers import (
    is_ordinal_chapter_heading,
)
from onyx.regulatory.labeling.chapter_caption_pdf import caption_font_metrics
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepairAudit,
    CaptionRepairSource,
)

CHAPTER_CAPTION_KEY = "intake_source_only_chapter_caption"
_SHA = r"^[0-9a-f]{64}$"
type CaptionProfileName = Literal[
    "closing_chapter_service_caption_v1",
    "performance_chapter_caption_v1",
    "congested_infrastructure_chapter_caption_v1",
    "closing_chapter_inspection_caption_v1",
    "closing_chapter_regulatory_acts_caption_v1",
    "closing_chapter_committee_caption_v1",
]


@dataclass(frozen=True)
class _CaptionProfile:
    name: CaptionProfileName
    chapter_lines: tuple[str, ...]
    article_caption: str
    title_basis: Literal["metadata_body_fact", "native_caption"]
    section_basis: Literal["heading_marker", "body_chapter_caption"] = "heading_marker"
    following_section_origin: Literal[
        "following_article", "chapter_caption", "absent_native_section"
    ] = "following_article"
    document_title_basis: Literal["root_body_fact", "context_heading_via_root_body"] = (
        "root_body_fact"
    )

    masthead_qualifier: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join((*self.chapter_lines, self.article_caption))


_PROFILES = (
    _CaptionProfile(
        "closing_chapter_service_caption_v1",
        ("Çeşitli ve Son Hükümler",),
        "Sorumlu hizmet birimi",
        "metadata_body_fact",
    ),
    _CaptionProfile(
        "performance_chapter_caption_v1",
        (
            "Performans Değerlendirme, Performans Planı, Kapasite Hakları ve",
            "Tahsisli Kapasitenin Kullanılmaması",
        ),
        "Performans değerlendirme",
        "native_caption",
    ),
    _CaptionProfile(
        "congested_infrastructure_chapter_caption_v1",
        ("Sıkışık Altyapı, Program Dışı Talepler ve Sınıflandırılmış Altyapı",),
        "Sıkışık altyapı",
        "native_caption",
        following_section_origin="chapter_caption",
    ),
    _CaptionProfile(
        "closing_chapter_inspection_caption_v1",
        ("Çeşitli ve Son Hükümler",),
        "Denetim",
        "metadata_body_fact",
        "body_chapter_caption",
        "chapter_caption",
        "context_heading_via_root_body",
    ),
    _CaptionProfile(
        "closing_chapter_regulatory_acts_caption_v1",
        ("Çeşitli ve Son Hükümler",),
        "Alt düzenleyici işlemler",
        "metadata_body_fact",
        "body_chapter_caption",
        "absent_native_section",
    ),
    _CaptionProfile(
        "closing_chapter_committee_caption_v1",
        ("Çeşitli ve Son Hükümler",),
        "Komite işlemleri ve tecrübe değişimi",
        "native_caption",
        "body_chapter_caption",
        "chapter_caption",
        masthead_qualifier=("(2023/1230/AB)",),
    ),
)
_SNAPSHOT = (
    "position",
    "heading_path",
    "chunk_type",
    "chunk_metadata",
    "validity_start_date",
    "validity_end_date",
)
_LOCATORS = (
    "article_no",
    "paragraph_no",
    "clause_label",
    "subclause_label",
    "table_index",
    "table_row_index",
    "appendix_label",
)
MAX_CAPTION_PDF_BYTES = 16 * 1024 * 1024
_MAX_PAGE_CONTENT_BYTES = 2 * 1024 * 1024
_MAX_TOTAL_CONTENT_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_TEXT = 2_000_000


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def chapter_caption_candidate(text: str) -> bool:
    """A closed nominal routing hint; it never proves a source disposition."""
    return any(text == profile.text for profile in _PROFILES)


def _profile(text: str) -> _CaptionProfile:
    found = next((profile for profile in _PROFILES if profile.text == text), None)
    if found is None:
        raise ValueError("The nominal chapter/article caption pair is unsupported")
    return found


def chapter_caption_qualification_sha256(text: str) -> str:
    return publication_digest(asdict(_profile(text)))


def _metadata(source: CaptionRepairSource) -> Mapping[str, Any]:
    value = source.snapshot.get("chunk_metadata")
    return value if isinstance(value, Mapping) else {}


def _fact(
    source: CaptionRepairSource, field: str, quote: str, scope: str, source_id: str
) -> bool:
    audit = _metadata(source).get("intake_legal_metadata")
    fields = audit.get("fields") if isinstance(audit, Mapping) else None
    fact = fields.get(field) if isinstance(fields, Mapping) else None
    return bool(
        isinstance(audit, Mapping)
        and audit.get("input_stale") is False
        and isinstance(fact, Mapping)
        and fact.get("stale") is False
        and re.fullmatch(_SHA, str(fact.get("request_hash", "")))
        and all(
            fact.get(k) == v
            for k, v in {
                "source": "llm",
                "basis": "explicit",
                "evidence_quote": quote,
                "evidence_scope": scope,
                "source_chunk_id": source_id,
            }.items()
        )
    )


def chapter_caption_sources(
    *, chunk_id: str, sources: Sequence[CaptionRepairSource]
) -> tuple[CaptionRepairSource, CaptionRepairSource, str]:
    """Require the exact current context and the following own article's facts."""
    by_id = {source.chunk_id: source for source in sources}
    target = by_id.get(chunk_id)
    if target is None or not 2 <= len(sources) <= 40 or len(by_id) != len(sources):
        raise ValueError("The paired-caption frozen source scope is incomplete")
    members = target.snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or len(members) != len(set(members))
        or set(members) != set(by_id)
    ):
        raise ValueError("The paired-caption context members changed")
    if any(
        not s.active
        or s.file_id != target.file_id
        or s.text != s.current_text
        or context_hash(s.text) != s.canonical_text_sha256
        or any(s.snapshot.get(k) != s.current_snapshot.get(k) for k in _SNAPSHOT)
        for s in sources
    ):
        raise ValueError("The paired-caption frozen and current sources differ")
    meta = _metadata(target)
    profile = _profile(target.text)
    headings = target.snapshot.get("heading_path")
    position = target.snapshot.get("position")
    if (
        not chapter_caption_candidate(target.text)
        or target.snapshot.get("chunk_type") != "free_text"
        or meta.get("chunk_variant") != "atomic"
        or meta.get("intake_chunking_version") not in ("2", "3")
        or type(position) is not int
        or position < 0
        or any(meta.get(k) is not None for k in _LOCATORS)
        or not isinstance(headings, list)
        or len(headings) != 2
        or not all(isinstance(h, str) for h in headings)
        or not is_ordinal_chapter_heading(headings[-1])
        or meta.get("heading_path") != headings
    ):
        raise ValueError("The target is not a proved legacy paired chapter caption")
    section = (
        headings[-1]
        if profile.section_basis == "heading_marker"
        else "\n".join(profile.chapter_lines)
    )
    section_scope = "heading" if profile.section_basis == "heading_marker" else "body"
    if meta.get("section_title") != section or not _fact(
        target, "section_title", section, section_scope, chunk_id
    ):
        raise ValueError("The paired chapter caption has no honest section fact")
    title = meta.get("document_title")
    roots = [s for s in sources if s.snapshot.get("position") == 0]
    if (
        not isinstance(title, str)
        or title != headings[0]
        or len(roots) != 1
        or _normal(title) not in _normal(roots[0].text)
    ):
        raise ValueError("The paired caption has no own document masthead proof")
    if profile.document_title_basis == "root_body_fact":
        if not _fact(target, "document_title", title, "body", roots[0].chunk_id):
            raise ValueError("The own document title has no direct root BODY fact")
    else:
        audit = meta.get("intake_legal_metadata")
        fields = audit.get("fields") if isinstance(audit, Mapping) else None
        title_fact = (
            fields.get("document_title") if isinstance(fields, Mapping) else None
        )
        reference = (
            by_id.get(str(title_fact.get("source_chunk_id")))
            if isinstance(title_fact, Mapping)
            else None
        )
        reference_headings = (
            reference.snapshot.get("heading_path") if reference else None
        )
        if (
            reference is None
            or reference.chunk_id in (chunk_id, roots[0].chunk_id)
            or not isinstance(reference_headings, list)
            or not reference_headings
            or reference_headings[0] != title
            or _metadata(reference).get("heading_path") != reference_headings
            or _metadata(reference).get("document_title") != title
            or not _fact(target, "document_title", title, "heading", reference.chunk_id)
            or not _fact(reference, "document_title", title, "body", roots[0].chunk_id)
        ):
            raise ValueError(
                "The own title has no depth-one current heading to root BODY chain"
            )
    following = [s for s in sources if s.snapshot.get("position") == position + 1]
    if len(following) != 1:
        raise ValueError("The paired caption has no unique immediate own article")
    own = following[0]
    own_meta = _metadata(own)
    own_headings = own.snapshot.get("heading_path")
    number = own_meta.get("article_no")
    marker = re.match(r"^MADDE\s+([1-9]\d*)\s*[-–—:]", own.text)
    if profile.following_section_origin == "absent_native_section":
        audit = own_meta.get("intake_legal_metadata")
        fields = audit.get("fields") if isinstance(audit, Mapping) else None
        section_supported = (
            "section_title" not in own_meta
            and isinstance(fields, Mapping)
            and "section_title" not in fields
        )
    else:
        section_supported = own_meta.get("section_title") == section and _fact(
            own,
            "section_title",
            section,
            section_scope,
            own.chunk_id
            if profile.following_section_origin == "following_article"
            else chunk_id,
        )
    if (
        not isinstance(number, str)
        or marker is None
        or marker[1] != number
        or own.snapshot.get("chunk_type") not in ("article", "paragraph")
        or own_meta.get("chunk_variant") != "atomic"
        or own_meta.get("intake_chunking_version")
        != meta.get("intake_chunking_version")
        or own_meta.get("document_title") != title
        or not isinstance(own_headings, list)
        or own_headings[:2] != headings
        or own_meta.get("heading_path") != own_headings
        or not section_supported
        or own_meta.get("appendix_label") is not None
        or not _fact(own, "article_no", f"MADDE {number}", "body", own.chunk_id)
    ):
        raise ValueError("The following own article or honest metadata proof differs")
    if profile.title_basis == "metadata_body_fact":
        if own_meta.get("article_title") != profile.article_caption or not _fact(
            own, "article_title", profile.article_caption, "body", chunk_id
        ):
            raise ValueError("The own article caption has no honest existing BODY fact")
    else:
        audit = own_meta.get("intake_legal_metadata")
        fields = audit.get("fields") if isinstance(audit, Mapping) else None
        if (
            "article_title" not in own_meta
            or own_meta["article_title"] is not None
            or not isinstance(fields, Mapping)
            or "article_title" in fields
        ):
            raise ValueError(
                "The native-only caption contradicts existing article-title metadata"
            )
    return target, own, headings[-1]


class ChapterCaptionSourceProof(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    profile: CaptionProfileName
    user_file_id: UUID
    source_asset_id: UUID
    source_sha256: str = Field(pattern=_SHA)
    source_byte_count: int = Field(gt=0, le=MAX_CAPTION_PDF_BYTES)
    chunk_id: str
    following_chunk_id: str
    following_article_number: str
    chapter_marker: str
    chapter_caption: str
    article_caption: str
    article_title_evidence: Literal["metadata_body_fact", "native_caption"]
    section_title_evidence: Literal["heading_marker", "body_chapter_caption"]
    following_section_evidence_origin: Literal[
        "following_article", "chapter_caption", "absent_native_section"
    ]
    document_title_evidence: Literal["root_body_fact", "context_heading_via_root_body"]
    document_title_source_chunk_id: str
    document_root_chunk_id: str
    native_page: int = Field(ge=1, le=64)
    native_region_sha256: str = Field(pattern=_SHA)
    native_article_hierarchy_sha256: str = Field(pattern=_SHA)
    native_opening_sha256: str = Field(pattern=_SHA)
    following_body_sha256: str = Field(pattern=_SHA)
    context_sources_sha256: str = Field(pattern=_SHA)

    @model_validator(mode="after")
    def closed_profile(self) -> "ChapterCaptionSourceProof":
        profile = next(p for p in _PROFILES if p.name == self.profile)
        if (
            self.chapter_caption != "\n".join(profile.chapter_lines)
            or self.article_caption != profile.article_caption
            or self.article_title_evidence != profile.title_basis
            or self.section_title_evidence != profile.section_basis
            or self.following_section_evidence_origin
            != profile.following_section_origin
            or self.document_title_evidence != profile.document_title_basis
        ):
            raise ValueError("The native caption proof mixes closed profiles")
        return self


class SourceOnlyChapterCaptionReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    disposition: Literal["source_only_chapter_caption"] = "source_only_chapter_caption"
    classification: Literal["abstained"] = "abstained"
    proof: ChapterCaptionSourceProof
    frozen_target_sha256: str = Field(pattern=_SHA)
    taxonomy_sha256: str = Field(pattern=_SHA)
    provider_binding_sha256: str = Field(pattern=_SHA)
    provider_abstention: AbstentionRepairAudit

    @model_validator(mode="after")
    def empty_continued_abstention(self) -> "SourceOnlyChapterCaptionReceipt":
        if (
            not self.provider_abstention.outcome.abstained
            or self.provider_abstention.outcome.labels
        ):
            raise ValueError(
                "A paired caption preserves an actual empty reconsideration"
            )
        return self


@dataclass(frozen=True)
class _Span:
    text: str
    x: float
    y: float
    font: str
    size: float
    visible: bool
    order: int
    box: tuple[float, float, float, float]


def _native_page(page: Any, *, deadline: float) -> list[_Span]:
    """Support simple visible native print text; ambiguous PDF features fail closed."""
    resources = page.get("/Resources", {}).get_object()
    if (
        page.get("/Rotate", 0) != 0
        or page.get("/UserUnit", 1) != 1
        or tuple(page.cropbox) != tuple(page.mediabox)
        or resources.get("/XObject")
        or resources.get("/Properties")
    ):
        raise ValueError("The native caption page has unsupported visibility features")
    for ref in (
        resources.get("/ExtGState", {}).get_object().values()
        if resources.get("/ExtGState")
        else []
    ):
        state = ref.get_object()
        if (
            set(state) - {"/Type", "/ca", "/CA", "/BM"}
            or float(state.get("/ca", 1)) != 1
            or float(state.get("/CA", 1)) != 1
            or state.get("/BM", "/Normal") != "/Normal"
        ):
            raise ValueError("The native caption page has ambiguous transparency")
    for ref in page.get("/Annots", []):
        annot = ref.get_object()
        if annot.get("/Subtype") != "/Link" or annot.get("/AP"):
            raise ValueError("The native caption page has a visual annotation")
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    state: dict[str, Any] = {
        "render": 0,
        "color": (0.0,),
        "clip": (0.0, 0.0, width, height),
    }
    stack: list[dict[str, Any]] = []
    path: tuple[float, float, float, float] | None = None
    path_points: list[tuple[float, float]] = []
    compound_path = False
    paints: list[tuple[int, tuple[float, float, float, float]]] = []
    fonts: dict[int, tuple[dict[str, float], float, float]] = {}
    order = 0
    text_active = False
    spans: list[_Span] = []

    def operand(op: bytes, args: list[Any], cm: list[float], _tm: list[float]) -> None:
        nonlocal order, path, path_points, compound_path, text_active
        order += 1
        if order > 50_000 or time.monotonic() > deadline:
            raise ValueError("The native caption operation budget is exhausted")
        if op not in {
            b"q",
            b"Q",
            b"cm",
            b"gs",
            b"re",
            b"m",
            b"l",
            b"h",
            b"W",
            b"W*",
            b"n",
            b"f",
            b"F",
            b"f*",
            b"g",
            b"rg",
            b"k",
            b"RG",
            b"G",
            b"K",
            b"Tr",
            b"Tc",
            b"Tw",
            b"Tz",
            b"TJ",
            b"BT",
            b"ET",
            b"Td",
            b"Tf",
            b"Tj",
            b"Tm",
            b"BDC",
            b"EMC",
        }:
            raise ValueError("The native caption uses an unsupported print operator")
        # Text callbacks may flush later; visibility changes inside BT are unsupported.
        if text_active and op not in {
            b"ET",
            b"Tf",
            b"Tm",
            b"Td",
            b"Tj",
            b"BDC",
            b"EMC",
            b"Tr",
            b"Tc",
            b"Tw",
            b"Tz",
            b"TJ",
        }:
            raise ValueError(
                "The native caption changes visibility inside a text object"
            )
        if op == b"BT":
            text_active = True
        elif op == b"ET":
            if not text_active:
                raise ValueError("Unbalanced native caption text object")
            text_active = False
        if op == b"BDC":
            properties = args[1] if len(args) == 2 else None
            if (
                str(args[0]) == "/OC"
                or not isinstance(properties, Mapping)
                or set(properties) - {"/MCID", "/Type", "/Subtype", "/ActualText"}
                or (
                    "/ActualText" in properties
                    and _normal(str(properties["/ActualText"]))
                )
            ):
                raise ValueError("The native caption has unsupported marked content")
        if op == b"q":
            stack.append(dict(state))
        elif op == b"Q":
            if not stack:
                raise ValueError("Unbalanced native caption graphics state")
            state.clear()
            state.update(stack.pop())
        elif op == b"Tr":
            if float(args[0]) != 0:
                raise ValueError(
                    "The native caption uses an unsupported text render mode"
                )
            state["render"] = 0
        elif (
            op == b"TJ"
            or (op in (b"Tc", b"Tw") and float(args[0]) != 0)
            or (op == b"Tz" and float(args[0]) != 100)
        ):
            raise ValueError("The native caption uses unsupported glyph spacing")
        elif op in (b"g", b"rg", b"k"):
            values = tuple(float(v) for v in args)
            state["color"] = (
                ((0.0,) if values == (0, 0, 0, 1) else (1.0,)) if op == b"k" else values
            )
        elif op == b"re":
            compound_path = compound_path or path is not None or bool(path_points)
            path_points = []
            x, y, w, h = (float(a) for a in args)
            points = [
                (a * cm[0] + b * cm[2] + cm[4], a * cm[1] + b * cm[3] + cm[5])
                for a, b in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
            ]
            path = (
                min(p[0] for p in points),
                min(p[1] for p in points),
                max(p[0] for p in points),
                max(p[1] for p in points),
            )
        elif op in (b"m", b"l"):
            if op == b"l" and not path_points:
                raise ValueError("The native caption path has no start")
            if op == b"m":
                compound_path = compound_path or path is not None or bool(path_points)
            path = None
            x, y = (float(a) for a in args)
            point = (x * cm[0] + y * cm[2] + cm[4], x * cm[1] + y * cm[3] + cm[5])
            path_points = [point] if op == b"m" else [*path_points, point]
            if len(path_points) > 5:
                path_points = []
        elif op == b"h":
            points = path_points
            if len(points) == 4:
                points = [*points, points[0]]
            if len(points) == 5 and points[0] == points[-1]:
                xs, ys = {p[0] for p in points}, {p[1] for p in points}
                if (
                    len(xs) == len(ys) == 2
                    and set(points[:-1]) == {(x, y) for x in xs for y in ys}
                    and all(
                        a[0] == b[0] or a[1] == b[1] for a, b in zip(points, points[1:])
                    )
                ):
                    path = (min(xs), min(ys), max(xs), max(ys))
            path_points = []
        elif op in (b"c", b"v", b"y"):
            path = None
            path_points = []
        elif op in (b"W", b"W*"):
            if path is None:
                raise ValueError(
                    "Nonrectangular native caption clipping is unsupported"
                )
            c = state["clip"]
            state["clip"] = (
                (0.0, 0.0, 0.0, 0.0)
                if compound_path
                else (
                    max(c[0], path[0]),
                    max(c[1], path[1]),
                    min(c[2], path[2]),
                    min(c[3], path[3]),
                )
            )
        elif op in (b"f", b"F", b"f*", b"B", b"B*", b"S", b"s"):
            if path is None or compound_path:
                raise ValueError("Ambiguous native caption painting is unsupported")
            paints.append((order, path))
            path = None
            compound_path = False
        elif op == b"n":
            path = None
            path_points = []
            compound_path = False

    def visitor(
        text: str, cm: list[float], tm: list[float], font: Any, size: float
    ) -> None:
        text = _normal(text)
        if not text:
            return
        if len(spans) >= 10_000 or time.monotonic() > deadline:
            raise ValueError("The native caption span budget is exhausted")
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        scale = math.hypot(cm[0], cm[1])
        effective = size * scale
        if id(font) not in fonts:
            fonts[id(font)] = caption_font_metrics(font)
        widths, descent, ascent = fonts[id(font)]
        if any(char not in widths or widths[char] <= 0 for char in text):
            raise ValueError("The native caption glyph extent is unavailable")
        advance = sum(widths[char] for char in text) * size * abs(cm[0] * tm[0]) / 1000
        box = (
            x,
            y - descent * effective / 1000,
            x + advance,
            y + ascent * effective / 1000,
        )
        clip = state["clip"]
        visible = (
            state["render"] == 0
            and max(state["color"]) <= 0.25
            and all(math.isfinite(v) for v in (x, y, effective))
            and abs(cm[1]) < 1e-6
            and abs(cm[2]) < 1e-6
            and abs(tm[1]) < 1e-6
            and abs(tm[2]) < 1e-6
            and abs(abs(tm[0]) - 1) < 1e-6
            and abs(abs(tm[3]) - 1) < 1e-6
            and cm[0] * tm[0] > 0
            and cm[3] * tm[3] > 0
            and abs(cm[0] * tm[0] - cm[3] * tm[3]) < 1e-6
            and clip[0] <= box[0] <= box[2] <= clip[2]
            and clip[1] <= box[1] <= box[3] <= clip[3]
            and 6 <= effective <= 30
            and font is not None
        )
        spans.append(
            _Span(
                text,
                x,
                y,
                str(font.get("/BaseFont", "")) if font else "",
                effective,
                visible,
                order,
                box,
            )
        )

    page.extract_text(visitor_operand_before=operand, visitor_text=visitor)
    if stack or text_active:
        raise ValueError("Unbalanced native caption graphics/text state")
    # Prior backgrounds make contrast ambiguous; later paint may cover glyphs.
    if len(spans) * len(paints) > 1_000_000:
        raise ValueError("The native caption paint-intersection budget is exhausted")
    checked = []
    for span in spans:
        if time.monotonic() > deadline:
            raise ValueError("The native caption paint budget is exhausted")
        overlaps = [
            n
            for n, box in paints
            if box[0] < span.box[2]
            and box[2] > span.box[0]
            and box[1] < span.box[3]
            and box[3] > span.box[1]
        ]
        if span.visible and any(n > span.order for n in overlaps):
            raise ValueError("The native caption is covered by later painting")
        checked.append(replace(span, visible=span.visible and not overlaps))
    return checked


def _opening_masthead(
    page: Any,
    text: str,
    title: str,
    *,
    deadline: float,
    qualifier: tuple[str, ...] = (),
) -> str:
    marker = re.search(r"(?m)^\s*MADDE\s+1\s*[-–—:\u2011]+", text)
    if marker is None:
        raise ValueError("The own native document has no opening article")
    lines = [
        _normal(line) for line in text[: marker.start()].splitlines() if line.strip()
    ]
    chapter = next((i for i, line in enumerate(lines) if line == "BİRİNCİ BÖLÜM"), -1)
    if (
        chapter < 4
        or chapter > 6
        or len(lines) != chapter + 3
        or not re.fullmatch(
            r"[0-9]{1,2} [A-Za-zÇĞİÖŞÜçğıöşü]+ [0-9]{4} [A-ZÇĞİÖŞÜ]+ Resm[îi] Gazete Sayı\s*:\s*[0-9]+",
            lines[0],
        )
        or lines[1] != "YÖNETMELİK"
        or not re.fullmatch(
            r"[A-ZÇĞİÖŞÜ][^\n:;\"'‘’“”]{1,180} Bakanlığından:", lines[2]
        )
        or _normal(" ".join(lines[3 : chapter - len(qualifier)])) != title
        or tuple(lines[chapter - len(qualifier) : chapter]) != qualifier
        or any(any(char in line for char in "\"'‘’“”") for line in lines)
    ):
        raise ValueError("The native own title is not its opening regulation masthead")
    spans = _native_page(page, deadline=deadline)
    ordered = sorted(
        (s for s in spans if s.visible), key=lambda s: (-round(s.y, 1), s.x)
    )
    starts = [i for i, span in enumerate(ordered) if span.text == title]
    if len(starts) != 1:
        raise ValueError("The native own opening title is not uniquely visible")
    count = 5 + len(qualifier)
    opening = ordered[starts[0] : starts[0] + count]
    if (
        len(opening) != count
        or [s.text for s in opening[1:-1]] != [*qualifier, *lines[chapter:]]
        or not re.fullmatch(r"MADDE\s+1\s*[-–—:\u2011]+", opening[-1].text)
        or any("Bold" not in s.font for s in opening)
        or any(sum(s.text == value for s in spans) != 1 for value in qualifier)
        or any(
            abs((s.box[0] + s.box[2]) / 2 - (opening[0].box[0] + opening[0].box[2]) / 2)
            > 2
            for s in opening[1 : 1 + len(qualifier)]
        )
        or any(opening[i].y <= opening[i + 1].y for i in range(count - 1))
    ):
        raise ValueError(
            "The native own title has no complete visible opening hierarchy"
        )
    return publication_digest(
        {"prefix": lines, "visible_opening": [s.__dict__ for s in opening]}
    )


def build_chapter_caption_source_proof(
    *,
    raw_pdf: bytes,
    source_sha256: str,
    source_asset_id: UUID,
    user_file_id: UUID,
    chunk_id: str,
    sources: Sequence[CaptionRepairSource],
) -> ChapterCaptionSourceProof:
    from pypdf.errors import PyPdfError

    try:
        return _build_chapter_caption_source_proof(
            raw_pdf=raw_pdf,
            source_sha256=source_sha256,
            source_asset_id=source_asset_id,
            user_file_id=user_file_id,
            chunk_id=chunk_id,
            sources=sources,
        )
    except (
        PyPdfError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        OverflowError,
    ) as error:
        raise ValueError("The native paired-caption PDF cannot be proved") from error


def _build_chapter_caption_source_proof(
    *,
    raw_pdf: bytes,
    source_sha256: str,
    source_asset_id: UUID,
    user_file_id: UUID,
    chunk_id: str,
    sources: Sequence[CaptionRepairSource],
) -> ChapterCaptionSourceProof:
    from pypdf import PdfReader
    from pypdf.generic import DictionaryObject

    target, following, chapter = chapter_caption_sources(
        chunk_id=chunk_id, sources=sources
    )
    deadline = time.monotonic() + 8
    if (
        not raw_pdf.startswith(b"%PDF-")
        or not 0 < len(raw_pdf) <= MAX_CAPTION_PDF_BYTES
        or hashlib.sha256(raw_pdf).hexdigest() != source_sha256
        or target.file_id != str(user_file_id)
    ):
        raise ValueError("The retained paired-caption PDF bytes or ownership changed")
    reader = PdfReader(BytesIO(raw_pdf), strict=True)
    root = reader.trailer["/Root"]
    if (
        reader.is_encrypted
        or not 1 <= len(reader.pages) <= 64
        or not isinstance(root, DictionaryObject)
        or root.get("/OCProperties")
        or root.get("/AcroForm")
    ):
        raise ValueError("The retained caption PDF has ambiguous native content")
    texts: list[str] = []
    decoded = 0
    for page in reader.pages:
        contents = page.get("/Contents")
        streams = contents.get_object() if contents is not None else []
        streams = streams if isinstance(streams, list) else [streams]
        if len(streams) > 32:
            raise ValueError("The native caption page has too many content streams")
        page_bytes = sum(len(stream.get_object().get_data()) for stream in streams)
        decoded += page_bytes
        if (
            page_bytes > _MAX_PAGE_CONTENT_BYTES
            or decoded > _MAX_TOTAL_CONTENT_BYTES
            or time.monotonic() > deadline
        ):
            raise ValueError("The native caption decoded-content budget is exhausted")
        text = page.extract_text()
        texts.append(text)
        if sum(map(len, texts)) > _MAX_TOTAL_TEXT or time.monotonic() > deadline:
            raise ValueError("The native caption text budget is exhausted")
    numbers = [
        int(value)
        for value in re.findall(
            r"(?m)^\s*MADDE\s+([1-9]\d*)\s*[-–—:\u2011]+", "\n".join(texts)
        )
    ]
    title = str(_metadata(target)["document_title"])
    if not numbers or len(numbers) > 3000 or numbers != list(range(1, numbers[-1] + 1)):
        raise ValueError("The native caption has no unambiguous own article hierarchy")
    profile = _profile(target.text)
    opening_sha256 = _opening_masthead(
        reader.pages[0],
        texts[0],
        title,
        deadline=deadline,
        qualifier=profile.masthead_qualifier,
    )
    article = str(_metadata(following)["article_no"])
    profile = _profile(target.text)
    if not 1 < int(article) < numbers[-1]:
        raise ValueError(
            "The native caption has no preceding and following own article"
        )
    candidates: list[tuple[int, list[_Span]]] = []
    for number, page in enumerate(reader.pages, 1):
        if _normal(
            " ".join((chapter, *profile.chapter_lines, profile.article_caption))
        ) not in _normal(texts[number - 1]):
            continue
        spans = _native_page(page, deadline=deadline)
        ordered = sorted(
            (s for s in spans if s.visible), key=lambda s: (-round(s.y, 1), s.x)
        )
        for i, first in enumerate(ordered):
            count = len(profile.chapter_lines) + 3
            if first.text != chapter or i + count > len(ordered):
                continue
            region = ordered[i : i + count]
            if [s.text for s in region[1:-1]] != [
                *profile.chapter_lines,
                profile.article_caption,
            ]:
                continue
            marker = re.fullmatch(r"MADDE\s+([1-9]\d*)\s*[-–—:]", region[-1].text)
            if marker is None or marker[1] != article:
                continue
            if (
                any("Bold" not in s.font for s in region)
                or any(
                    not 3 <= region[j].y - region[j + 1].y <= 32
                    for j in range(len(region) - 1)
                )
                or max((s.box[0] + s.box[2]) / 2 for s in region[:-2])
                - min((s.box[0] + s.box[2]) / 2 for s in region[:-2])
                > 2
                or (region[0].box[0] + region[0].box[2]) / 2 - region[-2].x < 50
                or abs(region[-2].x - region[-1].x) > 2
                or any(
                    not s.visible and region[-1].y - 2 <= s.y <= region[0].y + 20
                    for s in spans
                )
            ):
                raise ValueError("The native paired-caption layout is not unambiguous")
            body_spans = [
                s for s in ordered[i + count - 1 :] if s.y <= region[-1].y + 0.5
            ]
            native = _normal(" ".join(s.text for s in body_spans))
            expected = _normal(following.text)
            if not native.startswith(expected):
                raise ValueError(
                    "The following canonical article differs from the native PDF"
                )
            candidates.append((number, region))
    if len(candidates) != 1:
        raise ValueError("The native paired-caption region is missing or duplicated")
    number, region = candidates[0]
    return ChapterCaptionSourceProof(
        profile=profile.name,
        user_file_id=user_file_id,
        source_asset_id=source_asset_id,
        source_sha256=source_sha256,
        source_byte_count=len(raw_pdf),
        chunk_id=chunk_id,
        following_chunk_id=following.chunk_id,
        following_article_number=article,
        chapter_marker=chapter,
        chapter_caption="\n".join(profile.chapter_lines),
        article_caption=profile.article_caption,
        article_title_evidence=profile.title_basis,
        section_title_evidence=profile.section_basis,
        following_section_evidence_origin=profile.following_section_origin,
        document_title_evidence=profile.document_title_basis,
        document_title_source_chunk_id=str(
            _metadata(target)["intake_legal_metadata"]["fields"]["document_title"][
                "source_chunk_id"
            ]
        ),
        document_root_chunk_id=next(
            s.chunk_id for s in sources if s.snapshot.get("position") == 0
        ),
        native_page=number,
        native_region_sha256=publication_digest([s.__dict__ for s in region]),
        native_article_hierarchy_sha256=publication_digest(numbers),
        native_opening_sha256=opening_sha256,
        following_body_sha256=context_hash(following.text),
        context_sources_sha256=publication_digest(
            [
                {
                    "id": s.chunk_id,
                    "text": s.text,
                    "canonical_text_sha256": s.canonical_text_sha256,
                    "snapshot": {key: s.snapshot.get(key) for key in _SNAPSHOT},
                }
                for s in sorted(sources, key=lambda s: s.chunk_id)
            ]
        ),
    )
