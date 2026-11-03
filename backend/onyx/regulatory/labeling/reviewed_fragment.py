"""A closed public-source qualification preserves a genuinely abstained fragment."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.duplicate_ack_evidence import (
    DuplicateAckEvidence,
    complete_ack_response,
)
from onyx.regulatory.labeling.intake_abstention_repair import AbstentionRepairAudit
from onyx.regulatory.labeling.provider import (
    MAX_RESPONSE_BYTES,
    TaxonomyDefinition,
    validate_labeling_response,
)

REVIEWED_FRAGMENT_KEY = "intake_source_only_reviewed_fragment"
_MANIFEST = (
    Path(__file__).parent / "qualifications/tmgdk_shareholder_blank_rows_v1.json"
)
_MANIFEST_SHA256 = "7883f71a52bd6687d553f760215066497757a8de98de257d19c20465216da492"
_FIGURE_MANIFEST = _MANIFEST.with_name("dryer_label_figure_tail_v1.json")
_FIGURE_MANIFEST_SHA256 = (
    "546f6a1f3795e3bbc91e64c4e4dbf9b0eb1108ba4a7f4947f41cb5781b271d20"
)
SHA = r"^[0-9a-f]{64}$"
_SOURCE_FIELDS = (
    "position",
    "heading_path",
    "chunk_type",
    "chunk_metadata",
    "validity_start_date",
    "validity_end_date",
)
_FORBIDDEN = (
    "article_no",
    "article_title",
    "paragraph_no",
    "clause_label",
    "subclause_label",
    "table_row_index",
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def qualification() -> dict[str, Any]:
    try:
        raw = _MANIFEST.read_bytes()
    except OSError as error:
        raise ValueError(
            "The reviewed public-source qualification is unavailable"
        ) from error
    if hashlib.sha256(raw).hexdigest() != _MANIFEST_SHA256:
        raise ValueError("The reviewed public-source qualification changed")
    return json.loads(raw)


def reviewed_fragment_candidate(text: str) -> bool:
    return text == qualification()["target_rows"]


def figure_qualification() -> dict[str, Any]:
    try:
        raw = _FIGURE_MANIFEST.read_bytes()
    except OSError as error:
        raise ValueError("The reviewed figure qualification is unavailable") from error
    if hashlib.sha256(raw).hexdigest() != _FIGURE_MANIFEST_SHA256:
        raise ValueError("The reviewed figure qualification changed")
    return json.loads(raw)


def reviewed_figure_fragment_candidate(text: str) -> bool:
    return text == figure_qualification()["target_text"]


def selected_qualification(text: str) -> dict[str, Any]:
    if reviewed_fragment_candidate(text):
        return qualification()
    if reviewed_figure_fragment_candidate(text):
        return figure_qualification()
    raise ValueError("The source fragment has no closed reviewed qualification")


@dataclass(frozen=True)
class ReviewedFragmentSource:
    chunk_id: str
    file_id: str
    text: str
    canonical_text_sha256: str
    snapshot: dict[str, Any]
    current_text: str
    current_snapshot: dict[str, Any]
    active: bool


def _metadata(source: ReviewedFragmentSource) -> Mapping[str, Any]:
    raw = source.snapshot.get("chunk_metadata")
    return raw if isinstance(raw, Mapping) else {}


def _fact(
    source: ReviewedFragmentSource,
    field: str,
    quote: str,
    scope: str,
    witness: ReviewedFragmentSource,
) -> bool:
    audit = _metadata(source).get("intake_legal_metadata")
    fields = audit.get("fields") if isinstance(audit, Mapping) else None
    fact = fields.get(field) if isinstance(fields, Mapping) else None
    return bool(
        isinstance(audit, Mapping)
        and audit.get("input_stale") is False
        and isinstance(fact, Mapping)
        and re.fullmatch(SHA, str(fact.get("request_hash", "")))
        and all(
            fact.get(k) == v
            for k, v in {
                "source": "llm",
                "basis": "explicit",
                "stale": False,
                "evidence_quote": quote,
                "evidence_scope": scope,
                "source_chunk_id": witness.chunk_id,
                "reference_quote": None,
                "reference_chunk_id": None,
            }.items()
        )
        and (
            quote in witness.text
            if scope == "body"
            else quote in witness.snapshot.get("heading_path", [])
        )
    )


class ReviewedFragmentProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    profile_kind: Literal["blank_form_rows"] = "blank_form_rows"
    profile_id: Literal["tmgdk_shareholder_blank_rows_v1"] = (
        "tmgdk_shareholder_blank_rows_v1"
    )
    qualification_sha256: str = Field(pattern=SHA)
    user_file_id: UUID
    source_asset_id: UUID
    source_sha256: str = Field(pattern=SHA)
    source_byte_count: int = Field(gt=0, le=82928)
    chunk_id: str
    transcription_sha256: str = Field(pattern=SHA)
    context_sources_sha256: str = Field(pattern=SHA)
    reviewed_page_sha256: tuple[str, str]
    review_basis: Literal["content_addressed_public_source_visual_review"] = (
        "content_addressed_public_source_visual_review"
    )


def build_reviewed_fragment_proof(
    *,
    raw_pdf: bytes,
    source_asset_id: UUID,
    user_file_id: UUID,
    chunk_id: str,
    sources: Sequence[ReviewedFragmentSource],
) -> ReviewedFragmentProof:
    profile = qualification()
    if (
        len(raw_pdf) != profile["source_byte_count"]
        or hashlib.sha256(raw_pdf).hexdigest() != profile["source_sha256"]
        or not raw_pdf.startswith(b"%PDF-")
    ):
        raise ValueError("The retained source differs from the reviewed public bytes")
    by_id = {source.chunk_id: source for source in sources}
    target = by_id.get(chunk_id)
    if len(sources) != 9 or len(by_id) != 9 or target is None:
        raise ValueError("The reviewed fragment requires all nine source entries")
    members = target.snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or not all(isinstance(v, str) for v in members)
        or len(members) != len(set(members))
        or set(members) != set(by_id)
    ):
        raise ValueError("The reviewed fragment context membership changed")
    if any(
        not s.active
        or s.file_id != str(user_file_id)
        or s.text != s.current_text
        or context_hash(s.text) != s.canonical_text_sha256
        or any(s.snapshot.get(k) != s.current_snapshot.get(k) for k in _SOURCE_FIELDS)
        for s in sources
    ):
        raise ValueError("The reviewed fragment frozen/current source changed")
    if any(type(s.snapshot.get("position")) is not int for s in sources):
        raise ValueError("The reviewed source positions are invalid")
    ordered = sorted(sources, key=lambda s: s.snapshot["position"])
    if [s.snapshot["position"] for s in ordered] != list(range(9)):
        raise ValueError("The reviewed source order changed")
    transcription = [
        {
            "position": s.snapshot["position"],
            "chunk_type": s.snapshot.get("chunk_type"),
            "heading_path": s.snapshot.get("heading_path"),
            "text": s.text,
        }
        for s in ordered
    ]
    if digest(transcription) != profile["transcription_sha256"]:
        raise ValueError("The complete reviewed source transcription changed")
    index = ordered.index(target)
    if (
        not 2 <= index < 8
        or target.text != profile["target_rows"]
        or ordered[index - 2].text != profile["preceding_section"]
        or ordered[index - 1].text != profile["preceding_columns"]
        or ordered[index + 1].text != profile["following_section"]
    ):
        raise ValueError(
            "The target is not the reviewed shareholder blank-row fragment"
        )
    meta = _metadata(target)
    audit = meta.get("intake_legal_metadata")
    fields = audit.get("fields") if isinstance(audit, Mapping) else None
    structural = fields.get("table_index") if isinstance(fields, Mapping) else None
    if (
        meta.get("chunk_variant") != "atomic"
        or meta.get("intake_chunking_version") != "3"
        or meta.get("heading_path") != target.snapshot["heading_path"]
        or meta.get("table_index") != 3
        or not isinstance(fields, Mapping)
        or any(meta.get(k) is not None or k in fields for k in _FORBIDDEN)
        or not isinstance(structural, Mapping)
        or any(
            structural.get(k) != v
            for k, v in {
                "source": "structural",
                "basis": "chunker",
                "stale": False,
                "source_chunk_id": chunk_id,
                "evidence_quote": None,
            }.items()
        )
        or meta.get("annex_title") != profile["annex_title"]
        or not _fact(target, "annex_title", profile["annex_title"], "body", ordered[0])
        or meta.get("appendix_label") != profile["appendix_label"]
        or not _fact(
            target, "appendix_label", profile["appendix_label"], "heading", target
        )
    ):
        raise ValueError("The reviewed fragment metadata provenance changed")
    if profile["signature_heading"] not in ordered[-1].snapshot["heading_path"]:
        raise ValueError("The form signature heading disappeared")
    return ReviewedFragmentProof(
        qualification_sha256=_MANIFEST_SHA256,
        user_file_id=user_file_id,
        source_asset_id=source_asset_id,
        source_sha256=profile["source_sha256"],
        source_byte_count=len(raw_pdf),
        chunk_id=chunk_id,
        transcription_sha256=digest(transcription),
        context_sources_sha256=digest(
            [
                {
                    "id": s.chunk_id,
                    "text_sha256": s.canonical_text_sha256,
                    "snapshot": {k: s.snapshot.get(k) for k in _SOURCE_FIELDS},
                }
                for s in ordered
            ]
        ),
        reviewed_page_sha256=tuple(
            p["render_sha256"] for p in profile["reviewed_pages"]
        ),
    )


class ReviewedBaseAbstention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy: Literal["genuine_base_abstention"] = "genuine_base_abstention"
    origin: Literal["same_item", "prior_failed_item"]
    source_run_id: UUID
    source_item_id: UUID
    source_frozen_sha256: str = Field(pattern=SHA)
    prior_history_sha256: str | None = Field(default=None, pattern=SHA)
    ack: DuplicateAckEvidence
    response: str = Field(min_length=1, max_length=MAX_RESPONSE_BYTES)


class ReviewedFigureFragmentProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    profile_kind: Literal["figure_caption"] = "figure_caption"
    profile_id: Literal["dryer_label_figure_tail_v1"] = "dryer_label_figure_tail_v1"
    qualification_sha256: str = Field(pattern=SHA)
    user_file_id: UUID
    source_asset_id: UUID
    source_sha256: str = Field(pattern=SHA)
    source_byte_count: int = Field(gt=0, le=4526449)
    chunk_id: str
    transcription_sha256: str = Field(pattern=SHA)
    context_sources_sha256: str = Field(pattern=SHA)
    reviewed_page_sha256: tuple[str]
    review_basis: Literal["content_addressed_public_source_visual_review"] = (
        "content_addressed_public_source_visual_review"
    )


def build_reviewed_figure_proof(
    *,
    raw_pdf: bytes,
    source_asset_id: UUID,
    user_file_id: UUID,
    chunk_id: str,
    sources: Sequence[ReviewedFragmentSource],
) -> ReviewedFigureFragmentProof:
    profile = figure_qualification()
    if (
        len(raw_pdf) != profile["source_byte_count"]
        or hashlib.sha256(raw_pdf).hexdigest() != profile["source_sha256"]
        or not raw_pdf.startswith(b"%PDF-")
    ):
        raise ValueError("The retained figure source differs from its reviewed bytes")
    by_id = {s.chunk_id: s for s in sources}
    target = by_id.get(chunk_id)
    if len(sources) != 27 or len(by_id) != 27 or target is None:
        raise ValueError("The reviewed figure requires all 27 frozen source entries")
    members = target.snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or not all(isinstance(v, str) for v in members)
        or len(members) != len(set(members))
        or set(members) != set(by_id)
        or any(
            not s.active
            or s.file_id != str(user_file_id)
            or s.text != s.current_text
            or context_hash(s.text) != s.canonical_text_sha256
            or any(
                s.snapshot.get(k) != s.current_snapshot.get(k) for k in _SOURCE_FIELDS
            )
            or type(s.snapshot.get("position")) is not int
            for s in sources
        )
    ):
        raise ValueError("The reviewed figure frozen/current context changed")
    ordered = sorted(sources, key=lambda s: s.snapshot["position"])
    transcription = [
        {
            "position": s.snapshot["position"],
            "chunk_type": s.snapshot.get("chunk_type"),
            "heading_path": s.snapshot.get("heading_path"),
            "text": s.text,
        }
        for s in ordered
    ]
    if digest(transcription) != profile["transcription_sha256"]:
        raise ValueError("The complete reviewed figure transcription changed")
    index = ordered.index(target)
    if (
        not 1 <= index < len(ordered) - 1
        or target.text != profile["target_text"]
        or ordered[index - 1].text != profile["preceding_diagram"]
        or ordered[index + 1].text != profile["following_clause"]
    ):
        raise ValueError("The reviewed diagram/caption/operative boundary changed")
    witnesses = [s for s in ordered if s.text == profile["annex_title_witness_text"]]
    meta = _metadata(target)
    audit = meta.get("intake_legal_metadata")
    fields = audit.get("fields") if isinstance(audit, Mapping) else None
    structural = fields.get("document_type") if isinstance(fields, Mapping) else None
    if (
        len(witnesses) != 1
        or meta.get("chunk_variant") != "atomic"
        or meta.get("intake_chunking_version") != "2"
        or meta.get("heading_path") != target.snapshot.get("heading_path")
        or meta.get("paragraph_no") != "2"
        or meta.get("section_title") != profile["section_title"]
        or meta.get("appendix_label") != profile["appendix_label"]
        or meta.get("annex_title") != profile["annex_title"]
        or meta.get("document_type") != "unknown"
        or not isinstance(fields, Mapping)
        or any(
            meta.get(k) is not None or k in fields
            for k in (
                "article_no",
                "article_title",
                "clause_label",
                "subclause_label",
                "table_index",
                "table_row_index",
            )
        )
        or not isinstance(structural, Mapping)
        or any(
            structural.get(k) != v
            for k, v in {
                "source": "structural",
                "basis": "chunker",
                "stale": False,
                "source_chunk_id": chunk_id,
                "evidence_quote": None,
            }.items()
        )
        or not _fact(
            target, "section_title", profile["section_title"], "heading", target
        )
        or not _fact(
            target, "paragraph_no", profile["section_title"], "heading", target
        )
        or not _fact(
            target, "appendix_label", profile["appendix_label"], "heading", target
        )
        or not _fact(
            target, "annex_title", profile["annex_title"], "body", witnesses[0]
        )
    ):
        raise ValueError("The reviewed figure metadata witnesses changed")
    return ReviewedFigureFragmentProof(
        qualification_sha256=_FIGURE_MANIFEST_SHA256,
        user_file_id=user_file_id,
        source_asset_id=source_asset_id,
        source_sha256=profile["source_sha256"],
        source_byte_count=len(raw_pdf),
        chunk_id=chunk_id,
        transcription_sha256=digest(transcription),
        context_sources_sha256=digest(
            [
                {
                    "id": s.chunk_id,
                    "text_sha256": s.canonical_text_sha256,
                    "snapshot": {k: s.snapshot.get(k) for k in _SOURCE_FIELDS},
                }
                for s in ordered
            ]
        ),
        reviewed_page_sha256=(profile["reviewed_pages"][0]["render_sha256"],),
    )


class ReviewedContinuedAbstention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy: Literal["genuine_continued_semantic_abstention"] = (
        "genuine_continued_semantic_abstention"
    )
    origin: Literal["same_item", "prior_failed_item"]
    source_run_id: UUID
    source_item_id: UUID
    source_frozen_sha256: str = Field(pattern=SHA)
    prior_history_sha256: str | None = Field(default=None, pattern=SHA)
    ack: DuplicateAckEvidence
    response: str = Field(min_length=1, max_length=MAX_RESPONSE_BYTES)
    semantic_ack: DuplicateAckEvidence
    semantic_audit: AbstentionRepairAudit


class SourceOnlyReviewedFragmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    disposition: Literal["source_only_reviewed_fragment"] = (
        "source_only_reviewed_fragment"
    )
    classification: Literal["abstained"] = "abstained"
    proof: Annotated[
        ReviewedFragmentProof | ReviewedFigureFragmentProof,
        Field(discriminator="profile_kind"),
    ]
    frozen_target_sha256: str = Field(pattern=SHA)
    taxonomy_sha256: str = Field(pattern=SHA)
    provider_binding_sha256: str = Field(pattern=SHA)
    base_abstention: Annotated[
        ReviewedBaseAbstention | ReviewedContinuedAbstention,
        Field(discriminator="policy"),
    ]


def require_base_abstention(
    response: str, *, text: str, taxonomy: TaxonomyDefinition
) -> None:
    outcome = validate_labeling_response(response, text=text, taxonomy=taxonomy)
    if not outcome.abstained or outcome.labels:
        raise ValueError(
            "A reviewed source-only fragment requires a genuine empty abstention"
        )


def require_base_ack_line(
    line: str, request: VertexBatchRequest, *, text: str, taxonomy: TaxonomyDefinition
) -> str:
    row = json.loads(line)
    if (
        not isinstance(row, dict)
        or row.get("error") is not None
        or row.get("status", "") != ""
    ):
        raise ValueError("The reviewed-fragment ACK contains a provider error")
    response = complete_ack_response(line, request)
    require_base_abstention(response, text=text, taxonomy=taxonomy)
    return response
