"""Current owned native-PDF proof and actual reconsideration for paired captions."""

import hashlib

from botocore.exceptions import ClientError
from google.api_core.exceptions import GoogleAPICallError
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

from onyx.db.document_intake_source_binding import require_owned_intake_source
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
)
from onyx.db.regulatory_labeling_source_only import (
    require_single_source_only_kind,
    source_only_target_digest,
    validate_source_only_abstention,
)
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.chapter_caption import (
    CHAPTER_CAPTION_KEY,
    MAX_CAPTION_PDF_BYTES,
    ChapterCaptionSourceProof,
    SourceOnlyChapterCaptionReceipt,
    build_chapter_caption_source_proof,
    chapter_caption_candidate,
    chapter_caption_sources,
)
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepairAudit,
    CaptionRepairSource,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request


def _current_inputs(
    session: Session, *, item: RegulatoryLabelingItem, run: RegulatoryLabelingRun
) -> tuple[TaxonomyDefinition, VertexBatchRequest, ChapterCaptionSourceProof]:
    if (
        item.run_id != run.id
        or run.document_set_id is None
        or str(item.user_file_id) not in run.file_ids
        or not chapter_caption_candidate(item.text_snapshot)
    ):
        raise ValueError("The paired-caption target differs from its frozen run")
    members = item.source_snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or not 2 <= len(members) <= 40
        or not all(isinstance(value, str) for value in members)
        or len(members) != len(set(members))
    ):
        raise ValueError("The paired-caption context membership is incomplete")
    rows = session.execute(
        select(RegulatoryChunk, RegulatoryLabelingItem)
        .join(
            RegulatoryLabelingItem,
            (RegulatoryLabelingItem.regulatory_chunk_id == RegulatoryChunk.id)
            & (RegulatoryLabelingItem.user_file_id == RegulatoryChunk.user_file_id),
        )
        .where(
            RegulatoryLabelingItem.run_id == run.id,
            RegulatoryChunk.user_file_id == item.user_file_id,
            RegulatoryChunk.id.in_(members),
        )
        .execution_options(populate_existing=True)
    ).all()
    sources = [
        CaptionRepairSource(
            chunk_id=row.id,
            file_id=str(row.user_file_id),
            text=frozen.text_snapshot,
            canonical_text_sha256=frozen.canonical_text_sha256 or "",
            snapshot=frozen.source_snapshot,
            current_text=row.text,
            current_snapshot={
                "position": row.position,
                "heading_path": row.heading_path,
                "chunk_type": row.chunk_type,
                "chunk_metadata": row.chunk_metadata,
                "validity_start_date": row.validity_start_date.isoformat()
                if row.validity_start_date
                else None,
                "validity_end_date": row.validity_end_date.isoformat()
                if row.validity_end_date
                else None,
            },
            active=row.status == "active" and row.superseded_by_chunk_id is None,
        )
        for row, frozen in rows
    ]
    chapter_caption_sources(chunk_id=item.regulatory_chunk_id, sources=sources)
    asset = require_owned_intake_source(
        session, user_file_id=item.user_file_id, document_set_id=run.document_set_id
    )
    if (
        asset.mime_type != "application/pdf"
        or not 0 < asset.byte_count <= MAX_CAPTION_PDF_BYTES
    ):
        raise ValueError("A paired legacy caption requires its retained bounded PDF")
    try:
        with get_default_file_store().read_file(asset.file_id) as stream:
            raw = stream.read(MAX_CAPTION_PDF_BYTES + 1)
    except (OSError, ClientError, GoogleAPICallError, NoResultFound) as error:
        raise ValueError("The retained native caption PDF is unavailable") from error
    if len(raw) != asset.byte_count or hashlib.sha256(raw).hexdigest() != asset.sha256:
        raise ValueError("The retained native caption PDF bytes changed")
    taxonomy_row = session.get_one(
        RegulatoryLabelTaxonomy, run.taxonomy_id, populate_existing=True
    )
    taxonomy = TaxonomyDefinition.model_validate(taxonomy_row.definition)
    if (
        taxonomy.version_hash != taxonomy_row.version_hash
        or run.provider_binding.get("model") != run.model
        or not item.request_payload
        or not item.context_sha256
        or item.context_snapshot is None
    ):
        raise ValueError("The paired-caption frozen taxonomy, model or request changed")
    request = VertexBatchRequest.model_validate(item.request_payload)
    expected = build_labeling_request(
        chunk_id=item.regulatory_chunk_id,
        text=item.text_snapshot,
        context=item.context_snapshot,
        taxonomy=taxonomy,
        source_hash=item.canonical_text_sha256 or "",
    )
    if request != expected or request.request_hash != item.request_hash:
        raise ValueError("The paired-caption frozen request changed")
    proof = build_chapter_caption_source_proof(
        raw_pdf=raw,
        source_sha256=asset.sha256,
        source_asset_id=asset.id,
        user_file_id=item.user_file_id,
        chunk_id=item.regulatory_chunk_id,
        sources=sources,
    )
    return taxonomy, request, proof


def require_current_chapter_caption(
    session: Session, *, item: RegulatoryLabelingItem, run: RegulatoryLabelingRun
) -> None:
    """Eligibility only; classification still requires one strict reconsideration."""
    _current_inputs(session, item=item, run=run)


def build_source_only_chapter_caption_receipt(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    provider_abstention: AbstentionRepairAudit,
) -> SourceOnlyChapterCaptionReceipt:
    taxonomy, _, proof = _current_inputs(session, item=item, run=run)
    receipt = SourceOnlyChapterCaptionReceipt(
        proof=proof,
        frozen_target_sha256=source_only_target_digest(
            item, receipt_key=CHAPTER_CAPTION_KEY
        ),
        taxonomy_sha256=taxonomy.version_hash,
        provider_binding_sha256=publication_digest(run.provider_binding),
        provider_abstention=provider_abstention,
    )
    validate_item_chapter_caption_receipt(session, item=item, run=run, receipt=receipt)
    return receipt


def validate_item_chapter_caption_receipt(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    receipt: SourceOnlyChapterCaptionReceipt | None = None,
) -> None:
    require_single_source_only_kind(item.source_snapshot)
    if receipt is None and CHAPTER_CAPTION_KEY not in item.source_snapshot:
        return
    require_single_source_only_kind(
        item.source_snapshot, prospective_key=CHAPTER_CAPTION_KEY
    )
    stored = receipt or SourceOnlyChapterCaptionReceipt.model_validate(
        item.source_snapshot[CHAPTER_CAPTION_KEY]
    )
    if (
        receipt is not None
        and CHAPTER_CAPTION_KEY in item.source_snapshot
        and item.source_snapshot[CHAPTER_CAPTION_KEY] != receipt.model_dump(mode="json")
    ):
        raise ValueError("A different paired-caption disposition is already recorded")
    if item.labels or item.assignments:
        raise ValueError("A paired caption cannot invent taxonomy assignments")
    taxonomy, request, proof = _current_inputs(session, item=item, run=run)
    if (
        stored.proof != proof
        or stored.frozen_target_sha256
        != source_only_target_digest(item, receipt_key=CHAPTER_CAPTION_KEY)
        or stored.taxonomy_sha256 != taxonomy.version_hash
        or stored.provider_binding_sha256 != publication_digest(run.provider_binding)
    ):
        raise ValueError("The paired-caption native proof or frozen authority changed")
    validate_source_only_abstention(
        session,
        item=item,
        run=run,
        receipt_key=CHAPTER_CAPTION_KEY,
        taxonomy=taxonomy,
        request=request,
        provider_abstention=stored.provider_abstention,
        prior_failed=None,
    )
