"""Bind a dependent article subject to unchanged source and provider abstention."""

from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryLabelingItem, RegulatoryLabelingRun
from onyx.db.regulatory_labeling_source_only import (
    build_source_only_prior_failure,
    load_current_source_only_inputs,
    recover_failed_source_only_items,
    require_single_source_only_kind,
    source_only_target_digest,
    validate_source_only_abstention,
)
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.article_lead_in import (
    ARTICLE_LEAD_IN_KEY,
    ArticleLeadInSourceProof,
    SourceOnlyArticleLeadInReceipt,
    article_lead_in_candidate,
    build_article_lead_in_source_proof,
)
from onyx.regulatory.labeling.intake_abstention_repair import AbstentionRepairAudit
from onyx.regulatory.labeling.provider import TaxonomyDefinition


def _current_inputs(
    session: Session, item: RegulatoryLabelingItem, run: RegulatoryLabelingRun
) -> tuple[TaxonomyDefinition, VertexBatchRequest, ArticleLeadInSourceProof]:
    if not article_lead_in_candidate(item.text_snapshot):
        raise ValueError("Only a verified dependent article lead-in is source-only")
    inputs = load_current_source_only_inputs(
        session,
        item=item,
        run=run,
        receipt_key=ARTICLE_LEAD_IN_KEY,
        store_factory=get_default_file_store,
    )
    return (
        inputs.taxonomy,
        inputs.request,
        build_article_lead_in_source_proof(
            raw_html=inputs.raw_html,
            source_sha256=inputs.source_sha256,
            source_asset_id=inputs.source_asset_id,
            user_file_id=item.user_file_id,
            chunk_id=item.regulatory_chunk_id,
            rows=inputs.rows,
        ),
    )


def build_source_only_article_lead_in_receipt(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    provider_abstention: AbstentionRepairAudit | None = None,
    prior_failed: RegulatoryLabelingItem | None = None,
) -> SourceOnlyArticleLeadInReceipt:
    taxonomy, _, proof = _current_inputs(session, item, run)
    receipt = SourceOnlyArticleLeadInReceipt(
        proof=proof,
        frozen_target_sha256=source_only_target_digest(
            item, receipt_key=ARTICLE_LEAD_IN_KEY
        ),
        taxonomy_sha256=taxonomy.version_hash,
        provider_binding_sha256=publication_digest(run.provider_binding),
        provider_abstention=provider_abstention,
        prior_failed_abstention=build_source_only_prior_failure(prior_failed)
        if prior_failed is not None
        else None,
    )
    validate_item_article_lead_in_receipt(session, item=item, run=run, receipt=receipt)
    return receipt


def validate_item_article_lead_in_receipt(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    receipt: SourceOnlyArticleLeadInReceipt | None = None,
) -> None:
    require_single_source_only_kind(item.source_snapshot)
    if receipt is None and ARTICLE_LEAD_IN_KEY not in item.source_snapshot:
        return
    require_single_source_only_kind(
        item.source_snapshot, prospective_key=ARTICLE_LEAD_IN_KEY
    )
    stored = receipt or SourceOnlyArticleLeadInReceipt.model_validate(
        item.source_snapshot[ARTICLE_LEAD_IN_KEY]
    )
    if (
        receipt is not None
        and ARTICLE_LEAD_IN_KEY in item.source_snapshot
        and item.source_snapshot[ARTICLE_LEAD_IN_KEY] != receipt.model_dump(mode="json")
    ):
        raise ValueError(
            "A different structural lead-in disposition is already recorded"
        )
    if item.labels or item.assignments:
        raise ValueError("A structural lead-in cannot invent taxonomy assignments")
    taxonomy, request, proof = _current_inputs(session, item, run)
    if (
        stored.proof != proof
        or stored.frozen_target_sha256
        != source_only_target_digest(item, receipt_key=ARTICLE_LEAD_IN_KEY)
        or stored.taxonomy_sha256 != taxonomy.version_hash
        or stored.provider_binding_sha256 != publication_digest(run.provider_binding)
    ):
        raise ValueError("The structural lead-in evidence or frozen context changed")
    validate_source_only_abstention(
        session,
        item=item,
        run=run,
        receipt_key=ARTICLE_LEAD_IN_KEY,
        taxonomy=taxonomy,
        request=request,
        provider_abstention=stored.provider_abstention,
        prior_failed=stored.prior_failed_abstention,
    )


def recover_failed_article_lead_in_items(
    session: Session,
    *,
    previous: RegulatoryLabelingRun,
    replacement: RegulatoryLabelingRun,
) -> int:
    return recover_failed_source_only_items(
        session,
        previous=previous,
        replacement=replacement,
        receipt_key=ARTICLE_LEAD_IN_KEY,
        build_receipt=build_source_only_article_lead_in_receipt,
    )
