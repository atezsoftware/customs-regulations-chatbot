"""Source-only completion retains provider abstention and immutable legal text."""

import hashlib
from copy import deepcopy

from botocore.exceptions import ClientError
from google.api_core.exceptions import GoogleAPICallError
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

from onyx.db.document_intake_source_binding import require_owned_v4_intake_source
from onyx.db.models import (
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling_semantic_repair import (
    require_current_intake_repair_item,
)
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes.sources import MAX_ASSET_BYTES
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepairAudit,
    validate_stored_abstention_evidence,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from onyx.regulatory.labeling.self_repeal import (
    SOURCE_ONLY_KEY,
    PriorFailedAbstention,
    SelfRepealSourceProof,
    SourceOnlySelfRepealReceipt,
    build_self_repeal_source_proof,
    self_repeal_candidate,
)

CONTINUED_ABSTENTION = (
    "Model abstained: insufficient evidence after bounded semantic reconsideration"
)


def _target_digest(item: RegulatoryLabelingItem) -> str:
    return publication_digest(
        {
            "chunk_id": item.regulatory_chunk_id,
            "file_id": str(item.user_file_id),
            "text": item.text_snapshot,
            "canonical_text_sha256": item.canonical_text_sha256,
            "context_sha256": item.context_sha256,
            "context": item.context_snapshot,
            "request_hash": item.request_hash,
            "request_payload": item.request_payload,
            "source_snapshot": {
                key: value
                for key, value in item.source_snapshot.items()
                if key != SOURCE_ONLY_KEY
            },
        }
    )


def _current_inputs(
    session: Session,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
) -> tuple[TaxonomyDefinition, VertexBatchRequest, SelfRepealSourceProof]:
    if not self_repeal_candidate(item.text_snapshot):
        raise ValueError("Only a whole unconditional self-repeal marker is source-only")
    require_current_intake_repair_item(session, item=item, run=run)
    taxonomy_row = session.get_one(
        RegulatoryLabelTaxonomy, run.taxonomy_id, populate_existing=True
    )
    taxonomy = TaxonomyDefinition.model_validate(taxonomy_row.definition)
    if (
        taxonomy.version_hash != taxonomy_row.version_hash
        or run.provider_binding.get("model") != run.model
    ):
        raise ValueError("The source-only frozen taxonomy or model changed")
    if (
        not item.request_payload
        or not item.context_sha256
        or item.context_snapshot is None
    ):
        raise ValueError(
            "The source-only target has no complete frozen request and context"
        )
    request = VertexBatchRequest.model_validate(item.request_payload)
    expected = build_labeling_request(
        chunk_id=item.regulatory_chunk_id,
        text=item.text_snapshot,
        context=item.context_snapshot,
        taxonomy=taxonomy,
        source_hash=item.canonical_text_sha256 or "",
    )
    if request != expected or request.request_hash != item.request_hash:
        raise ValueError("The source-only frozen request changed")
    rows = capture_canonical_scope(session, item.user_file_id)
    row = next((row for row in rows if row.id == item.regulatory_chunk_id), None)
    if row is None:
        raise ValueError("The source-only canonical target disappeared")
    asset = require_owned_v4_intake_source(
        session,
        user_file_id=item.user_file_id,
        metadata=row.metadata,
        chunk_type=row.chunk_type,
        document_set_id=run.document_set_id,
    )
    if asset.mime_type != "text/html":
        raise ValueError("Source-only self-repeal requires the retained native HTML")
    try:
        with get_default_file_store().read_file(asset.file_id) as stream:
            raw = stream.read(MAX_ASSET_BYTES + 1)
    except (OSError, ClientError, GoogleAPICallError, NoResultFound) as error:
        raise ValueError("The retained source-only original is unavailable") from error
    if len(raw) != asset.byte_count or hashlib.sha256(raw).hexdigest() != asset.sha256:
        raise ValueError("The retained source-only original bytes changed")
    return (
        taxonomy,
        request,
        build_self_repeal_source_proof(
            raw_html=raw,
            source_sha256=asset.sha256,
            source_asset_id=asset.id,
            user_file_id=item.user_file_id,
            chunk_id=item.regulatory_chunk_id,
            rows=rows,
        ),
    )


def build_source_only_self_repeal_receipt(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    provider_abstention: AbstentionRepairAudit | None = None,
    prior_failed: RegulatoryLabelingItem | None = None,
) -> SourceOnlySelfRepealReceipt:
    taxonomy, _, proof = _current_inputs(session, item, run)
    prior = None
    if prior_failed is not None:
        if prior_failed.error != CONTINUED_ABSTENTION or not prior_failed.request_hash:
            raise ValueError(
                "The prior item does not retain a bounded provider abstention"
            )
        prior = PriorFailedAbstention(
            run_id=prior_failed.run_id,
            item_id=prior_failed.id,
            request_hash=prior_failed.request_hash,
            error=CONTINUED_ABSTENTION,
            source_snapshot_sha256=publication_digest(prior_failed.source_snapshot),
        )
    receipt = SourceOnlySelfRepealReceipt(
        proof=proof,
        frozen_target_sha256=_target_digest(item),
        taxonomy_sha256=taxonomy.version_hash,
        provider_binding_sha256=publication_digest(run.provider_binding),
        provider_abstention=provider_abstention,
        prior_failed_abstention=prior,
    )
    validate_item_source_only_receipt(session, item=item, run=run, receipt=receipt)
    return receipt


def validate_item_source_only_receipt(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    receipt: SourceOnlySelfRepealReceipt | None = None,
) -> None:
    if receipt is None and SOURCE_ONLY_KEY not in item.source_snapshot:
        return
    stored = receipt or SourceOnlySelfRepealReceipt.model_validate(
        item.source_snapshot[SOURCE_ONLY_KEY]
    )
    if (
        receipt is not None
        and SOURCE_ONLY_KEY in item.source_snapshot
        and item.source_snapshot[SOURCE_ONLY_KEY] != receipt.model_dump(mode="json")
    ):
        raise ValueError("A different source-only disposition is already recorded")
    if item.labels or item.assignments:
        raise ValueError("A source-only disposition cannot invent taxonomy assignments")
    taxonomy, request, proof = _current_inputs(session, item, run)
    if (
        stored.proof != proof
        or stored.frozen_target_sha256 != _target_digest(item)
        or stored.taxonomy_sha256 != taxonomy.version_hash
        or stored.provider_binding_sha256 != publication_digest(run.provider_binding)
    ):
        raise ValueError(
            "The source-only evidence or frozen classification context changed"
        )
    if stored.provider_abstention is not None:
        validate_stored_abstention_evidence(
            stored.provider_abstention,
            base_request=request,
            text=item.text_snapshot,
            chunk_id=item.regulatory_chunk_id,
            source_hash=item.canonical_text_sha256 or "",
            taxonomy=taxonomy,
            provider_binding=run.provider_binding,
        )
    else:
        prior = stored.prior_failed_abstention
        if prior is None:
            raise ValueError("The original provider abstention is absent")
        old = session.get(RegulatoryLabelingItem, prior.item_id, populate_existing=True)
        old_run = session.get(
            RegulatoryLabelingRun, prior.run_id, populate_existing=True
        )
        if (
            old is None
            or old_run is None
            or old.run_id != old_run.id
            or old.status != "failed"
            or old.error != prior.error
            or old.request_hash != prior.request_hash
            or old.user_file_id != item.user_file_id
            or old_run.document_set_id != run.document_set_id
            or old_run.taxonomy_id != run.taxonomy_id
            or old_run.model != run.model
            or old_run.provider_binding != run.provider_binding
            or old.labels
            or old.assignments
            or publication_digest(old.source_snapshot) != prior.source_snapshot_sha256
            or _target_digest(old) != _target_digest(item)
        ):
            raise ValueError("The immutable prior provider abstention lineage changed")


def recover_failed_self_repeal_items(
    session: Session,
    *,
    previous: RegulatoryLabelingRun,
    replacement: RegulatoryLabelingRun,
) -> int:
    """Resolve only verified old abstentions, without changing their provider run."""
    from onyx.db.regulatory_labeling import current_labeling_items_for_search

    failed = list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == previous.id,
                RegulatoryLabelingItem.status == "failed",
                RegulatoryLabelingItem.error == CONTINUED_ABSTENTION,
            )
        )
    )
    recovered = 0
    for old in failed:
        target = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id == old.regulatory_chunk_id,
                RegulatoryLabelingItem.status == "pending",
            )
        )
        if (
            target is None
            or target.user_file_id != old.user_file_id
            or target.text_snapshot != old.text_snapshot
            or any(
                old.source_snapshot.get(key) != value
                for key, value in target.source_snapshot.items()
            )
        ):
            continue
        candidate = RegulatoryLabelingItem(
            id=target.id,
            run_id=replacement.id,
            regulatory_chunk_id=old.regulatory_chunk_id,
            user_file_id=old.user_file_id,
            status="completed",
            labels=[],
            assignments=[],
            canonical_text_sha256=old.canonical_text_sha256,
            context_sha256=old.context_sha256,
            text_snapshot=old.text_snapshot,
            context_snapshot=old.context_snapshot,
            request_hash=old.request_hash,
            request_payload=deepcopy(old.request_payload),
            source_snapshot=deepcopy(old.source_snapshot),
        )
        try:
            receipt = build_source_only_self_repeal_receipt(
                session,
                item=candidate,
                run=replacement,
                prior_failed=old,
            )
            candidate.source_snapshot = {
                **candidate.source_snapshot,
                SOURCE_ONLY_KEY: receipt.model_dump(mode="json"),
            }
            if current_labeling_items_for_search(
                session, run=replacement, items=[candidate]
            ) != {candidate.regulatory_chunk_id}:
                continue
        except ValueError:
            continue
        for field in (
            "canonical_text_sha256",
            "context_sha256",
            "context_snapshot",
            "request_hash",
            "request_payload",
            "source_snapshot",
        ):
            setattr(target, field, deepcopy(getattr(candidate, field)))
        target.status, target.labels, target.assignments, target.error = (
            "completed",
            [],
            [],
            None,
        )
        recovered += 1
    session.flush()
    return recovered
