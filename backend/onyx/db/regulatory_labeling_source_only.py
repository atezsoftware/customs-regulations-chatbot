"""Current owned inputs and unchanged abstention lineage for source-only receipts."""

import hashlib
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Literal, Protocol
from uuid import UUID

from botocore.exceptions import ClientError
from google.api_core.exceptions import GoogleAPICallError
from pydantic import BaseModel, ConfigDict
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
from onyx.file_store.file_store import FileStore
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.sources import MAX_ASSET_BYTES
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.article_lead_in import ARTICLE_LEAD_IN_KEY
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepairAudit,
    validate_stored_abstention_evidence,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from onyx.regulatory.labeling.self_repeal import SOURCE_ONLY_KEY, PriorFailedAbstention

SourceOnlyReceiptKey = Literal[
    "intake_source_only_self_repeal", "intake_source_only_article_lead_in"
]

CONTINUED_ABSTENTION = (
    "Model abstained: insufficient evidence after bounded semantic reconsideration"
)


class CurrentSourceOnlyInputs(BaseModel):
    model_config = ConfigDict(frozen=True)

    taxonomy: TaxonomyDefinition
    request: VertexBatchRequest
    rows: list[AnnexCanonicalSnapshot]
    raw_html: bytes
    source_sha256: str
    source_asset_id: UUID


class SourceOnlyReceiptBuilder(Protocol):
    def __call__(
        self,
        session: Session,
        *,
        item: RegulatoryLabelingItem,
        run: RegulatoryLabelingRun,
        prior_failed: RegulatoryLabelingItem,
    ) -> BaseModel: ...


def require_single_source_only_kind(
    snapshot: Mapping[str, object],
    *,
    prospective_key: SourceOnlyReceiptKey | None = None,
) -> None:
    supported = {SOURCE_ONLY_KEY, ARTICLE_LEAD_IN_KEY}
    if prospective_key is not None and prospective_key not in supported:
        raise ValueError("Unknown source-only receipt kind")
    keys = supported.intersection(snapshot)
    if prospective_key is not None:
        keys.add(prospective_key)
    if len(keys) > 1:
        raise ValueError("A classification item cannot contain two source-only kinds")


def source_only_target_digest(
    item: RegulatoryLabelingItem, *, receipt_key: SourceOnlyReceiptKey
) -> str:
    require_single_source_only_kind(item.source_snapshot, prospective_key=receipt_key)
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
                if key != receipt_key
            },
        }
    )


def load_current_source_only_inputs(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    receipt_key: SourceOnlyReceiptKey,
    store_factory: Callable[[], FileStore],
) -> CurrentSourceOnlyInputs:
    require_single_source_only_kind(item.source_snapshot, prospective_key=receipt_key)
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
        raise ValueError("Source-only classification requires the retained native HTML")
    try:
        with store_factory().read_file(asset.file_id) as stream:
            raw = stream.read(MAX_ASSET_BYTES + 1)
    except (OSError, ClientError, GoogleAPICallError, NoResultFound) as error:
        raise ValueError("The retained source-only original is unavailable") from error
    if len(raw) != asset.byte_count or hashlib.sha256(raw).hexdigest() != asset.sha256:
        raise ValueError("The retained source-only original bytes changed")
    return CurrentSourceOnlyInputs(
        taxonomy=taxonomy,
        request=request,
        rows=rows,
        raw_html=raw,
        source_sha256=asset.sha256,
        source_asset_id=asset.id,
    )


def build_source_only_prior_failure(
    old: RegulatoryLabelingItem,
) -> PriorFailedAbstention:
    if old.error != CONTINUED_ABSTENTION or not old.request_hash:
        raise ValueError("The prior item does not retain a bounded provider abstention")
    return PriorFailedAbstention(
        run_id=old.run_id,
        item_id=old.id,
        request_hash=old.request_hash,
        error=CONTINUED_ABSTENTION,
        source_snapshot_sha256=publication_digest(old.source_snapshot),
    )


def validate_source_only_abstention(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    receipt_key: SourceOnlyReceiptKey,
    taxonomy: TaxonomyDefinition,
    request: VertexBatchRequest,
    provider_abstention: AbstentionRepairAudit | None,
    prior_failed: PriorFailedAbstention | None,
) -> None:
    require_single_source_only_kind(item.source_snapshot, prospective_key=receipt_key)
    if provider_abstention is not None:
        validate_stored_abstention_evidence(
            provider_abstention,
            base_request=request,
            text=item.text_snapshot,
            chunk_id=item.regulatory_chunk_id,
            source_hash=item.canonical_text_sha256 or "",
            taxonomy=taxonomy,
            provider_binding=run.provider_binding,
        )
        return
    if prior_failed is None:
        raise ValueError("The original provider abstention is absent")
    old = session.get(
        RegulatoryLabelingItem, prior_failed.item_id, populate_existing=True
    )
    old_run = session.get(
        RegulatoryLabelingRun, prior_failed.run_id, populate_existing=True
    )
    if (
        old is None
        or old_run is None
        or old.run_id != old_run.id
        or old.status != "failed"
        or old.error != prior_failed.error
        or old.request_hash != prior_failed.request_hash
        or old.user_file_id != item.user_file_id
        or old_run.document_set_id != run.document_set_id
        or old_run.taxonomy_id != run.taxonomy_id
        or old_run.model != run.model
        or old_run.provider_binding != run.provider_binding
        or old.labels
        or old.assignments
        or publication_digest(old.source_snapshot)
        != prior_failed.source_snapshot_sha256
        or source_only_target_digest(old, receipt_key=receipt_key)
        != source_only_target_digest(item, receipt_key=receipt_key)
    ):
        raise ValueError("The immutable prior provider abstention lineage changed")


def recover_failed_source_only_items(
    session: Session,
    *,
    previous: RegulatoryLabelingRun,
    replacement: RegulatoryLabelingRun,
    receipt_key: SourceOnlyReceiptKey,
    build_receipt: SourceOnlyReceiptBuilder,
) -> int:
    require_single_source_only_kind({}, prospective_key=receipt_key)
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
            receipt = build_receipt(
                session, item=candidate, run=replacement, prior_failed=old
            )
            candidate.source_snapshot = {
                **candidate.source_snapshot,
                receipt_key: receipt.model_dump(mode="json"),
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
