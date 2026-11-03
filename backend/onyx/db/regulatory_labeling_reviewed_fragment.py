"""Reviewed public-source fragments and actual base-ACK abstention authority."""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from uuid import UUID

from botocore.exceptions import ClientError
from google.api_core.exceptions import GoogleAPICallError
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound

from onyx.db.document_intake_source_binding import require_owned_intake_source
from onyx.db.enums import UserFileStatus
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    FileRecord,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryIndexingJob,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatoryLabelTaxonomy,
    RegulatoryTemporalProjection,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_labeling_duplicate_evidence import (
    load_duplicate_ack,
    validate_duplicate_ack_header,
)
from onyx.db.regulatory_labeling_source_only import (
    require_single_source_only_kind,
    source_only_target_digest,
)
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
    vertex_batch_submission_key,
)
from onyx.regulatory.labeling.duplicate_ack_evidence import MAX_ACK_BYTES
from onyx.regulatory.labeling.intake_abstention_repair import (
    ABSTENTION_REPAIR_KEY,
    AbstentionRepair,
    AbstentionRepairAudit,
    build_abstention_repair,
    build_abstention_repair_audit,
    parse_abstention_repair_output,
    validate_stored_abstention_evidence,
)
from onyx.regulatory.labeling.intake_duplicate_label_repair import DUPLICATE_REPAIR_KEY
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from onyx.regulatory.labeling.reviewed_fragment import (
    REVIEWED_FRAGMENT_KEY,
    ReviewedBaseAbstention,
    ReviewedContinuedAbstention,
    ReviewedFigureFragmentProof,
    ReviewedFragmentProof,
    ReviewedFragmentSource,
    SourceOnlyReviewedFragmentReceipt,
    build_reviewed_figure_proof,
    build_reviewed_fragment_proof,
    require_base_abstention,
    require_base_ack_line,
    reviewed_figure_fragment_candidate,
    reviewed_fragment_candidate,
    selected_qualification,
)
from shared_configs.contextvars import get_current_tenant_id


def _current_inputs(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    context_run_id: UUID | None = None,
) -> tuple[
    TaxonomyDefinition,
    VertexBatchRequest,
    ReviewedFragmentProof | ReviewedFigureFragmentProof,
]:
    profile = selected_qualification(item.text_snapshot)
    if (
        item.run_id != run.id
        or run.document_set_id is None
        or str(item.user_file_id) not in run.file_ids
    ):
        raise ValueError("The reviewed-fragment target differs from its frozen run")
    members = item.source_snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or len(members) != profile["canonical_entry_count"]
        or not all(isinstance(value, str) for value in members)
        or len(members) != len(set(members))
    ):
        raise ValueError("The reviewed-fragment context membership is incomplete")
    rows = session.execute(
        select(RegulatoryChunk, RegulatoryLabelingItem)
        .join(
            RegulatoryLabelingItem,
            (RegulatoryLabelingItem.regulatory_chunk_id == RegulatoryChunk.id)
            & (RegulatoryLabelingItem.user_file_id == RegulatoryChunk.user_file_id),
        )
        .where(
            RegulatoryLabelingItem.run_id == (context_run_id or run.id),
            RegulatoryChunk.user_file_id == item.user_file_id,
            RegulatoryChunk.id.in_(members),
        )
        .execution_options(populate_existing=True)
    ).all()
    sources = [
        ReviewedFragmentSource(
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
        for row, saved in rows
        for frozen in [item if row.id == item.regulatory_chunk_id else saved]
    ]
    asset = require_owned_intake_source(
        session, user_file_id=item.user_file_id, document_set_id=run.document_set_id
    )
    retained = session.get(FileRecord, asset.file_id, populate_existing=True)
    if (
        asset.mime_type != "application/pdf"
        or asset.byte_count != profile["source_byte_count"]
        or retained is None
        or retained.file_type != "application/pdf"
    ):
        raise ValueError("A reviewed fragment requires its retained bounded PDF")
    try:
        with get_default_file_store().read_file(asset.file_id) as stream:
            raw = stream.read(profile["source_byte_count"] + 1)
    except (OSError, ClientError, GoogleAPICallError, NoResultFound) as error:
        raise ValueError("The reviewed public-source PDF is unavailable") from error
    if len(raw) != asset.byte_count or hashlib.sha256(raw).hexdigest() != asset.sha256:
        raise ValueError("The reviewed public-source PDF bytes changed")
    taxonomy_row = session.get_one(
        RegulatoryLabelTaxonomy, run.taxonomy_id, populate_existing=True
    )
    taxonomy = TaxonomyDefinition.model_validate(taxonomy_row.definition)
    if (
        context_hash(item.text_snapshot) != item.canonical_text_sha256
        or taxonomy.version_hash != taxonomy_row.version_hash
        or run.provider_binding.get("model") != run.model
        or not item.request_payload
        or not item.context_sha256
        or item.context_snapshot is None
    ):
        raise ValueError(
            "The reviewed-fragment frozen taxonomy, model or request changed"
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
        raise ValueError("The reviewed-fragment frozen request changed")
    builder = (
        build_reviewed_fragment_proof
        if profile["profile_kind"] == "blank_form_rows"
        else build_reviewed_figure_proof
    )
    proof = builder(
        raw_pdf=raw,
        source_asset_id=asset.id,
        user_file_id=item.user_file_id,
        chunk_id=item.regulatory_chunk_id,
        sources=sources,
    )
    return taxonomy, request, proof


def _history(item: RegulatoryLabelingItem) -> str:
    return publication_digest(
        {
            "id": str(item.id),
            "run_id": str(item.run_id),
            "status": item.status,
            "error": item.error,
            "labels": item.labels,
            "assignments": item.assignments,
            "frozen": source_only_target_digest(
                item, receipt_key=REVIEWED_FRAGMENT_KEY
            ),
        }
    )


def _same_runs(left: RegulatoryLabelingRun, right: RegulatoryLabelingRun) -> bool:
    return (
        left.document_set_id == right.document_set_id
        and left.taxonomy_id == right.taxonomy_id
        and left.model_configuration_id == right.model_configuration_id
        and left.model == right.model
        and left.provider_binding == right.provider_binding
        and left.requested_by_id == right.requested_by_id
        and len(left.file_ids)
        == len(set(left.file_ids))
        == len(right.file_ids)
        == len(set(right.file_ids))
        and set(left.file_ids) == set(right.file_ids)
    )


def _submission(
    session: Session, source: RegulatoryLabelingItem, run: RegulatoryLabelingRun
) -> str:
    shard = (
        session.get(RegulatoryLabelingShard, source.shard_id, populate_existing=True)
        if source.shard_id
        else None
    )
    if shard is None or shard.run_id != run.id or not shard.submission_key:
        raise ValueError("The genuine base ACK has no matching frozen shard")
    return shard.submission_key


def _line_from_cache(
    session: Session,
    *,
    run: RegulatoryLabelingRun,
    source: RegulatoryLabelingItem,
    request: VertexBatchRequest,
    submission_key: str | None = None,
) -> str:
    binding = LabelingProviderBinding.model_validate(run.provider_binding)
    submission = submission_key or _submission(session, source, run)
    file_id = f"regulatory-labeling-direct/{binding.fingerprint}/{submission.removeprefix('regulatory-labeling-')}/{request.request_hash}.json"
    try:
        with get_default_file_store().read_file(file_id) as stream:
            raw = stream.read(MAX_ACK_BYTES + 1)
        if not 0 < len(raw) <= MAX_ACK_BYTES:
            raise ValueError("The base ACK exceeds its byte budget")
        value = json.loads(raw)
        line = value.get("line") if isinstance(value, dict) else None
        if not isinstance(line, str):
            raise ValueError("The genuine base ACK row is absent")
        return line
    except (OSError, ClientError, GoogleAPICallError, NoResultFound) as error:
        raise ValueError("The genuine base ACK bytes are unavailable") from error


def _semantic_repair(
    *,
    request: VertexBatchRequest,
    response: str,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    taxonomy: TaxonomyDefinition,
) -> tuple[AbstentionRepair, str]:
    if item.shard_id is None:
        raise ValueError("The actual semantic attempt has no frozen source shard")
    repair = build_abstention_repair(
        base_request=request,
        rejected_response=response,
        text=item.text_snapshot,
        chunk_id=item.regulatory_chunk_id,
        source_hash=item.canonical_text_sha256 or "",
        taxonomy=taxonomy,
        provider_binding=run.provider_binding,
    )
    submission = vertex_batch_submission_key(
        [repair.request],
        tenant_id=get_current_tenant_id(),
        job_id=run.id,
        output_prefix=f"regulatory-labeling/{run.id}/abstention-repair/{item.shard_id}/{item.id}",
        submission_attempt=1,
    ).replace("regulatory-context-", "regulatory-labeling-", 1)
    return repair, submission


def _require_unpublished_mutation(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    source_run: RegulatoryLabelingRun,
) -> None:
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile)
        .where(RegulatoryDocumentIntakeFile.user_file_id == item.user_file_id)
        .execution_options(populate_existing=True)
    )
    batch = (
        session.get(RegulatoryDocumentIntake, member.intake_id, populate_existing=True)
        if member
        else None
    )
    file = session.get(UserFile, item.user_file_id, populate_existing=True)
    publication = session.get(
        RegulatoryFilePublication, item.user_file_id, populate_existing=True
    )
    if (
        member is None
        or batch is None
        or file is None
        or publication is None
        or batch.labeling_run_id not in {run.id, source_run.id}
        or batch.created_by != run.requested_by_id
        or batch.document_set_id != run.document_set_id
        or batch.status not in {"preparing", "partial", "ready"}
        or batch.stage not in {"waiting_labeling", "labeling", "review"}
        or member.status not in {"chunked", "labeling", "ready"}
        or member.approval_attempt_id is not None
        or file.status != UserFileStatus.CHUNKED
        or publication.epoch != 0
        or publication.gate_closed
        or publication.writer_manifest is not None
        or publication.lease_expires_at > datetime.now(timezone.utc)
        or any(
            session.scalar(
                select(model).where(model.user_file_id == item.user_file_id).limit(1)
            )
            is not None
            for model in (
                RegulatoryIndexingJob,
                RegulatoryTemporalProjection,
                UserFileProjectionRepair,
            )
        )
    ):
        raise ValueError(
            "A reviewed-fragment mutation requires its current unpublished intake"
        )


def build_source_only_reviewed_fragment_receipt(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    original_line: str | None = None,
    prior_failed: RegulatoryLabelingItem | None = None,
    provider_abstention: AbstentionRepairAudit | None = None,
) -> SourceOnlyReviewedFragmentReceipt:
    require_single_source_only_kind(
        item.source_snapshot, prospective_key=REVIEWED_FRAGMENT_KEY
    )
    if ABSTENTION_REPAIR_KEY in item.source_snapshot:
        raise ValueError(
            "A base-only reviewed fragment cannot replace a semantic repair"
        )
    source = prior_failed or item
    source_run = session.get_one(
        RegulatoryLabelingRun, source.run_id, populate_existing=True
    )
    _require_unpublished_mutation(session, item=item, run=run, source_run=source_run)
    taxonomy, request, proof = _current_inputs(
        session, item=item, run=run, context_run_id=source_run.id
    )
    if prior_failed is not None and (
        source.status != "failed"
        or source.labels
        or source.assignments
        or not _same_runs(source_run, run)
        or source_only_target_digest(source, receipt_key=REVIEWED_FRAGMENT_KEY)
        != source_only_target_digest(item, receipt_key=REVIEWED_FRAGMENT_KEY)
    ):
        raise ValueError("The prior failed base outcome changed its frozen identity")
    if original_line is None:
        original_line = _line_from_cache(
            session, run=source_run, source=source, request=request
        )
    response = require_base_ack_line(
        original_line, request, text=item.text_snapshot, taxonomy=taxonomy
    )
    ack = load_duplicate_ack(
        session,
        binding_fingerprint=LabelingProviderBinding.model_validate(
            source_run.provider_binding
        ).fingerprint,
        submission_key=_submission(session, source, source_run),
        request=request,
        expected_line=original_line,
    )
    base_evidence = ReviewedBaseAbstention(
        origin="prior_failed_item" if prior_failed else "same_item",
        source_run_id=source_run.id,
        source_item_id=source.id,
        source_frozen_sha256=source_only_target_digest(
            source, receipt_key=REVIEWED_FRAGMENT_KEY
        ),
        prior_history_sha256=_history(source) if prior_failed else None,
        ack=ack,
        response=response,
    )
    evidence: ReviewedBaseAbstention | ReviewedContinuedAbstention = base_evidence
    if proof.profile_kind == "figure_caption":
        repair, submission = _semantic_repair(
            request=request,
            response=response,
            item=source,
            run=source_run,
            taxonomy=taxonomy,
        )
        line = _line_from_cache(
            session,
            run=source_run,
            source=source,
            request=repair.request,
            submission_key=submission,
        )
        require_base_ack_line(
            line, repair.request, text=item.text_snapshot, taxonomy=taxonomy
        )
        semantic_ack = load_duplicate_ack(
            session,
            binding_fingerprint=ack.binding_fingerprint,
            submission_key=submission,
            request=repair.request,
            expected_line=line,
        )
        audit = provider_abstention or build_abstention_repair_audit(
            output=parse_abstention_repair_output(
                [line], repair=repair, on_progress=lambda: None
            ),
            repair=repair,
            output_uri=f"ack-cache://{semantic_ack.file_id}",
            text=item.text_snapshot,
            taxonomy=taxonomy,
        )
        evidence = ReviewedContinuedAbstention(
            **base_evidence.model_dump(exclude={"policy"}),
            semantic_ack=semantic_ack,
            semantic_audit=audit,
        )
    elif provider_abstention is not None:
        raise ValueError(
            "A blank-form qualification requires only its genuine base ACK"
        )
    receipt = SourceOnlyReviewedFragmentReceipt(
        proof=proof,
        frozen_target_sha256=source_only_target_digest(
            item, receipt_key=REVIEWED_FRAGMENT_KEY
        ),
        taxonomy_sha256=taxonomy.version_hash,
        provider_binding_sha256=publication_digest(run.provider_binding),
        base_abstention=evidence,
    )
    validate_item_reviewed_fragment_receipt(
        session, item=item, run=run, receipt=receipt
    )
    return receipt


def validate_item_reviewed_fragment_receipt(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    receipt: SourceOnlyReviewedFragmentReceipt | None = None,
) -> None:
    require_single_source_only_kind(item.source_snapshot)
    if receipt is None and REVIEWED_FRAGMENT_KEY not in item.source_snapshot:
        return
    require_single_source_only_kind(
        item.source_snapshot, prospective_key=REVIEWED_FRAGMENT_KEY
    )
    if (
        ABSTENTION_REPAIR_KEY in item.source_snapshot
        or DUPLICATE_REPAIR_KEY in item.source_snapshot
        or item.labels
        or item.assignments
    ):
        raise ValueError(
            "A reviewed fragment cannot contain a positive or semantic classification"
        )
    stored = receipt or SourceOnlyReviewedFragmentReceipt.model_validate(
        item.source_snapshot[REVIEWED_FRAGMENT_KEY]
    )
    if (
        receipt is not None
        and REVIEWED_FRAGMENT_KEY in item.source_snapshot
        and item.source_snapshot[REVIEWED_FRAGMENT_KEY]
        != receipt.model_dump(mode="json")
    ):
        raise ValueError(
            "A different reviewed-fragment disposition is already recorded"
        )
    taxonomy, request, proof = _current_inputs(
        session,
        item=item,
        run=run,
        context_run_id=stored.base_abstention.source_run_id,
    )
    if (
        stored.proof != proof
        or stored.frozen_target_sha256
        != source_only_target_digest(item, receipt_key=REVIEWED_FRAGMENT_KEY)
        or stored.taxonomy_sha256 != taxonomy.version_hash
        or stored.provider_binding_sha256 != publication_digest(run.provider_binding)
    ):
        raise ValueError("The reviewed-fragment source/request authority changed")
    evidence = stored.base_abstention
    source = (
        item
        if evidence.source_item_id == item.id and evidence.source_run_id == run.id
        else session.get(
            RegulatoryLabelingItem, evidence.source_item_id, populate_existing=True
        )
    )
    source_run = (
        run
        if evidence.source_run_id == run.id
        else session.get(
            RegulatoryLabelingRun, evidence.source_run_id, populate_existing=True
        )
    )
    if (
        source is None
        or source_run is None
        or source.run_id != source_run.id
        or source.user_file_id != item.user_file_id
        or source.regulatory_chunk_id != item.regulatory_chunk_id
        or not _same_runs(source_run, run)
        or source_only_target_digest(source, receipt_key=REVIEWED_FRAGMENT_KEY)
        != stored.frozen_target_sha256
        or evidence.source_frozen_sha256 != stored.frozen_target_sha256
    ):
        raise ValueError("The genuine base ACK source lineage changed")
    if source_run.id != run.id and (
        source_run.status
        not in {"completed", "completed_with_errors", "failed", "cancelled"}
        or source_run.lease_token is not None
        or source_run.lease_expires_at is not None
    ):
        raise ValueError("The prior base source run is not terminal and idle")
    if evidence.origin == "prior_failed_item":
        if (
            source.status != "failed"
            or source.labels
            or source.assignments
            or evidence.prior_history_sha256 != _history(source)
        ):
            raise ValueError("The original failed base history changed")
    elif evidence.prior_history_sha256 is not None or (
        source.id != item.id
        and (
            source.status != "completed"
            or source.source_snapshot.get(REVIEWED_FRAGMENT_KEY)
            != stored.model_dump(mode="json")
        )
    ):
        raise ValueError("A reused base receipt lost its original accepted lineage")
    if (
        evidence.ack.request_hash != request.request_hash
        or evidence.ack.binding_fingerprint
        != LabelingProviderBinding.model_validate(run.provider_binding).fingerprint
        or evidence.ack.submission_key != _submission(session, source, source_run)
        or hashlib.sha256(evidence.response.encode()).hexdigest()
        != evidence.ack.response_sha256
    ):
        raise ValueError("The genuine base ACK/request binding changed")
    require_base_abstention(
        evidence.response, text=item.text_snapshot, taxonomy=taxonomy
    )
    validate_duplicate_ack_header(session, evidence.ack)
    if stored.proof.profile_kind == "blank_form_rows":
        if not isinstance(evidence, ReviewedBaseAbstention):
            raise ValueError("The reviewed blank-form abstention policy changed")
    else:
        if not isinstance(evidence, ReviewedContinuedAbstention):
            raise ValueError("A reviewed figure requires actual continued abstention")
        repair, submission = _semantic_repair(
            request=request,
            response=evidence.response,
            item=source,
            run=source_run,
            taxonomy=taxonomy,
        )
        audit = validate_stored_abstention_evidence(
            evidence.semantic_audit,
            base_request=request,
            text=item.text_snapshot,
            chunk_id=item.regulatory_chunk_id,
            source_hash=item.canonical_text_sha256 or "",
            taxonomy=taxonomy,
            provider_binding=run.provider_binding,
        )
        if (
            not audit.outcome.abstained
            or audit.outcome.labels
            or audit.rejected_response != evidence.response
            or evidence.semantic_ack.binding_fingerprint
            != evidence.ack.binding_fingerprint
            or evidence.semantic_ack.request_hash != repair.request.request_hash
            or evidence.semantic_ack.submission_key != submission
            or evidence.semantic_ack.line_sha256 != audit.output_row_sha256
            or evidence.semantic_ack.response_sha256 != audit.output_response_sha256
        ):
            raise ValueError(
                "The reviewed figure lost its actual same-item bounded semantic ACK"
            )
        validate_duplicate_ack_header(session, evidence.semantic_ack)


def recover_failed_reviewed_fragment_items(
    session: Session,
    *,
    previous: RegulatoryLabelingRun,
    replacement: RegulatoryLabelingRun,
) -> int:
    if not _same_runs(previous, replacement):
        return 0
    from onyx.db.regulatory_labeling import current_labeling_items_for_search

    recovered = 0
    for old in session.scalars(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == previous.id,
            RegulatoryLabelingItem.status == "failed",
        )
    ):
        if not (
            reviewed_fragment_candidate(old.text_snapshot)
            or reviewed_figure_fragment_candidate(old.text_snapshot)
        ):
            continue
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
                old.source_snapshot.get(k) != v
                for k, v in target.source_snapshot.items()
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
            receipt = build_source_only_reviewed_fragment_receipt(
                session, item=candidate, run=replacement, prior_failed=old
            )
            candidate.source_snapshot = {
                **candidate.source_snapshot,
                REVIEWED_FRAGMENT_KEY: receipt.model_dump(mode="json"),
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
