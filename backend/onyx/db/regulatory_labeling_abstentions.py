"""Correct legacy no-label failures from retained ACKs, without provider work."""

import json
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling as labeling
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    RegulatoryDerivedLabelProjection,
    RegulatoryDocumentIntake,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
)
from onyx.db.regulatory_direct_ack import read_direct_ack
from onyx.db.regulatory_labeling_duplicate_evidence import load_duplicate_ack
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.abstention import accept_provider_abstention
from onyx.regulatory.labeling.accepted_result import AcceptedProviderAbstention
from onyx.regulatory.labeling.duplicate_ack_evidence import (
    MAX_ACK_BYTES,
    complete_ack_response,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request

_ABSTENTION_ERRORS = {
    "Model abstained: insufficient evidence",
    "Model abstained: insufficient evidence after bounded semantic reconsideration",
}


def complete_retained_abstentions(
    session: Session, run_id: UUID, *, limit: int = 128
) -> int:
    if not 1 <= limit <= 128:
        raise ValueError("Retained abstention correction requires a bounded page")
    run = session.get_one(
        RegulatoryLabelingRun, run_id, with_for_update=True, populate_existing=True
    )
    now = datetime.now(timezone.utc)
    if (
        run.status not in {"completed", "completed_with_errors", "failed"}
        or run.cancel_requested
        or (run.lease_expires_at is not None and run.lease_expires_at > now)
    ):
        raise ValueError("Only idle terminal label runs can be corrected")
    if (
        run.error is not None
        and run.error
        != f"{run.unresolved_derived_chunks} derived chunk(s) could not be projected safely"
    ):
        raise ValueError(
            "An operational labeling failure cannot be cleared as abstention"
        )
    items = list(
        session.scalars(
            select(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.status == "failed",
                RegulatoryLabelingItem.error.in_(_ABSTENTION_ERRORS),
            )
            .order_by(RegulatoryLabelingItem.id)
            .limit(limit + 1)
            .with_for_update()
        )
    )
    if not items:
        return 0
    if len(items) > limit:
        raise ValueError("Retained abstentions exceed the bounded correction scope")
    binding = LabelingProviderBinding.model_validate(run.provider_binding)
    if binding.transport != "vertex_direct_v1":
        raise ValueError("Retained abstention correction requires direct ACK evidence")
    taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
    plans = []
    for item in items:
        if item.labels or item.assignments or item.context_snapshot is None:
            raise ValueError("A failed abstention has inconsistent stored results")
        request = VertexBatchRequest.model_validate(item.request_payload)
        rebuilt = build_labeling_request(
            chunk_id=item.regulatory_chunk_id,
            text=item.text_snapshot,
            context=item.context_snapshot,
            taxonomy=taxonomy,
            source_hash=item.canonical_text_sha256 or "",
        )
        if request != rebuilt or item.request_hash != request.request_hash:
            raise ValueError("The frozen abstention request changed")
        shard = session.get_one(RegulatoryLabelingShard, item.shard_id)
        if shard.run_id != run_id:
            raise ValueError("The abstention shard scope changed")
        file_id = f"regulatory-labeling-direct/{binding.fingerprint}/{shard.submission_key.removeprefix('regulatory-labeling-')}/{request.request_hash}.json"
        raw = read_direct_ack(session, file_id, max_bytes=MAX_ACK_BYTES)
        stored = json.loads(raw)
        line = stored.get("line")
        if not isinstance(line, str):
            raise ValueError("The original provider acknowledgment is absent")
        evidence = load_duplicate_ack(
            session,
            binding_fingerprint=binding.fingerprint,
            submission_key=shard.submission_key,
            request=request,
            expected_line=line,
        )
        receipt = accept_provider_abstention(
            request.request_hash, complete_ack_response(line, request)
        )
        plans.append((item, receipt, evidence))
    with session.begin_nested():
        for item, receipt, evidence in plans:
            old_error = item.error
            lower, upper = labeling._snapshot_context_window(item.source_snapshot)
            labeling._apply_frozen_window_results(
                session,
                run,
                items=[item],
                file_id=item.user_file_id,
                lower=lower,
                upper=upper,
                outcomes={receipt.request_hash: AcceptedProviderAbstention(receipt)},
            )
            if item.status != "completed":
                raise ValueError("The retained abstention source context changed")
            item.source_snapshot = {
                **item.source_snapshot,
                "retained_abstention_correction": {
                    "prior_error": old_error,
                    "ack": evidence.model_dump(mode="json"),
                },
            }
        session.flush()
        # The normal SQL-only projection rules recheck every canonical dependency.
        for target in session.scalars(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run_id,
                RegulatoryDerivedLabelProjection.resolution == "unresolved",
                RegulatoryDerivedLabelProjection.unresolved_reason
                == "source_label_result_unavailable",
            )
        ):
            target.resolution = "pending"
        run.status, run.stage = "running", "projecting"
        run.lease_generation += 1
        run.lease_token, run.lease_expires_at = uuid4(), now + timedelta(minutes=5)
        session.flush()
        assert run.lease_token is not None
        lease = labeling.RunLease(run_id, run.lease_generation, run.lease_token)
        for _ in range(100):
            if not labeling.has_pending_derived_projections(session, lease):
                break
            labeling.project_derived_labels(session, lease, limit=128)
            session.flush()
        else:
            raise ValueError("Retained abstention projection exceeds its bounded scope")
        status = labeling.final_status(session, lease)
        labeling.release_run(
            session,
            lease,
            status=status,
            stage="finished",
            error=run.error,
            finished=True,
        )
        session.refresh(run)
        for batch in session.scalars(
            select(RegulatoryDocumentIntake).where(
                RegulatoryDocumentIntake.labeling_run_id == run_id,
                RegulatoryDocumentIntake.background_labeling_enabled.is_(True),
            )
        ):
            from onyx.db.document_intake_background_labels import (
                reconcile_background_labels,
            )

            batch.background_label_status = "running"
            reconcile_background_labels(session, batch)
        session.flush()
    return len(plans)
