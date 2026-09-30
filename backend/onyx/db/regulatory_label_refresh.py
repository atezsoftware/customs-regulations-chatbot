"""Durable, file-scoped label refresh after canonical amendment publication."""

import os
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling
from onyx.db.labeling_configuration import (
    LabelingProviderBinding,
    resolve_labeling_provider_binding,
)
from onyx.db.models import (
    AmendmentProposal,
    DocumentSet__UserFile,
    RegulatoryAmendmentLabelRefresh,
    RegulatoryLabelingRun,
    RegulatoryLabelSettings,
    User,
)
from onyx.regulatory.labeling.api_models import AmendmentLabelRefreshSnapshot

_TERMINAL_RUN_STATUSES = frozenset(
    {"completed", "completed_with_errors", "failed", "cancelled"}
)
_MAX_ATTEMPTS = 3


def amendment_label_refresh_enabled() -> bool:
    return (
        os.environ.get("REGULATORY_AMENDMENT_LABEL_REFRESH_ENABLED", "false").lower()
        == "true"
    )


def refresh_snapshot(
    row: RegulatoryAmendmentLabelRefresh,
) -> AmendmentLabelRefreshSnapshot:
    return AmendmentLabelRefreshSnapshot(
        id=str(row.id),
        proposal_id=row.proposal_id,
        file_id=str(row.user_file_id),
        new_chunk_ids=list(row.new_chunk_ids),
        run_id=str(row.run_id) if row.run_id is not None else None,
        status=row.status,
        attempt_count=row.attempt_count,
        next_retry_at=row.next_retry_at,
        error=row.error,
        created_at=row.created_at,
    )


def list_refreshes(
    session: Session, *, document_set_id: int
) -> list[RegulatoryAmendmentLabelRefresh]:
    return list(
        session.scalars(
            select(RegulatoryAmendmentLabelRefresh)
            .where(RegulatoryAmendmentLabelRefresh.document_set_id == document_set_id)
            .order_by(RegulatoryAmendmentLabelRefresh.created_at.desc())
            .limit(100)
        ).all()
    )


def retry_failed_refresh(
    session: Session, *, document_set_id: int, refresh_id: UUID
) -> RegulatoryAmendmentLabelRefresh | None:
    row = session.scalar(
        select(RegulatoryAmendmentLabelRefresh)
        .where(
            RegulatoryAmendmentLabelRefresh.id == refresh_id,
            RegulatoryAmendmentLabelRefresh.document_set_id == document_set_id,
        )
        .with_for_update()
    )
    if row is None:
        return None
    if row.status != "failed":
        raise ValueError("Only a failed amendment label refresh can be retried")
    proposal = session.get(AmendmentProposal, row.proposal_id)
    if proposal is None or proposal.status != "approved":
        raise ValueError("The amendment is no longer approved")
    row.status = "pending"
    row.attempt_count = 0
    row.run_id = None
    row.next_retry_at = None
    row.error = None
    session.flush()
    return row


def record_published_amendment_refresh(
    session: Session, *, proposal_id: int, user_file_id: UUID
) -> None:
    """Append once in the same transaction that marks the proposal approved."""
    if not amendment_label_refresh_enabled():
        return
    proposal = session.get(AmendmentProposal, proposal_id)
    if proposal is None:
        raise ValueError("Label refresh requires a published amendment")
    session.refresh(proposal)
    if proposal.status != "approved":
        raise ValueError("Label refresh requires a published amendment")
    new_chunk_ids = list(
        dict.fromkeys(
            chunk_id
            for chunk_id in (
                proposal.applied_new_chunk_ids or [proposal.applied_new_chunk_id]
            )
            if chunk_id is not None
        )
    )
    if not new_chunk_ids:
        raise ValueError("Published amendment has no new canonical chunks")
    if session.get(RegulatoryLabelSettings, 1) is None:
        return
    document_set_ids = session.scalars(
        select(DocumentSet__UserFile.document_set_id).where(
            DocumentSet__UserFile.user_file_id == user_file_id
        )
    ).all()
    for document_set_id in set(document_set_ids):
        prior_run = session.scalar(
            select(RegulatoryLabelingRun.id)
            .where(RegulatoryLabelingRun.document_set_id == document_set_id)
            .limit(1)
        )
        if prior_run is None:
            continue
        session.execute(
            pg_insert(RegulatoryAmendmentLabelRefresh)
            .values(
                id=uuid4(),
                proposal_id=proposal_id,
                document_set_id=document_set_id,
                user_file_id=user_file_id,
                new_chunk_ids=new_chunk_ids,
                status="pending",
            )
            .on_conflict_do_nothing(
                constraint="uq_regulatory_amendment_label_refresh_proposal_set"
            )
        )


def _retry(refresh: RegulatoryAmendmentLabelRefresh, error: str) -> None:
    refresh.attempt_count += 1
    refresh.run_id = None
    refresh.error = error[:4000]
    if refresh.attempt_count >= _MAX_ATTEMPTS:
        refresh.status = "failed"
        refresh.next_retry_at = None
    else:
        refresh.status = "pending"
        refresh.next_retry_at = datetime.now(timezone.utc) + timedelta(
            seconds=min(60 * 2**refresh.attempt_count, 3600)
        )


def reconcile_finished_refreshes(session: Session, *, limit: int = 32) -> int:
    rows = list(
        session.scalars(
            select(RegulatoryAmendmentLabelRefresh)
            .outerjoin(
                RegulatoryLabelingRun,
                RegulatoryLabelingRun.id == RegulatoryAmendmentLabelRefresh.run_id,
            )
            .where(
                RegulatoryAmendmentLabelRefresh.status == "running",
                or_(
                    RegulatoryAmendmentLabelRefresh.run_id.is_(None),
                    RegulatoryLabelingRun.id.is_(None),
                    RegulatoryLabelingRun.status.in_(_TERMINAL_RUN_STATUSES),
                ),
            )
            .order_by(RegulatoryAmendmentLabelRefresh.created_at)
            .limit(limit)
            .with_for_update(of=RegulatoryAmendmentLabelRefresh, skip_locked=True)
        ).all()
    )
    reconciled = 0
    for refresh in rows:
        run = (
            session.get(RegulatoryLabelingRun, refresh.run_id)
            if refresh.run_id is not None
            else None
        )
        if run is not None and run.status not in _TERMINAL_RUN_STATUSES:
            continue
        reconciled += 1
        if run is not None and run.status == "completed":
            refresh.status = "completed"
            refresh.error = None
            refresh.next_retry_at = None
        else:
            _retry(
                refresh,
                f"Labeling run {run.status if run is not None else 'missing'}",
            )
    return reconciled


def start_next_refresh(session: Session) -> UUID | None:
    now = datetime.now(timezone.utc)
    first = session.scalar(
        select(RegulatoryAmendmentLabelRefresh)
        .where(
            RegulatoryAmendmentLabelRefresh.status == "pending",
            or_(
                RegulatoryAmendmentLabelRefresh.next_retry_at.is_(None),
                RegulatoryAmendmentLabelRefresh.next_retry_at <= now,
            ),
        )
        .order_by(RegulatoryAmendmentLabelRefresh.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if first is None:
        return None
    if regulatory_labeling.get_active_run_id(session, first.document_set_id):
        first.next_retry_at = now + timedelta(minutes=1)
        return None
    grouped = list(
        session.scalars(
            select(RegulatoryAmendmentLabelRefresh)
            .where(
                RegulatoryAmendmentLabelRefresh.status == "pending",
                RegulatoryAmendmentLabelRefresh.document_set_id
                == first.document_set_id,
                RegulatoryAmendmentLabelRefresh.user_file_id == first.user_file_id,
            )
            .order_by(RegulatoryAmendmentLabelRefresh.created_at)
            .with_for_update(skip_locked=True)
        ).all()
    )
    template = session.scalar(
        select(RegulatoryLabelingRun)
        .where(
            RegulatoryLabelingRun.document_set_id == first.document_set_id,
            RegulatoryLabelingRun.model_configuration_id.is_not(None),
            RegulatoryLabelingRun.requested_by_id.is_not(None),
        )
        .order_by(RegulatoryLabelingRun.created_at.desc())
        .limit(1)
    )
    try:
        with session.begin_nested():
            if template is None or template.model_configuration_id is None:
                raise ValueError("No previous labeling model is available")
            user = session.get(User, template.requested_by_id)
            if user is None:
                raise ValueError("The previous labeling requester is unavailable")
            settings = session.get(RegulatoryLabelSettings, 1)
            if settings is None:
                raise ValueError("Current label settings are unavailable")
            binding = resolve_labeling_provider_binding(
                session, template.model_configuration_id, user=user
            )
            previous = LabelingProviderBinding.model_validate(template.provider_binding)
            if binding.fingerprint != previous.fingerprint:
                raise ValueError("The labeling model connection changed")
            run, _ = regulatory_labeling.create_labeling_run(
                session,
                document_set_id=first.document_set_id,
                taxonomy=settings.taxonomy,
                model_configuration_id=template.model_configuration_id,
                model=template.model,
                provider_binding=binding.model_dump(mode="json"),
                requested_by_id=user.id,
                idempotency_key=uuid5(
                    NAMESPACE_URL,
                    f"amendment-label-refresh:{first.id}:{first.attempt_count}",
                ),
                uses_current_labels=True,
                target_file_ids=[first.user_file_id],
            )
    except (ValueError, regulatory_labeling.LabelingStateConflictError) as error:
        for refresh in grouped:
            _retry(refresh, str(error))
        return None
    for refresh in grouped:
        refresh.status = "running"
        refresh.run_id = run.id
        refresh.error = None
        refresh.next_retry_at = None
    return run.id
