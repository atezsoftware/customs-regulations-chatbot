"""Read-only label overlay. All candidate IDs still require indexed ACL filtering."""

from collections import defaultdict
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date
from itertools import zip_longest
from time import monotonic
from uuid import UUID

from sqlalchemy import func, or_, select, text, tuple_
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.orm import Session, aliased, load_only, selectinload

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelSettings,
)
from onyx.db.regulatory_labeling import (
    current_derived_label_sources_for_search,
    current_labeling_items_for_search,
)
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    validate_labeling_response,
)
from onyx.regulatory.labeling.search_models import (
    LabelSearchMode,
    LabelSearchOverlay,
    LabelSearchSnapshot,
    SearchLabelEvidence,
)
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

logger = setup_logger()

MAX_OVERLAY_ITEMS = 64
MAX_LABEL_CANDIDATES = 32
MAX_DERIVED_ITEMS = 8
TERMINAL_RUN_STATUSES = ("completed", "completed_with_errors")


@contextmanager
def label_read_session() -> Generator[Session, None, None]:
    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        session.execute(text("SET LOCAL statement_timeout = '1500ms'"))
        with session.no_autoflush:
            yield session


def load_search_snapshot(
    session: Session,
    *,
    tenant_id: str,
    run_ids: tuple[UUID, ...],
    mode: LabelSearchMode,
    document_set_id: int | None = None,
) -> LabelSearchSnapshot | None:
    if mode == "off" or not run_ids or len(run_ids) > 32:
        return None
    runs = list(
        session.scalars(
            select(RegulatoryLabelingRun)
            .options(
                load_only(
                    RegulatoryLabelingRun.id,
                    RegulatoryLabelingRun.document_set_id,
                    RegulatoryLabelingRun.taxonomy_id,
                    RegulatoryLabelingRun.status,
                    RegulatoryLabelingRun.stage,
                    RegulatoryLabelingRun.created_at,
                ),
                selectinload(RegulatoryLabelingRun.taxonomy),
            )
            .where(RegulatoryLabelingRun.id.in_(run_ids))
        )
    )
    if len(runs) != len(set(run_ids)) or any(
        run.status not in TERMINAL_RUN_STATUSES or run.stage != "finished"
        for run in runs
    ):
        return None
    if document_set_id is not None and any(
        run.document_set_id != document_set_id for run in runs
    ):
        return None
    if len({run.taxonomy_id for run in runs}) != 1:
        return None
    settings = session.get(RegulatoryLabelSettings, 1)
    if settings is not None and settings.taxonomy_id != runs[0].taxonomy_id:
        return None
    taxonomy = TaxonomyDefinition.model_validate(runs[0].taxonomy.definition)
    if taxonomy.version_hash != runs[0].taxonomy.version_hash:
        return None
    return LabelSearchSnapshot(
        tenant_id=tenant_id,
        run_ids=tuple(dict.fromkeys(run_ids)),
        taxonomy=taxonomy,
        mode=mode,
        document_set_id=document_set_id,
    )


def load_document_set_search_snapshot(
    session: Session,
    *,
    tenant_id: str,
    document_set_id: int,
) -> LabelSearchSnapshot | None:
    """Reuse finalized current-taxonomy labels in an already authorized source scope."""
    run_ids = tuple(
        session.scalars(
            select(RegulatoryLabelingRun.id)
            .join(
                RegulatoryLabelSettings,
                RegulatoryLabelSettings.taxonomy_id
                == RegulatoryLabelingRun.taxonomy_id,
            )
            .where(
                RegulatoryLabelSettings.id == 1,
                RegulatoryLabelingRun.document_set_id == document_set_id,
                RegulatoryLabelingRun.status.in_(TERMINAL_RUN_STATUSES),
                RegulatoryLabelingRun.stage == "finished",
            )
            .order_by(RegulatoryLabelingRun.created_at.desc())
            .limit(33)
        )
    )
    if not run_ids or len(run_ids) > 32:
        return None
    return load_search_snapshot(
        session,
        tenant_id=tenant_id,
        run_ids=run_ids,
        mode="hybrid",
        document_set_id=document_set_id,
    )


def _select_items(
    session: Session,
    snapshot: LabelSearchSnapshot,
    chunk_ids: Sequence[str],
    labels: Sequence[str],
    candidate_chunk_ids: Sequence[str] | None = None,
) -> list[RegulatoryLabelingItem]:
    statement = (
        select(RegulatoryLabelingItem)
        .options(
            load_only(
                RegulatoryLabelingItem.run_id,
                RegulatoryLabelingItem.regulatory_chunk_id,
                RegulatoryLabelingItem.user_file_id,
                RegulatoryLabelingItem.status,
                RegulatoryLabelingItem.text_snapshot,
                RegulatoryLabelingItem.source_snapshot,
                RegulatoryLabelingItem.canonical_text_sha256,
                RegulatoryLabelingItem.context_sha256,
                RegulatoryLabelingItem.labels,
                RegulatoryLabelingItem.assignments,
            )
        )
        .where(
            RegulatoryLabelingItem.run_id.in_(snapshot.run_ids),
            RegulatoryLabelingItem.status == "completed",
        )
        .order_by(RegulatoryLabelingItem.regulatory_chunk_id)
    )
    if len(snapshot.run_ids) > 1:
        newer_item = aliased(RegulatoryLabelingItem)
        newer_run = aliased(RegulatoryLabelingRun)
        newer_result = (
            select(newer_item.id)
            .join(
                newer_run,
                newer_run.id == newer_item.run_id,
            )
            .where(
                newer_item.regulatory_chunk_id
                == RegulatoryLabelingItem.regulatory_chunk_id,
                newer_item.run_id.in_(snapshot.run_ids),
                tuple_(newer_run.created_at, newer_run.id)
                > tuple_(RegulatoryLabelingRun.created_at, RegulatoryLabelingRun.id),
            )
            .correlate(RegulatoryLabelingItem, RegulatoryLabelingRun)
            .exists()
        )
        # Newer results suppress older ones regardless of their labels or status.
        statement = statement.join(
            RegulatoryLabelingRun,
            RegulatoryLabelingRun.id == RegulatoryLabelingItem.run_id,
        ).where(~newer_result)
    baseline = list(
        session.scalars(
            statement.where(RegulatoryLabelingItem.regulatory_chunk_id.in_(chunk_ids))
            .order_by(None)
            .order_by(
                func.array_position(
                    array(list(chunk_ids)),
                    RegulatoryLabelingItem.regulatory_chunk_id,
                )
            )
            .limit(MAX_OVERLAY_ITEMS)
        )
    )
    if not labels:
        return baseline
    extra_statement = statement.where(
        RegulatoryLabelingItem.regulatory_chunk_id.not_in(chunk_ids),
        RegulatoryLabelingItem.labels.has_any(array(list(labels))),
    )
    if candidate_chunk_ids is not None:
        if not candidate_chunk_ids:
            return baseline
        extra_statement = (
            extra_statement.where(
                RegulatoryLabelingItem.regulatory_chunk_id.in_(candidate_chunk_ids)
            )
            .order_by(None)
            .order_by(
                func.array_position(
                    array(list(candidate_chunk_ids)),
                    RegulatoryLabelingItem.regulatory_chunk_id,
                )
            )
        )
    additions = list(session.scalars(extra_statement.limit(MAX_LABEL_CANDIDATES)))
    # Alternate lanes so a shared validation deadline cannot reserve every
    # context window for baseline results before considering relevant additions.
    return [
        item
        for pair in zip_longest(baseline, additions)
        for item in pair
        if item is not None
    ]


def _validate_items(
    session: Session,
    *,
    runs: Mapping[UUID, RegulatoryLabelingRun],
    items: Sequence[RegulatoryLabelingItem],
    baseline_ids: set[str],
    deadline: float,
) -> set[str]:
    selected_runs = {item.run_id for item in items}
    if len(selected_runs) == 1:
        run_id = next(iter(selected_runs))
        return set(
            current_labeling_items_for_search(
                session, run=runs[run_id], items=items, deadline=deadline
            )
        )
    baseline: dict[UUID, list[RegulatoryLabelingItem]] = defaultdict(list)
    extras: dict[UUID, list[RegulatoryLabelingItem]] = defaultdict(list)
    for item in items:
        lane = baseline if item.regulatory_chunk_id in baseline_ids else extras
        lane[item.run_id].append(item)
    now = monotonic()
    extra_deadline = now + max(0.0, deadline - now) / 2
    valid: set[str] = set()
    # Reserve half the remaining cooperative budget for relevant additions,
    # even when they belong to a different run. Unused time returns to baseline.
    for groups, lane_deadline in ((extras, extra_deadline), (baseline, deadline)):
        for run_id, group in groups.items():
            if monotonic() >= lane_deadline:
                break
            valid.update(
                current_labeling_items_for_search(
                    session, run=runs[run_id], items=group, deadline=lane_deadline
                )
            )
    return valid


def load_label_overlay(
    session: Session,
    *,
    snapshot: LabelSearchSnapshot,
    chunk_ids: Sequence[str],
    candidate_label_ids: Sequence[str],
    as_of_date: date,
    candidate_chunk_ids: Sequence[str] | None = None,
) -> LabelSearchOverlay:
    import json

    deadline = monotonic() + 1.5
    if get_current_tenant_id() != snapshot.tenant_id:
        return LabelSearchOverlay()
    current = load_search_snapshot(
        session,
        tenant_id=snapshot.tenant_id,
        run_ids=snapshot.run_ids,
        mode=snapshot.mode,
        document_set_id=snapshot.document_set_id,
    )
    if (
        current is None
        or current.taxonomy.version_hash != snapshot.taxonomy.version_hash
    ):
        return LabelSearchOverlay()
    ids = tuple(dict.fromkeys(chunk_ids))[:MAX_OVERLAY_ITEMS]
    allowed = {label.id for label in snapshot.taxonomy.labels}
    labels = tuple(label for label in candidate_label_ids if label in allowed)[:12]
    runs = {
        run.id: run
        for run in session.scalars(
            select(RegulatoryLabelingRun)
            .options(
                load_only(
                    RegulatoryLabelingRun.id,
                    RegulatoryLabelingRun.document_set_id,
                    RegulatoryLabelingRun.taxonomy_id,
                    RegulatoryLabelingRun.status,
                    RegulatoryLabelingRun.stage,
                    RegulatoryLabelingRun.created_at,
                )
            )
            .where(RegulatoryLabelingRun.id.in_(snapshot.run_ids))
        )
    }
    projections = list(
        session.scalars(
            select(RegulatoryDerivedLabelProjection)
            .join(
                RegulatoryLabelingRun,
                RegulatoryLabelingRun.id == RegulatoryDerivedLabelProjection.run_id,
            )
            .where(
                RegulatoryDerivedLabelProjection.run_id.in_(snapshot.run_ids),
                RegulatoryDerivedLabelProjection.regulatory_chunk_id.in_(ids),
            )
            .order_by(
                RegulatoryLabelingRun.created_at.desc(), RegulatoryLabelingRun.id.desc()
            )
            .limit(MAX_DERIVED_ITEMS * len(snapshot.run_ids))
        )
    )
    sources: dict[str, tuple[str, ...]] = {}
    projection_runs: dict[str, UUID] = {}
    for target in projections:
        if monotonic() >= deadline:
            break
        if target.regulatory_chunk_id in sources or len(sources) >= MAX_DERIVED_ITEMS:
            continue
        sources[target.regulatory_chunk_id] = current_derived_label_sources_for_search(
            session,
            run=runs[target.run_id],
            target=target,
        )
        projection_runs[target.regulatory_chunk_id] = target.run_id
    all_ids = tuple(
        dict.fromkeys(
            [*ids, *(source for members in sources.values() for source in members)]
        )
    )
    if len(all_ids) > MAX_OVERLAY_ITEMS:
        sources = {}
        all_ids = ids
    items = _select_items(session, snapshot, all_ids, labels, candidate_chunk_ids)
    # Context validation has its own budget after the bounded metadata lookups.
    deadline = monotonic() + 1.5
    valid = _validate_items(
        session, runs=runs, items=items, baseline_ids=set(all_ids), deadline=deadline
    )
    rows = {
        row.id: row
        for row in session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.id.in_(valid | set(sources)),
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of_date,
                ),
            )
        )
    }
    evidence: dict[str, tuple[SearchLabelEvidence, ...]] = {}
    item_runs: dict[str, UUID] = {}
    candidates: list[str] = []
    for item in items:
        identifier = item.regulatory_chunk_id
        if (
            identifier not in rows
            or not item.context_sha256
            or not item.canonical_text_sha256
        ):
            continue
        try:
            outcome = validate_labeling_response(
                json.dumps(
                    {
                        "labels": item.assignments,
                        "abstained": False,
                    }
                ),
                text=rows[identifier].text,
                taxonomy=snapshot.taxonomy,
            )
            if set(item.labels) != {
                assignment.label_id for assignment in outcome.labels
            }:
                continue
            evidence[identifier] = tuple(
                SearchLabelEvidence(
                    chunk_id=identifier,
                    source_chunk_id=identifier,
                    label_id=assignment.label_id,
                    evidence_quote=assignment.evidence_quote,
                    source_text_sha256=item.canonical_text_sha256,
                    context_sha256=item.context_sha256,
                    run_id=item.run_id,
                )
                for assignment in outcome.labels
            )
        except ValueError:
            continue
        item_runs[identifier] = item.run_id
        if (
            identifier not in ids
            and set(item.labels).intersection(labels)
            and len(candidates) < MAX_LABEL_CANDIDATES
        ):
            candidates.append(identifier)
    for derived, members in sources.items():
        if (
            derived not in rows
            or not members
            or any(
                member not in evidence or item_runs[member] != projection_runs[derived]
                for member in members
            )
        ):
            continue
        evidence[derived] = tuple(
            entry.model_copy(update={"chunk_id": derived})
            for member in members
            for entry in evidence[member]
        )
    logger.info(
        "Label overlay selected=%d validated=%d candidates=%d deadline_reached=%s",
        len(items),
        len(valid),
        len(candidates),
        monotonic() >= deadline,
    )
    return LabelSearchOverlay(
        evidence_by_chunk=evidence,
        candidate_ids=tuple(candidates),
        source_texts={
            identifier: row.text
            for identifier, row in rows.items()
            if identifier in evidence
        },
    )
