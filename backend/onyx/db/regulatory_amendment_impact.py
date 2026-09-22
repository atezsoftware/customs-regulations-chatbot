"""Read-only source-usage inspection; no leases, preview writes or model calls."""

from collections.abc import Callable
from datetime import date
from typing import cast
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryContextSnapshot, RegulatoryTemporalProjection
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.document_index.publication_models import FileOwnership, publication_digest
from onyx.regulatory.amendment_projection_impact import (
    ContextImpactResult,
    find_source_consumers,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    ContextSourceSnapshot,
    FrozenContextProjection,
)
from onyx.regulatory.writer_publication_models import (
    AmendmentConsumer,
    AmendmentSourceUsage,
)


def resolve_context_impact_audit(
    owner: FileOwnership,
    request_sha256: str,
    generate: Callable[[], ContextImpactResult],
) -> ContextImpactResult:
    """Checkpoint only validated decisions; no provider call holds a DB lock."""
    from sqlalchemy.dialects.postgresql import insert

    from onyx.db.engine.sql_engine import get_session_with_tenant
    from onyx.db.models import KVStore
    from onyx.db.regulatory_publication import PublicationStore

    key = f"regulatory-impact-v1:{owner.user_file_id}:{request_sha256}"
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        stored = session.get(KVStore, key)
        if stored is not None:
            if not isinstance(stored.value, dict):
                raise ValueError("context impact checkpoint changed")
            value = cast(dict[str, JsonValue], stored.value)
            if publication_digest(value["result"]) != value["sha256"]:
                raise ValueError("context impact checkpoint changed")
            return ContextImpactResult.model_validate(value["result"])
    result = generate()
    payload = result.model_dump(mode="json")
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        session.execute(
            insert(KVStore)
            .values(
                key=key,
                value={"result": payload, "sha256": publication_digest(payload)},
            )
            .on_conflict_do_nothing(index_elements=[KVStore.key])
        )
        session.commit()
    return result


def inspect_amendment_source_usage(
    session: Session,
    *,
    user_file_id: UUID,
    source_ids: set[str],
    as_of_date: date,
    index_uuid: str | None = None,
) -> AmendmentSourceUsage:
    """Requires a fresh transaction; includes input candidates without calling them affected."""
    session.execute(text("SET TRANSACTION READ ONLY"))
    canonical = capture_canonical_scope(session, user_file_id)
    rows = [
        r
        for r in canonical
        if (r.validity_start_date or date.min)
        <= as_of_date
        < (r.validity_end_date or date.max)
    ]
    visible_ids = {r.id for r in rows}
    if not source_ids <= visible_ids:
        raise ValueError("selected source is not visible on inspection date")
    consumers = [
        c
        for c in find_source_consumers(canonical, source_ids)
        if c.consumer_id in visible_ids
    ]
    snapshots = {
        s.sha256: s
        for payload in session.scalars(
            select(RegulatoryContextSnapshot.payload).where(
                RegulatoryContextSnapshot.user_file_id == user_file_id
            )
        )
        if (s := ContextSourceSnapshot.model_validate(payload))
    }
    bindings = session.execute(
        select(
            RegulatoryTemporalProjection.canonical_chunk_id,
            RegulatoryTemporalProjection.payload["context"],
            RegulatoryTemporalProjection.index_uuid,
        ).where(
            RegulatoryTemporalProjection.user_file_id == user_file_id,
            RegulatoryTemporalProjection.retired_at.is_(None),
            (RegulatoryTemporalProjection.effective_start.is_(None))
            | (RegulatoryTemporalProjection.effective_start <= as_of_date),
            (RegulatoryTemporalProjection.effective_end.is_(None))
            | (RegulatoryTemporalProjection.effective_end > as_of_date),
        )
    ).all()
    indexes = {r[2] for r in bindings}
    if index_uuid is None:
        if len(indexes) > 1:
            raise ValueError("multiple physical indexes; select --index-uuid")
        index_uuid = next(iter(indexes), None)
    bindings = [r for r in bindings if r[2] == index_uuid]
    warnings: list[str] = []
    contextual_ids: set[str] = set()
    for identifier, payload, _ in bindings:
        if payload is None:
            warnings.append(f"context provenance unavailable: {identifier}")
            continue
        context = FrozenContextProjection.model_validate(payload)
        if not context.doc_summary.strip() and not context.chunk_context.strip():
            continue
        contextual_ids.add(identifier)
        snapshot = snapshots.get(context.source_snapshot_sha256)
        if snapshot is None:
            warnings.append(f"context source snapshot unavailable: {identifier}")
            continue
        sources = {span.canonical_chunk_id for span in snapshot.ordered_ranges}
        consumers.extend(
            AmendmentConsumer(
                source_id=source,
                consumer_id=identifier,
                relation="context_input",
                path=[source, identifier],
            )
            for source in sorted(source_ids & sources)
            if source != identifier
        )
    if not bindings:
        warnings.append("No stored temporal context evidence; canonical usage only.")
    return AmendmentSourceUsage(
        user_file_id=user_file_id,
        index_uuid=index_uuid,
        as_of_date=as_of_date,
        source_ids=sorted(source_ids),
        canonical_sha256=context_hash([r.model_dump(mode="json") for r in canonical]),
        canonical_count=len(rows),
        consumers=list({r.model_dump_json(): r for r in consumers}.values()),
        contextual_consumer_count=len(contextual_ids),
        warnings=sorted(set(warnings)),
    )
