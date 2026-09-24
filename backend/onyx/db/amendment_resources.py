"""Lease-fenced resource deferrals, separate from legal matching failures."""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import AmendmentBatch, KVStore


def owns_analysis(db_session: Session, *, batch_id: int, lease_generation: int) -> bool:
    return (
        db_session.scalar(
            select(AmendmentBatch.id).where(
                AmendmentBatch.id == batch_id,
                AmendmentBatch.lease_generation == lease_generation,
                AmendmentBatch.status == "analyzing",
            )
        )
        is not None
    )


def _key(batch_id: int) -> str:
    return f"amendment_resource_stops:{batch_id}"


def _stops(row: KVStore | None) -> int:
    if row is None:
        return 0
    value = row.value
    stops = (
        cast(Mapping[str, object], value).get("stops")
        if isinstance(value, Mapping)
        else None
    )
    if not isinstance(stops, int) or isinstance(stops, bool) or stops < 0:
        raise ValueError("Invalid amendment resource recovery state")
    return stops


def parallel_analysis_allowed(db_session: Session, batch_id: int) -> bool:
    row = db_session.get(KVStore, _key(batch_id))
    return _stops(row) == 0


def defer_analysis(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    reason: str,
    started: bool,
) -> bool:
    batch = db_session.scalar(
        select(AmendmentBatch).where(AmendmentBatch.id == batch_id).with_for_update()
    )
    if (
        batch is None
        or batch.status != "analyzing"
        or batch.lease_generation != lease_generation
    ):
        db_session.rollback()
        return False
    row = db_session.get(KVStore, _key(batch_id))
    stops = _stops(row)
    stops += int(started)
    if row is None:
        row = KVStore(key=_key(batch_id), value={})
        db_session.add(row)
    row.value = {"stops": stops, "reason": reason}
    # One automatic continuation, in serial mode. Never repeatedly kill/retry
    # a single oversized instruction; subsequent recovery requires review.
    batch.status = "paused" if stops >= 2 else "queued"
    batch.stage = "waiting_resources"
    batch.lease_generation += 1
    batch.heartbeat_at = datetime.now(timezone.utc)
    batch.completed_at = None
    batch.error_message = None
    db_session.commit()
    return True
