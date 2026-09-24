"""Lease-fenced resource deferrals, separate from legal matching failures."""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.db.models import AmendmentBatch, KVStore
from onyx.utils.special_types import JSON_ro


def record_analysis_resources(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    measurements: dict[str, int],
) -> None:
    """Retain one numeric snapshot, fenced against a superseded analysis."""
    db_session.execute(text("SET LOCAL lock_timeout = '200ms'"))
    db_session.execute(text("SET LOCAL statement_timeout = '500ms'"))
    batch = db_session.scalar(
        select(AmendmentBatch.id)
        .where(
            AmendmentBatch.id == batch_id,
            AmendmentBatch.status == "analyzing",
            AmendmentBatch.lease_generation == lease_generation,
        )
        .with_for_update()
    )
    if batch is None:
        db_session.rollback()
        return
    key = f"amendment_runtime:{batch_id}"
    row = db_session.get(KVStore, key)
    if row is None:
        row = KVStore(key=key, value={})
        db_session.add(row)
    previous: Mapping[str, JSON_ro] = (
        cast(Mapping[str, JSON_ro], row.value) if isinstance(row.value, Mapping) else {}
    )
    if previous.get("lease_generation") != lease_generation:
        previous = {}
    previous_peak = previous.get("peak_bytes", 0)
    row.value = {
        **previous,
        **measurements,
        "peak_bytes": max(
            previous_peak if isinstance(previous_peak, int) else 0,
            measurements.get("peak_bytes", 0),
        ),
        "lease_generation": lease_generation,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    db_session.commit()


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
