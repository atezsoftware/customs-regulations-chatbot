"""Lease-fenced resource deferrals, separate from legal matching failures."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from onyx.db.models import AmendmentBatch, KVStore
from onyx.regulatory.amendments.runtime import AmendmentRuntime
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
    checked_at = datetime.now(timezone.utc).isoformat()
    row.value = {
        **previous,
        **measurements,
        "peak_bytes": max(
            previous_peak if isinstance(previous_peak, int) else 0,
            measurements.get("peak_bytes", 0),
        ),
        "lease_generation": lease_generation,
        "checked_at": checked_at,
        **(
            {"memory_checked_at": checked_at} if "current_bytes" in measurements else {}
        ),
        **({"activity_checked_at": checked_at} if "active" in measurements else {}),
    }
    db_session.commit()


@dataclass(frozen=True)
class RuntimeHeader:
    id: int
    document_set_id: int
    status: str
    stage: str
    lease_generation: int
    raw_text_chars: int


def load_runtime_header(db_session: Session, batch_id: int) -> RuntimeHeader | None:
    row = db_session.execute(
        select(
            AmendmentBatch.id,
            AmendmentBatch.document_set_id,
            AmendmentBatch.status,
            AmendmentBatch.stage,
            AmendmentBatch.lease_generation,
            func.char_length(AmendmentBatch.raw_text),
        ).where(AmendmentBatch.id == batch_id)
    ).one_or_none()
    return RuntimeHeader(*row) if row is not None else None


def load_runtime_snapshot(
    db_session: Session, header: RuntimeHeader
) -> AmendmentRuntime:
    from onyx.db.amendment_analysis_settings import load_analysis_model

    row = db_session.get(KVStore, f"amendment_runtime:{header.id}")
    raw_value = row.value if row is not None else None
    value: Mapping[str, object] = (
        cast(Mapping[str, object], raw_value) if isinstance(raw_value, Mapping) else {}
    )
    if value.get("lease_generation") != header.lease_generation:
        value = {}

    def counter(key: str) -> int | None:
        number = value.get(key)
        return (
            number
            if isinstance(number, int) and not isinstance(number, bool) and number >= 0
            else None
        )

    def timestamp(key: str) -> datetime | None:
        candidate = value.get(key)
        if isinstance(candidate, str):
            try:
                parsed = datetime.fromisoformat(candidate)
                if parsed.tzinfo is not None:
                    return parsed
            except ValueError:
                pass
        return None

    return AmendmentRuntime(
        analysis_model=load_analysis_model(db_session, header.id),
        batch_id=header.id,
        status=header.status,
        stage=header.stage,
        lease_generation=header.lease_generation,
        raw_text_chars=header.raw_text_chars,
        current_bytes=counter("current_bytes"),
        limit_bytes=counter("limit_bytes"),
        peak_bytes=counter("peak_bytes"),
        reserve_bytes=counter("reserve_bytes"),
        active=counter("active"),
        peak_active=counter("peak_active"),
        max_parallel=counter("max_parallel"),
        memory_checked_at=timestamp("memory_checked_at"),
        activity_checked_at=timestamp("activity_checked_at"),
        admission_limited=bool(value["admission_limited"])
        if value.get("admission_limited") in (0, 1)
        else None,
        dependency_limited=bool(value["dependency_limited"])
        if value.get("dependency_limited") in (0, 1)
        else None,
        calibrating=bool(value["calibrating"])
        if value.get("calibrating") in (0, 1)
        else None,
    )


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
    # One automatic continuation with bounded concurrency. Never repeatedly kill/retry
    # a single oversized instruction; subsequent recovery requires review.
    batch.status = "paused" if stops >= 2 else "queued"
    batch.stage = "waiting_resources"
    batch.lease_generation += 1
    batch.heartbeat_at = datetime.now(timezone.utc)
    batch.completed_at = None
    batch.error_message = None
    db_session.commit()
    return True
