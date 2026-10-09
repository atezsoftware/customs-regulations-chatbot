"""LC catalogue publication checks select authority scalars without wide JSON rows."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.legal_composite_preparation import (
    load_prepared_source_lane_catalogue as load_default_catalogue,
)
from onyx.db.legal_composite_sources import SourceLaneCatalogue
from onyx.db.models import RegulatoryFilePublication, User
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import PublicationScope, ReadObservation
from onyx.regulatory.publication_reads import public_read_store

_STAGES = frozenset(
    {
        "scope_validation_seconds",
        "inventory_query_seconds",
        "file_acl_seconds",
        "user_acl_seconds",
        "publication_observation_seconds",
        "publication_guard_seconds",
        "publication_session_scope_seconds",
        "publication_scalar_query_seconds",
        "inventory_page_seconds",
        "prepared_records_seconds",
        "catalogue_build_seconds",
        "catalogue_total_seconds",
    }
)


@dataclass
class CatalogueTimings:
    _seconds: dict[str, float] = field(default_factory=dict)
    _pages: int = 0

    def record(self, stage: str, seconds: float) -> None:
        if stage not in _STAGES or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid aggregate catalogue timing")
        total = self._seconds.get(stage, 0.0) + seconds
        if not math.isfinite(total):
            raise ValueError("Invalid aggregate catalogue timing")
        self._seconds[stage] = total
        if stage == "inventory_page_seconds":
            self._pages += 1

    def snapshot(self) -> dict[str, JsonValue]:
        return {
            "capture_mode": "legal_composite_catalogue_scalar_publication_v1",
            "inventory_page_count": self._pages,
            "durations_overlap": True,
            "stage_seconds": dict(self._seconds),
        }

    def safe_summary(self) -> str:
        def seconds(stage: str) -> float:
            return self._seconds.get(stage, 0.0)

        def formatted(value: float) -> str:
            if not math.isfinite(value):
                raise ValueError("Invalid aggregate catalogue timing")
            fixed = f"{value:.2f}"
            return fixed if len(fixed) <= 10 else f"{value:.2e}"

        values = (
            ("total", seconds("catalogue_total_seconds")),
            ("inventory", seconds("inventory_page_seconds")),
            ("acl", seconds("file_acl_seconds") + seconds("user_acl_seconds")),
            ("pub", seconds("publication_guard_seconds")),
            ("types", seconds("prepared_records_seconds")),
        )
        return (
            "catalogue "
            + " ".join(f"{name}={formatted(value)}s" for name, value in values)
            + f" pages={self._pages}"
        )[:160]


class _ScalarPublicationStore(PublicationStore):
    def __init__(self, scope: PublicationScope, timings: CatalogueTimings) -> None:
        super().__init__(scope)
        self._timings = timings

    def unavailable(
        self, observation: ReadObservation, candidate_files: tuple[UUID, ...]
    ) -> frozenset[UUID]:
        if observation.scope != self.scope:
            raise ValueError("read observation scope mismatch")
        started = time.perf_counter()
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            self._check_session(session)
            self._timings.record(
                "publication_session_scope_seconds", time.perf_counter() - started
            )
            started = time.perf_counter()
            rows = session.execute(
                select(
                    RegulatoryFilePublication.user_file_id,
                    RegulatoryFilePublication.scope_key,
                    RegulatoryFilePublication.gate_closed,
                    RegulatoryFilePublication.epoch,
                ).where(RegulatoryFilePublication.user_file_id.in_(candidate_files))
            ).all()
            self._timings.record(
                "publication_scalar_query_seconds", time.perf_counter() - started
            )
            return frozenset(
                row.user_file_id
                for row in rows
                if row.gate_closed
                or (
                    row.scope_key == self.scope_key
                    and row.epoch > observation.committed_epoch
                )
            )


def load_prepared_source_lane_catalogue(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    check_active: Callable[[], None],
    timings: CatalogueTimings | None = None,
) -> SourceLaneCatalogue:
    """Opt in only this catalogue load; later shared public-read guards remain unchanged."""
    measured = timings if timings is not None else CatalogueTimings()
    started = time.perf_counter()
    store = _ScalarPublicationStore(public_read_store().scope, measured)
    catalogue = load_default_catalogue(
        session,
        user=user,
        filters=filters,
        check_active=check_active,
        publication_store=store,
        record_timing=measured.record,
    )
    measured.record("catalogue_total_seconds", time.perf_counter() - started)
    return catalogue
