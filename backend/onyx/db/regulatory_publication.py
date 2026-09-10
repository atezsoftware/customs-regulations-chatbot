"""File ownership and committed read observations.

Each ordinary operation owns a short transaction. Finalize joins the caller's
canonical activation transaction. Lock order: file authority, canonical rows,
then observation clock; never hold the clock over network/model/index work.
All writers must use this authority before acquiring canonical/UserFile locks.
"""

from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryPublicationClock,
    RegulatoryPublicationOrdinal,
    UserFile,
)
from onyx.document_index.publication_models import (
    FileOwnership,
    FileReservations,
    PublicationScope,
    PublicationVerification,
    ReadObservation,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.config import ANNEX_DATABASE_IDENTITY


class PublicationStore:
    def __init__(self, scope: PublicationScope) -> None:
        if scope.database_identity != ANNEX_DATABASE_IDENTITY:
            raise ValueError("publication database scope mismatch")
        self.scope = scope
        self.scope_key = publication_digest(scope.model_dump(mode="json"))

    def _check_session(self, session: Session) -> None:
        connection = session.connection()
        translations = (
            connection.get_execution_options().get("schema_translate_map") or {}
        )
        if translations.get(None, "public") != self.scope.tenant_id:
            raise ValueError("publication session tenant scope mismatch")
        url = connection.engine.url
        if (
            f"{url.host}:{url.port or 5432}/{url.database}"
            != self.scope.database_identity
        ):
            raise ValueError("publication session database scope mismatch")
        if connection.get_isolation_level() != "READ COMMITTED":
            raise ValueError("publication requires READ COMMITTED observations")

    def _locked(
        self, session: Session, owner: FileOwnership
    ) -> RegulatoryFilePublication:
        self._check_session(session)
        if owner.scope != self.scope:
            raise ValueError("publication ownership scope mismatch")
        row = session.scalar(
            select(RegulatoryFilePublication)
            .where(RegulatoryFilePublication.user_file_id == owner.user_file_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        now = session.scalar(select(func.clock_timestamp()))
        if (
            row is None
            or row.scope_key != self.scope_key
            or row.owner_id != owner.owner_id
            or row.fencing_token != owner.fencing_token
            or now is None
            or row.lease_expires_at <= now
        ):
            raise ValueError("publication ownership lost or expired")
        return row

    def _ownership(self, row: RegulatoryFilePublication) -> FileOwnership:
        return FileOwnership(
            scope=self.scope,
            user_file_id=row.user_file_id,
            owner_id=row.owner_id,
            fencing_token=row.fencing_token,
            expires_at=row.lease_expires_at,
        )

    def acquire(
        self, user_file_id: UUID, *, owner_id: UUID, ttl: timedelta
    ) -> FileOwnership:
        if ttl <= timedelta(0) or ttl > timedelta(hours=1):
            raise ValueError("lease duration must be positive and at most one hour")
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            self._check_session(session)
            if session.get(UserFile, user_file_id) is None:
                raise ValueError("publication file scope missing")
            now = session.scalar(select(func.clock_timestamp()))
            assert isinstance(now, datetime)
            session.execute(
                insert(RegulatoryFilePublication)
                .values(
                    user_file_id=user_file_id,
                    scope_key=self.scope_key,
                    owner_id=owner_id,
                    fencing_token=0,
                    lease_expires_at=now,
                    gate_closed=False,
                    epoch=0,
                    next_ordinal=0,
                )
                .on_conflict_do_nothing()
            )
            row = session.scalars(
                select(RegulatoryFilePublication)
                .where(RegulatoryFilePublication.user_file_id == user_file_id)
                .with_for_update()
            ).one()
            now = session.scalar(select(func.clock_timestamp()))
            assert isinstance(now, datetime)
            if row.scope_key != self.scope_key:
                raise ValueError("publication file environment scope mismatch")
            if row.lease_expires_at > now:
                if row.owner_id != owner_id:
                    raise ValueError("publication ownership already leased")
                return self._ownership(row)
            row.owner_id = owner_id
            row.fencing_token += 1
            row.lease_expires_at = now + ttl
            # Import existing sparse identities once; never renumber or recycle them.
            for chunk in session.scalars(
                select(RegulatoryChunk).where(
                    RegulatoryChunk.user_file_id == user_file_id
                )
            ):
                key = "canonical:" + chunk.id
                stored = session.get(RegulatoryPublicationOrdinal, (user_file_id, key))
                if stored is not None and stored.ordinal != chunk.projection_ordinal:
                    raise ValueError("existing canonical ordinal changed")
                if stored is None:
                    session.add(
                        RegulatoryPublicationOrdinal(
                            user_file_id=user_file_id,
                            allocation_key=key,
                            ordinal=chunk.projection_ordinal,
                        )
                    )
                row.next_ordinal = max(row.next_ordinal, chunk.projection_ordinal + 1)
            result = self._ownership(row)
            session.commit()
            return result

    def heartbeat(self, owner: FileOwnership, *, ttl: timedelta) -> FileOwnership:
        if ttl <= timedelta(0) or ttl > timedelta(hours=1):
            raise ValueError("invalid lease duration")
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            row = self._locked(session, owner)
            now = session.scalar(select(func.clock_timestamp()))
            assert isinstance(now, datetime)
            row.lease_expires_at = now + ttl
            result = self._ownership(row)
            session.commit()
            return result

    def release(self, owner: FileOwnership) -> None:
        """Release ownership; a closed gate survives expiry, release and takeover."""
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            row = self._locked(session, owner)
            now = session.scalar(select(func.clock_timestamp()))
            assert isinstance(now, datetime)
            row.lease_expires_at = now
            session.commit()

    def allocate(self, owner: FileOwnership, allocation_key: str) -> int:
        if not allocation_key:
            raise ValueError("allocation key required")
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            row = self._locked(session, owner)
            allocation = session.get(
                RegulatoryPublicationOrdinal, (owner.user_file_id, allocation_key)
            )
            if allocation is None:
                allocation = RegulatoryPublicationOrdinal(
                    user_file_id=owner.user_file_id,
                    allocation_key=allocation_key,
                    ordinal=row.next_ordinal,
                )
                row.next_ordinal += 1
                session.add(allocation)
            result = allocation.ordinal
            session.commit()
            return result

    def _ordinals(self, session: Session, user_file_id: UUID) -> tuple[int, ...]:
        return tuple(
            session.scalars(
                select(RegulatoryPublicationOrdinal.ordinal)
                .where(RegulatoryPublicationOrdinal.user_file_id == user_file_id)
                .order_by(RegulatoryPublicationOrdinal.ordinal)
            )
        )

    def lock_owned_snapshot(
        self, session: Session, owner: FileOwnership
    ) -> FileReservations:
        """Lock authority in caller transaction BEFORE canonical/settings row locks.

        Use this session for activation work, then finalize and commit. Do not call
        independent ownership/heartbeat sessions while retaining this row lock.
        """
        row = self._locked(session, owner)
        return FileReservations(
            ownership=owner,
            ordinals=self._ordinals(session, owner.user_file_id),
            gate_closed=row.gate_closed,
        )

    def lock_recovery_reservations(
        self, session: Session, user_file_id: UUID
    ) -> tuple[int, ...]:
        """Serialize intent recovery without replacing an active writer's lease."""
        self._check_session(session)
        row = session.scalar(
            select(RegulatoryFilePublication)
            .where(RegulatoryFilePublication.user_file_id == user_file_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if row is None or row.scope_key != self.scope_key or not row.gate_closed:
            raise ValueError("publication recovery requires its scoped closed gate")
        return self._ordinals(session, user_file_id)

    def reservations(self, owner: FileOwnership) -> FileReservations:
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            return self.lock_owned_snapshot(session, owner)

    def owned_chunks(self, owner: FileOwnership) -> list[RegulatoryChunk]:
        """Explicit writer-only snapshot; returned rows are detached, never public reads."""
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            self._locked(session, owner)
            rows = list(
                session.scalars(
                    select(RegulatoryChunk)
                    .where(RegulatoryChunk.user_file_id == owner.user_file_id)
                    .order_by(RegulatoryChunk.position)
                )
            )
            session.expunge_all()
            return rows

    def record_event(self, session: Session, owner: FileOwnership) -> None:
        """Close gate in caller transaction; the counter lock must remain until commit."""
        row = self._locked(session, owner)
        row.gate_closed = True
        self._event(session, row)

    def _event(self, session: Session, row: RegulatoryFilePublication) -> None:
        session.execute(
            insert(RegulatoryPublicationClock)
            .values(scope_key=self.scope_key, epoch=0)
            .on_conflict_do_nothing()
        )
        clock = session.scalars(
            select(RegulatoryPublicationClock)
            .where(RegulatoryPublicationClock.scope_key == self.scope_key)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).one()
        clock.epoch += 1
        row.epoch = clock.epoch
        session.flush()

    def close_gate(self, owner: FileOwnership) -> None:
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            self.record_event(session, owner)
            session.commit()

    def observe(self) -> ReadObservation:
        """Call before launching any retrieval, even if candidate files are unknown."""
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            self._check_session(session)
            epoch = session.scalar(
                select(RegulatoryPublicationClock.epoch).where(
                    RegulatoryPublicationClock.scope_key == self.scope_key
                )
            )
            return ReadObservation(scope=self.scope, committed_epoch=epoch or 0)

    def unavailable(
        self, observation: ReadObservation, candidate_files: tuple[UUID, ...]
    ) -> frozenset[UUID]:
        if observation.scope != self.scope:
            raise ValueError("read observation scope mismatch")
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            self._check_session(session)
            rows = session.scalars(
                select(RegulatoryFilePublication).where(
                    RegulatoryFilePublication.user_file_id.in_(candidate_files)
                )
            )
            return frozenset(
                row.user_file_id
                for row in rows
                if row.scope_key != self.scope_key
                or row.gate_closed
                or row.epoch > observation.committed_epoch
            )

    def finalize(
        self,
        session: Session,
        owner: FileOwnership,
        verification: PublicationVerification,
    ) -> None:
        """Join canonical activation transaction AFTER exact ES verification, no commit.

        Caller must validate frozen PRESENT/FUTURE settings and finalize every required
        index. No network work may follow this call before commit/rollback.
        """
        row = self._locked(session, owner)
        if (
            not row.gate_closed
            or verification.reservations.ownership != owner
            or verification.reservations.ordinals
            != self._ordinals(session, owner.user_file_id)
        ):
            raise ValueError("verification does not cover owned reserved inventory")
        row.gate_closed = False
        self._event(session, row)
