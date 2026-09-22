"""File ownership and committed read observations.

Each ordinary operation owns a short transaction. Finalize joins the caller's
canonical activation transaction. Lock order: file authority, canonical rows,
then observation clock; never hold the clock over network/model/index work.
All writers must use this authority before acquiring canonical/UserFile locks.
Acquisition/recovery first take the shared transaction acquisition lock, including
when the authority row has not yet been created.
"""

from datetime import datetime, timedelta
from hashlib import sha256
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, defer

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

PUBLICATION_LEASE_TTL = timedelta(minutes=2)


def lock_publication_acquisition(
    session: Session, user_file_id: UUID, *, blocking: bool = True
) -> bool:
    """Serialize acquisition/recovery even before the first authority row exists."""
    key = int.from_bytes(
        sha256(f"file-publication-acquisition:{user_file_id}".encode()).digest()[:8],
        byteorder="big",
        signed=True,
    )
    if blocking:
        session.execute(select(func.pg_advisory_xact_lock(key)))
        return True
    return bool(session.scalar(select(func.pg_try_advisory_xact_lock(key))))


class PublicationLeaseConflict(ValueError):
    """Another live writer owns this file; delivery can retry later."""


class PublicationOwnershipLost(ValueError):
    """This writer must stop; its expired or superseded work is recoverable."""


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
            # Heartbeats and per-ordinal checks need authority, not frozen vectors.
            .options(defer(RegulatoryFilePublication.writer_manifest))
            .where(RegulatoryFilePublication.user_file_id == owner.user_file_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        now = session.scalar(select(func.clock_timestamp()))
        transaction = (session.get_transaction(), session.get_nested_transaction())
        lock_identity = (
            *transaction,
            self.scope_key,
            owner.owner_id,
            owner.fencing_token,
        )
        held_locks = session.info.setdefault("publication_transaction_locks", {})
        if (
            row is None
            or row.scope_key != self.scope_key
            or row.owner_id != owner.owner_id
            or row.fencing_token != owner.fencing_token
            or now is None
            or (
                row.lease_expires_at <= now
                and held_locks.get(owner.user_file_id) != lock_identity
            )
        ):
            raise PublicationOwnershipLost("publication ownership lost or expired")
        # A valid lease becomes exclusive transaction ownership under FOR UPDATE.
        # Heartbeats/takeovers cannot acquire this row until commit or rollback;
        # savepoint rollback must also invalidate authority acquired inside it.
        held_locks[owner.user_file_id] = lock_identity
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
            lock_publication_acquisition(session, user_file_id)
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
                    raise PublicationLeaseConflict(
                        "publication ownership already leased"
                    )
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
            # Importing a large sparse inventory can outlive the initial lease.
            session.flush()
            now = session.scalar(select(func.clock_timestamp()))
            assert isinstance(now, datetime)
            row.lease_expires_at = now + ttl
            result = self._ownership(row)
            session.commit()
            return result

    def renew_in_session(
        self,
        session: Session,
        owner: FileOwnership,
        *,
        ttl: timedelta = PUBLICATION_LEASE_TTL,
    ) -> FileOwnership:
        """Hand off a live lease before releasing validated transaction ownership."""
        if ttl <= timedelta(0) or ttl > timedelta(hours=1):
            raise ValueError("invalid lease duration")
        row = self._locked(session, owner)
        session.flush()
        now = session.scalar(select(func.clock_timestamp()))
        assert isinstance(now, datetime)
        row.lease_expires_at = now + ttl
        return self._ownership(row)

    def heartbeat(self, owner: FileOwnership, *, ttl: timedelta) -> FileOwnership:
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            result = self.renew_in_session(session, owner, ttl=ttl)
            session.commit()
            return result

    def advance_after_publication(self, owner: FileOwnership) -> FileOwnership:
        """A different logical payload requires a fresh token, even in one caller."""
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            row = self._locked(session, owner)
            if row.gate_closed or row.writer_manifest is not None:
                raise ValueError("cannot advance unfinished publication ownership")
            row.fencing_token += 1
            session.commit()
            return self._ownership(row)

    def release(self, owner: FileOwnership) -> None:
        """Release ownership; a closed gate survives expiry, release and takeover."""
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            row = self._locked(session, owner)
            now = session.scalar(select(func.clock_timestamp()))
            assert isinstance(now, datetime)
            row.lease_expires_at = now
            session.commit()

    def allocate(self, owner: FileOwnership, allocation_key: str) -> int:
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            result = self.allocate_in_session(session, owner, allocation_key)
            session.commit()
            return result

    def allocate_in_session(
        self, session: Session, owner: FileOwnership, allocation_key: str
    ) -> int:
        """Reserve inside a short canonical transaction; no independent heartbeat."""
        if not allocation_key:
            raise ValueError("allocation key required")
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
        return allocation.ordinal

    def reserve_existing_ordinals(
        self, owner: FileOwnership, ordinals: tuple[int, ...]
    ) -> None:
        """Adopt positively identified legacy ES IDs before the first fenced write."""
        if any(
            type(value) is not int or value < 0 or value >= 2**63 for value in ordinals
        ):
            raise ValueError("legacy projection ordinal is invalid")
        with get_session_with_tenant(tenant_id=self.scope.tenant_id) as session:
            row = self._locked(session, owner)
            missing = set(ordinals) - set(self._ordinals(session, owner.user_file_id))
            if missing and row.gate_closed:
                raise ValueError(
                    "a staged publication cannot acquire unplanned legacy IDs"
                )
            for ordinal in sorted(missing):
                session.add(
                    RegulatoryPublicationOrdinal(
                        user_file_id=owner.user_file_id,
                        allocation_key=f"legacy-index:{ordinal}",
                        ordinal=ordinal,
                    )
                )
                row.next_ordinal = max(row.next_ordinal, ordinal + 1)
            session.commit()

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

    def lock_clock(self, session: Session) -> RegulatoryPublicationClock:
        """Serialize a short local transition; never retain this lock over I/O."""
        self._check_session(session)
        session.execute(
            insert(RegulatoryPublicationClock)
            .values(scope_key=self.scope_key, epoch=0)
            .on_conflict_do_nothing()
        )
        return session.scalars(
            select(RegulatoryPublicationClock)
            .where(RegulatoryPublicationClock.scope_key == self.scope_key)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).one()

    def _event(self, session: Session, row: RegulatoryFilePublication) -> None:
        clock = self.lock_clock(session)
        from onyx.db.regulatory_physical_indexes import require_physical_index_available

        require_physical_index_available(session)
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
        """Files this snapshot must not serve.

        Two things make a file unsafe to read: it is mid-publication, or it has
        moved past the snapshot this read is pinned to. Owning the publication
        is a third, separate question, and answering it here hides content for
        the wrong reason: the scope key folds in how a process dialled the
        database, so an API server and a worker on one deployment can derive
        different keys for the rows describing the very publication they are
        both reading, and every file then looks foreign and disappears.

        Ownership is still enforced wherever a file is acquired or finalized,
        which is where writing under another scope actually matters. Epochs are
        only comparable inside the scope that issued them, so a foreign row's
        epoch is not tested against this observation — but a closed gate is
        honoured whoever closed it, because the file is being rewritten
        underneath this read either way.
        """

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
                if row.gate_closed
                or (
                    row.scope_key == self.scope_key
                    and row.epoch > observation.committed_epoch
                )
            )

    def lock_public_read(
        self,
        session: Session,
        observation: ReadObservation,
        candidate_files: tuple[UUID, ...],
    ) -> bool:
        """Order a short local save/finalization before any subsequent publication."""
        self._check_session(session)
        if observation.scope != self.scope:
            raise ValueError("read observation scope mismatch")
        rows = list(
            session.scalars(
                select(RegulatoryFilePublication)
                .where(RegulatoryFilePublication.user_file_id.in_(candidate_files))
                .order_by(RegulatoryFilePublication.user_file_id)
                .with_for_update(read=True)
                .execution_options(populate_existing=True)
            )
        )
        session.execute(
            insert(RegulatoryPublicationClock)
            .values(scope_key=self.scope_key, epoch=0)
            .on_conflict_do_nothing()
        )
        session.execute(
            select(RegulatoryPublicationClock)
            .where(RegulatoryPublicationClock.scope_key == self.scope_key)
            .with_for_update(read=True)
        ).one()
        rows = list(
            session.scalars(
                select(RegulatoryFilePublication)
                .where(RegulatoryFilePublication.user_file_id.in_(candidate_files))
                .execution_options(populate_existing=True)
            )
        )
        return not any(
            row.scope_key != self.scope_key
            or row.gate_closed
            or row.epoch > observation.committed_epoch
            for row in rows
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


def archive_canonical_revisions(
    session: Session, owner: FileOwnership
) -> dict[str, UUID]:
    """Freeze current canonical authority before an owned correction/deletion."""
    from onyx.db.regulatory_annex_changes import capture_canonical_scope
    from onyx.db.regulatory_canonical_revisions import retain_canonical_revisions

    PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
    return retain_canonical_revisions(
        session, capture_canonical_scope(session, owner.user_file_id)
    )
