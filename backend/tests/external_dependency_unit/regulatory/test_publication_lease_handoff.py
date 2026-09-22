"""Long frozen-manifest reads retain fenced ownership across transactions."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from onyx.db import regulatory_publication, regulatory_writer_publication
from onyx.db.engine.sql_engine import SqlEngine, get_sqlalchemy_engine
from onyx.db.models import RegulatoryFilePublication
from onyx.db.regulatory_publication import PublicationOwnershipLost, PublicationStore
from onyx.document_index.publication_models import (
    PublicationVerification,
    publication_digest,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from tests.external_dependency_unit.regulatory.test_revision_batching import (
    owned_canonical,
)
from tests.unit.onyx.document_index.elasticsearch.test_observed_publication import (
    observed,
)


@pytest.mark.parametrize("phase", ["read", "stage", "finalize"])
@pytest.mark.parametrize("expire_during_read", [False, True])
def test_frozen_manifest_read_hands_back_live_ownership(
    monkeypatch: pytest.MonkeyPatch, expire_during_read: bool, phase: str
) -> None:
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    with get_sqlalchemy_engine().connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                bind=connection, join_transaction_mode="create_savepoint"
            ) as seed:
                owner, _ = owned_canonical(seed, 0)
                owner = owner.model_copy(
                    update={
                        "scope": owner.scope.model_copy(
                            update={"environment": "lease-handoff-" + str(uuid4())}
                        )
                    }
                )
                manifest = WriterPublicationManifest(
                    id=uuid4(),
                    scope=owner.scope,
                    user_file_id=owner.user_file_id,
                    kind="baseline",
                    canonical_before_sha256=publication_digest([]),
                    indexes=[observed().observed_index],
                    previous_binding_ids=[],
                    bindings=[],
                )
                payload = manifest.model_dump(mode="json")
                seed.execute(
                    update(RegulatoryFilePublication)
                    .where(RegulatoryFilePublication.user_file_id == owner.user_file_id)
                    .values(
                        scope_key=publication_digest(
                            owner.scope.model_dump(mode="json")
                        ),
                        gate_closed=phase == "finalize",
                        writer_manifest=payload,
                        writer_manifest_sha256=publication_digest(payload),
                        lease_expires_at=func.clock_timestamp() + timedelta(seconds=3),
                    )
                )
                seed.commit()

            active: list[Session] = []

            @contextmanager
            def sessions(*, tenant_id: str) -> Generator[Session, None, None]:
                assert tenant_id == "public"
                with Session(
                    bind=connection, join_transaction_mode="create_savepoint"
                ) as session:
                    active.append(session)
                    try:
                        yield session
                    finally:
                        active.pop()

            monkeypatch.setattr(
                regulatory_publication, "get_session_with_tenant", sessions
            )
            monkeypatch.setattr(
                regulatory_writer_publication, "get_session_with_tenant", sessions
            )
            validate = WriterPublicationManifest.model_validate
            dump = WriterPublicationManifest.model_dump
            delayed = False

            def delay() -> None:
                nonlocal delayed
                if expire_during_read and not delayed:
                    active[-1].execute(select(func.pg_sleep(3.1)))
                    delayed = True

            def slow_validation(value: Any) -> WriterPublicationManifest:
                result = validate(value)
                delay()
                return result

            def slow_dump(
                self: WriterPublicationManifest, **kwargs: Any
            ) -> dict[str, Any]:
                result = dump(self, **kwargs)
                delay()
                return result

            authority = PublicationStore(owner.scope)
            if phase == "read":
                monkeypatch.setattr(
                    WriterPublicationManifest, "model_validate", slow_validation
                )
                assert (
                    regulatory_writer_publication.pending_writer_manifest(owner)
                    == manifest
                )
            else:
                monkeypatch.setattr(WriterPublicationManifest, "model_dump", slow_dump)
                if phase == "stage":
                    regulatory_writer_publication.stage_writer_publication(
                        owner, manifest
                    )
                else:
                    proof = PublicationVerification(
                        reservations=authority.reservations(owner),
                        index=manifest.indexes[0],
                        live_ordinals=(),
                        canonical_chunk_ids=frozenset(),
                        manifest_sha256="a" * 64,
                    )
                    regulatory_writer_publication.finalize_writer_publication(
                        owner, manifest, [proof]
                    )
                    owner = authority.advance_after_publication(owner)
                    assert owner.fencing_token == 2
            assert (
                authority.reservations(owner).ownership.fencing_token
                == owner.fencing_token
            )
            # A new transaction cannot resurrect already expired ownership.
            with sessions(tenant_id="public") as session:
                session.execute(
                    update(RegulatoryFilePublication)
                    .where(RegulatoryFilePublication.user_file_id == owner.user_file_id)
                    .values(
                        lease_expires_at=func.clock_timestamp() - timedelta(seconds=1)
                    )
                )
                session.commit()
            with pytest.raises(PublicationOwnershipLost):
                authority.heartbeat(owner, ttl=timedelta(minutes=2))
        finally:
            transaction.rollback()
