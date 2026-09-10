"""Immutable canonical authority for corrections and retained publication evidence."""

from datetime import timedelta
from uuid import UUID, uuid4

from onyx.db.engine.sql_engine import get_session_with_tenant
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    owned_file as owned_file,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    store,
)


def test_owned_correction_retains_exact_old_revision_and_stable_identity(
    owned_file: UUID,
) -> None:
    from onyx.db import regulatory_publication
    from onyx.db.regulatory_canonical_revisions import get_canonical_revision

    assert hasattr(regulatory_publication, "archive_canonical_revisions"), (
        "corrections have no immutable canonical revision authority"
    )
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    original = authority.owned_chunks(owner)[0]
    with get_session_with_tenant(tenant_id="public") as session:
        revisions = regulatory_publication.archive_canonical_revisions(session, owner)
        session.commit()
        old_id = revisions[original.id]
        old = get_canonical_revision(session, old_id)
        assert old.snapshot.text == original.text
        assert old.snapshot.id == original.id
        assert old.snapshot.projection_ordinal == original.projection_ordinal
        assert (
            regulatory_publication.archive_canonical_revisions(session, owner)
            == revisions
        )
        session.commit()
        from onyx.db.regulatory_chunks import get_chunk_snapshot_by_id, update_chunk

        current = get_chunk_snapshot_by_id(session, original.id)
        assert current is not None
        update_chunk(current, text="administratively corrected text")
        session.flush()
        corrected = regulatory_publication.archive_canonical_revisions(session, owner)
        session.commit()
        assert corrected[original.id] != old_id
        assert get_canonical_revision(session, old_id).snapshot == old.snapshot
        new = get_canonical_revision(session, corrected[original.id])
        assert new.snapshot.id == old.snapshot.id
        assert new.snapshot.projection_ordinal == old.snapshot.projection_ordinal
        assert new.snapshot.text == "administratively corrected text"
        assert new.snapshot.validity_start_date == old.snapshot.validity_start_date
        assert new.snapshot.validity_end_date == old.snapshot.validity_end_date
    authority.release(owner)
