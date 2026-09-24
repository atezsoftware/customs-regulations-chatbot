"""Persistent one-shot maintenance claims; a crash never grants an implicit retry."""

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.models import KVStore


def claim_regulatory_maintenance(
    *, tenant_id: str, key: str, expected_database: str, selection_sha256: str
) -> None:
    if not key.startswith("regulatory_maintenance:") or not key.endswith(":claim"):
        raise ValueError("maintenance claim requires its own operational namespace")
    with get_session_with_tenant(tenant_id=tenant_id) as session:
        if session.scalar(text("SELECT current_database()")) != expected_database:
            raise ValueError("maintenance database identity mismatch")
        claimed = session.scalar(
            insert(KVStore)
            .values(key=key, value={"selection_sha256": selection_sha256})
            .on_conflict_do_nothing(index_elements=[KVStore.key])
            .returning(KVStore.key)
        )
        if claimed is None:
            raise ValueError(
                "maintenance was already claimed; inspect before any retry"
            )
        session.commit()
