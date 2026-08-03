"""Read-only helpers for checking a tenant schema's migration revision."""

from sqlalchemy import Column, MetaData, String, Table, select
from sqlalchemy.orm import Session

_ALEMBIC_VERSION = Table(
    "alembic_version",
    MetaData(),
    Column("version_num", String(32), nullable=False),
)


def get_database_alembic_heads(db_session: Session) -> frozenset[str]:
    """Return every Alembic head stamped in the current tenant schema.

    The unqualified table intentionally participates in the session connection's
    ``schema_translate_map``. This keeps the check on the same physical shard and
    tenant schema as the importer's subsequent ORM operations.
    """

    return frozenset(db_session.scalars(select(_ALEMBIC_VERSION.c.version_num)).all())
