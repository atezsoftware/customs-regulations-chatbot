"""Scope untouched default chunk labels to the four labeling vocabularies.

Revision ID: b27e6a4c1d90
Revises: 9f3a7c2e5d18
"""

import json
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "b27e6a4c1d90"
down_revision = "9f3a7c2e5d18"
branch_labels = None
depends_on = None

_LEGACY_DEFAULT_HASH = (
    "5a89e4d393c2974a900e15bb57914814633e70ce65b0f5f39f02b7d24b7b50bd"
)
_CHUNK_DEFAULT_HASH = "6ff25f4865bcd1107dd7f7dc51120476f05339327498d5d9a2b422b130193193"


def upgrade() -> None:
    connection = op.get_bind()
    taxonomy = sa.table(
        "regulatory_label_taxonomy",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("name", sa.String()),
        sa.column("version_hash", sa.String()),
        sa.column("definition", postgresql.JSONB()),
        sa.column("label_count", sa.Integer()),
    )
    settings = sa.table(
        "regulatory_label_settings",
        sa.column("id", sa.Integer()),
        sa.column("taxonomy_id", postgresql.UUID(as_uuid=True)),
        sa.column("revision", sa.Integer()),
        sa.column("updated_at", sa.DateTime(timezone=True)),
        sa.column("updated_by_id", postgresql.UUID(as_uuid=True)),
    )
    untouched_id = connection.execute(
        sa.select(settings.c.id)
        .select_from(settings.join(taxonomy, settings.c.taxonomy_id == taxonomy.c.id))
        .where(
            settings.c.id == 1,
            settings.c.revision == 1,
            settings.c.updated_by_id.is_(None),
            taxonomy.c.version_hash == _LEGACY_DEFAULT_HASH,
        )
        .with_for_update(of=settings)
    ).scalar_one_or_none()
    if untouched_id is None:
        return

    # Versioned migration assets must remain immutable for future tenant upgrades.
    seed_path = (
        Path(__file__).resolve().parents[2]
        / "onyx/regulatory/labeling/data/tariff-regulatory-intelligence-chunk-labels-v1.json"
    )
    definition = json.loads(seed_path.read_text(encoding="utf-8"))
    version_hash = sha256(
        json.dumps(
            definition, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if version_hash != _CHUNK_DEFAULT_HASH:
        raise ValueError("The immutable chunk-label migration seed was changed")
    connection.execute(
        postgresql.insert(taxonomy)
        .values(
            id=uuid4(),
            name=definition["name"],
            version_hash=version_hash,
            definition=definition,
            label_count=len(definition["labels"]),
        )
        .on_conflict_do_nothing(index_elements=["version_hash"])
    )
    taxonomy_id = connection.execute(
        sa.select(taxonomy.c.id).where(taxonomy.c.version_hash == version_hash)
    ).scalar_one()
    connection.execute(
        settings.update()
        .where(settings.c.id == untouched_id)
        .values(
            taxonomy_id=taxonomy_id,
            revision=settings.c.revision + 1,
            updated_at=sa.func.now(),
        )
    )


def downgrade() -> None:
    # Preserve historical snapshots and edits; never restore excluded vocabularies.
    pass
