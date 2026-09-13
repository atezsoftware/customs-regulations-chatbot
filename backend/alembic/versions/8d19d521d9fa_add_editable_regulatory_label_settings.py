"""add editable regulatory label settings

Revision ID: 8d19d521d9fa
Revises: c8b7a6d5e4f3
Create Date: 2026-09-13 21:14:04.820491

"""

import json
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "8d19d521d9fa"
down_revision = "c8b7a6d5e4f3"
branch_labels = None
depends_on = None


def _settings_already_exists() -> bool:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    schema = connection.scalar(sa.text("SELECT current_schema()"))
    table = "regulatory_label_settings"
    if not inspector.has_table(table, schema=schema):
        return False
    columns = {
        column["name"]: (
            column["type"].compile(dialect=connection.dialect),
            column["nullable"],
        )
        for column in inspector.get_columns(table, schema=schema)
    }
    foreign_keys = {
        (
            tuple(foreign_key["constrained_columns"]),
            foreign_key["referred_table"],
            tuple(foreign_key["referred_columns"]),
            foreign_key.get("options", {}).get("ondelete"),
        )
        for foreign_key in inspector.get_foreign_keys(table, schema=schema)
        if foreign_key["referred_schema"] in (None, schema)
    }
    checks = {
        "".join(check["sqltext"].split()).replace("(", "").replace(")", "")
        for check in inspector.get_check_constraints(table, schema=schema)
    }
    defaults = {
        column["name"]: column.get("default")
        for column in inspector.get_columns(table, schema=schema)
    }
    if (
        columns
        != {
            "id": ("INTEGER", False),
            "taxonomy_id": ("UUID", False),
            "revision": ("INTEGER", False),
            "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
            "updated_by_id": ("UUID", True),
        }
        or inspector.get_pk_constraint(table, schema=schema)["constrained_columns"]
        != ["id"]
        or inspector.get_unique_constraints(table, schema=schema)
        or len(inspector.get_foreign_keys(table, schema=schema)) != 2
        or foreign_keys
        != {
            (("taxonomy_id",), "regulatory_label_taxonomy", ("id",), "RESTRICT"),
            (("updated_by_id",), "user", ("id",), "SET NULL"),
        }
        or checks != {"id=1", "revision>0"}
        or defaults["revision"] != "1"
        or defaults["updated_at"] != "now()"
    ):
        raise RuntimeError("Existing regulatory_label_settings has incompatible schema")
    return True


def _create_settings() -> None:
    op.create_table(
        "regulatory_label_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("taxonomy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("revision", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["taxonomy_id"], ["regulatory_label_taxonomy.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["updated_by_id"], ["user.id"], ondelete="SET NULL"),
        sa.CheckConstraint("id = 1", name="regulatory_label_settings_singleton"),
        sa.CheckConstraint(
            "revision > 0", name="regulatory_label_settings_revision_positive"
        ),
    )


def upgrade() -> None:
    if not _settings_already_exists():
        _create_settings()
    op.add_column(
        "regulatory_labeling_run",
        sa.Column(
            "uses_current_labels",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    # Retain this versioned seed asset for future tenant migrations.
    seed_path = (
        Path(__file__).resolve().parents[2]
        / "onyx/regulatory/labeling/data/tariff-regulatory-intelligence-v2.1.json"
    )
    definition = json.loads(seed_path.read_text(encoding="utf-8"))
    version_hash = sha256(
        json.dumps(
            definition, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if (
        version_hash
        != "5a89e4d393c2974a900e15bb57914814633e70ce65b0f5f39f02b7d24b7b50bd"
    ):
        raise ValueError("The immutable TARIFF v2.1 migration seed was changed")
    taxonomy = sa.table(
        "regulatory_label_taxonomy",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("name", sa.String()),
        sa.column("version_hash", sa.String()),
        sa.column("definition", postgresql.JSONB()),
        sa.column("label_count", sa.Integer()),
    )
    connection = op.get_bind()
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
        sa.text(
            "UPDATE regulatory_labeling_run SET uses_current_labels = true "
            "WHERE taxonomy_id = :taxonomy_id"
        ),
        {"taxonomy_id": taxonomy_id},
    )
    settings = sa.table(
        "regulatory_label_settings",
        sa.column("id", sa.Integer()),
        sa.column("taxonomy_id", postgresql.UUID(as_uuid=True)),
    )
    connection.execute(
        postgresql.insert(settings)
        .values(id=1, taxonomy_id=taxonomy_id)
        .on_conflict_do_nothing(index_elements=["id"])
    )


def downgrade() -> None:
    op.drop_table("regulatory_label_settings")
    op.drop_column("regulatory_labeling_run", "uses_current_labels")
