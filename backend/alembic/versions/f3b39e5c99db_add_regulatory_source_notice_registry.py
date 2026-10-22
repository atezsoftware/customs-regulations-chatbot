"""Add a catalogue-only regulatory source notice registry.

Revision ID: f3b39e5c99db
Revises: 5da101e73adc
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection

revision = "f3b39e5c99db"
down_revision = "5da101e73adc"
branch_labels = None
depends_on = None


def _validate_precreated_registry(connection: Connection) -> None:
    """Accept the approved manual bootstrap without altering its table or data."""
    inspector = sa.inspect(connection)
    expected = {
        "id": ("UUID", False),
        "document_set_id": ("INTEGER", False),
        "identity_key": ("TEXT", False),
        "instrument_type": ("TEXT", False),
        "instrument_number": ("TEXT", True),
        "instrument_year": ("INTEGER", True),
        "instrument_date": ("DATE", True),
        "catalog_title": ("TEXT", False),
        "aliases": ("JSONB", False),
        "legistation_metadata": ("JSONB", False),
        "provenance": ("JSONB", False),
        "content_state": ("VARCHAR(32)", False),
        "user_file_id": ("UUID", True),
        "revision": ("INTEGER", False),
        "created_at": ("TIMESTAMP WITH TIME ZONE", False),
        "updated_at": ("TIMESTAMP WITH TIME ZONE", False),
    }
    actual = {
        column["name"]: (
            str(column["type"].compile(dialect=connection.dialect)).upper(),
            bool(column["nullable"]),
        )
        for column in inspector.get_columns("regulatory_source_notice")
    }
    if actual != expected:
        raise ValueError(
            "Precreated source notice registry columns differ from this revision"
        )
    if inspector.get_pk_constraint("regulatory_source_notice").get(
        "constrained_columns"
    ) != ["id"]:
        raise ValueError("Precreated source notice registry primary key differs")
    unique = {
        (value["name"], tuple(value["column_names"]))
        for value in inspector.get_unique_constraints("regulatory_source_notice")
    }
    if unique != {
        ("uq_regulatory_source_notice_identity", ("document_set_id", "identity_key"))
    }:
        raise ValueError("Precreated source notice registry unique identity differs")
    checks = {
        value["name"]
        for value in inspector.get_check_constraints("regulatory_source_notice")
    }
    if checks != {
        "ck_regulatory_source_notice_revision",
        "ck_regulatory_source_notice_content_state",
    }:
        raise ValueError("Precreated source notice registry content constraints differ")
    foreign_keys = {
        (
            tuple(value["constrained_columns"]),
            value["referred_table"],
            tuple(value["referred_columns"]),
            value.get("options", {}).get("ondelete"),
        )
        for value in inspector.get_foreign_keys("regulatory_source_notice")
    }
    if foreign_keys != {
        (("document_set_id",), "document_set", ("id",), "RESTRICT"),
        (("user_file_id",), "user_file", ("id",), "RESTRICT"),
    }:
        raise ValueError("Precreated source notice registry ownership differs")
    indexes = {
        value["name"]: (value["column_names"], bool(value["unique"]))
        for value in inspector.get_indexes("regulatory_source_notice")
        if not value.get("duplicates_constraint")
    }
    if indexes != {
        "ix_regulatory_source_notice_instrument": (
            [
                "document_set_id",
                "instrument_type",
                "instrument_year",
                "instrument_number",
            ],
            False,
        )
    }:
        raise ValueError("Precreated source notice registry lookup indexes differ")


def upgrade() -> None:
    if not op.get_context().as_sql:
        connection = op.get_bind()
        if sa.inspect(connection).has_table("regulatory_source_notice"):
            _validate_precreated_registry(connection)
            return
    op.create_table(
        "regulatory_source_notice",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_set_id", sa.Integer(), nullable=False),
        sa.Column("identity_key", sa.Text(), nullable=False),
        sa.Column("instrument_type", sa.Text(), nullable=False),
        sa.Column("instrument_number", sa.Text(), nullable=True),
        sa.Column("instrument_year", sa.Integer(), nullable=True),
        sa.Column("instrument_date", sa.Date(), nullable=True),
        sa.Column("catalog_title", sa.Text(), nullable=False),
        sa.Column(
            "aliases",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "legistation_metadata",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "provenance",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "content_state",
            sa.String(32),
            server_default=sa.text("'metadata_only'"),
            nullable=False,
        ),
        sa.Column("user_file_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "revision", sa.Integer(), server_default=sa.text("1"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_set_id"], ["document_set.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["user_file_id"], ["user_file.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_set_id",
            "identity_key",
            name="uq_regulatory_source_notice_identity",
        ),
        sa.CheckConstraint(
            "revision >= 1", name="ck_regulatory_source_notice_revision"
        ),
        sa.CheckConstraint(
            "(content_state = 'metadata_only' AND user_file_id IS NULL) OR "
            "(content_state = 'materialized' AND user_file_id IS NOT NULL)",
            name="ck_regulatory_source_notice_content_state",
        ),
    )
    op.create_index(
        "ix_regulatory_source_notice_instrument",
        "regulatory_source_notice",
        ["document_set_id", "instrument_type", "instrument_year", "instrument_number"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_regulatory_source_notice_instrument", table_name="regulatory_source_notice"
    )
    op.drop_table("regulatory_source_notice")
