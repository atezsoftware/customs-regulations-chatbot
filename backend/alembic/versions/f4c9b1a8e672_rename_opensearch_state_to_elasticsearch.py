"""rename OpenSearch migration state to Elasticsearch

Revision ID: f4c9b1a8e672
Revises: a71c9d4e2f30
"""

from alembic import op

revision = "f4c9b1a8e672"
down_revision = "a71c9d4e2f30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.rename_table(
        "opensearch_document_migration_record",
        "elasticsearch_document_migration_record",
    )
    op.rename_table(
        "opensearch_tenant_migration_record",
        "elasticsearch_tenant_migration_record",
    )
    op.alter_column(
        "elasticsearch_tenant_migration_record",
        "enable_opensearch_retrieval",
        new_column_name="enable_elasticsearch_retrieval",
    )
    op.execute(
        "ALTER INDEX ix_opensearch_document_migration_record_status "
        "RENAME TO ix_elasticsearch_document_migration_record_status"
    )
    op.execute(
        "ALTER INDEX ix_opensearch_document_migration_record_attempts_count "
        "RENAME TO ix_elasticsearch_document_migration_record_attempts_count"
    )
    op.execute(
        "ALTER INDEX ix_opensearch_document_migration_record_created_at "
        "RENAME TO ix_elasticsearch_document_migration_record_created_at"
    )
    op.execute(
        "ALTER INDEX idx_opensearch_tenant_migration_singleton "
        "RENAME TO idx_elasticsearch_tenant_migration_singleton"
    )


def downgrade() -> None:
    op.execute(
        "ALTER INDEX idx_elasticsearch_tenant_migration_singleton "
        "RENAME TO idx_opensearch_tenant_migration_singleton"
    )
    op.execute(
        "ALTER INDEX ix_elasticsearch_document_migration_record_created_at "
        "RENAME TO ix_opensearch_document_migration_record_created_at"
    )
    op.execute(
        "ALTER INDEX ix_elasticsearch_document_migration_record_attempts_count "
        "RENAME TO ix_opensearch_document_migration_record_attempts_count"
    )
    op.execute(
        "ALTER INDEX ix_elasticsearch_document_migration_record_status "
        "RENAME TO ix_opensearch_document_migration_record_status"
    )
    op.alter_column(
        "elasticsearch_tenant_migration_record",
        "enable_elasticsearch_retrieval",
        new_column_name="enable_opensearch_retrieval",
    )
    op.rename_table(
        "elasticsearch_tenant_migration_record",
        "opensearch_tenant_migration_record",
    )
    op.rename_table(
        "elasticsearch_document_migration_record",
        "opensearch_document_migration_record",
    )
