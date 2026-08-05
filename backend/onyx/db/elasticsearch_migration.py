"""Database operations for Elasticsearch migration tracking.

This module provides functions to track the progress of migrating documents
from Vespa to Elasticsearch.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.background.celery.tasks.elasticsearch_migration.constants import (
    GET_VESPA_CHUNKS_SLICE_COUNT,
    TOTAL_ALLOWABLE_DOC_MIGRATION_ATTEMPTS_BEFORE_PERMANENT_FAILURE,
)
from onyx.configs.app_configs import (
    ENABLE_ELASTICSEARCH_RETRIEVAL_FOR_ONYX,
    ONYX_DISABLE_VESPA,
)
from onyx.db.enums import ElasticsearchDocumentMigrationStatus
from onyx.db.models import (
    Document,
    ElasticsearchDocumentMigrationRecord,
    ElasticsearchTenantMigrationRecord,
)
from onyx.document_index.vespa.shared_utils.utils import (
    replace_invalid_doc_id_characters,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()


def get_paginated_document_batch(
    db_session: Session,
    limit: int,
    prev_ending_document_id: str | None = None,
) -> list[str]:
    """Gets a paginated batch of document IDs from the Document table.

    We need some deterministic ordering to ensure that we don't miss any
    documents when paginating. This function uses the document ID. It is
    possible a document is inserted above a spot this function has already
    passed. In that event we assume that the document will be indexed into
    Elasticsearch anyway and we don't need to migrate.
    TODO(andrei): Consider ordering on last_modified in addition to ID to better
    match get_elasticsearch_migration_records_needing_migration.

    Args:
        db_session: SQLAlchemy session.
        limit: Number of document IDs to fetch.
        prev_ending_document_id: Document ID to start after (for pagination). If
            None, returns the first batch of documents. If not None, this should
            be the last ordered ID which was fetched in a previous batch.
            Defaults to None.

    Returns:
        List of document IDs.
    """
    stmt = select(Document.id).order_by(Document.id.asc()).limit(limit)
    if prev_ending_document_id is not None:
        stmt = stmt.where(Document.id > prev_ending_document_id)
    return list(db_session.scalars(stmt).all())


def get_last_elasticsearch_migration_document_id(
    db_session: Session,
) -> str | None:
    """
    Gets the last document ID in the ElasticsearchDocumentMigrationRecord table.

    Returns None if no records are found.
    """
    stmt = (
        select(ElasticsearchDocumentMigrationRecord.document_id)
        .order_by(ElasticsearchDocumentMigrationRecord.document_id.desc())
        .limit(1)
    )
    return db_session.scalars(stmt).first()


def create_elasticsearch_migration_records_with_commit(
    db_session: Session,
    document_ids: list[str],
) -> None:
    """Creates new ElasticsearchDocumentMigrationRecord records.

    Silently skips any document IDs that already have records.
    """
    if not document_ids:
        return

    values = [
        {
            "document_id": document_id,
            "status": ElasticsearchDocumentMigrationStatus.PENDING,
        }
        for document_id in document_ids
    ]

    stmt = insert(ElasticsearchDocumentMigrationRecord).values(values)
    stmt = stmt.on_conflict_do_nothing(index_elements=["document_id"])

    db_session.execute(stmt)
    db_session.commit()


def get_elasticsearch_migration_records_needing_migration(
    db_session: Session,
    limit: int,
) -> list[ElasticsearchDocumentMigrationRecord]:
    """Gets records of documents that need to be migrated.

    Properties:
    - First tries documents with status PENDING.
    - Of these, orders documents with the oldest last_modified to prioritize
      documents that were modified a long time ago, as they are presumed to be
      stable. This column is modified in many flows so is not a guarantee of the
      document having been indexed.
    - Then if there's room in the result, tries documents with status FAILED.
    - Of these, first orders documents on the least attempts_count so as to have
      a backoff for recently-failed docs. Then orders on last_modified as
      before.
    """
    result: list[ElasticsearchDocumentMigrationRecord] = []

    # Step 1: Fetch as many PENDING status records as possible ordered by
    # last_modified (oldest first). last_modified lives on Document, so we join.
    stmt_pending = (
        select(ElasticsearchDocumentMigrationRecord)
        .join(Document, ElasticsearchDocumentMigrationRecord.document_id == Document.id)
        .where(
            ElasticsearchDocumentMigrationRecord.status
            == ElasticsearchDocumentMigrationStatus.PENDING
        )
        .order_by(Document.last_modified.asc())
        .limit(limit)
    )
    result.extend(list(db_session.scalars(stmt_pending).all()))
    remaining = limit - len(result)

    # Step 2: If more are needed, fetch records with status FAILED, ordered by
    # attempts_count (lowest first), then last_modified (oldest first).
    if remaining > 0:
        stmt_failed = (
            select(ElasticsearchDocumentMigrationRecord)
            .join(
                Document,
                ElasticsearchDocumentMigrationRecord.document_id == Document.id,
            )
            .where(
                ElasticsearchDocumentMigrationRecord.status
                == ElasticsearchDocumentMigrationStatus.FAILED
            )
            .order_by(
                ElasticsearchDocumentMigrationRecord.attempts_count.asc(),
                Document.last_modified.asc(),
            )
            .limit(remaining)
        )
        result.extend(list(db_session.scalars(stmt_failed).all()))

    return result


def get_total_elasticsearch_migration_record_count(
    db_session: Session,
) -> int:
    """Gets the total number of Elasticsearch migration records.

    Used to check whether every document has been tracked for migration.
    """
    return db_session.query(ElasticsearchDocumentMigrationRecord).count()


def get_total_document_count(db_session: Session) -> int:
    """Gets the total number of documents.

    Used to check whether every document has been tracked for migration.
    """
    return db_session.query(Document).count()


def try_insert_elasticsearch_tenant_migration_record_with_commit(
    db_session: Session,
) -> None:
    """Tries to insert the singleton row on ElasticsearchTenantMigrationRecord.

    Does nothing if the row already exists.
    """
    stmt = insert(ElasticsearchTenantMigrationRecord).on_conflict_do_nothing(
        index_elements=[text("(true)")]
    )
    db_session.execute(stmt)
    db_session.commit()


def increment_num_times_observed_no_additional_docs_to_migrate_with_commit(
    db_session: Session,
) -> None:
    """Increments the number of times observed no additional docs to migrate.

    Requires the ElasticsearchTenantMigrationRecord to exist.

    Used to track when to stop the migration task.
    """
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    if record is None:
        raise RuntimeError("ElasticsearchTenantMigrationRecord not found.")
    record.num_times_observed_no_additional_docs_to_migrate += 1
    db_session.commit()


def increment_num_times_observed_no_additional_docs_to_populate_migration_table_with_commit(
    db_session: Session,
) -> None:
    """
    Increments the number of times observed no additional docs to populate the
    migration table.

    Requires the ElasticsearchTenantMigrationRecord to exist.

    Used to track when to stop the migration check task.
    """
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    if record is None:
        raise RuntimeError("ElasticsearchTenantMigrationRecord not found.")
    record.num_times_observed_no_additional_docs_to_populate_migration_table += 1
    db_session.commit()


def should_document_migration_be_permanently_failed(
    elasticsearch_document_migration_record: ElasticsearchDocumentMigrationRecord,
) -> bool:
    return (
        elasticsearch_document_migration_record.status
        == ElasticsearchDocumentMigrationStatus.PERMANENTLY_FAILED
        or (
            elasticsearch_document_migration_record.status
            == ElasticsearchDocumentMigrationStatus.FAILED
            and elasticsearch_document_migration_record.attempts_count
            >= TOTAL_ALLOWABLE_DOC_MIGRATION_ATTEMPTS_BEFORE_PERMANENT_FAILURE
        )
    )


def get_vespa_visit_state(
    db_session: Session,
) -> tuple[dict[int, str | None], int]:
    """Gets the current Vespa migration state from the tenant migration record.

    Requires the ElasticsearchTenantMigrationRecord to exist.

    Returns:
        Tuple of (continuation_token_map, total_chunks_migrated).
    """
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    if record is None:
        raise RuntimeError("ElasticsearchTenantMigrationRecord not found.")
    if record.vespa_visit_continuation_token is None:
        continuation_token_map: dict[int, str | None] = {
            slice_id: None for slice_id in range(GET_VESPA_CHUNKS_SLICE_COUNT)
        }
    else:
        json_loaded_continuation_token_map = json.loads(
            record.vespa_visit_continuation_token
        )
        continuation_token_map = {
            int(key): value for key, value in json_loaded_continuation_token_map.items()
        }
    return continuation_token_map, record.total_chunks_migrated


def update_vespa_visit_progress_with_commit(
    db_session: Session,
    continuation_token_map: dict[int, str | None],
    chunks_processed: int,
    chunks_errored: int,
    approx_chunk_count_in_vespa: int | None,
) -> None:
    """Updates the Vespa migration progress and commits.

    Requires the ElasticsearchTenantMigrationRecord to exist.

    Args:
        db_session: SQLAlchemy session.
        continuation_token_map: The new continuation token map. None entry means
            the visit is complete for that slice.
        chunks_processed: Number of chunks processed in this batch (added to
            the running total).
        chunks_errored: Number of chunks errored in this batch (added to the
            running errored total).
        approx_chunk_count_in_vespa: Approximate number of chunks in Vespa. If
            None, the existing value is used.
    """
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    if record is None:
        raise RuntimeError("ElasticsearchTenantMigrationRecord not found.")
    record.vespa_visit_continuation_token = json.dumps(continuation_token_map)
    record.total_chunks_migrated += chunks_processed
    record.total_chunks_errored += chunks_errored
    record.approx_chunk_count_in_vespa = (
        approx_chunk_count_in_vespa
        if approx_chunk_count_in_vespa is not None
        else record.approx_chunk_count_in_vespa
    )
    db_session.commit()


def mark_migration_completed_time_if_not_set_with_commit(
    db_session: Session,
) -> None:
    """Marks the migration completed time if not set.

    Requires the ElasticsearchTenantMigrationRecord to exist.
    """
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    if record is None:
        raise RuntimeError("ElasticsearchTenantMigrationRecord not found.")
    if record.migration_completed_at is not None:
        return
    record.migration_completed_at = datetime.now(timezone.utc)
    db_session.commit()


def is_migration_completed(db_session: Session) -> bool:
    """Returns True if the migration is completed.

    Can be run even if the migration record does not exist.
    """
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    return record is not None and record.migration_completed_at is not None


def build_sanitized_to_original_doc_id_mapping(
    db_session: Session,
) -> dict[str, str]:
    """Pre-computes a mapping of sanitized -> original document IDs.

    Only includes documents whose ID contains single quotes (the only character
    that gets sanitized by replace_invalid_doc_id_characters). For all other
    documents, sanitized == original and no mapping entry is needed.

    Scans over all documents.

    Checks if the sanitized ID already exists as a genuine separate document in
    the Document table. If so, raises as there is no way of resolving the
    conflict in the migration. The user will need to reindex.

    Args:
        db_session: SQLAlchemy session.

    Returns:
        Dict mapping sanitized_id -> original_id, only for documents where
        the IDs differ. Empty dict means no documents have single quotes
        in their IDs.
    """
    # Find all documents with single quotes in their ID.
    stmt = select(Document.id).where(Document.id.contains("'"))
    ids_with_quotes = list(db_session.scalars(stmt).all())

    result: dict[str, str] = {}
    for original_id in ids_with_quotes:
        sanitized_id = replace_invalid_doc_id_characters(original_id)
        if sanitized_id != original_id:
            result[sanitized_id] = original_id

    # See if there are any documents whose ID is a sanitized ID of another
    # document. If there is even one match, we cannot proceed.
    stmt = select(Document.id).where(Document.id.in_(result.keys()))
    ids_with_matches = list(db_session.scalars(stmt).all())
    if ids_with_matches:
        raise RuntimeError(
            f"Documents with IDs {ids_with_matches} have sanitized IDs that match other documents. "
            "This is not supported and the user will need to reindex."
        )

    return result


def get_elasticsearch_migration_state(
    db_session: Session,
) -> tuple[int, datetime | None, datetime | None, int | None]:
    """Returns the state of the Vespa to Elasticsearch migration.

    If the tenant migration record is not found, returns defaults of 0, None,
    None, None.

    Args:
        db_session: SQLAlchemy session.

    Returns:
        Tuple of (total_chunks_migrated, created_at, migration_completed_at,
            approx_chunk_count_in_vespa).
    """
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    if record is None:
        return 0, None, None, None
    return (
        record.total_chunks_migrated,
        record.created_at,
        record.migration_completed_at,
        record.approx_chunk_count_in_vespa,
    )


def get_elasticsearch_retrieval_state(
    db_session: Session,
) -> bool:
    """Returns the state of the Elasticsearch retrieval.

    If the tenant migration record is not found, defaults to
    ENABLE_ELASTICSEARCH_RETRIEVAL_FOR_ONYX.

    If ONYX_DISABLE_VESPA is True, always returns True.
    """
    if ONYX_DISABLE_VESPA:
        return True
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    if record is None:
        return ENABLE_ELASTICSEARCH_RETRIEVAL_FOR_ONYX
    return record.enable_elasticsearch_retrieval


def set_enable_elasticsearch_retrieval_with_commit(
    db_session: Session,
    enable: bool,
) -> None:
    """Sets the enable_elasticsearch_retrieval flag on the singleton record.

    Creates the record if it doesn't exist yet.
    """
    try_insert_elasticsearch_tenant_migration_record_with_commit(db_session)
    record = db_session.query(ElasticsearchTenantMigrationRecord).first()
    if record is None:
        raise RuntimeError("ElasticsearchTenantMigrationRecord not found.")
    record.enable_elasticsearch_retrieval = enable
    db_session.commit()
