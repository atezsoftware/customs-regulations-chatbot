"""Metadata-only projection of explicit regulatory validity windows."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from onyx.configs.app_configs import ENABLE_OPENSEARCH_INDEXING_FOR_ONYX
from onyx.db.models import UserFile
from onyx.db.regulatory_chunks import (
    RegulatoryChunkValidityState,
    RegulatoryFileValidityWindow,
    common_reindex_validity_window,
    get_chunks_for_file,
)
from onyx.db.search_settings import get_active_search_settings_list
from onyx.document_index.factory import build_opensearch_document_index
from onyx.document_index.opensearch.opensearch_document_index import (
    OpenSearchDocumentIndex,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()


@dataclass(frozen=True, slots=True)
class AppliedRegulatoryValidityPatch:
    """Enough state to compensate index writes if PostgreSQL commit fails."""

    document_id: str
    expected_regulatory_chunk_ids: tuple[str, ...]
    previous_window: RegulatoryFileValidityWindow
    updated_window: RegulatoryFileValidityWindow
    updated_indices: tuple[OpenSearchDocumentIndex, ...]


def patch_user_file_validity_in_active_indices(
    db_session: Session,
    user_file: UserFile,
    *,
    previous_window: RegulatoryFileValidityWindow,
    updated_window: RegulatoryFileValidityWindow,
) -> AppliedRegulatoryValidityPatch | None:
    """Patch PRESENT/FUTURE without embedding when projection identity is exact.

    Returns ``None`` before any index write when the metadata-only path is not
    provably safe, allowing the caller to use the canonical full projection.
    PRESENT failures are raised. FUTURE failures retain PRESENT and mark the
    file for the existing secondary reconciliation workflow.

    The caller must hold the ``UserFile`` projection row lock for the duration
    of this function and the surrounding PostgreSQL transaction.
    """

    if not ENABLE_OPENSEARCH_INDEXING_FOR_ONYX:
        return None

    rows = get_chunks_for_file(db_session, user_file.id)
    if not rows or user_file.chunk_count is None:
        return None
    if user_file.chunk_count != len(rows):
        return None

    canonical_window = common_reindex_validity_window(
        [
            RegulatoryChunkValidityState(
                source=row.source,
                status=row.status,
                validity_start_date=row.validity_start_date,
                validity_end_date=row.validity_end_date,
                supersedes_chunk_id=row.supersedes_chunk_id,
                superseded_by_chunk_id=row.superseded_by_chunk_id,
            )
            for row in rows
        ]
    )
    if canonical_window != updated_window:
        return None

    search_settings_list = get_active_search_settings_list(db_session)
    if not any(settings.status.is_current() for settings in search_settings_list):
        raise RuntimeError("No current search settings found")

    expected_regulatory_chunk_ids = tuple(row.id for row in rows)
    wrote_current = False
    updated_indices: list[OpenSearchDocumentIndex] = []
    for search_settings in search_settings_list:
        try:
            document_index = build_opensearch_document_index(search_settings)
            document_index.update_regulatory_validity(
                document_id=str(user_file.id),
                expected_regulatory_chunk_ids=list(expected_regulatory_chunk_ids),
                previous_start_date=previous_window.start,
                previous_end_date=previous_window.end,
                updated_start_date=updated_window.start,
                updated_end_date=updated_window.end,
            )
        except Exception:
            if search_settings.status.is_current():
                raise
            user_file.secondary_reconcile_pending = True
            logger.exception(
                "Deferred FUTURE regulatory validity patch for user_file=%s "
                "search_settings=%s",
                user_file.id,
                search_settings.id,
            )
            continue

        if search_settings.status.is_current():
            wrote_current = True
        # A validity-only preflight does not prove FUTURE content/ACL parity.
        # Preserve an existing broad reconcile flag; only a full canonical
        # projection/reconcile may clear it.
        updated_indices.append(document_index)

    db_session.add(user_file)
    if not wrote_current:
        raise RuntimeError("Current regulatory validity index was not updated")
    return AppliedRegulatoryValidityPatch(
        document_id=str(user_file.id),
        expected_regulatory_chunk_ids=expected_regulatory_chunk_ids,
        previous_window=previous_window,
        updated_window=updated_window,
        updated_indices=tuple(updated_indices),
    )


def compensate_regulatory_validity_patch(
    applied_patch: AppliedRegulatoryValidityPatch,
) -> None:
    """Best-effort reversal used only when the surrounding PG commit fails."""

    rollback_errors: list[Exception] = []
    for document_index in applied_patch.updated_indices:
        try:
            document_index.update_regulatory_validity(
                document_id=applied_patch.document_id,
                expected_regulatory_chunk_ids=list(
                    applied_patch.expected_regulatory_chunk_ids
                ),
                previous_start_date=applied_patch.updated_window.start,
                previous_end_date=applied_patch.updated_window.end,
                updated_start_date=applied_patch.previous_window.start,
                updated_end_date=applied_patch.previous_window.end,
            )
        except Exception as error:
            rollback_errors.append(error)
            logger.exception(
                "Failed to compensate regulatory validity patch for document=%s",
                applied_patch.document_id,
            )
    if rollback_errors:
        raise RuntimeError(
            "Failed to compensate one or more regulatory validity index writes."
        ) from rollback_errors[0]
