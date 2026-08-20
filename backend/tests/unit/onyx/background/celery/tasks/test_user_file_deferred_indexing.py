"""Chunking is separable from indexing.

Uploading a file writes its regulatory chunks so an operator can inspect them
before anything reaches the search index. That first phase must not depend on
the contextual retrieval model, which only the indexing phase needs.
"""

from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.background.celery.tasks.user_file_processing.tasks import (
    _chunk_user_file_without_indexing,
    index_user_file_impl,
)
from onyx.connectors.models import Document
from onyx.db.enums import UserFileStatus
from onyx.db.models import UserFile
from onyx.indexing.contextual_settings import ContextualIndexingConfigurationError

TASKS_MODULE = "onyx.background.celery.tasks.user_file_processing.tasks"


def _session_returning(user_file: UserFile) -> MagicMock:
    """The session context manager does not commit on exit, so callers must."""

    db_session = MagicMock()
    db_session.get.return_value = user_file
    session_context = MagicMock()
    session_context.__enter__.return_value = db_session
    session_context.db_session = db_session
    return session_context


def _user_file(status: UserFileStatus) -> MagicMock:
    user_file = MagicMock(spec=UserFile)
    user_file.id = uuid4()
    user_file.status = status
    user_file.chunk_count = None
    user_file.regulatory_chunk_generation_hash = None
    return user_file


def test_chunking_writes_rows_without_a_contextual_model() -> None:
    """The reason this phase exists: no LLM is needed to produce chunks."""

    user_file = _user_file(UserFileStatus.PROCESSING)
    documents = cast(list[Document], [MagicMock(name="document")])
    rows = [MagicMock(name="row") for _ in range(223)]

    session_context = _session_returning(user_file)
    with (
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(f"{TASKS_MODULE}.get_active_search_settings_list") as search_settings,
        patch(f"{TASKS_MODULE}.DefaultIndexingEmbedder"),
        patch(f"{TASKS_MODULE}.effective_contextual_rag_enabled", return_value=True),
        patch(
            f"{TASKS_MODULE}.compute_regulatory_chunk_generation_hash",
            return_value="a" * 64,
        ) as generation_hash,
        # Patched at its source: reaching it at all would mean the chunking
        # phase depends on a contextual model, which is exactly what it must not.
        patch(
            "onyx.indexing.contextual_settings.require_contextual_rag_llm",
            side_effect=ContextualIndexingConfigurationError("no model"),
        ),
        patch(f"{TASKS_MODULE}.RegulatoryIndexingChunker") as chunker_cls,
        patch(f"{TASKS_MODULE}.process_image_sections", side_effect=lambda docs: docs),
        patch(f"{TASKS_MODULE}.get_chunks_for_file", return_value=rows),
        patch(f"{TASKS_MODULE}.store_user_file_plaintext"),
        patch(f"{TASKS_MODULE}.run_indexing_pipeline") as run_pipeline,
    ):
        settings = MagicMock()
        settings.status.is_current.return_value = True
        search_settings.return_value = [settings]

        _chunk_user_file_without_indexing(
            user_file_id=str(user_file.id),
            documents=documents,
            tenant_id="tenant",
        )

    # Chunk boundaries come from the same chunker the indexing path uses, so a
    # later index pass cannot shift them.
    chunker_cls.return_value.chunk.assert_called_once_with(documents)
    run_pipeline.assert_not_called()
    assert user_file.status == UserFileStatus.CHUNKED
    assert user_file.chunk_count == 223
    assert user_file.regulatory_chunk_generation_hash == "a" * 64
    generation_hash.assert_called_once_with(
        embedding_provider=settings.provider_type,
        embedding_model_name=settings.model_name,
        enable_contextual_rag=True,
    )
    # The chunk rows share this transaction; without a commit they never land.
    session_context.db_session.commit.assert_called_once()


def test_chunking_leaves_the_search_index_untouched() -> None:
    user_file = _user_file(UserFileStatus.PROCESSING)

    with (
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=_session_returning(user_file),
        ),
        patch(f"{TASKS_MODULE}.get_active_search_settings_list") as search_settings,
        patch(f"{TASKS_MODULE}.DefaultIndexingEmbedder"),
        patch(f"{TASKS_MODULE}.effective_contextual_rag_enabled", return_value=False),
        patch(
            f"{TASKS_MODULE}.compute_regulatory_chunk_generation_hash",
            return_value="b" * 64,
        ),
        patch(f"{TASKS_MODULE}.RegulatoryIndexingChunker"),
        patch(f"{TASKS_MODULE}.process_image_sections", side_effect=lambda docs: docs),
        patch(f"{TASKS_MODULE}.get_chunks_for_file", return_value=[MagicMock()]),
        patch(f"{TASKS_MODULE}.store_user_file_plaintext"),
        patch(f"{TASKS_MODULE}.get_all_document_indices") as document_indices,
        patch(f"{TASKS_MODULE}.project_user_file_to_index") as project,
    ):
        settings = MagicMock()
        settings.status.is_current.return_value = True
        search_settings.return_value = [settings]

        _chunk_user_file_without_indexing(
            user_file_id=str(user_file.id),
            documents=[MagicMock()],
            tenant_id="tenant",
        )

    document_indices.assert_not_called()
    project.assert_not_called()


def test_indexing_promotes_a_chunked_file_to_completed() -> None:
    user_file = _user_file(UserFileStatus.CHUNKED)

    session_context = _session_returning(user_file)
    with (
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            f"{TASKS_MODULE}.project_user_file_to_index", return_value=223
        ) as project,
    ):
        index_user_file_impl(user_file_id=str(user_file.id), tenant_id="tenant")

    project.assert_called_once()
    assert user_file.status == UserFileStatus.COMPLETED
    session_context.db_session.commit.assert_called_once()


def test_feature_enabled_indexing_creates_a_durable_job_without_legacy_projection() -> (
    None
):
    user_file = _user_file(UserFileStatus.CHUNKED)
    job_id = uuid4()
    session_context = _session_returning(user_file)

    with (
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(f"{TASKS_MODULE}.app_configs.REGULATORY_BATCH_INDEXING_ENABLED", True),
        patch(
            f"{TASKS_MODULE}.prepare_regulatory_indexing_job_from_chunks",
            return_value=job_id,
        ) as prepare,
        patch(f"{TASKS_MODULE}._enqueue_durable_regulatory_indexing") as enqueue,
        patch(f"{TASKS_MODULE}.project_user_file_to_index") as project,
    ):
        index_user_file_impl(user_file_id=str(user_file.id), tenant_id="tenant-a")

    prepare.assert_called_once_with(
        user_file.id, "tenant-a", session_context.db_session
    )
    enqueue.assert_called_once_with(
        job_id=job_id,
        tenant_id="tenant-a",
        user_file_id=str(user_file.id),
    )
    project.assert_not_called()
    assert user_file.status == UserFileStatus.CHUNKED


def test_indexing_marks_the_file_failed_when_projection_raises() -> None:
    user_file = _user_file(UserFileStatus.CHUNKED)

    session_context = _session_returning(user_file)
    with (
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            f"{TASKS_MODULE}.project_user_file_to_index",
            side_effect=RuntimeError("elasticsearch is down"),
        ),
        pytest.raises(RuntimeError),
    ):
        index_user_file_impl(user_file_id=str(user_file.id), tenant_id="tenant")

    assert user_file.status == UserFileStatus.FAILED
    session_context.db_session.commit.assert_called_once()
