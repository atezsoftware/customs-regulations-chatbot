"""Unit coverage for PostgreSQL-backed regulatory secondary projection."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.background.celery.tasks.user_file_processing.tasks import (
    _index_user_file_to_secondary,
)
from onyx.db.enums import UserFileStatus
from onyx.db.models import SearchSettings, UserFile

TASKS_MODULE = "onyx.background.celery.tasks.user_file_processing.tasks"


@pytest.mark.parametrize("pass_uuid", [False, True])
def test_secondary_projection_uses_postgres_rows_without_parsing_source(
    pass_uuid: bool,
) -> None:
    user_file_id = uuid4()
    input_user_file_id = user_file_id if pass_uuid else str(user_file_id)
    tenant_id = "test-tenant"

    detached_secondary = MagicMock(spec=SearchSettings)
    detached_secondary.id = 41
    bound_secondary = MagicMock(spec=SearchSettings)
    bound_secondary.id = 41

    user_file = MagicMock(spec=UserFile)
    user_file.id = user_file_id
    user_file.status = UserFileStatus.COMPLETED
    user_file.chunk_count = 5

    rows = [MagicMock(name="row_0"), MagicMock(name="row_1")]
    project_ids = {str(user_file_id): [7]}
    persona_ids = {str(user_file_id): [11]}

    db_session = MagicMock()

    def get_model(model: type[object], _identity: object) -> object | None:
        if model is SearchSettings:
            return bound_secondary
        return None

    db_session.get.side_effect = get_model
    session_context = MagicMock()
    session_context.__enter__.return_value = db_session

    with (
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            f"{TASKS_MODULE}.lock_completed_user_file_for_projection",
            return_value=user_file,
        ) as lock_user_file,
        patch(f"{TASKS_MODULE}.get_chunks_for_file", return_value=rows) as get_rows,
        patch(
            f"{TASKS_MODULE}.fetch_user_project_ids_for_user_files",
            return_value=project_ids,
        ) as fetch_project_ids,
        patch(
            f"{TASKS_MODULE}.fetch_persona_ids_for_user_files",
            return_value=persona_ids,
        ) as fetch_persona_ids,
        patch(f"{TASKS_MODULE}._project_rows_to_search_settings") as project_rows,
        patch(f"{TASKS_MODULE}._load_user_file_documents") as load_documents,
        patch(f"{TASKS_MODULE}.RegulatoryIndexingChunker") as parse_chunks,
        patch(f"{TASKS_MODULE}.run_indexing_pipeline") as run_pipeline,
    ):
        _index_user_file_to_secondary(
            input_user_file_id,
            detached_secondary,
            tenant_id,
        )

    lock_user_file.assert_called_once_with(db_session, user_file_id)
    get_rows.assert_called_once_with(db_session, user_file_id)
    fetch_project_ids.assert_called_once_with([str(user_file_id)], db_session)
    fetch_persona_ids.assert_called_once_with([str(user_file_id)], db_session)
    project_rows.assert_called_once()
    projection_kwargs = project_rows.call_args.kwargs
    assert projection_kwargs["user_file"] is user_file
    assert projection_kwargs["rows"] is rows
    assert projection_kwargs["search_settings"] is bound_secondary
    assert projection_kwargs["tenant_id"] == tenant_id
    assert projection_kwargs["project_ids"] == project_ids
    assert projection_kwargs["persona_ids"] == persona_ids

    chunk_count_diffs = projection_kwargs["indexing_metadata"].doc_id_to_chunk_cnt_diff
    assert list(chunk_count_diffs) == [str(user_file_id)]
    counts = chunk_count_diffs[str(user_file_id)]
    assert counts.old_chunk_cnt == 5
    assert counts.new_chunk_cnt == 2

    load_documents.assert_not_called()
    parse_chunks.assert_not_called()
    run_pipeline.assert_not_called()


def test_secondary_projection_skips_non_completed_file_under_row_lock() -> None:
    user_file_id = uuid4()
    detached_secondary = MagicMock(spec=SearchSettings)
    detached_secondary.id = 41
    bound_secondary = MagicMock(spec=SearchSettings)
    bound_secondary.id = 41
    db_session = MagicMock()
    db_session.get.return_value = bound_secondary
    session_context = MagicMock()
    session_context.__enter__.return_value = db_session

    with (
        patch(
            f"{TASKS_MODULE}.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            f"{TASKS_MODULE}.lock_completed_user_file_for_projection",
            return_value=None,
        ) as lock_user_file,
        patch(f"{TASKS_MODULE}.get_chunks_for_file") as get_rows,
        patch(f"{TASKS_MODULE}._project_rows_to_search_settings") as project_rows,
    ):
        _index_user_file_to_secondary(
            str(user_file_id), detached_secondary, "test-tenant"
        )

    lock_user_file.assert_called_once_with(db_session, user_file_id)
    get_rows.assert_not_called()
    project_rows.assert_not_called()
