from unittest.mock import MagicMock, call, patch

import pytest

from onyx.context.search.models import SearchSettingsCreationRequest
from onyx.db import swap_index
from onyx.db.enums import EmbeddingPrecision, SwitchoverType
from onyx.db.models import IndexModelStatus
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.server.manage import search_settings
from shared_configs.enums import EmbeddingProvider

_MODULE = "onyx.server.manage.search_settings"


class ImportUnavailable(Exception):
    pass


def _cloud_request() -> SearchSettingsCreationRequest:
    return SearchSettingsCreationRequest(
        model_name="openai/text-embedding-3-small",
        model_dim=1536,
        normalize=True,
        query_prefix="",
        passage_prefix="",
        provider_type=EmbeddingProvider.OPENROUTER,
        index_name=None,
        multipass_indexing=False,
        embedding_precision=EmbeddingPrecision.FLOAT,
        reduced_dimension=None,
        enable_contextual_rag=False,
        contextual_rag_model_configuration_id=None,
        switchover_type=SwitchoverType.REINDEX,
    )


def test_parser_free_empty_production_allows_cloud_embedding_bootstrap() -> None:
    current = MagicMock(
        id=1,
        use_port_flow=False,
        port_backfill_source_id=None,
        model_name="nomic-ai/nomic-embed-text-v1",
        index_name="danswer_chunk_nomic",
    )
    future = MagicMock(
        id=2,
        index_name="danswer_chunk_openai_text_embedding_3_small",
        use_port_flow=False,
        switchover_type=SwitchoverType.REINDEX,
    )
    promoted = MagicMock(id=2)
    db_session = MagicMock()

    with (
        patch.object(search_settings, "DOCUMENT_IMPORT_ENABLED", False),
        patch(f"{_MODULE}.check_docs_exist", return_value=False),
        patch(f"{_MODULE}.check_connectors_exist", return_value=False),
        patch(f"{_MODULE}.check_user_files_exist", return_value=False),
        patch(f"{_MODULE}.ensure_document_import_available") as ensure_import,
        patch(
            f"{_MODULE}.get_embedding_provider_from_provider_type",
            return_value=MagicMock(),
        ),
        patch(f"{_MODULE}.validate_contextual_rag_model"),
        patch(
            f"{_MODULE}.get_current_search_settings",
            side_effect=[current, promoted],
        ),
        patch(f"{_MODULE}.get_secondary_search_settings", return_value=None),
        patch(f"{_MODULE}.create_search_settings", return_value=future) as create,
        patch(f"{_MODULE}._opensearch_index_exists", return_value=False),
        patch(f"{_MODULE}.get_all_document_indices", return_value=[]),
        patch(f"{_MODULE}.check_and_perform_index_swap") as perform_swap,
    ):
        result = search_settings.set_new_search_settings(
            _cloud_request(), _=MagicMock(), db_session=db_session
        )

    assert result.id == 2
    ensure_import.assert_not_called()
    create.assert_called_once()
    assert create.call_args.kwargs["use_port_flow"] is False
    db_session.commit.assert_called_once_with()
    perform_swap.assert_called_once_with(db_session)


def test_empty_cloud_bootstrap_cleans_up_if_promotion_does_not_complete() -> None:
    current = MagicMock(
        id=1,
        use_port_flow=False,
        port_backfill_source_id=None,
        model_name="nomic-ai/nomic-embed-text-v1",
        index_name="danswer_chunk_nomic",
    )
    future = MagicMock(
        id=2,
        index_name="danswer_chunk_openai_text_embedding_3_small",
        use_port_flow=False,
        switchover_type=SwitchoverType.REINDEX,
    )
    db_session = MagicMock()

    with (
        patch.object(search_settings, "DOCUMENT_IMPORT_ENABLED", False),
        patch(f"{_MODULE}.check_docs_exist", return_value=False),
        patch(f"{_MODULE}.check_connectors_exist", return_value=False),
        patch(f"{_MODULE}.check_user_files_exist", return_value=False),
        patch(f"{_MODULE}.ensure_document_import_available"),
        patch(
            f"{_MODULE}.get_embedding_provider_from_provider_type",
            return_value=MagicMock(),
        ),
        patch(f"{_MODULE}.validate_contextual_rag_model"),
        patch(
            f"{_MODULE}.get_current_search_settings",
            side_effect=[current, current],
        ),
        patch(f"{_MODULE}.get_secondary_search_settings", return_value=None),
        patch(f"{_MODULE}.create_search_settings", return_value=future),
        patch(f"{_MODULE}._opensearch_index_exists", return_value=False),
        patch(f"{_MODULE}.get_all_document_indices", return_value=[]),
        patch(f"{_MODULE}.check_and_perform_index_swap"),
        patch(f"{_MODULE}.reclaim_index_data") as reclaim_index,
        patch(
            f"{_MODULE}.delete_search_settings_if_not_present", return_value=True
        ) as delete_settings,
        pytest.raises(OnyxError) as exc_info,
    ):
        search_settings.set_new_search_settings(
            _cloud_request(), _=MagicMock(), db_session=db_session
        )

    assert exc_info.value.error_code == OnyxErrorCode.INTERNAL_ERROR
    delete_settings.assert_called_once_with(
        db_session=db_session,
        search_settings_id=future.id,
    )
    reclaim_index.assert_called_once()
    assert reclaim_index.call_args.kwargs["index_name"] == future.index_name


def test_empty_cloud_bootstrap_cleans_up_if_swap_raises_before_promotion() -> None:
    current = MagicMock(
        id=1,
        use_port_flow=False,
        port_backfill_source_id=None,
        model_name="nomic-ai/nomic-embed-text-v1",
        index_name="danswer_chunk_nomic",
    )
    future = MagicMock(
        id=2,
        index_name="danswer_chunk_openai_text_embedding_3_small",
        use_port_flow=False,
        switchover_type=SwitchoverType.REINDEX,
    )
    db_session = MagicMock()

    with (
        patch.object(search_settings, "DOCUMENT_IMPORT_ENABLED", False),
        patch(f"{_MODULE}.check_docs_exist", return_value=False),
        patch(f"{_MODULE}.check_connectors_exist", return_value=False),
        patch(f"{_MODULE}.check_user_files_exist", return_value=False),
        patch(f"{_MODULE}.ensure_document_import_available"),
        patch(
            f"{_MODULE}.get_embedding_provider_from_provider_type",
            return_value=MagicMock(),
        ),
        patch(f"{_MODULE}.validate_contextual_rag_model"),
        patch(
            f"{_MODULE}.get_current_search_settings",
            side_effect=[current, current],
        ),
        patch(f"{_MODULE}.get_secondary_search_settings", return_value=None),
        patch(f"{_MODULE}.create_search_settings", return_value=future),
        patch(f"{_MODULE}._opensearch_index_exists", return_value=False),
        patch(f"{_MODULE}.get_all_document_indices", return_value=[]),
        patch(
            f"{_MODULE}.check_and_perform_index_swap",
            side_effect=RuntimeError("swap failed"),
        ),
        patch(f"{_MODULE}.reclaim_index_data") as reclaim_index,
        patch(
            f"{_MODULE}.delete_search_settings_if_not_present", return_value=True
        ) as delete_settings,
        pytest.raises(OnyxError) as exc_info,
    ):
        search_settings.set_new_search_settings(
            _cloud_request(), _=MagicMock(), db_session=db_session
        )

    assert exc_info.value.error_code == OnyxErrorCode.INTERNAL_ERROR
    db_session.rollback.assert_called_once_with()
    reclaim_index.assert_called_once()
    delete_settings.assert_called_once_with(
        db_session=db_session,
        search_settings_id=future.id,
    )


def test_empty_cloud_bootstrap_never_reclaims_a_promoted_index_after_swap_error() -> (
    None
):
    current = MagicMock(
        id=1,
        use_port_flow=False,
        port_backfill_source_id=None,
        model_name="nomic-ai/nomic-embed-text-v1",
        index_name="danswer_chunk_nomic",
    )
    future = MagicMock(
        id=2,
        index_name="danswer_chunk_openai_text_embedding_3_small",
        use_port_flow=False,
        switchover_type=SwitchoverType.REINDEX,
    )
    promoted = MagicMock(id=2)
    db_session = MagicMock()

    with (
        patch.object(search_settings, "DOCUMENT_IMPORT_ENABLED", False),
        patch(f"{_MODULE}.check_docs_exist", return_value=False),
        patch(f"{_MODULE}.check_connectors_exist", return_value=False),
        patch(f"{_MODULE}.check_user_files_exist", return_value=False),
        patch(f"{_MODULE}.ensure_document_import_available"),
        patch(
            f"{_MODULE}.get_embedding_provider_from_provider_type",
            return_value=MagicMock(),
        ),
        patch(f"{_MODULE}.validate_contextual_rag_model"),
        patch(
            f"{_MODULE}.get_current_search_settings",
            side_effect=[current, promoted],
        ),
        patch(f"{_MODULE}.get_secondary_search_settings", return_value=None),
        patch(f"{_MODULE}.create_search_settings", return_value=future),
        patch(f"{_MODULE}._opensearch_index_exists", return_value=False),
        patch(f"{_MODULE}.get_all_document_indices", return_value=[]),
        patch(
            f"{_MODULE}.check_and_perform_index_swap",
            side_effect=RuntimeError("late swap failure"),
        ),
        patch(f"{_MODULE}.reclaim_index_data") as reclaim_index,
        patch(
            f"{_MODULE}.delete_search_settings_if_not_present", return_value=False
        ) as delete_settings,
    ):
        result = search_settings.set_new_search_settings(
            _cloud_request(), _=MagicMock(), db_session=db_session
        )

    assert result.id == future.id
    reclaim_index.assert_not_called()
    delete_settings.assert_called_once_with(
        db_session=db_session,
        search_settings_id=future.id,
    )


def test_empty_cloud_bootstrap_never_reclaims_a_preexisting_index() -> None:
    future = MagicMock(
        id=2,
        index_name="danswer_chunk_shared",
    )
    db_session = MagicMock()

    with (
        patch(f"{_MODULE}.delete_search_settings_if_not_present", return_value=True),
        patch(f"{_MODULE}.reclaim_index_data") as reclaim_index,
    ):
        removed = search_settings._cleanup_unpromoted_empty_cloud_bootstrap(
            db_session=db_session,
            new_search_settings=future,
            opensearch_index_preexisted=True,
        )

    assert removed is True
    reclaim_index.assert_not_called()


def test_empty_cloud_bootstrap_never_reclaims_when_conditional_delete_loses_race() -> (
    None
):
    future = MagicMock(
        id=2,
        index_name="danswer_chunk_new",
    )
    db_session = MagicMock()

    with (
        patch(f"{_MODULE}.delete_search_settings_if_not_present", return_value=False),
        patch(f"{_MODULE}.reclaim_index_data") as reclaim_index,
    ):
        removed = search_settings._cleanup_unpromoted_empty_cloud_bootstrap(
            db_session=db_session,
            new_search_settings=future,
            opensearch_index_preexisted=False,
        )

    assert removed is False
    reclaim_index.assert_not_called()


def test_index_swap_commits_present_and_past_statuses_atomically() -> None:
    current = MagicMock(id=1)
    future = MagicMock(
        id=2,
        use_port_flow=False,
        enable_contextual_rag=False,
        contextual_rag_model_configuration_id=None,
    )
    db_session = MagicMock()

    with (
        patch.object(swap_index, "get_current_search_settings", return_value=current),
        patch.object(swap_index, "update_search_settings_status") as update_status,
        patch.object(swap_index, "update_default_contextual_model"),
        patch.object(swap_index, "get_all_document_indices", return_value=[]),
    ):
        result = swap_index._perform_index_swap(
            db_session=db_session,
            new_search_settings=future,
            all_cc_pairs=[],
        )

    assert result is current
    assert update_status.call_args_list == [
        call(
            search_settings=current,
            new_status=IndexModelStatus.PAST,
            db_session=db_session,
            commit=False,
        ),
        call(
            search_settings=future,
            new_status=IndexModelStatus.PRESENT,
            db_session=db_session,
            commit=False,
        ),
    ]
    db_session.commit.assert_called_once_with()


@pytest.mark.parametrize(
    ("docs_exist", "connectors_exist", "user_files_exist"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_parser_free_production_rejects_embedding_change_when_data_exists(
    docs_exist: bool,
    connectors_exist: bool,
    user_files_exist: bool,
) -> None:
    with (
        patch.object(search_settings, "DOCUMENT_IMPORT_ENABLED", False),
        patch(f"{_MODULE}.check_docs_exist", return_value=docs_exist),
        patch(f"{_MODULE}.check_connectors_exist", return_value=connectors_exist),
        patch(f"{_MODULE}.check_user_files_exist", return_value=user_files_exist),
        patch(
            f"{_MODULE}.ensure_document_import_available",
            side_effect=ImportUnavailable,
        ),
        pytest.raises(ImportUnavailable),
    ):
        search_settings.set_new_search_settings(
            _cloud_request(), _=MagicMock(), db_session=MagicMock()
        )
