"""Tests for no-vector-DB user file processing paths.

Verifies that when DISABLE_VECTOR_DB is True:
- process_user_file_impl calls _process_user_file_without_vector_db (not indexing)
- _process_user_file_without_vector_db extracts text, counts tokens, stores plaintext,
  sets status=COMPLETED and chunk_count=0
- delete_user_file_impl skips vector DB chunk deletion
- project_sync_user_file_impl skips vector DB metadata update
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest

from onyx.background.celery.tasks.user_file_processing.tasks import (
    _process_user_file_without_vector_db,
    delete_user_file_impl,
    process_user_file_impl,
    project_sync_user_file_impl,
)
from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db.enums import UserFileStatus

TASKS_MODULE = "onyx.background.celery.tasks.user_file_processing.tasks"
LLM_FACTORY_MODULE = "onyx.llm.factory"


def _make_documents(texts: list[str]) -> list[Document]:
    """Build a list of Document objects with the given section texts."""
    return [
        Document(
            id=str(uuid4()),
            source=DocumentSource.USER_FILE,
            sections=[TextSection(text=t)],
            semantic_identifier=f"test-doc-{i}",
            metadata={},
        )
        for i, t in enumerate(texts)
    ]


def _make_user_file(
    *,
    status: UserFileStatus = UserFileStatus.PROCESSING,
    file_id: str = "test-file-id",
    name: str = "test.txt",
) -> MagicMock:
    """Return a MagicMock mimicking a UserFile ORM instance."""
    uf = MagicMock()
    uf.id = uuid4()
    uf.file_id = file_id
    uf.name = name
    uf.status = status
    uf.token_count = None
    uf.chunk_count = None
    uf.last_project_sync_at = None
    uf.projects = []
    uf.assistants = []
    uf.needs_project_sync = True
    uf.needs_persona_sync = True
    uf.needs_document_set_sync = False
    uf.secondary_reconcile_pending = False
    return uf


def _mock_session(uf: MagicMock, mock_get_session: MagicMock) -> MagicMock:
    """Wire a session mock so db_session.get(...) returns uf."""
    session = MagicMock()
    session.get.return_value = uf
    mock_get_session.return_value.__enter__.return_value = session
    return session


# ------------------------------------------------------------------
# _process_user_file_without_vector_db — direct tests
# ------------------------------------------------------------------


class TestProcessUserFileWithoutVectorDb:
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.store_user_file_plaintext")
    @patch(f"{LLM_FACTORY_MODULE}.get_llm_tokenizer_encode_func")
    @patch(f"{LLM_FACTORY_MODULE}.get_default_llm")
    def test_extracts_and_combines_text(
        self,
        mock_get_llm: MagicMock,  # noqa: ARG002
        mock_get_encode: MagicMock,
        mock_store_plaintext: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        mock_encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        mock_get_encode.return_value = mock_encode

        uf = _make_user_file()
        docs = _make_documents(["hello world", "foo bar"])
        _mock_session(uf, mock_get_session)

        _process_user_file_without_vector_db(user_file_id=uf.id, documents=docs)

        stored_text = mock_store_plaintext.call_args.kwargs["plaintext_content"]
        assert "hello world" in stored_text
        assert "foo bar" in stored_text

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.store_user_file_plaintext")
    @patch(f"{LLM_FACTORY_MODULE}.get_llm_tokenizer_encode_func")
    @patch(f"{LLM_FACTORY_MODULE}.get_default_llm")
    def test_computes_token_count(
        self,
        mock_get_llm: MagicMock,  # noqa: ARG002
        mock_get_encode: MagicMock,
        mock_store_plaintext: MagicMock,  # noqa: ARG002
        mock_get_session: MagicMock,
    ) -> None:
        mock_encode = MagicMock(return_value=list(range(42)))
        mock_get_encode.return_value = mock_encode

        uf = _make_user_file()
        docs = _make_documents(["some text content"])
        _mock_session(uf, mock_get_session)

        _process_user_file_without_vector_db(user_file_id=uf.id, documents=docs)

        assert uf.token_count == 42

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.store_user_file_plaintext")
    @patch(f"{LLM_FACTORY_MODULE}.get_llm_tokenizer_encode_func")
    @patch(f"{LLM_FACTORY_MODULE}.get_default_llm")
    def test_token_count_falls_back_to_none_on_error(
        self,
        mock_get_llm: MagicMock,
        mock_get_encode: MagicMock,  # noqa: ARG002
        mock_store_plaintext: MagicMock,  # noqa: ARG002
        mock_get_session: MagicMock,
    ) -> None:
        mock_get_llm.side_effect = RuntimeError("No LLM configured")

        uf = _make_user_file()
        docs = _make_documents(["text"])
        _mock_session(uf, mock_get_session)

        _process_user_file_without_vector_db(user_file_id=uf.id, documents=docs)

        assert uf.token_count is None

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.store_user_file_plaintext")
    @patch(f"{LLM_FACTORY_MODULE}.get_llm_tokenizer_encode_func")
    @patch(f"{LLM_FACTORY_MODULE}.get_default_llm")
    def test_stores_plaintext(
        self,
        mock_get_llm: MagicMock,  # noqa: ARG002
        mock_get_encode: MagicMock,
        mock_store_plaintext: MagicMock,
        mock_get_session: MagicMock,
    ) -> None:
        mock_get_encode.return_value = MagicMock(return_value=[1])

        uf = _make_user_file()
        docs = _make_documents(["content to store"])
        _mock_session(uf, mock_get_session)

        _process_user_file_without_vector_db(user_file_id=uf.id, documents=docs)

        mock_store_plaintext.assert_called_once_with(
            user_file_id=uf.id,
            plaintext_content="content to store",
        )

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.store_user_file_plaintext")
    @patch(f"{LLM_FACTORY_MODULE}.get_llm_tokenizer_encode_func")
    @patch(f"{LLM_FACTORY_MODULE}.get_default_llm")
    def test_sets_completed_status_and_zero_chunk_count(
        self,
        mock_get_llm: MagicMock,  # noqa: ARG002
        mock_get_encode: MagicMock,
        mock_store_plaintext: MagicMock,  # noqa: ARG002
        mock_get_session: MagicMock,
    ) -> None:
        mock_get_encode.return_value = MagicMock(return_value=[1])

        uf = _make_user_file()
        docs = _make_documents(["text"])
        session = _mock_session(uf, mock_get_session)

        _process_user_file_without_vector_db(user_file_id=uf.id, documents=docs)

        assert uf.status == UserFileStatus.COMPLETED
        assert uf.chunk_count == 0
        assert uf.last_project_sync_at is not None
        session.add.assert_called_once_with(uf)
        session.commit.assert_called_once()

    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    @patch(f"{TASKS_MODULE}.store_user_file_plaintext")
    @patch(f"{LLM_FACTORY_MODULE}.get_llm_tokenizer_encode_func")
    @patch(f"{LLM_FACTORY_MODULE}.get_default_llm")
    def test_preserves_deleting_status(
        self,
        mock_get_llm: MagicMock,  # noqa: ARG002
        mock_get_encode: MagicMock,
        mock_store_plaintext: MagicMock,  # noqa: ARG002
        mock_get_session: MagicMock,
    ) -> None:
        mock_get_encode.return_value = MagicMock(return_value=[1])

        uf = _make_user_file(status=UserFileStatus.DELETING)
        docs = _make_documents(["text"])
        _mock_session(uf, mock_get_session)

        _process_user_file_without_vector_db(user_file_id=uf.id, documents=docs)

        assert uf.status == UserFileStatus.DELETING
        assert uf.chunk_count == 0


# ------------------------------------------------------------------
# process_user_file_impl — branching on DISABLE_VECTOR_DB
# ------------------------------------------------------------------


class TestProcessImplBranching:
    @patch(f"{TASKS_MODULE}._chunk_user_file_without_indexing")
    @patch(f"{TASKS_MODULE}._process_user_file_without_vector_db")
    @patch(f"{TASKS_MODULE}._process_user_file_with_indexing")
    @patch(f"{TASKS_MODULE}.DEFER_USER_FILE_INDEXING", True)
    @patch(f"{TASKS_MODULE}.app_configs.REGULATORY_BATCH_INDEXING_ENABLED", True)
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    def test_durable_flag_preserves_the_production_chunk_phase(
        self,
        mock_get_session: MagicMock,
        mock_with_indexing: MagicMock,
        mock_without_vdb: MagicMock,
        mock_chunk: MagicMock,
    ) -> None:
        uf = _make_user_file(name="regulation.md")
        session = MagicMock()
        session.get.return_value = uf
        mock_get_session.return_value.__enter__.return_value = session
        documents = _make_documents(["MADDE 1 - durable"])
        connector_mock = MagicMock()
        connector_mock.load_from_state.return_value = [documents]

        with patch(f"{TASKS_MODULE}.LocalFileConnector", return_value=connector_mock):
            process_user_file_impl(
                user_file_id=str(uf.id),
                tenant_id="test-tenant",
                redis_locking=False,
            )

        mock_chunk.assert_called_once_with(
            user_file_id=str(uf.id),
            documents=documents,
            tenant_id="test-tenant",
        )
        mock_with_indexing.assert_not_called()
        mock_without_vdb.assert_not_called()

    @patch(f"{TASKS_MODULE}._process_user_file_without_vector_db")
    @patch(f"{TASKS_MODULE}._process_user_file_with_indexing")
    @patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", True)
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    def test_calls_without_vector_db_when_disabled(
        self,
        mock_get_session: MagicMock,
        mock_with_indexing: MagicMock,
        mock_without_vdb: MagicMock,
    ) -> None:
        uf = _make_user_file()
        session = MagicMock()
        session.get.return_value = uf
        mock_get_session.return_value.__enter__.return_value = session

        connector_mock = MagicMock()
        connector_mock.load_from_state.return_value = [_make_documents(["hello"])]

        with patch(f"{TASKS_MODULE}.LocalFileConnector", return_value=connector_mock):
            process_user_file_impl(
                user_file_id=str(uf.id),
                tenant_id="test-tenant",
                redis_locking=False,
            )

        mock_without_vdb.assert_called_once()
        mock_with_indexing.assert_not_called()

    @patch(f"{TASKS_MODULE}._process_user_file_without_vector_db")
    @patch(f"{TASKS_MODULE}._process_user_file_with_indexing")
    @patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", False)
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    def test_calls_with_indexing_when_vector_db_enabled(
        self,
        mock_get_session: MagicMock,
        mock_with_indexing: MagicMock,
        mock_without_vdb: MagicMock,
    ) -> None:
        uf = _make_user_file()
        session = MagicMock()
        session.get.return_value = uf
        mock_get_session.return_value.__enter__.return_value = session

        connector_mock = MagicMock()
        connector_mock.load_from_state.return_value = [_make_documents(["hello"])]

        with patch(f"{TASKS_MODULE}.LocalFileConnector", return_value=connector_mock):
            process_user_file_impl(
                user_file_id=str(uf.id),
                tenant_id="test-tenant",
                redis_locking=False,
            )

        mock_with_indexing.assert_called_once()
        mock_without_vdb.assert_not_called()

    @patch(f"{TASKS_MODULE}.run_indexing_pipeline")
    @patch(f"{TASKS_MODULE}.store_user_file_plaintext")
    @patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", True)
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    def test_indexing_pipeline_not_called_when_disabled(
        self,
        mock_get_session: MagicMock,
        mock_store_plaintext: MagicMock,  # noqa: ARG002
        mock_run_pipeline: MagicMock,
    ) -> None:
        """End-to-end: verify run_indexing_pipeline is never invoked."""
        uf = _make_user_file()
        session = MagicMock()
        session.get.return_value = uf
        mock_get_session.return_value.__enter__.return_value = session

        connector_mock = MagicMock()
        connector_mock.load_from_state.return_value = [_make_documents(["content"])]

        with (
            patch(f"{TASKS_MODULE}.LocalFileConnector", return_value=connector_mock),
            patch(f"{LLM_FACTORY_MODULE}.get_default_llm"),
            patch(
                f"{LLM_FACTORY_MODULE}.get_llm_tokenizer_encode_func",
                return_value=MagicMock(return_value=[1, 2, 3]),
            ),
        ):
            process_user_file_impl(
                user_file_id=str(uf.id),
                tenant_id="test-tenant",
                redis_locking=False,
            )

        mock_run_pipeline.assert_not_called()


# ------------------------------------------------------------------
# delete_user_file_impl — vector DB skip
# ------------------------------------------------------------------


@pytest.fixture
def owned_delete_boundary(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from onyx.configs import app_configs
    from onyx.db import regulatory_writer_publication as repository
    from onyx.document_index.elasticsearch import client
    from onyx.file_store import file_store
    from onyx.regulatory import writer_publication

    authority = MagicMock()
    owner = authority.acquire.return_value
    monkeypatch.setattr(writer_publication, "PublicationStore", lambda _: authority)
    monkeypatch.setattr(repository, "writer_file_exists", lambda *_: True)
    monkeypatch.setattr(writer_publication, "pending_writer_manifest", lambda _: None)
    monkeypatch.setattr(
        writer_publication, "recover_owned_writer_before_next", lambda _: owner
    )
    plan = MagicMock(ready_to_delete=True, deliveries=())
    begin = MagicMock(return_value=plan)
    monkeypatch.setattr(repository, "begin_owned_deletion", begin)
    absence = MagicMock(return_value="original")
    monkeypatch.setattr(repository, "owned_unindexed_deletion_file_id", absence)
    finish = MagicMock()
    monkeypatch.setattr(repository, "finish_owned_deletion", finish)
    storage = MagicMock()
    monkeypatch.setattr(file_store, "get_default_file_store", lambda: storage)
    heartbeat = MagicMock()
    heartbeat.return_value.__enter__.return_value.is_set.return_value = False
    monkeypatch.setattr(writer_publication, "publication_heartbeat", heartbeat)
    es = MagicMock(side_effect=AssertionError("no ES for proven unindexed deletion"))
    monkeypatch.setattr(client, "ElasticsearchClient", es)
    monkeypatch.setattr(app_configs, "DISABLE_VECTOR_DB", True)
    return SimpleNamespace(
        owner=owner, plan=plan, storage=storage, finish=finish, absence=absence, es=es
    )


class TestDeleteImplNoVectorDb:
    def test_waits_without_deleting_anything_until_durable_cancellation_finishes(
        self, owned_delete_boundary: SimpleNamespace
    ) -> None:
        state = owned_delete_boundary
        state.plan.ready_to_delete = False
        delete_user_file_impl(
            user_file_id=str(uuid4()), tenant_id="test-tenant", redis_locking=False
        )
        state.storage.delete_file.assert_not_called()
        state.absence.assert_not_called()
        state.finish.assert_not_called()

    def test_pending_cancellation_is_delivered_without_hard_delete(
        self, owned_delete_boundary: SimpleNamespace
    ) -> None:
        state = owned_delete_boundary
        job_id = uuid4()
        state.plan.ready_to_delete = False
        state.plan.deliveries = (
            SimpleNamespace(job_id=job_id, expected_generation=12),
        )
        with patch(
            "onyx.background.celery.tasks.regulatory_indexing.tasks.enqueue_regulatory_indexing_step"
        ) as enqueue:
            delete_user_file_impl(
                user_file_id=str(uuid4()), tenant_id="test-tenant", redis_locking=False
            )
        enqueue.assert_called_once()
        assert enqueue.call_args.kwargs["job_id"] == job_id
        assert enqueue.call_args.kwargs["expected_generation"] == 12
        state.storage.delete_file.assert_not_called()
        state.absence.assert_not_called()
        state.finish.assert_not_called()

    def test_skips_vector_db_deletion(
        self, owned_delete_boundary: SimpleNamespace
    ) -> None:
        state = owned_delete_boundary
        delete_user_file_impl(
            user_file_id=str(uuid4()), tenant_id="test-tenant", redis_locking=False
        )
        state.es.assert_not_called()
        state.absence.assert_called_once_with(state.owner)
        state.finish.assert_called_once_with(state.owner, without_index_authority=True)

    def test_still_deletes_file_store_and_db_record(
        self, owned_delete_boundary: SimpleNamespace
    ) -> None:
        from onyx.file_store.utils import user_file_id_to_plaintext_file_name

        state = owned_delete_boundary
        identifier = uuid4()
        delete_user_file_impl(
            user_file_id=str(identifier), tenant_id="test-tenant", redis_locking=False
        )
        assert state.storage.delete_file.call_args_list == [
            call("original", error_on_missing=False),
            call(
                user_file_id_to_plaintext_file_name(identifier), error_on_missing=False
            ),
        ]
        state.finish.assert_called_once_with(state.owner, without_index_authority=True)


# ------------------------------------------------------------------
# project_sync_user_file_impl — vector DB skip
# ------------------------------------------------------------------


class TestProjectSyncImplNoVectorDb:
    @patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", True)
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    def test_skips_vector_db_update(
        self,
        mock_get_session: MagicMock,
    ) -> None:
        uf = _make_user_file(status=UserFileStatus.COMPLETED)
        session = MagicMock()
        session.get.return_value = uf
        mock_get_session.return_value.__enter__.return_value = session

        with (
            patch(
                f"{TASKS_MODULE}.fetch_user_files_with_access_relationships",
                return_value=[uf],
            ),
            patch(f"{TASKS_MODULE}.get_all_document_indices") as mock_get_indices,
            patch(f"{TASKS_MODULE}.get_active_search_settings") as mock_get_ss,
            patch(f"{TASKS_MODULE}.httpx_init_vespa_pool") as mock_vespa_pool,
        ):
            project_sync_user_file_impl(
                user_file_id=str(uf.id),
                tenant_id="test-tenant",
                redis_locking=False,
            )

            mock_get_indices.assert_not_called()
            mock_get_ss.assert_not_called()
            mock_vespa_pool.assert_not_called()

    @patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", True)
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    def test_still_clears_sync_flags(
        self,
        mock_get_session: MagicMock,
    ) -> None:
        uf = _make_user_file(status=UserFileStatus.COMPLETED)
        session = MagicMock()
        session.get.return_value = uf
        mock_get_session.return_value.__enter__.return_value = session

        with patch(
            f"{TASKS_MODULE}.fetch_user_files_with_access_relationships",
            return_value=[uf],
        ):
            project_sync_user_file_impl(
                user_file_id=str(uf.id),
                tenant_id="test-tenant",
                redis_locking=False,
            )

        assert uf.needs_project_sync is False
        assert uf.needs_persona_sync is False
        assert uf.last_project_sync_at is not None
        session.add.assert_called_once_with(uf)
        session.commit.assert_called_once()

    @patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", True)
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    def test_stale_task_skips_completed_file_without_pending_flags(
        self,
        mock_get_session: MagicMock,
    ) -> None:
        uf = _make_user_file(status=UserFileStatus.COMPLETED)
        uf.needs_project_sync = False
        uf.needs_persona_sync = False
        uf.secondary_reconcile_pending = False
        session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = session

        with patch(
            f"{TASKS_MODULE}.fetch_user_files_with_access_relationships",
            return_value=[uf],
        ):
            project_sync_user_file_impl(
                user_file_id=str(uf.id),
                tenant_id="test-tenant",
                redis_locking=False,
            )

        mock_get_session.assert_called_once_with()
        session.add.assert_not_called()
        session.commit.assert_not_called()

    @patch(f"{TASKS_MODULE}.DISABLE_VECTOR_DB", True)
    @patch(f"{TASKS_MODULE}.get_session_with_current_tenant")
    def test_failed_document_set_sync_clears_only_document_set_flag(
        self,
        mock_get_session: MagicMock,
    ) -> None:
        uf = _make_user_file(status=UserFileStatus.FAILED)
        uf.needs_document_set_sync = True
        uf.secondary_reconcile_pending = True
        session = MagicMock()
        session.get.return_value = uf
        mock_get_session.return_value.__enter__.return_value = session

        with patch(
            f"{TASKS_MODULE}.fetch_user_files_with_access_relationships",
            return_value=[uf],
        ):
            project_sync_user_file_impl(
                user_file_id=str(uf.id),
                tenant_id="test-tenant",
                redis_locking=False,
            )

        assert uf.needs_document_set_sync is False
        assert uf.needs_project_sync is True
        assert uf.needs_persona_sync is True
        assert uf.secondary_reconcile_pending is True
        session.add.assert_called_once_with(uf)
        session.commit.assert_called_once()
