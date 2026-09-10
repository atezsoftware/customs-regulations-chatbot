"""Actual PG/ES promotion orders staged writers and current dated history."""

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from elasticsearch import Elasticsearch
from sqlalchemy import delete, select
from sqlalchemy.schema import DropSchema

from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.enums import IndexModelStatus, SwitchoverType, UserFileStatus
from onyx.db.models import (
    CloudEmbeddingProvider,
    RegulatoryPublicationClock,
    SearchSettings,
    UserFile,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import (
    load_owned_writer_inputs,
    stage_writer_publication,
)
from onyx.document_index.elasticsearch.schema import DocumentSchema
from onyx.document_index.publication_models import PublicationScope
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.writer_projection import prepare_owned_correction
from onyx.regulatory.writer_publication import (
    complete_writer_index_inventory,
    execute_writer_publication,
)
from shared_configs.configs import MULTI_TENANT
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from shared_configs.enums import EmbeddingProvider
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    create_owned_file,
)


@pytest.mark.parametrize(
    "case",
    [
        "pending",
        "history",
        "preparing",
        "legacy",
        "port",
        "legacy_secondary",
        "settings",
        "metadata_flags",
        "durable_intent",
    ],
)
def test_promotion_waits_for_staged_writer_and_refuses_old_settings_after_swap(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    if not MULTI_TENANT:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                f"{Path(__file__).resolve()}::test_promotion_waits_for_staged_writer_and_refuses_old_settings_after_swap[{case}]",
            ],
            env={**os.environ, "MULTI_TENANT": "true"},
            check=True,
        )
        return
    from onyx.configs.app_configs import POSTGRES_HOST, POSTGRES_PORT
    from onyx.db import swap_index

    assert POSTGRES_HOST in {"localhost", "127.0.0.1"} and str(POSTGRES_PORT) == "25432"
    tenant = "tenant_5d_swap_" + uuid4().hex
    names = ["annex-5d-swap-" + uuid4().hex for _ in range(2)]
    client = Elasticsearch("http://127.0.0.1:29200")
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "-x",
                f"schemas={tenant}",
                "upgrade",
                "head",
            ],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for name in names:
            client.indices.create(
                index=name, mappings=DocumentSchema.get_document_schema(3, True)
            )
        with create_owned_file(tenant) as file_id:
            with get_session_with_tenant(tenant_id=tenant) as session:
                session.execute(delete(SearchSettings))
                session.add(
                    CloudEmbeddingProvider(
                        provider_type=EmbeddingProvider.OPENROUTER,
                        api_key=None,
                        api_url=None,
                    )
                )
                settings = [
                    SearchSettings(
                        model_name="openai/text-embedding-3-large",
                        provider_type=EmbeddingProvider.OPENROUTER,
                        model_dim=3,
                        normalize=True,
                        query_prefix="",
                        passage_prefix="",
                        status=status,
                        index_name=name,
                        enable_contextual_rag=False,
                        switchover_type=SwitchoverType.INSTANT,
                    )
                    for name, status in zip(
                        names, [IndexModelStatus.PRESENT, IndexModelStatus.FUTURE]
                    )
                ]
                if case == "legacy_secondary":
                    settings[1].use_port_flow = True
                session.add_all(settings)
                file = session.get_one(UserFile, file_id)
                assert file is not None
                file.status = UserFileStatus.COMPLETED
                if case in {"legacy", "legacy_secondary"}:
                    from onyx.db.models import RegulatoryChunk

                    session.execute(
                        delete(RegulatoryChunk).where(
                            RegulatoryChunk.user_file_id == file_id
                        )
                    )
                    file.needs_document_set_sync = True
                session.commit()
                ids = [item.id for item in settings]
                for item in settings:
                    monkeypatch.setattr(
                        type(
                            DefaultIndexingEmbedder.from_db_search_settings(
                                search_settings=item
                            ).embedding_model
                        ),
                        "encode",
                        lambda _self, texts, **_: [[0.25, 0.5, 0.75] for _ in texts],
                    )
            external_calls: list[str] = []

            def verify_without_clock(**_: object) -> None:
                with get_session_with_tenant(tenant_id=tenant) as check:
                    check.execute(
                        select(RegulatoryPublicationClock).with_for_update(nowait=True)
                    ).all()
                external_calls.append("index-readiness")

            monkeypatch.setattr(
                swap_index,
                "get_all_document_indices",
                lambda *_args, **_kwargs: [
                    SimpleNamespace(
                        verify_and_create_index_if_necessary=verify_without_clock
                    )
                ],
            )
            authority = PublicationStore(
                PublicationScope(
                    tenant_id=tenant,
                    environment=config.REGULATORY_ANNEX_ENVIRONMENT,
                    database_identity=config.ANNEX_DATABASE_IDENTITY,
                )
            )
            owner = authority.acquire(
                file_id, owner_id=uuid4(), ttl=timedelta(minutes=2)
            )
            inputs = load_owned_writer_inputs(owner)
            if case not in {"legacy", "legacy_secondary"}:
                manifest = prepare_owned_correction(
                    owner, client, inputs, inputs.canonical, changed_id=None
                )
                execute_writer_publication(owner, client, manifest)
            authority.release(owner)
            if case in {"legacy", "legacy_secondary"}:
                from onyx.background.celery.tasks.user_file_processing import (
                    tasks as file_tasks,
                )
                from onyx.configs.constants import DocumentSource
                from onyx.connectors.models import Document, TextSection
                from onyx.db.regulatory_annex_publication import (
                    load_file_temporal_bindings,
                )

                documents = [
                    Document(
                        id=str(file_id),
                        semantic_identifier="Ordinary report",
                        source=DocumentSource.USER_FILE,
                        sections=[
                            TextSection(
                                text="An ordinary unversioned report with useful plain text.",
                                link=None,
                            )
                        ],
                        metadata={},
                    )
                ]
                monkeypatch.setattr(
                    "onyx.file_processing.user_file_loader.load_user_file_documents",
                    lambda **_kwargs: (documents, []),
                )
                if case == "legacy_secondary":

                    def reject_unowned_pipeline(**_kwargs: object) -> None:
                        raise AssertionError(
                            "legacy secondary supply used the unfenced indexing pipeline"
                        )

                    monkeypatch.setattr(
                        file_tasks,
                        "run_indexing_pipeline",
                        reject_unowned_pipeline,
                        raising=False,
                    )
                    monkeypatch.setattr(
                        file_tasks,
                        "_load_user_file_documents",
                        lambda *_args: (documents, []),
                    )
                    assert file_tasks._supply_user_file_to_secondary(
                        str(file_id), tenant
                    )
                    with get_session_with_tenant(tenant_id=tenant) as session:
                        assert session.get_one(
                            UserFile, file_id
                        ).needs_document_set_sync
                        assert swap_index.check_and_perform_index_swap(session) is None
                    file_tasks.project_sync_user_file_impl(
                        user_file_id=str(file_id), tenant_id=tenant, redis_locking=False
                    )
                else:
                    file_tasks.project_sync_user_file_impl(
                        user_file_id=str(file_id), tenant_id=tenant, redis_locking=False
                    )
                with get_session_with_tenant(tenant_id=tenant) as session:
                    adopted = load_file_temporal_bindings(session, file_id)
                    assert adopted, (
                        "legacy metadata supply must publish canonical content, not silently clear flags on an empty index"
                    )
                    assert {item.index.index_name for item in adopted} == set(names)
                    assert (
                        session.get_one(UserFile, file_id).status
                        == UserFileStatus.COMPLETED
                    )
            if case in {"history", "port"}:
                owner = authority.acquire(
                    file_id, owner_id=uuid4(), ttl=timedelta(minutes=2)
                )
                inputs = load_owned_writer_inputs(owner)
                target = inputs.canonical[0]
                after = [
                    row.model_copy(
                        update={"text": "Corrected PRESENT authoritative text"}
                    )
                    if row.id == target.id
                    else row
                    for row in inputs.canonical
                ]
                correction = prepare_owned_correction(
                    owner,
                    client,
                    inputs,
                    after,
                    changed_id=target.id,
                    target_settings_ids={ids[0]},
                )
                execute_writer_publication(owner, client, correction)
                authority.release(owner)
                with get_session_with_tenant(tenant_id=tenant) as session:
                    assert swap_index.check_and_perform_index_swap(session) is None, (
                        "INSTANT promotion must first reconcile stale FUTURE legal history"
                    )
                    session.expire_all()
                    assert session.get_one(
                        UserFile, file_id
                    ).secondary_reconcile_pending
                from onyx.background.celery.tasks.user_file_processing.tasks import (
                    project_sync_user_file_impl,
                )

                if case == "port":
                    from onyx.document_index.elasticsearch.port_copy import PortCopier

                    with get_session_with_tenant(tenant_id=tenant) as session:
                        copier = PortCopier(
                            session.get_one(SearchSettings, ids[0]),
                            session.get_one(SearchSettings, ids[1]),
                        )

                    def reject_raw(*_args: object, **_kwargs: object) -> None:
                        raise AssertionError(
                            "user-file port bypassed publication ownership"
                        )

                    monkeypatch.setattr(
                        copier._future_index, "index_raw_chunks", reject_raw
                    )
                    count, aborted = copier.copy_doc_batch(
                        [str(file_id)],
                        surviving_doc_ids=lambda: {str(file_id)},
                        should_abort=lambda: False,
                    )
                    assert count > 0 and not aborted
                    assert copier.delete_port_written([str(file_id)]) == 0
                else:
                    project_sync_user_file_impl(
                        user_file_id=str(file_id), tenant_id=tenant, redis_locking=False
                    )
                from onyx.db.regulatory_annex_publication import (
                    load_file_temporal_bindings,
                )

                with get_session_with_tenant(tenant_id=tenant) as session:
                    current = load_file_temporal_bindings(session, file_id)
                    expected = {
                        (
                            item.representation_text,
                            item.effective_start,
                            item.effective_end,
                        )
                        for item in current
                        if item.index.index_name == names[0]
                    }
                    actual = {
                        (
                            item.representation_text,
                            item.effective_start,
                            item.effective_end,
                        )
                        for item in current
                        if item.index.index_name == names[1]
                    }
                    assert actual == expected, (
                        "FUTURE reindex must follow current PRESENT history, not its stale former representation"
                    )
            if case == "durable_intent":
                from onyx.db.models import RegulatoryIndexingJob
                from tests.unit.onyx.regulatory.indexing_jobs.test_publisher import (
                    _snapshot,
                )

                configuration = _snapshot().model_dump(mode="json")
                with get_session_with_tenant(tenant_id=tenant) as session:
                    job = RegulatoryIndexingJob(
                        id=uuid4(),
                        user_file_id=file_id,
                        content_hash="a" * 64,
                        chunk_generation_hash="b" * 64,
                        search_settings_id=ids[1],
                        prompt_hash="c" * 64,
                        config_snapshot=configuration,
                        status="QUEUED",
                        stage="PREPARING",
                        lease_generation=0,
                    )
                    session.add(job)
                    session.commit()
                    assert (
                        swap_index._perform_index_swap(
                            session, session.get_one(SearchSettings, ids[1]), []
                        )
                        is None
                    ), "the final clock barrier must include pending durable writers"
                    session.delete(job)
                    session.commit()
            if case == "metadata_flags":
                with get_session_with_tenant(tenant_id=tenant) as session:
                    session.get_one(UserFile, file_id).needs_document_set_sync = True
                    session.commit()
                    assert swap_index.check_and_perform_index_swap(session) is None, (
                        "pending metadata must reconcile before FUTURE promotion"
                    )
                from onyx.background.celery.tasks.user_file_processing.tasks import (
                    project_sync_user_file_impl,
                )

                project_sync_user_file_impl(
                    user_file_id=str(file_id), tenant_id=tenant, redis_locking=False
                )
            owner = authority.acquire(
                file_id, owner_id=uuid4(), ttl=timedelta(minutes=2)
            )
            inputs = load_owned_writer_inputs(owner)
            manifest = complete_writer_index_inventory(
                owner,
                client,
                prepare_owned_correction(
                    owner, client, inputs, inputs.canonical, changed_id=None
                ),
            )
            stage_writer_publication(owner, manifest)
            assert authority.reservations(owner).gate_closed
            if case == "settings":
                from onyx.context.search.models import SavedSearchSettings
                from onyx.db.search_settings import (
                    create_search_settings,
                    delete_search_settings,
                    update_search_settings_status,
                )

                for operation in ("cancel", "delete", "create"):
                    with get_session_with_tenant(tenant_id=tenant) as session:
                        setting = session.get_one(SearchSettings, ids[1])
                        assert setting is not None
                        with pytest.raises(ValueError, match="publication"):
                            if operation == "cancel":
                                update_search_settings_status(
                                    setting, IndexModelStatus.PAST, session
                                )
                            elif operation == "delete":
                                delete_search_settings(session, ids[1])
                            else:
                                create_search_settings(
                                    SavedSearchSettings.from_db_model(setting), session
                                )
                        session.rollback()
                        assert (
                            session.get_one(SearchSettings, ids[1]).status
                            == IndexModelStatus.FUTURE
                        )
            connector_reader = swap_index.get_connector_credential_pairs
            monkeypatch.setattr(
                swap_index,
                "get_connector_credential_pairs",
                lambda _: [SimpleNamespace(id=1)],
            )

            def reject_connector_mutation() -> object:
                raise AssertionError(
                    "connector state changed before promotion was allowed"
                )

            monkeypatch.setattr(swap_index, "get_kv_store", reject_connector_mutation)
            with get_session_with_tenant(tenant_id=tenant) as session:
                assert swap_index.check_and_perform_index_swap(session) is None, (
                    "staged publication must hold promotion before any released ES write"
                )
                session.rollback()
                assert (
                    session.get_one(SearchSettings, ids[0]).status
                    == IndexModelStatus.PRESENT
                )
                assert (
                    session.get_one(SearchSettings, ids[1]).status
                    == IndexModelStatus.FUTURE
                )
            monkeypatch.setattr(
                swap_index, "get_connector_credential_pairs", connector_reader
            )
            assert external_calls == []
            execute_writer_publication(owner, client)
            authority.release(owner)
            owner = authority.acquire(
                file_id, owner_id=uuid4(), ttl=timedelta(minutes=2)
            )
            inputs = load_owned_writer_inputs(owner)
            old_settings_manifest = prepare_owned_correction(
                owner, client, inputs, inputs.canonical, changed_id=None
            )
            if case != "preparing":
                old_settings_manifest = complete_writer_index_inventory(
                    owner, client, old_settings_manifest
                )
            with get_session_with_tenant(tenant_id=tenant) as session:
                old = swap_index.check_and_perform_index_swap(session)
                assert old is not None and old.id == ids[0]
                assert (
                    session.get_one(SearchSettings, ids[1]).status
                    == IndexModelStatus.PRESENT
                )
            assert external_calls == ["index-readiness"]
            with pytest.raises(ValueError, match="settings changed before staging"):
                if case == "preparing":
                    execute_writer_publication(owner, client, old_settings_manifest)
                else:
                    stage_writer_publication(owner, old_settings_manifest)
            assert not authority.reservations(owner).gate_closed
            authority.release(owner)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
        for name in names:
            if client.indices.exists(index=name):
                client.indices.delete(index=name)
        client.close()
        with SqlEngine.get_engine().begin() as connection:
            connection.execute(DropSchema(tenant, cascade=True, if_exists=True))
