"""Actual empty-index replacement cannot cross an owned writer or name reuse."""

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from sqlalchemy import delete
from sqlalchemy.schema import DropSchema

from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.enums import IndexModelStatus
from onyx.db.models import CloudEmbeddingProvider, SearchSettings
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import (
    load_owned_writer_inputs,
    stage_writer_publication,
)
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from onyx.document_index.elasticsearch.schema import DocumentSchema
from onyx.document_index.publication_models import PublicationScope
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.writer_projection import prepare_owned_correction
from onyx.regulatory.writer_publication import complete_writer_index_inventory
from shared_configs.configs import MULTI_TENANT
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from shared_configs.enums import EmbeddingProvider
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    create_owned_file,
)


@pytest.mark.parametrize("case", ["pending", "repair", "shared", "resume", "cleanup"])
def test_physical_replacement_authority(
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
                f"{Path(__file__).resolve()}::test_physical_replacement_authority[{case}]",
            ],
            env={**os.environ, "MULTI_TENANT": "true"},
            check=True,
        )
        return
    from onyx.context.search.models import SavedSearchSettings
    from onyx.db.search_settings import create_search_settings
    from onyx.document_index.elasticsearch import (
        elasticsearch_document_index as physical,
    )

    tenant = "tenant_5d_physical_" + uuid4().hex
    name = "annex-5d-physical-" + uuid4().hex
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
        desired = DocumentSchema.get_document_schema(3, True)
        old = DocumentSchema.get_document_schema(3, True)
        old["properties"]["content"] = {"type": "keyword"}
        client.indices.create(index=name, mappings=old)
        original_uuid = client.indices.get(index=name)[name]["settings"]["index"][
            "uuid"
        ]
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
                setting = SearchSettings(
                    model_name="openai/text-embedding-3-large",
                    provider_type=EmbeddingProvider.OPENROUTER,
                    model_dim=3,
                    normalize=True,
                    query_prefix="",
                    passage_prefix="",
                    status=IndexModelStatus.PRESENT,
                    index_name=name,
                    enable_contextual_rag=False,
                )
                session.add(setting)
                session.commit()
                setting_id = setting.id
                monkeypatch.setattr(
                    type(
                        DefaultIndexingEmbedder.from_db_search_settings(
                            setting
                        ).embedding_model
                    ),
                    "encode",
                    lambda _self, texts, **_: [[0.25, 0.5, 0.75] for _ in texts],
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
            manifest = complete_writer_index_inventory(
                owner,
                client,
                prepare_owned_correction(
                    owner, client, inputs, inputs.canonical, changed_id=None
                ),
            )
            if case == "pending":
                stage_writer_publication(owner, manifest)
            if case == "cleanup":
                from onyx.db.regulatory_physical_indexes import (
                    pending_physical_operation,
                )
                from onyx.server.manage import search_settings as settings_api

                monkeypatch.setattr(settings_api, "MULTI_TENANT", False)
                with get_session_with_tenant(tenant_id=tenant) as session:
                    setting = session.get(SearchSettings, setting_id)
                    assert setting is not None
                    setting.status = IndexModelStatus.FUTURE
                    session.commit()
                    saved = SavedSearchSettings.from_db_model(setting)
                    session.expunge(setting)
                raw_delete = type(client.indices).delete
                calls: list[str] = []
                observed_responses: list[object] = []

                def cleanup_delete(transport, **kwargs):
                    with get_session_with_tenant(tenant_id=tenant) as session:
                        assert session.get(SearchSettings, setting_id) is None
                        assert (
                            pending_physical_operation(owner.scope, name) is not None
                        ), (
                            "bootstrap removed its row without retaining the physical name/UUID claim"
                        )
                        with pytest.raises(ValueError, match="physical"):
                            create_search_settings(saved, session)
                    calls.append(str(kwargs["index"]))
                    response = raw_delete(transport, **kwargs)
                    observed_responses.append(response)
                    raise RuntimeError("lost terminal delete response")

                with monkeypatch.context() as scoped:
                    scoped.setattr(type(client.indices), "delete", cleanup_delete)
                    with get_session_with_tenant(tenant_id=tenant) as session:
                        with pytest.raises(
                            RuntimeError, match="lost terminal delete response"
                        ):
                            settings_api._cleanup_unpromoted_empty_cloud_bootstrap(
                                db_session=session,
                                new_search_settings=setting,
                                elasticsearch_index_preexisted=False,
                            )
                assert calls == [name]
                assert not client.indices.exists(index=name)
                with get_session_with_tenant(tenant_id=tenant) as session:
                    with pytest.raises(ValueError, match="indeterminate"):
                        settings_api._cleanup_unpromoted_empty_cloud_bootstrap(
                            db_session=session,
                            new_search_settings=setting,
                            elasticsearch_index_preexisted=False,
                        )
                    with pytest.raises(ValueError, match="physical"):
                        create_search_settings(saved, session)
                from onyx.document_index.elasticsearch import physical_operations

                retained = pending_physical_operation(owner.scope, name)
                assert retained is not None
                response = observed_responses[0]
                assert isinstance(response, ObjectApiResponse)
                physical_operations.reconcile_physical_delete_after_writer_exit(
                    client,
                    operation_id=retained.id,
                    index_name=name,
                    multitenant=False,
                    original_response=response,
                    writer_exit_evidence_sha256=__import__("hashlib")
                    .sha256(
                        str(
                            (
                                retained.owner_id,
                                calls,
                                "controlled invocation exited before recovery",
                            )
                        ).encode()
                    )
                    .hexdigest(),
                )
                assert pending_physical_operation(owner.scope, name) is None
                authority.release(owner)
                return
            monkeypatch.setattr(physical, "MULTI_TENANT", case == "shared")
            with ElasticsearchIndexClient(index_name=name) as index:
                if case in {"pending", "shared"}:
                    with pytest.raises(ValueError, match="publication|shared"):
                        physical.ensure_current_schema(
                            index_client=index,
                            expected_mappings=desired,
                            index_settings={},
                            database_has_indexed_documents=False,
                        )
                    assert (
                        client.indices.get(index=name)[name]["settings"]["index"][
                            "uuid"
                        ]
                        == original_uuid
                    )
                else:
                    original_delete = type(index.publication_client().indices).delete

                    def deleting(transport, **kwargs: object):
                        from onyx.db.regulatory_physical_indexes import (
                            pending_physical_operation,
                        )

                        if pending_physical_operation(owner.scope, name) is None:
                            return original_delete(transport, **kwargs)
                        with get_session_with_tenant(tenant_id=tenant) as session:
                            setting = session.get(SearchSettings, setting_id)
                            assert setting is not None
                            with pytest.raises(ValueError, match="physical"):
                                create_search_settings(
                                    SavedSearchSettings.from_db_model(setting), session
                                )
                        with pytest.raises(ValueError, match="physical"):
                            stage_writer_publication(owner, manifest)
                        assert not authority.reservations(owner).gate_closed
                        return original_delete(transport, **kwargs)

                    monkeypatch.setattr(
                        type(index.publication_client().indices), "delete", deleting
                    )
                    if case == "resume":
                        from onyx.db.regulatory_physical_indexes import (
                            pending_physical_operation,
                        )
                        from onyx.document_index.elasticsearch import (
                            physical_operations,
                        )

                        create = physical_operations._create_owned_index

                        def interrupted(*_args: object) -> str:
                            raise RuntimeError("interrupted before owned recreate")

                        monkeypatch.setattr(
                            physical_operations, "_create_owned_index", interrupted
                        )
                        with pytest.raises(
                            RuntimeError, match="interrupted before owned recreate"
                        ):
                            physical.ensure_current_schema(
                                index_client=index,
                                expected_mappings=desired,
                                index_settings={},
                                database_has_indexed_documents=False,
                            )
                        retained = pending_physical_operation(owner.scope, name)
                        assert retained is not None and retained.phase == "creating"
                        assert retained.index_uuid == original_uuid
                        assert not client.indices.exists(index=name)
                        monkeypatch.setattr(
                            physical_operations, "_create_owned_index", create
                        )
                    physical.ensure_current_schema(
                        index_client=index,
                        expected_mappings=desired,
                        index_settings={},
                        database_has_indexed_documents=False,
                    )
                    assert (
                        client.indices.get(index=name)[name]["settings"]["index"][
                            "uuid"
                        ]
                        != original_uuid
                    )
                    assert (
                        client.indices.get_mapping(index=name)[name]["mappings"][
                            "properties"
                        ]["content"]
                        == desired["properties"]["content"]
                    )
            authority.release(owner)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
        if client.indices.exists(index=name):
            client.indices.delete(index=name)
        client.close()
        with SqlEngine.get_engine().begin() as connection:
            connection.execute(DropSchema(tenant, cascade=True, if_exists=True))
