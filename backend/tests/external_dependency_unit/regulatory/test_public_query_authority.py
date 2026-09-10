"""Configured public queries against owned PG schemas and real ES indices."""

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from elasticsearch import Elasticsearch
from sqlalchemy import delete, select
from sqlalchemy.schema import DropSchema

from onyx.configs.app_configs import POSTGRES_HOST, POSTGRES_PORT
from onyx.context.search.enums import QueryType
from onyx.context.search.models import IndexFilters
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.enums import EmbeddingPrecision, IndexModelStatus
from onyx.db.models import RegulatoryChunk, RegulatoryTemporalProjection, SearchSettings
from onyx.db.regulatory_context_projections import activate_temporal_projection
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchDocumentIndex,
)
from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
from onyx.document_index.elasticsearch.schema import DocumentSchema
from onyx.document_index.interfaces_new import DocumentSectionRequest, TenantState
from onyx.document_index.publication_models import (
    FrozenPublicationProjection,
    PublicationEncoderAuthority,
    PublicationEncoderReceipt,
    PublicationIndexSnapshot,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.publication_reads import public_read_store
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    create_owned_file,
    frozen_projection,
)


def test_public_queries_resolve_actual_present_future_authority_without_injected_snapshot() -> (
    None
):
    from shared_configs.configs import MULTI_TENANT

    if not MULTI_TENANT:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                str(Path(__file__).resolve()),
            ],
            env={**os.environ, "MULTI_TENANT": "true"},
            check=True,
        )
        return
    assert POSTGRES_HOST in ("localhost", "127.0.0.1") and str(POSTGRES_PORT) == "25432"
    tenant = "tenant_5c_authority_" + uuid4().hex
    backend = Path(__file__).resolve().parents[3]
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    with SqlEngine.get_engine().connect() as connection:
        from sqlalchemy import text

        assert (
            connection.scalar(
                text("SELECT count(*) FROM pg_namespace WHERE nspname=:name"),
                {"name": tenant},
            )
            == 0
        )
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant)
    client = Elasticsearch("http://127.0.0.1:29200")
    created_indices: list[str] = []
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
            cwd=backend,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        with create_owned_file(tenant) as file_id:
            try:
                authority = public_read_store()
                owner = authority.acquire(
                    file_id, owner_id=uuid4(), ttl=timedelta(seconds=120)
                )
                authority.close_gate(owner)
                inventory = authority.reservations(owner)
                with get_session_with_tenant(tenant_id=tenant) as session:
                    canonical = list(
                        session.scalars(
                            select(RegulatoryChunk)
                            .where(RegulatoryChunk.user_file_id == file_id)
                            .order_by(RegulatoryChunk.position)
                        )
                    )
                    canonical_data = [
                        (row.id, row.text, row.position) for row in canonical
                    ]
                    # Migration-seeded settings, if any, are confined to this new schema.
                    session.execute(delete(SearchSettings))
                    session.commit()
                for status in (IndexModelStatus.PRESENT, IndexModelStatus.FUTURE):
                    authority.close_gate(owner)
                    inventory = authority.reservations(owner)
                    name = "annex-5c-authority-" + uuid4().hex
                    client.indices.create(
                        index=name, mappings=DocumentSchema.get_document_schema(3, True)
                    )
                    created_indices.append(name)
                    with get_session_with_tenant(tenant_id=tenant) as session:
                        settings = SearchSettings(
                            model_name="fixture",
                            model_dim=3,
                            normalize=True,
                            status=status,
                            index_name=name,
                            provider_type=None,
                            query_prefix="",
                            passage_prefix="",
                        )
                        session.add(settings)
                        session.commit()
                        settings_id = settings.id
                    encoder = PublicationEncoderAuthority(
                        provider=None,
                        model="fixture",
                        effective_dimension=3,
                        endpoint_sha256=context_hash(None),
                        deployment_name=None,
                        api_version=None,
                        normalize=True,
                        passage_prefix="",
                    )
                    resolved = encoder.model_dump(
                        mode="json", exclude={"model", "effective_dimension"}
                    )
                    resolved["dimension"] = 3
                    bindings: list[AnnexTemporalProjection] = []
                    for number, (canonical_id, canonical_text, ordinal) in enumerate(
                        canonical_data
                    ):
                        config = {
                            "model": "fixture",
                            "formatter": f"compatible-{number}",
                        }
                        receipt = PublicationEncoderReceipt(
                            configuration_json=json.dumps(config),
                            authority=encoder,
                            resolved_fields=resolved,
                            resolution_sha256=context_hash(config),
                        )
                        snapshot = PublicationIndexSnapshot(
                            index_name=name,
                            index_uuid=client.indices.get(index=name)[name]["settings"][
                                "index"
                            ]["uuid"],
                            search_settings_id=settings_id,
                            model_provider="",
                            model_name="fixture",
                            vector_dimension=3,
                            embedding_config_sha256=publication_digest(config),
                            multitenant=True,
                            encoder_authority=encoder,
                            encoder_receipts=(receipt,),
                        )
                        source = json.loads(
                            frozen_projection(file_id, ordinal).source_json
                        )
                        summary, context = (
                            f"{status.value} summary\n",
                            f"\n{status.value} context",
                        )
                        source.update(
                            max_chunk_size=512,
                            tenant_id=tenant,
                            regulatory_chunk_id=canonical_id,
                            content=summary + canonical_text + context,
                            doc_summary=summary,
                            chunk_context=context,
                            source_links=json.dumps({0: ""}),
                            validity_start_date=1577836800,
                            validity_end_date=None,
                            public=True,
                        )
                        identity = uuid4()
                        projection = FrozenPublicationProjection(
                            ordinal=ordinal,
                            context_projection_id=str(identity),
                            source_json=json.dumps(source),
                            embedding_inputs=(source["content"],),
                            embedding_config_json=json.dumps(config),
                        )
                        bindings.append(
                            AnnexTemporalProjection(
                                id=identity,
                                index=snapshot,
                                projection=projection,
                                canonical_base_sha256=context_hash(canonical_text),
                                derived_role="canonical",
                                dependency_ids=[],
                                representation_text=canonical_text,
                                representation_metadata={
                                    "image_file_id": "image-evidence"
                                },
                                reference_date=date(2020, 1, 1),
                                effective_start=date(2020, 1, 1),
                                effective_end=None,
                                semantic_position=number,
                            )
                        )
                    adapter = FencedPublicationIndex(
                        client,
                        bindings[0].index.model_copy(
                            update={
                                "encoder_receipts": tuple(
                                    item.index.encoder_receipts[0] for item in bindings
                                )
                            }
                        ),
                    )
                    adapter.seal(inventory)
                    for binding in bindings:
                        adapter.upsert(inventory, binding.projection)
                    proof = adapter.verify(
                        inventory, tuple(item.projection for item in bindings)
                    )
                    with get_session_with_tenant(tenant_id=tenant) as session:
                        for binding in bindings:
                            activate_temporal_projection(
                                session, user_file_id=file_id, binding=binding
                            )
                        authority.finalize(session, owner, proof)
                        session.commit()
                    index = ElasticsearchDocumentIndex(
                        TenantState(tenant_id=tenant, multitenant=True),
                        name,
                        3,
                        EmbeddingPrecision.FLOAT,
                    )
                    filters = IndexFilters(
                        access_control_list=None, as_of_date=date(2026, 9, 10)
                    )
                    # The vector is the sole controlled query-embedding boundary. No resolver,
                    # search transport, query builder or ES result is replaced.
                    results = [
                        index.keyword_retrieval("existing", filters, 10),
                        index.semantic_retrieval([0.1, 0.2, 0.3], filters, 10),
                        index.hybrid_retrieval(
                            "existing",
                            [0.1, 0.2, 0.3],
                            None,
                            QueryType.SEMANTIC,
                            filters,
                            10,
                        ),
                        index.id_based_retrieval(
                            [DocumentSectionRequest(document_id=str(file_id))], filters
                        ),
                    ]
                    for method, hits in zip(
                        ("keyword", "vector", "hybrid", "ID"), results, strict=True
                    ):
                        assert {hit.chunk_id for hit in hits} == {0, 1_000_000_042}, (
                            method
                        )
                        assert all(
                            hit.doc_summary == f"{status.value} summary\n"
                            for hit in hits
                        )
                        assert all(
                            hit.publication_index is not None
                            and hit.publication_index.search_settings_id == settings_id
                            and len(hit.publication_index.encoder_receipts) == 2
                            for hit in hits
                        )
                    if status == IndexModelStatus.FUTURE:
                        old_uuid = bindings[0].index.index_uuid
                        client.indices.delete(index=name)
                        client.indices.create(
                            index=name,
                            mappings=DocumentSchema.get_document_schema(3, True),
                        )
                        assert (
                            client.indices.get(index=name)[name]["settings"]["index"][
                                "uuid"
                            ]
                            != old_uuid
                        )
                        for binding in bindings:
                            client.index(
                                index=name,
                                id=str(binding.projection.ordinal),
                                document=json.loads(binding.projection.source_json),
                            )
                        client.indices.refresh(index=name)
                        with pytest.raises(
                            ValueError, match="no activated compatible index authority"
                        ):
                            index.keyword_retrieval("existing", filters, 10)
            finally:
                with get_session_with_tenant(tenant_id=tenant) as session:
                    session.execute(
                        delete(RegulatoryTemporalProjection).where(
                            RegulatoryTemporalProjection.user_file_id == file_id
                        )
                    )
                    session.commit()
    finally:
        for name in created_indices:
            client.indices.delete(index=name)
        client.close()
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
        assert tenant.startswith("tenant_5c_authority_") and len(tenant) == 52
        with SqlEngine.get_engine().begin() as connection:
            connection.execute(DropSchema(tenant, cascade=True, if_exists=True))
