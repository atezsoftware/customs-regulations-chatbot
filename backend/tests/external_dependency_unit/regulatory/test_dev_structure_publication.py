"""Explicit DEV acceptance using an owned fixture and a disposable physical index."""

import json
import os
from datetime import timedelta
from io import BytesIO
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import delete

from onyx.db.engine.sql_engine import (
    SqlEngine,
    get_session_with_tenant,
    get_sqlalchemy_engine,
)
from onyx.db.models import (
    DocumentSet,
    RegulatoryCanonicalRevision,
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryPublicationOrdinal,
    RegulatoryTemporalProjection,
    SearchSettings,
    User,
    UserFile,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import (
    IndexedProjectionEvidence,
    PublicationScope,
    publication_digest,
    publication_source,
)
from onyx.regulatory.amendments.annexes.config import ANNEX_DATABASE_IDENTITY
from onyx.regulatory.publication_baseline import (
    observed_baseline_binding,
    observed_index_snapshot,
)
from onyx.regulatory.structure_metadata_repair import (
    plan_structure_repair,
    prepare_owned_structure_metadata,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file
from tests.unit.onyx.regulatory.test_publication_baseline import baseline_case


@pytest.mark.skipif(
    os.getenv("RUN_DEV_PUBLICATION_ACCEPTANCE") != "1",
    reason="Explicit owned DEV acceptance only",
)
def test_metadata_publication_recovers_after_index_write_and_preserves_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import search_settings as settings_repository
    from onyx.db.enums import IndexModelStatus
    from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
    from onyx.db.regulatory_canonical_revisions import (
        validate_temporal_canonical_revision,
    )
    from onyx.db.regulatory_writer_publication import (
        load_owned_writer_inputs,
        pending_writer_manifest,
    )
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.elasticsearch.schema import DocumentSchema
    from onyx.file_store import file_store
    from onyx.regulatory import writer_publication
    from onyx.regulatory.writer_publication_models import WriterPublicationManifest

    SqlEngine.init_engine(pool_size=5, max_overflow=3)
    assert get_sqlalchemy_engine().url.database == "customs-regulations-dev"
    transport = ElasticsearchClient(timeout=30)
    client = transport.publication_client()
    name = "dev-amendment-acceptance-" + uuid4().hex
    client.indices.create(
        index=name, mappings=DocumentSchema.get_document_schema(3, False)
    )
    file_id = user_id = group_id = settings_id = None
    authority = PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment="dev",
            database_identity=ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = None
    markdown = "KANUN\n\nMADDE 8 - (1) Tanımlar:\n\nğ) Tanım."
    try:
        with get_session_with_tenant(tenant_id="public") as session:
            group = DocumentSet(
                name=name,
                description="Owned publication acceptance",
                is_up_to_date=True,
            )
            session.add(group)
            session.flush()
            file = _file(session, group)
            file.name = "Regulation.md"
            canonical = _chunk(session, file, 0, "ğ) Tanım.")
            canonical.chunk_type = "clause"
            canonical.heading_path = ["KANUN", "MADDE 8", "g) Tanım"]
            canonical.chunk_metadata = {"article_no": "8", "clause_label": "g"}
            settings = SearchSettings(
                status=IndexModelStatus.PAST,
                index_name=name,
                model_name="fixture",
                model_dim=3,
                normalize=True,
                query_prefix="",
                passage_prefix="",
                enable_contextual_rag=False,
                multipass_indexing=False,
            )
            session.add(settings)
            session.flush()
            file_id, user_id, group_id, settings_id = (
                file.id,
                file.user_id,
                group.id,
                settings.id,
            )
            canonical_id = canonical.id
            session.commit()
            _ = settings.cloud_provider
            session.expunge(settings)
        monkeypatch.setattr(
            settings_repository, "get_active_search_settings_list", lambda _: [settings]
        )
        monkeypatch.setattr(
            file_store,
            "get_default_file_store",
            lambda: MagicMock(
                read_file=lambda *_args, **_kwargs: BytesIO(markdown.encode())
            ),
        )
        owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(minutes=3))
        inputs = load_owned_writer_inputs(owner)
        index = observed_index_snapshot(
            settings, client.indices.get(index=name)[name]["settings"]["index"]["uuid"]
        )
        _, example = baseline_case()
        source = json.loads(example.source_json)
        source["content_vector"] = [0.1, 0.2, 0.3]
        if "title_vector" in source:
            source["title_vector"] = [0.3, 0.2, 0.1]
        source.update(
            document_id=str(file_id),
            regulatory_chunk_id=canonical_id,
            chunk_index=0,
            content="ğ) Tanım.",
            blurb="ğ) Tanım.",
            heading_path=inputs.canonical[0].heading_path,
        )
        evidence = IndexedProjectionEvidence(
            index=index,
            source_json=json.dumps(source),
            frozen_projection=None,
            payload_sha256=None,
        )
        previous = observed_baseline_binding(evidence, inputs.canonical)
        baseline = WriterPublicationManifest(
            id=uuid4(),
            scope=owner.scope,
            user_file_id=file_id,
            kind="baseline",
            canonical_before_sha256=publication_digest(
                [row.model_dump(mode="json") for row in inputs.canonical]
            ),
            canonical_after=inputs.canonical,
            indexes=[index],
            previous_binding_ids=[],
            bindings=[previous],
            index_state_sha256=inputs.index_state_sha256,
        )
        writer_publication.execute_writer_publication(owner, client, baseline)
        authority.release(owner)
        owner = None
        owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(minutes=3))
        inputs = load_owned_writer_inputs(owner)
        plan = plan_structure_repair(
            inputs.canonical, markdown=markdown, source_file=inputs.file.name
        )
        assert len(plan.changes) == 1 and not plan.unresolved
        manifest = prepare_owned_structure_metadata(owner, client, plan)
        assert manifest.structure_repair == plan.model_dump(mode="json")
        finalizer = writer_publication.finalize_writer_publication

        def fail_before_activation(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("Injected interruption after verified index write")

        monkeypatch.setattr(
            writer_publication, "finalize_writer_publication", fail_before_activation
        )
        with pytest.raises(RuntimeError, match="Injected interruption"):
            writer_publication.execute_writer_publication(owner, client, manifest)
        assert pending_writer_manifest(owner) == manifest
        assert authority.reservations(owner).gate_closed
        with get_session_with_tenant(tenant_id="public") as session:
            row = session.get(RegulatoryChunk, canonical_id)
            assert row is not None and row.chunk_metadata["clause_label"] == "g"
        monkeypatch.setattr(
            writer_publication, "finalize_writer_publication", finalizer
        )
        writer_publication.execute_writer_publication(owner, client)
        assert not authority.reservations(owner).gate_closed
        with get_session_with_tenant(tenant_id="public") as session:
            row = session.get(RegulatoryChunk, canonical_id)
            assert (
                row is not None
                and row.chunk_metadata["clause_label"] == "ğ"
                and row.chunk_metadata["paragraph_no"] == "1"
            )
            assert row.text == "ğ) Tanım." and row.projection_ordinal == 0
            current = load_file_temporal_bindings(session, file_id)
            assert current == manifest.bindings
            retained = session.get(RegulatoryTemporalProjection, previous.id)
            assert (
                retained is not None
                and retained.retired_at is not None
                and retained.payload == previous.model_dump(mode="json")
            )
            validate_temporal_canonical_revision(session, retained)
        actual = FencedPublicationIndex(client, index).inventory_evidence(
            authority.reservations(owner)
        )
        assert len(actual) == 1
        after = publication_source(actual[0].source_json)
        assert after["heading_path"] == plan.changes[0].after.heading_path
        assert {k: v for k, v in after.items() if k != "heading_path"} == {
            k: v for k, v in source.items() if k != "heading_path"
        }
        from onyx.document_index.interfaces_new import MetadataUpdateRequest

        authority.release(owner)
        owner = None
        owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(minutes=3))
        writer_publication.publish_owned_metadata(
            owner,
            client,
            MetadataUpdateRequest(
                document_ids=[str(file_id)],
                doc_id_to_chunk_cnt={str(file_id): 1},
                project_ids={313},
            ),
            index_names=[name],
        )
        actual = FencedPublicationIndex(client, index).inventory_evidence(
            authority.reservations(owner)
        )
        synchronized = publication_source(actual[0].source_json)
        assert synchronized["user_projects"] == [313]
        assert {
            key: value for key, value in synchronized.items() if key != "user_projects"
        } == {key: value for key, value in after.items() if key != "user_projects"}
    finally:
        if owner is not None:
            authority.release(owner)
        client.indices.delete(index=name)
        client.close()
        if file_id is not None:
            with get_session_with_tenant(tenant_id="public") as session:
                for model in (
                    RegulatoryPublicationOrdinal,
                    RegulatoryFilePublication,
                    RegulatoryTemporalProjection,
                    RegulatoryCanonicalRevision,
                ):
                    session.execute(delete(model).where(model.user_file_id == file_id))
                session.execute(delete(UserFile).where(UserFile.id == file_id))
                user = session.get(User, user_id)
                if user is not None:
                    session.delete(user)
                session.execute(delete(DocumentSet).where(DocumentSet.id == group_id))
                session.execute(
                    delete(SearchSettings).where(SearchSettings.id == settings_id)
                )
                session.commit()
