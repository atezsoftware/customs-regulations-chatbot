"""Fresh, isolated DEV approval through the real PG/ES publication contract."""

import os
from datetime import timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from onyx.db.engine.sql_engine import (
    SqlEngine,
    get_session_with_tenant,
    get_sqlalchemy_engine,
)
from onyx.db.enums import IndexModelStatus
from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
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
from onyx.db.regulatory_amendment_order import load_amendment_order
from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
from onyx.document_index.publication_models import PublicationScope, publication_source
from onyx.regulatory import writer_projection, writer_publication
from onyx.regulatory.amendments.annexes.config import ANNEX_DATABASE_IDENTITY
from onyx.regulatory.amendments.insertion_order import plan_insertion
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file
from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
    _durable_embedding_model,
)


@pytest.mark.skipif(
    os.getenv("RUN_DEV_PUBLICATION_ACCEPTANCE") != "1",
    reason="Explicit owned DEV acceptance only",
)
def test_insertion_approval_preserves_existing_payloads_and_recovers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import search_settings as settings_repository
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.elasticsearch.schema import DocumentSchema
    from shared_configs.enums import EmbeddingProvider

    SqlEngine.init_engine(pool_size=5, max_overflow=3)
    assert get_sqlalchemy_engine().url.database == "customs-regulations-dev"
    transport = ElasticsearchClient(timeout=30)
    client = transport.publication_client()
    name = "dev-amendment-acceptance-" + uuid4().hex
    client.indices.create(
        index=name, mappings=DocumentSchema.get_document_schema(3, False)
    )
    file_id = user_id = group_id = settings_id = batch_id = None
    authority = PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment="dev",
            database_identity=ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = None
    try:
        with get_session_with_tenant(tenant_id="public") as session:
            group = DocumentSet(
                name=name, description="Owned insertion acceptance", is_up_to_date=True
            )
            session.add(group)
            session.flush()
            file = _file(session, group)
            rows = []
            for position, (article, paragraph, body) in enumerate(
                [
                    ("8", "1", "(1) First."),
                    ("8", "2", "(2) Second."),
                    ("8", "3", "(3) Third."),
                    ("9", "1", "MADDE 9- (1) Next article."),
                ]
            ):
                row = _chunk(session, file, position, body)
                row.chunk_type = "paragraph"
                row.heading_path = ["KANUN", f"MADDE {article}", f"Fıkra {paragraph}"]
                row.chunk_metadata = {"article_no": article, "paragraph_no": paragraph}
                rows.append(row)
            settings = SearchSettings(
                status=IndexModelStatus.PAST,
                index_name=name,
                model_name="embedding",
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
            existing_ids = [row.id for row in rows]
            existing_fields = {
                row.id: (row.text, row.projection_ordinal) for row in rows
            }
            session.commit()
            _ = settings.cloud_provider
            session.expunge(settings)
        # PRESENT is confined to this process; no live active index setting changes.
        settings.status = IndexModelStatus.PRESENT
        settings.provider_type = EmbeddingProvider.OPENROUTER
        monkeypatch.setattr(
            settings_repository, "get_active_search_settings_list", lambda _: [settings]
        )
        monkeypatch.setattr(
            writer_projection, "resolve_review_context_llm", lambda *_: None
        )
        monkeypatch.setattr(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            lambda _: False,
        )
        model = _durable_embedding_model()
        encode = MagicMock(
            side_effect=lambda *, texts, **_: [[0.25, 0.5, 0.75] for _ in texts]
        )
        monkeypatch.setattr(model, "encode", encode)
        monkeypatch.setattr(
            writer_projection.DefaultIndexingEmbedder,
            "from_db_search_settings",
            lambda **_: MagicMock(embedding_model=model),
        )
        monkeypatch.setattr(ElasticsearchClient, "publication_client", lambda _: client)
        owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(minutes=3))
        inputs = load_owned_writer_inputs(owner)
        baseline = writer_projection.prepare_owned_correction(
            owner, client, inputs, inputs.canonical, changed_id=None
        )
        writer_publication.execute_writer_publication(owner, client, baseline)
        before_evidence = FencedPublicationIndex(
            client, baseline.indexes[0]
        ).inventory_evidence(authority.reservations(owner))
        before_payloads = {
            publication_source(item.source_json)[
                "regulatory_chunk_id"
            ]: publication_source(item.source_json)
            for item in before_evidence
        }
        authority.release(owner)
        owner = None
        with get_session_with_tenant(tenant_id="public") as session:
            before_bindings = load_file_temporal_bindings(session, file_id)
            order = plan_insertion(
                load_amendment_order(session, file_id),
                article_no="8",
                paragraph_no="4",
                clause_label=None,
            )
            assert order.position == 3 and order.before_chunk_id == existing_ids[3]
            batch = AmendmentBatch(
                document_set_id=group_id,
                raw_text="Owned insertion",
                user_file_ids=[str(file_id)],
                status="analyzed",
            )
            session.add(batch)
            session.flush()
            proposal = AmendmentProposal(
                batch_id=batch.id,
                instruction_index=0,
                instruction_indices=[0],
                instruction_text="MADDE 1- Kanunun 8 inci maddesine aşağıdaki fıkra eklenmiştir. “(4) Added rule.”",
                old_chunk_id=None,
                old_chunk_snapshot={},
                status="approving",
                new_chunk_draft={
                    "user_file_id": str(file_id),
                    "position": order.position,
                    "text": "(4) Added rule.",
                    "chunk_type": "paragraph",
                    "heading_path": ["KANUN", "MADDE 8", "Fıkra 4"],
                    "metadata": {"article_no": "8", "paragraph_no": "4"},
                    "effective_start_date": "2027-01-01",
                    "effective_end_date": None,
                    "insertion_order": order.model_dump(mode="json"),
                },
            )
            session.add(proposal)
            session.commit()
            proposal_id, batch_id = proposal.id, batch.id
        encode.reset_mock()
        finalize = writer_publication.finalize_writer_publication

        def interrupt(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("Injected insertion interruption before activation")

        monkeypatch.setattr(
            writer_publication, "finalize_writer_publication", interrupt
        )
        with pytest.raises(RuntimeError, match="Injected insertion interruption"):
            writer_publication.approve_owned_amendment(
                proposal_id, "public", settings_id
            )
        assert encode.call_count == 1
        assert encode.call_args.kwargs["texts"] == ["(4) Added rule.", "Regulation"]
        with get_session_with_tenant(tenant_id="public") as session:
            assert session.get_one(AmendmentProposal, proposal_id).status == "approving"
            assert session.get_one(RegulatoryChunk, existing_ids[3]).position == 3
            assert (
                len(
                    list(
                        session.scalars(
                            select(RegulatoryChunk.id).where(
                                RegulatoryChunk.user_file_id == file_id
                            )
                        )
                    )
                )
                == 4
            )
        monkeypatch.setattr(writer_publication, "finalize_writer_publication", finalize)
        writer_publication.approve_owned_amendment(proposal_id, "public", settings_id)
        assert encode.call_count == 1
        with get_session_with_tenant(tenant_id="public") as session:
            proposal = session.get_one(AmendmentProposal, proposal_id)
            assert proposal.status == "approved"
            added = session.get_one(RegulatoryChunk, proposal.applied_new_chunk_id)
            assert added.position == 3 and added.chunk_metadata["paragraph_no"] == "4"
            assert session.get_one(RegulatoryChunk, existing_ids[3]).position == 4
            after_bindings = load_file_temporal_bindings(session, file_id)
            for identifier, expected in existing_fields.items():
                row = session.get_one(RegulatoryChunk, identifier)
                assert (row.text, row.projection_ordinal) == expected
            assert len(after_bindings) == 5
            for old in before_bindings:
                retained = session.get_one(RegulatoryTemporalProjection, old.id)
                assert retained.payload == old.model_dump(mode="json")
            assert (
                session.get_one(SearchSettings, settings_id).status
                == IndexModelStatus.PAST
            )
        owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(minutes=3))
        after_evidence = FencedPublicationIndex(
            client, baseline.indexes[0]
        ).inventory_evidence(authority.reservations(owner))
        after_payloads = {
            publication_source(item.source_json)[
                "regulatory_chunk_id"
            ]: publication_source(item.source_json)
            for item in after_evidence
        }
        assert len(after_payloads) == 5
        assert {
            identifier: after_payloads[identifier] for identifier in existing_ids
        } == before_payloads
    finally:
        if owner is not None:
            authority.release(owner)
        client.indices.delete(index=name)
        client.close()
        if file_id is not None:
            with get_session_with_tenant(tenant_id="public") as session:
                if batch_id is not None:
                    session.execute(
                        delete(AmendmentProposal).where(
                            AmendmentProposal.batch_id == batch_id
                        )
                    )
                    session.execute(
                        delete(AmendmentBatch).where(AmendmentBatch.id == batch_id)
                    )
                for table in (
                    RegulatoryPublicationOrdinal,
                    RegulatoryFilePublication,
                    RegulatoryTemporalProjection,
                    RegulatoryCanonicalRevision,
                ):
                    session.execute(delete(table).where(table.user_file_id == file_id))
                session.execute(delete(UserFile).where(UserFile.id == file_id))
                user = session.get(User, user_id)
                if user is not None:
                    session.delete(user)
                session.execute(delete(DocumentSet).where(DocumentSet.id == group_id))
                session.execute(
                    delete(SearchSettings).where(SearchSettings.id == settings_id)
                )
                session.commit()
