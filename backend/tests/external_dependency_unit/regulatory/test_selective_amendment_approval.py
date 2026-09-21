"""Real PG/ES approval, with only the embedding provider replaced by a fixture."""

import json
from datetime import date, timedelta
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from elasticsearch import Elasticsearch
from sqlalchemy import delete, select

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.enums import IndexModelStatus
from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    DocumentSet__UserFile,
    RegulatoryChunk,
    SearchSettings,
)
from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
from onyx.document_index.publication_models import PublicationScope
from onyx.regulatory import writer_projection, writer_publication
from onyx.regulatory.amendments.annexes import config
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    es as es,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    owned_file as owned_file,
)
from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
    _durable_embedding_model,
)


def test_selective_approval_preserves_other_binding_and_recovers_atomic_activation(
    owned_file: UUID, es: tuple[Elasticsearch, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.db import search_settings as settings_repository
    from onyx.document_index.elasticsearch.client import ElasticsearchClient

    with get_session_with_tenant(tenant_id="public") as session:
        settings = SearchSettings(
            model_name="embedding",
            model_dim=3,
            normalize=True,
            query_prefix="",
            passage_prefix="",
            status=IndexModelStatus.PRESENT,
            index_name=es[1],
            enable_contextual_rag=False,
        )
        session.add(settings)
        session.commit()
        setting_id = settings.id
        assert settings.cloud_provider is None
        session.expunge(settings)
    monkeypatch.setattr(
        settings_repository,
        "get_active_search_settings_list",
        lambda _session: [settings],
    )
    monkeypatch.setattr(
        writer_projection, "resolve_review_context_llm", lambda *_args: None
    )
    monkeypatch.setattr(
        "onyx.regulatory.projection.effective_contextual_rag_enabled", lambda _: False
    )
    model = _durable_embedding_model()
    encode = MagicMock(
        side_effect=lambda *, texts, **_kwargs: [[0.25, 0.5, 0.75] for _ in texts]
    )
    monkeypatch.setattr(model, "encode", encode)
    monkeypatch.setattr(
        writer_projection.DefaultIndexingEmbedder,
        "from_db_search_settings",
        lambda **_kwargs: MagicMock(embedding_model=model),
    )
    monkeypatch.setattr(ElasticsearchClient, "publication_client", lambda _self: es[0])
    authority = PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=3))
    inputs = load_owned_writer_inputs(owner)
    manifest = writer_projection.prepare_owned_correction(
        owner, es[0], inputs, inputs.canonical, changed_id=None
    )
    writer_publication.execute_writer_publication(owner, es[0], manifest)
    authority.release(owner)
    with get_session_with_tenant(tenant_id="public") as session:
        before = load_file_temporal_bindings(session, owned_file)
        old = session.get_one(RegulatoryChunk, inputs.canonical[0].id)
        batch = AmendmentBatch(
            document_set_id=session.scalar(
                select(DocumentSet__UserFile.document_set_id).where(
                    DocumentSet__UserFile.user_file_id == owned_file
                )
            ),
            raw_text="MADDE 1 changed",
            user_file_ids=[str(owned_file)],
            status="analyzed",
        )
        session.add(batch)
        session.flush()
        proposal = AmendmentProposal(
            batch_id=batch.id,
            instruction_index=0,
            instruction_text="MADDE 1 changed",
            instruction_indices=[0],
            instruction_texts=["MADDE 1 changed"],
            old_chunk_id=old.id,
            old_chunk_snapshot={"id": old.id, "text": old.text},
            new_chunk_draft={
                "user_file_id": str(owned_file),
                "position": old.position,
                "text": "Only approved provision changes",
                "chunk_type": old.chunk_type,
                "heading_path": old.heading_path,
                "metadata": old.chunk_metadata,
                "effective_start_date": "2027-01-01",
                "effective_end_date": None,
            },
            status="approving",
        )
        session.add(proposal)
        session.commit()
        proposal_id, batch_id = proposal.id, batch.id
    real_finalize = writer_publication.finalize_writer_publication

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("interrupted before atomic activation")

    encode.reset_mock()
    monkeypatch.setattr(writer_publication, "finalize_writer_publication", interrupt)
    with pytest.raises(RuntimeError, match="interrupted before atomic activation"):
        writer_publication.approve_owned_amendment(proposal_id, "public", setting_id)
    assert encode.call_count == 1
    with get_session_with_tenant(tenant_id="public") as session:
        assert session.get_one(AmendmentProposal, proposal_id).status == "approving"
        assert (
            session.get_one(RegulatoryChunk, inputs.canonical[0].id).validity_end_date
            is None
        )
    monkeypatch.setattr(
        writer_publication, "finalize_writer_publication", real_finalize
    )
    writer_publication.approve_owned_amendment(proposal_id, "public", setting_id)
    assert encode.call_count == 1
    with get_session_with_tenant(tenant_id="public") as session:
        assert session.get_one(AmendmentProposal, proposal_id).status == "approved"
        assert session.get_one(
            RegulatoryChunk, inputs.canonical[0].id
        ).validity_end_date == date(2027, 1, 1)
        after = load_file_temporal_bindings(session, owned_file)
        untouched = next(
            b
            for b in before
            if json.loads(b.projection.source_json)["regulatory_chunk_id"]
            == inputs.canonical[1].id
        )
        assert untouched in after
        assert len(after) == 3
        session.execute(
            delete(AmendmentProposal).where(AmendmentProposal.id == proposal_id)
        )
        session.execute(delete(AmendmentBatch).where(AmendmentBatch.id == batch_id))
        session.execute(delete(SearchSettings).where(SearchSettings.id == setting_id))
        session.commit()
