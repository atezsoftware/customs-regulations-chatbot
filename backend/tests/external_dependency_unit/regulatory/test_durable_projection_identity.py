"""Real PostgreSQL checkpoints retain dated identities across batch redelivery."""

from datetime import date, datetime, timezone
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import select

from onyx.db import regulatory_indexing_jobs as repository
from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.enums import RegulatoryIndexingStage, UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    UserFile,
)
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from onyx.regulatory.indexing_jobs.embedding import build_openrouter_embedding_batch
from onyx.regulatory.indexing_jobs.models import OpenRouterBatchConfig
from onyx.regulatory.indexing_jobs.projection_identity import DurableProjectionInput
from onyx.regulatory.indexing_jobs.publisher import (
    _expected_verification,
    _verification_request,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    owned_file as owned_file,
)
from tests.unit.onyx.regulatory.indexing_jobs.test_publisher import (
    _snapshot as configuration_snapshot,
)


def test_dated_items_survive_partial_batch_checkpoint_with_canonical_counts(
    owned_file: UUID,
) -> None:
    snapshot = configuration_snapshot()
    with get_session_with_tenant(tenant_id="public") as session:
        file = session.get(UserFile, owned_file)
        assert file is not None
        file.status = UserFileStatus.INDEXING
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == owned_file)
                .order_by(RegulatoryChunk.position)
            )
        )
        job = RegulatoryIndexingJob(
            id=uuid4(),
            user_file_id=owned_file,
            content_hash=snapshot.input_content_hash,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            search_settings_id=snapshot.search_settings_id,
            prompt_hash=snapshot.prompt_hash,
            config_snapshot=snapshot.model_dump(mode="json"),
            status="RUNNING",
            stage="PREPARING",
            lease_generation=1,
        )
        session.add(job)
        session.commit()
        job_id = job.id
        frozen_rows = [_snapshot(row) for row in rows]
        prepared = [
            repository.RegulatoryIndexingPreparedItem(
                regulatory_chunk_id=rows[0].id,
                request_hash="shared-context-request",
                skip_context=False,
                projection_id=uuid4(),
                projection_ordinal=ordinal,
                effective_start=start,
                effective_end=end,
                projection_input=DurableProjectionInput(
                    representation=frozen_rows[0],
                    context_rows=frozen_rows,
                    reference_date=start,
                ),
            )
            for ordinal, start, end in [
                (0, None, date(2030, 1, 1)),
                (17, date(2030, 1, 1), None),
            ]
        ]
        prepared.append(
            repository.RegulatoryIndexingPreparedItem(
                regulatory_chunk_id=rows[1].id,
                request_hash="legacy-pending",
                skip_context=False,
            )
        )
        assert repository.persist_regulatory_indexing_preparation(
            session,
            job_id=job_id,
            expected_generation=1,
            prepare_items=lambda: prepared,
            resolved_input_hash_version="canonical-v2",
            now=datetime.now(timezone.utc),
        )
        assert repository.claim_regulatory_indexing_job(
            session,
            job_id=job_id,
            expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
            expected_generation=1,
            now=datetime.now(timezone.utc),
        )
        items = list(
            session.scalars(
                select(RegulatoryIndexingItem).where(
                    RegulatoryIndexingItem.job_id == job_id
                )
            )
        )
        assert len(items) == 3
        for item in items:
            assert repository.persist_regulatory_indexing_item_context(
                session,
                item_id=item.id,
                expected_generation=2,
                context={"contextual_text": f"Context {item.request_hash}. "},
            )
        session.expire_all()
        job = session.get(RegulatoryIndexingJob, job_id)
        assert job is not None
        items = list(
            session.scalars(
                select(RegulatoryIndexingItem).where(
                    RegulatoryIndexingItem.job_id == job_id
                )
            )
        )
        config = OpenRouterBatchConfig(
            api_url="http://127.0.0.1:23124/api/beta/batches",
            model_name=snapshot.embedding_model_name,
            effective_dimension=3,
            max_inputs=1,
        )
        first = build_openrouter_embedding_batch(
            job=job, rows=rows, items=items, config=config, max_attempts=3
        )
        first_id = next(iter(first.item_ids_by_custom_id.values()))[0]
        assert first.selected_item_count == 1 and first.remaining_item_count == 2
        assert repository.persist_regulatory_indexing_item_vector(
            session, item_id=first_id, expected_generation=2, vector=[0.1, 0.2, 0.3]
        )
    # A fresh session models process loss after the first durable result commit.
    with get_session_with_tenant(tenant_id="public") as session:
        job = session.get(RegulatoryIndexingJob, job_id)
        assert job is not None
        rows = list(
            session.scalars(
                select(RegulatoryChunk).where(
                    RegulatoryChunk.user_file_id == owned_file
                )
            )
        )
        items = list(
            session.scalars(
                select(RegulatoryIndexingItem).where(
                    RegulatoryIndexingItem.job_id == job_id
                )
            )
        )
        resumed = build_openrouter_embedding_batch(
            job=job,
            rows=rows,
            items=items,
            config=config.model_copy(update={"max_inputs": 10}),
            max_attempts=3,
        )
        remaining_ids = [
            identifier
            for group in resumed.item_ids_by_custom_id.values()
            for identifier in group
        ]
        assert first_id not in remaining_ids and len(remaining_ids) == 2
        assert (
            resumed.requests[0].inputs[0].startswith("Context shared-context-request. ")
        )
        for ordinal, identifier in enumerate(remaining_ids, start=1):
            assert repository.persist_regulatory_indexing_item_vector(
                session,
                item_id=identifier,
                expected_generation=2,
                vector=[float(ordinal), 0.2, 0.3],
            )
        session.expire_all()
        items = list(
            session.scalars(
                select(RegulatoryIndexingItem).where(
                    RegulatoryIndexingItem.job_id == job_id
                )
            )
        )
        counts = _expected_verification(
            job_id=job_id,
            user_file_id=owned_file,
            rows=rows,
            items=items,
            snapshot=snapshot,
        )
        assert counts.canonical_chunk_count == 2 and counts.embedded_item_count == 3
        request = _verification_request(
            expected=counts, rows=rows, items=items, hidden=True
        )
        assert [value.chunk_index for value in request.expected_chunks] == [
            0,
            17,
            1_000_000_042,
        ]
        assert len({item.id for item in items}) == 3


def test_legacy_vectors_requeue_selectively_without_changing_batch_identity(
    owned_file: UUID,
    monkeypatch,
) -> None:
    from onyx.regulatory.indexing_jobs.embedding_receipts import (
        batch_embedding_receipts,
    )

    snapshot = configuration_snapshot()
    config = OpenRouterBatchConfig(
        api_url="http://127.0.0.1:23124/api/beta/batches",
        model_name=snapshot.embedding_model_name,
        effective_dimension=3,
    )
    with get_session_with_tenant(tenant_id="public") as session:
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(
                    RegulatoryChunk.user_file_id == owned_file,
                )
                .order_by(RegulatoryChunk.position)
            )
        )
        job = RegulatoryIndexingJob(
            id=uuid4(),
            user_file_id=owned_file,
            content_hash=snapshot.input_content_hash,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            search_settings_id=snapshot.search_settings_id,
            prompt_hash=snapshot.prompt_hash,
            config_snapshot=snapshot.model_copy(
                update={"openrouter_batch": config}
            ).model_dump(mode="json"),
            status="RUNNING",
            stage="EMBEDDING",
            lease_generation=1,
        )
        session.add(job)
        session.flush()
        items = [
            RegulatoryIndexingItem(
                id=uuid4(),
                job_id=job.id,
                regulatory_chunk_id=row.id,
                request_hash=f"context-{offset}",
                status="CONTEXT_READY",
                context={
                    "contextual_text": f"Context {offset}. ",
                    "raw_contextual_text": f"Context {offset}. ",
                },
            )
            for offset, row in enumerate(rows)
        ]
        session.add_all(items)
        session.commit()
        receipts = batch_embedding_receipts(
            job=job, rows=rows, items=items, config=config
        )
        identities = [
            (item.id, item.regulatory_chunk_id, item.request_hash) for item in items
        ]
        assert (
            repository.freeze_regulatory_embedding_receipts(
                session,
                job_id=job.id,
                expected_generation=1,
                receipts=receipts,
            )
            == 0
        )
        assert repository.persist_regulatory_indexing_item_vectors(
            session,
            job_id=job.id,
            expected_generation=1,
            item_vectors=[(items[0].id, [0.1, 0.2, 0.3])],
        )
        # An old checkpoint has a vector but never recorded its complete request.
        items[1].status = "EMBEDDED"
        items[1].embedding_attempt_count = 3
        items[1].vector = [0.4, 0.5, 0.6]
        items[1].context = dict[str, object](
            contextual_text="Context 1. ", raw_contextual_text="Context 1. "
        )
        session.commit()
        assert (
            repository.freeze_regulatory_embedding_receipts(
                session,
                job_id=job.id,
                expected_generation=1,
                receipts=receipts,
            )
            == 1
        )
        session.expire_all()
        assert items[0].status == "EMBEDDED" and items[0].vector == [0.1, 0.2, 0.3]
        assert items[1].status == "CONTEXT_READY" and items[1].vector is None
        assert _context(items[1])["raw_contextual_text"] == "Context 1. "
        assert identities == [
            (item.id, item.regulatory_chunk_id, item.request_hash) for item in items
        ]
        plan = build_openrouter_embedding_batch(
            job=job, rows=rows, items=items, config=config, max_attempts=3
        )
        assert list(plan.item_ids_by_custom_id.values()) == [(items[1].id,)]
        assert plan.requests[0].inputs == ["Context 1. " + rows[1].text]
        assert repository.record_openrouter_submission_intent(
            session,
            job_id=job.id,
            expected_generation=1,
            submission_key="unknown-post",
            submission_attempt=1,
            active_item_ids=[items[1].id],
            now=datetime.now(timezone.utc),
        )
        import pytest

        with pytest.raises(ValueError, match="active submission"):
            repository.freeze_regulatory_embedding_receipts(
                session,
                job_id=job.id,
                expected_generation=1,
                receipts=receipts,
            )
        session.expire_all()
        assert job.openrouter_submission_state == "SUBMITTING"
        assert job.openrouter_submission_key == "unknown-post"
        assert job.openrouter_active_item_ids == [str(items[1].id)]

        # Simulate an old completed vector at the actual transport entry point.
        job.openrouter_submission_state = "NONE"
        job.openrouter_submission_key = None
        job.openrouter_active_item_ids = []
        items[1].status = "EMBEDDED"
        items[1].embedding_attempt_count = 3
        items[1].vector = [0.4, 0.5, 0.6]
        items[1].context = dict[str, object](
            contextual_text="Context 1. ", raw_contextual_text="Context 1. "
        )
        session.commit()
        from onyx.regulatory.indexing_jobs import orchestrator
        from onyx.regulatory.indexing_jobs.models import (
            IndexingGatewayIndeterminateSubmissionError,
        )
        from onyx.regulatory.indexing_jobs.openrouter_batch import (
            OpenRouterBatchContractError,
        )

        submissions = []

        class Gateway:
            def submit(self, requests, *, submission_key):
                submissions.append(requests)
                raise IndexingGatewayIndeterminateSubmissionError(submission_key)

        monkeypatch.setattr(
            orchestrator, "_build_openrouter_gateway", lambda _runtime: Gateway()
        )
        runtime = repository.get_regulatory_indexing_runtime(session, job.id)
        assert runtime is not None
        with pytest.raises(OpenRouterBatchContractError, match="manual reconciliation"):
            orchestrator._openrouter_embedding_batch(
                runtime,
                tenant_id="public",
                db_session=session,
                now=datetime.now(timezone.utc),
            )
        session.expire_all()
        assert len(submissions) == 1
        assert submissions[0][0].inputs == ["Context 1. " + rows[1].text]
        assert items[0].vector == [0.1, 0.2, 0.3]
        assert items[1].vector is None
        assert _context(items[1])["embedding_receipt"] == receipts[
            items[1].id
        ].model_dump(mode="json")
        runtime = repository.get_regulatory_indexing_runtime(session, job.id)
        assert runtime is not None
        with pytest.raises(OpenRouterBatchContractError, match="manual reconciliation"):
            orchestrator._openrouter_embedding_batch(
                runtime,
                tenant_id="public",
                db_session=session,
                now=datetime.now(timezone.utc),
            )
        assert len(submissions) == 1


def test_synchronous_durable_transport_freezes_before_reencoding_legacy_vector(
    owned_file: UUID,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from onyx.regulatory.indexing_jobs import embedding
    from onyx.regulatory.indexing_jobs.embedding_receipts import (
        has_proven_vector,
        synchronous_embedding_receipts,
    )
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
    )
    from tests.unit.onyx.regulatory.indexing_jobs.test_embedding import _settings

    snapshot = configuration_snapshot()
    settings = _settings(snapshot)
    calls = []
    model = SimpleNamespace(
        provider_type=snapshot.embedding_provider,
        model_name=snapshot.embedding_model_name,
        reduced_dimension=3,
        normalize=True,
        passage_prefix=None,
        retrim_content=False,
        api_url=None,
        api_version=None,
        deployment_name=None,
        tokenizer=_CharacterTokenizer(),
    )
    monkeypatch.setattr(
        embedding,
        "DefaultIndexingEmbedder",
        lambda **_kwargs: SimpleNamespace(embedding_model=model),
    )
    with get_session_with_tenant(tenant_id="public") as session:
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(
                    RegulatoryChunk.user_file_id == owned_file,
                )
                .order_by(RegulatoryChunk.position)
            )
        )
        job = RegulatoryIndexingJob(
            id=uuid4(),
            user_file_id=owned_file,
            content_hash=snapshot.input_content_hash,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            search_settings_id=snapshot.search_settings_id,
            prompt_hash=snapshot.prompt_hash,
            config_snapshot=snapshot.model_dump(mode="json"),
            status="RUNNING",
            stage="EMBEDDING",
            lease_generation=1,
        )
        session.add(job)
        session.flush()
        items = [
            RegulatoryIndexingItem(
                id=uuid4(),
                job_id=job.id,
                regulatory_chunk_id=row.id,
                request_hash=f"context-{offset}",
                status="CONTEXT_READY",
                context={"contextual_text": f"Context {offset}. "},
            )
            for offset, row in enumerate(rows)
        ]
        session.add_all(items)
        session.commit()
        receipts = synchronous_embedding_receipts(
            job=job, rows=rows, items=items, model=cast(EmbeddingModel, model)
        )
        assert (
            repository.freeze_regulatory_embedding_receipts(
                session,
                job_id=job.id,
                expected_generation=1,
                receipts=receipts,
            )
            == 0
        )
        assert repository.persist_regulatory_indexing_item_vectors(
            session,
            job_id=job.id,
            expected_generation=1,
            item_vectors=[(items[0].id, [0.1, 0.2, 0.3])],
        )
        items[1].status = "EMBEDDED"
        items[1].vector = [0.4, 0.5, 0.6]
        items[1].context = dict[str, object](contextual_text="Context 1. ")
        session.commit()
        legacy_id = items[1].id

        def encode(*, texts, **_kwargs):
            with get_session_with_tenant(tenant_id="public") as other:
                persisted = other.get(RegulatoryIndexingItem, legacy_id)
                assert persisted is not None and persisted.vector is None
                assert _context(persisted)["embedding_receipt"] == receipts[
                    legacy_id
                ].model_dump(mode="json")
            calls.append(texts)
            return [[0.7, 0.8, 0.9]]

        model.encode = encode
        summary = embedding.embed_pending_regulatory_items(
            job=job,
            rows=rows,
            items=items,
            search_settings=settings,
            tenant_id="public",
            db_session=session,
        )
        assert summary.embedded_count == 1 and summary.reused_count == 1
        assert calls == [["Context 1. " + rows[1].text]]
        session.expire_all()
        assert has_proven_vector(items[1], receipts[legacy_id])
        resumed = embedding.embed_pending_regulatory_items(
            job=job,
            rows=rows,
            items=items,
            search_settings=settings,
            tenant_id="public",
            db_session=session,
        )
        assert resumed.embedded_count == 0 and resumed.reused_count == 2
        assert len(calls) == 1


def _context(item: RegulatoryIndexingItem) -> dict[str, object]:
    context = item.context
    assert context is not None
    return context
