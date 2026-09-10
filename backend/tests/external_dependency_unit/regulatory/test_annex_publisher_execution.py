"""Actual annex publisher and recovery with isolated PG/ES/FileStore fixtures."""

import json
from datetime import date
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.document_index.publication_models import (
    FileReservations,
    FrozenPublicationProjection,
)
from onyx.regulatory.amendments.annexes.publication_execution_models import (
    AnnexPublicationDelivery,
)
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_review_revisions import (
    LiveReview,
)
from tests.external_dependency_unit.regulatory.test_annex_review_revisions import (
    live_review as live_review,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    es as es,
)


def invoke(delivery: AnnexPublicationDelivery) -> str:
    from onyx.background.celery.tasks.regulatory_amendments.annex_publication import (
        publish_annex_change,
    )

    return publish_annex_change(
        intent_id=str(delivery.intent_id),
        change_set_id=str(delivery.change_set_id),
        logical_group_id=str(delivery.logical_group_id),
        review_revision=delivery.review_revision,
        review_sha256=delivery.review_sha256,
        publication_generation=delivery.publication_generation,
        tenant_id=delivery.tenant_id,
        environment=delivery.environment,
        database_identity=delivery.database_identity,
    )


def approve(
    session: Session, fixture: LiveReview, *, retry: bool = False
) -> AnnexPublicationDelivery:
    from onyx.db.regulatory_annex_changes import queue_annex_publication
    from onyx.regulatory.amendments.annexes import config
    from shared_configs.contextvars import get_current_tenant_id

    intent = queue_annex_publication(
        session,
        change_set_id=fixture.review.id,
        expected_review_sha256=fixture.review.review_sha256,
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        tenant_id=get_current_tenant_id(),
        database_identity=config.ANNEX_DATABASE_IDENTITY,
        decided_by=fixture.file.user_id,
        retry=retry,
    )
    return AnnexPublicationDelivery(
        intent_id=intent.id,
        change_set_id=intent.change_set_id,
        logical_group_id=intent.logical_group_id,
        review_revision=intent.review_revision,
        review_sha256=intent.review_sha256,
        publication_generation=intent.publication_generation,
        tenant_id=intent.tenant_id,
        environment=intent.environment,
        database_identity=intent.database_identity,
    )


@pytest.mark.parametrize(
    "live_review",
    [
        "verified",
        "source-only-verified",
        "temporary",
        "future",
        "historical-source-verified",
        "historical-derived-verified",
        "open-start",
        "cessation",
        "multipart",
        "verified-compatible-receipt",
    ],
    indirect=True,
)
def test_actual_worker_activates_frozen_complete_history(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.tasks.regulatory_amendments import (
        annex_publication as worker,
    )

    assert hasattr(worker, "publish_annex_change"), (
        "scoped durable annex publisher is missing"
    )
    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_evidence import (
        read_publication_preparation,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.publication is not None
    prepared = read_publication_preparation(draft.publication)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    calls: list[list[str]] = []

    def encode(*, texts: list[str], **_kwargs: object) -> list[list[float]]:
        calls.append(texts)
        return [[0.25, 0.5, 0.75] for _ in texts]

    for setting in settings:
        monkeypatch.setattr(
            DefaultIndexingEmbedder.from_db_search_settings(
                search_settings=setting
            ).embedding_model,
            "encode",
            encode,
        )
    payload = approve(source_session, live_review)
    result = invoke(payload)
    assert result == "approved"
    source_session.expire_all()
    assert live_review.review.status == "approved"
    from onyx.db.regulatory_annex_changes import capture_canonical_scope

    assert (
        capture_canonical_scope(source_session, live_review.file.id)
        == prepared.legal.canonical_rows
    )
    assert calls == [
        plan.context.embedding_texts
        for plan in prepared.projections
        if plan.reuse_from is None
    ]
    from onyx.db.regulatory_context_projections import get_indexed_temporal_projection
    from onyx.db.regulatory_publication import PublicationStore

    authority = PublicationStore(prepared.scope)
    assert not authority.unavailable(authority.observe(), (live_review.file.id,))
    for plan in prepared.projections:
        anchor = plan.effective_start or date(2010, 1, 1)
        binding = get_indexed_temporal_projection(
            source_session, plan.row.id, index=plan.index, as_of_date=anchor
        )
        assert binding is not None and binding.id == plan.id
        assert binding.representation_text == plan.row.text
        assert binding.representation_metadata == plan.row.metadata
        if plan.reuse_from:
            assert (
                json.loads(binding.projection.source_json)["content_vector"]
                == json.loads(plan.reuse_from.source_json)["content_vector"]
            )
    from sqlalchemy import select

    from onyx.db.models import RegulatoryContextGeneration, RegulatoryContextSnapshot

    for view in prepared.views.values():
        for snapshot in view.snapshots:
            persisted = source_session.scalar(
                select(RegulatoryContextSnapshot).where(
                    RegulatoryContextSnapshot.user_file_id == live_review.file.id,
                    RegulatoryContextSnapshot.sha256 == snapshot.sha256,
                )
            )
            assert persisted is not None and persisted.payload == snapshot.model_dump(
                mode="json"
            )
        for call in view.calls:
            persisted_call = source_session.scalar(
                select(RegulatoryContextGeneration).where(
                    RegulatoryContextGeneration.user_file_id == live_review.file.id,
                    RegulatoryContextGeneration.request_sha256 == call.request_sha256,
                )
            )
            assert (
                persisted_call is not None
                and persisted_call.payload == call.model_dump(mode="json")
            )
    from onyx.db.models import RegulatoryAnnexElementChunk, RegulatoryAnnexRevision
    from onyx.db.regulatory_annexes import (
        get_effective_annex_revision,
        get_revision_elements,
    )

    assert draft.baseline is not None and draft.baseline.revision_id is not None
    old_revision = source_session.get(
        RegulatoryAnnexRevision, UUID(draft.baseline.revision_id)
    )
    assert old_revision is not None and old_revision.effective_end == date(2026, 9, 10)
    before_source = get_effective_annex_revision(
        source_session, old_revision.annex_id, date(2020, 1, 1)
    )
    during_source = get_effective_annex_revision(
        source_session, old_revision.annex_id, date(2026, 9, 11)
    )
    assert before_source is not None and before_source.id == old_revision.id
    assert during_source is not None and during_source.id != old_revision.id
    assert draft.new_extraction is not None and draft.new_evidence_remapping is not None
    assert during_source.snapshot["extraction"] == draft.new_extraction.model_dump(
        mode="json"
    )
    assert during_source.snapshot[
        "new_evidence_remapping"
    ] == draft.new_evidence_remapping.model_dump(mode="json")
    assert {
        row.element_id
        for row in get_revision_elements(source_session, during_source.id)
    } == {item.element_id for item in draft.new_evidence_remapping.elements}
    source_links = list(
        source_session.scalars(
            select(RegulatoryAnnexElementChunk).where(
                RegulatoryAnnexElementChunk.revision_id == during_source.id
            )
        )
    )
    assert source_links
    if draft.after_window_authority:
        after_source = get_effective_annex_revision(
            source_session, old_revision.annex_id, date(2027, 1, 1)
        )
        if draft.after_window_authority.kind == "cessation":
            assert after_source is None
        else:
            assert (
                after_source is not None
                and after_source.snapshot == old_revision.snapshot
            )
    previous_calls = list(calls)
    assert invoke(payload) == "approved"
    assert calls == previous_calls


@pytest.mark.parametrize(
    "failure", ["second-item", "commit", "takeover", "new-generation"]
)
def test_partial_publication_recovery_retains_manifest_vectors_and_gate(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from datetime import timedelta
    from uuid import uuid4

    from sqlalchemy import event, select

    from onyx.db.models import AnnexPublicationEmbedding, AnnexPublicationManifest
    from onyx.db.regulatory_annex_changes import capture_canonical_scope
    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_evidence import (
        read_publication_preparation,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.publication is not None
    prepared = read_publication_preparation(draft.publication)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    calls: list[list[str]] = []

    def encode(*, texts: list[str], **_kwargs: object) -> list[list[float]]:
        calls.append(texts)
        return [[0.25, 0.5, 0.75] for _ in texts]

    for setting in settings:
        monkeypatch.setattr(
            DefaultIndexingEmbedder.from_db_search_settings(
                search_settings=setting
            ).embedding_model,
            "encode",
            encode,
        )
    payload = approve(source_session, live_review)
    authority = PublicationStore(prepared.scope)
    original = FencedPublicationIndex.upsert
    written = []

    def fail_item(
        self: FencedPublicationIndex,
        reservations: FileReservations,
        projection: FrozenPublicationProjection,
    ) -> None:
        if len(written) == 1:
            if failure == "takeover":
                previous = reservations.ownership
                authority.release(previous)
                successor = authority.acquire(
                    prepared.user_file_id, owner_id=uuid4(), ttl=timedelta(minutes=2)
                )
                self.seal(authority.reservations(successor))
                authority.release(successor)
                original(self, reservations, projection)
            raise RuntimeError("injected second item failure")
        original(self, reservations, projection)
        written.append(projection)

    def fail_commit(session: Session) -> None:
        if any(
            isinstance(row, AnnexPublicationManifest) and row.approved_at is not None
            for row in session.identity_map.values()
        ):
            raise RuntimeError("injected activation commit failure")

    if failure == "commit":
        event.listen(Session, "before_commit", fail_commit)
    else:
        monkeypatch.setattr(FencedPublicationIndex, "upsert", fail_item)
    try:
        with pytest.raises(Exception):
            invoke(payload)
    finally:
        if failure == "commit":
            event.remove(Session, "before_commit", fail_commit)
        monkeypatch.setattr(FencedPublicationIndex, "upsert", original)
    source_session.expire_all()
    assert (
        capture_canonical_scope(source_session, live_review.file.id)
        == live_review.before
    )
    assert live_review.file.id in authority.unavailable(
        authority.observe(), (live_review.file.id,)
    )
    manifest = source_session.get(AnnexPublicationManifest, live_review.review.id)
    assert manifest is not None and manifest.es_started and manifest.approved_at is None
    operations = manifest.operations
    vectors = list(
        source_session.scalars(
            select(AnnexPublicationEmbedding).where(
                AnnexPublicationEmbedding.change_set_id == live_review.review.id
            )
        )
    )
    assert len(vectors) == prepared.counts.embeddings and all(
        row.status == "complete" for row in vectors
    )
    assert live_review.review.status in ("preparing", "publishing")
    if failure != "takeover":
        assert (
            live_review.review.error_message
            == "Publication interrupted; the frozen manifest will be retried."
        )
    if failure == "new-generation":
        live_review.review.status = "failed"
        source_session.commit()
        next_payload = approve(source_session, live_review, retry=True)
        assert next_payload.publication_generation == payload.publication_generation + 1
        with pytest.raises(ValueError):
            invoke(payload)
        payload = next_payload
    first_calls = list(calls)
    assert invoke(payload) == "approved"
    source_session.expire_all()
    assert manifest.operations == operations
    assert calls == first_calls
    assert (
        capture_canonical_scope(source_session, live_review.file.id)
        == prepared.legal.canonical_rows
    )
    assert not authority.unavailable(authority.observe(), (live_review.file.id,))


@pytest.mark.parametrize("live_review", ["verified"], indirect=True)
def test_reconstructed_history_retires_unqualified_binding_without_erasing_evidence(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import select

    from onyx.db.models import RegulatoryTemporalProjection
    from onyx.db.regulatory_annex_changes import revise_annex_review
    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.document_index.publication_models import publication_digest
    from onyx.file_store.file_store import get_default_file_store
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes import analysis
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    prior = list(
        source_session.scalars(
            select(RegulatoryTemporalProjection).where(
                RegulatoryTemporalProjection.user_file_id == live_review.file.id
            )
        )
    )
    for row in prior:
        row.payload = {**row.payload, "context": None}
        row.payload_sha256 = publication_digest(row.payload)
    source_session.commit()
    old_payloads = {row.id: row.payload for row in prior}
    draft = analysis.prepare_review_context(
        AnnexChangeDraft.model_validate(live_review.review.review_payload)
    )
    assert draft.publication is not None
    try:
        live_review.review = revise_annex_review(
            source_session,
            change_set_id=live_review.review.id,
            expected_review_sha256=live_review.review.review_sha256,
            draft=draft,
            environment="local-test",
        )
        source_session.commit()
        _, settings, _ = load_annex_publication_inputs(source_session, draft)
        for setting in settings:
            monkeypatch.setattr(
                DefaultIndexingEmbedder.from_db_search_settings(
                    search_settings=setting
                ).embedding_model,
                "encode",
                lambda *, texts, **_kwargs: [[0.25, 0.5, 0.75] for _ in texts],
            )
        assert invoke(approve(source_session, live_review)) == "approved"
        source_session.expire_all()
        assert all(row.retired_at is not None for row in prior)
        assert {row.id: row.payload for row in prior} == old_payloads
    finally:
        get_default_file_store().delete_file(draft.publication.artifact_file_id)


@pytest.mark.parametrize(
    "transport",
    [
        "synchronous",
        "batch",
        "indeterminate",
        "wrong-model",
        "correlate",
        "correlate-wrong-id",
        "correlate-wrong-count",
        "correlate-wrong-model",
        "correlate-duplicate",
    ],
)
def test_durable_encoder_submissions_and_results_survive_redelivery(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
) -> None:
    from sqlalchemy import select

    from onyx.db.models import AnnexPublicationEmbedding, SearchSettings
    from onyx.db.regulatory_annex_changes import (
        capture_canonical_scope,
        revise_annex_review,
    )
    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.file_store.file_store import get_default_file_store
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes import analysis
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_evidence import (
        read_publication_preparation,
    )
    from onyx.regulatory.indexing_jobs.models import (
        IndexingGatewayIndeterminateSubmissionError,
        OpenRouterBatchConfig,
        RegulatoryIndexingConfigSnapshot,
        RegulatoryInputHashVersion,
        VertexAuthenticationMode,
        VertexBatchConfig,
    )
    from onyx.regulatory.indexing_jobs.openrouter_batch import (
        HttpxOpenRouterBatchGateway,
    )
    from shared_configs.enums import EmbeddingProvider

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    snapshot = RegulatoryIndexingConfigSnapshot(
        input_content_hash="a" * 64,
        input_hash_version=RegulatoryInputHashVersion.CANONICAL_V2,
        chunk_generation_hash="b" * 64,
        search_settings_id=settings[0].id,
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name="embedding",
        model_dimension=3,
        effective_dimension=3,
        index_name=settings[0].index_name,
        vertex=VertexBatchConfig(
            model_configuration_id=1,
            model_name="gemini-2.5-flash",
            project="test",
            location="us-central1",
            authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        ),
        prompt_version="test",
        prompt_hash="c" * 64,
        openrouter_batch=OpenRouterBatchConfig(
            api_url="https://openrouter.ai/api/beta/batches",
            model_name="embedding",
            effective_dimension=3,
        )
        if transport != "synchronous"
        else None,
    )
    monkeypatch.setattr(
        "onyx.configs.app_configs.REGULATORY_BATCH_INDEXING_ENABLED", True
    )
    monkeypatch.setattr(
        "onyx.regulatory.indexing_jobs.configuration.resolve_regulatory_indexing_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    live_review.llm.config = live_review.llm.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": snapshot.vertex.model_name}
    )
    live_review.llm.invoke.return_value.choice.message.content = (
        "Durable contextual output"
    )
    monkeypatch.setattr(
        analysis,
        "resolve_review_context_llm",
        lambda *_args, **_kwargs: live_review.llm,
    )
    model = DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=SearchSettings()
    ).embedding_model
    model.provider_type, model.reduced_dimension, model.api_key = (
        EmbeddingProvider.OPENROUTER,
        3,
        "fixture-only",
    )
    monkeypatch.setattr(
        model, "encode", lambda *, texts, **_kwargs: [[0.25, 0.5, 0.75] for _ in texts]
    )
    prepared_draft = analysis.prepare_review_context(draft)
    assert prepared_draft.publication is not None
    try:
        live_review.review = revise_annex_review(
            source_session,
            change_set_id=live_review.review.id,
            expected_review_sha256=live_review.review.review_sha256,
            draft=prepared_draft,
            environment="local-test",
        )
        source_session.commit()
        prepared = read_publication_preparation(prepared_draft.publication)
        payload = approve(source_session, live_review)
        sent: dict[str, dict[str, object]] = {}

        def request(
            _self: HttpxOpenRouterBatchGateway,
            method: str,
            url: str,
            *,
            json_body: dict[str, object] | None = None,
            indeterminate_key: str | None = None,
        ) -> dict[str, object]:
            if method == "POST":
                rows = list(
                    source_session.scalars(
                        select(AnnexPublicationEmbedding).where(
                            AnnexPublicationEmbedding.change_set_id
                            == live_review.review.id
                        )
                    )
                )
                source_session.expire_all()
                assert any(row.status == "submitting" for row in rows), (
                    "provider submission preceded durable checkpoint"
                )
                assert json_body is not None
                identity = f"batch-{len(sent)}"
                sent[identity] = json_body
                if (
                    transport == "indeterminate"
                    or transport.startswith("correlate")
                    and len(sent) == 1
                ):
                    raise IndexingGatewayIndeterminateSubmissionError(
                        indeterminate_key or ""
                    )
                return {"id": identity, "status": "validating"}
            identity = url.rsplit("/", 1)[-1]
            body = sent[identity]
            requests = body["requests"]
            assert isinstance(requests, list)
            results: list[dict[str, object]] = []
            for item in requests:
                assert isinstance(item, dict)
                request_item = cast(dict[str, object], item)
                custom_id = request_item["custom_id"]
                request_body = request_item["body"]
                assert isinstance(custom_id, str) and isinstance(request_body, dict)
                inputs = cast(dict[str, object], request_body)["input"]
                assert isinstance(inputs, list)
                count = len(inputs) + (transport == "correlate-wrong-count")
                results.append(
                    {
                        "custom_id": "foreign"
                        if transport == "correlate-wrong-id"
                        else custom_id,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "model": "wrong"
                                if transport in ("wrong-model", "correlate-wrong-model")
                                else "embedding",
                                "data": [
                                    {"index": i, "embedding": [0.25, 0.5, 0.75]}
                                    for i in range(count)
                                ],
                            },
                        },
                    }
                )
            if transport == "correlate-duplicate":
                results += results
            return {"id": identity, "status": "completed", "results": results}

        monkeypatch.setattr(HttpxOpenRouterBatchGateway, "_request", request)
        if transport.startswith("correlate"):
            with pytest.raises(IndexingGatewayIndeterminateSubmissionError):
                invoke(payload)
            from onyx.regulatory.amendments.annexes import publication_execution

            assert hasattr(publication_execution, "reconcile_annex_batch_submission"), (
                "bounded correlated batch resume is missing"
            )
            source_session.expire_all()
            from onyx.db.models import (
                AnnexPublicationManifest,
                RegulatoryFilePublication,
            )

            manifest = source_session.get(
                AnnexPublicationManifest, payload.change_set_id
            )
            gate = source_session.get(RegulatoryFilePublication, prepared.user_file_id)
            assert manifest is not None and not manifest.es_started
            assert gate is not None and not gate.gate_closed
            checkpoint = source_session.scalar(
                select(AnnexPublicationEmbedding).where(
                    AnnexPublicationEmbedding.change_set_id == live_review.review.id,
                    AnnexPublicationEmbedding.status == "indeterminate",
                )
            )
            assert checkpoint is not None and checkpoint.remote_id is None
            if transport != "correlate":
                with pytest.raises(ValueError):
                    publication_execution.reconcile_annex_batch_submission(
                        payload,
                        projection_id=checkpoint.projection_id,
                        candidate_remote_id="batch-0",
                    )
                source_session.expire_all()
                assert (
                    checkpoint.remote_id is None
                    and checkpoint.vectors is None
                    and live_review.review.status == "failed"
                )
                return
            assert (
                publication_execution.reconcile_annex_batch_submission(
                    payload,
                    projection_id=checkpoint.projection_id,
                    candidate_remote_id="batch-0",
                )
                == "preparing"
            )
            source_session.expire_all()
            assert checkpoint.remote_id == "batch-0" and checkpoint.status == "complete"
            assert checkpoint.provider_receipt is not None
            assert invoke(payload) == "preparing"
            assert invoke(payload) == "approved"
            assert len(sent) == prepared.counts.embeddings
            return
        if transport == "indeterminate":
            with pytest.raises(IndexingGatewayIndeterminateSubmissionError):
                invoke(payload)
            with pytest.raises(ValueError, match="manual provider reconciliation"):
                invoke(payload)
            assert len(sent) == 1
            source_session.expire_all()
            assert live_review.review.status == "failed"
            retried = approve(source_session, live_review, retry=True)
            with pytest.raises(ValueError, match="manual provider reconciliation"):
                invoke(retried)
            assert len(sent) == 1
            assert (
                capture_canonical_scope(source_session, live_review.file.id)
                == live_review.before
            )
            return
        if transport != "synchronous":
            assert invoke(payload) == "preparing"
            assert len(sent) == prepared.counts.embeddings
            assert (
                capture_canonical_scope(source_session, live_review.file.id)
                == live_review.before
            )
        if transport == "wrong-model":
            with pytest.raises(ValueError, match="model is mismatched"):
                invoke(payload)
            assert len(sent) == prepared.counts.embeddings
        else:
            assert invoke(payload) == "approved"
            assert (
                capture_canonical_scope(source_session, live_review.file.id)
                == prepared.legal.canonical_rows
            )
            assert all(
                plan.context.generation_path == "durable"
                for plan in prepared.projections
            )
    finally:
        get_default_file_store().delete_file(
            prepared_draft.publication.artifact_file_id
        )


def test_source_history_change_during_embedding_refuses_activation(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.models import RegulatoryAnnexRevision
    from onyx.db.regulatory_annex_changes import capture_canonical_scope
    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.baseline is not None and draft.baseline.revision_id is not None
    old = source_session.get(RegulatoryAnnexRevision, UUID(draft.baseline.revision_id))
    assert old is not None
    prior = old.effective_end
    _, settings, _ = load_annex_publication_inputs(source_session, draft)

    def encode(*, texts: list[str], **_kwargs: object) -> list[list[float]]:
        old.effective_end = date(2030, 1, 1)
        source_session.commit()
        return [[0.25, 0.5, 0.75] for _ in texts]

    monkeypatch.setattr(
        DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=settings[0]
        ).embedding_model,
        "encode",
        encode,
    )
    try:
        with pytest.raises(ValueError, match="source history changed"):
            invoke(approve(source_session, live_review))
        assert (
            capture_canonical_scope(source_session, live_review.file.id)
            == live_review.before
        )
    finally:
        source_session.rollback()
        old.effective_end = prior
        source_session.commit()


@pytest.mark.parametrize("change", ["canonical", "model", "index", "reservation"])
def test_publication_refuses_stale_preapproved_authority_before_encoding(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from datetime import timedelta
    from uuid import uuid4

    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.elasticsearch.client import ElasticsearchClient
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_evidence import (
        read_publication_preparation,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.publication is not None
    prepared = read_publication_preparation(draft.publication)
    _, settings, _ = load_annex_publication_inputs(source_session, draft)
    called = False

    def encode(**_kwargs: object) -> list[list[float]]:
        nonlocal called
        called = True
        raise AssertionError("stale approval reached encoder")

    monkeypatch.setattr(
        DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=settings[0]
        ).embedding_model,
        "encode",
        encode,
    )
    delivery = approve(source_session, live_review)
    if change == "canonical":
        live_review.outside.text = "unreviewed outside edit"
        source_session.commit()
    elif change == "model":
        settings[0].model_name = "wrong-space"
    elif change == "reservation":
        authority = PublicationStore(prepared.scope)
        owner = authority.acquire(
            live_review.file.id, owner_id=uuid4(), ttl=timedelta(minutes=2)
        )
        authority.allocate(owner, f"context:{uuid4()}")
        authority.release(owner)
    else:
        with ElasticsearchClient() as transport:
            client = transport.publication_client()
            name = prepared.indexes[0].index_name
            mapping = client.indices.get_mapping(index=name)[name]["mappings"]
            client.indices.delete(index=name)
            client.indices.create(index=name, mappings=mapping)
    with pytest.raises(ValueError):
        invoke(delivery)
    assert not called
    source_session.expire_all()
    assert live_review.review.status != "approved"


def test_scoped_recovery_redelivers_expired_or_lost_intent_without_new_generation(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery.tasks.regulatory_amendments.annex_publication import (
        ANNEX_PUBLICATION_TASK,
        recover_annex_publications,
    )
    from onyx.background.celery.versioned_apps.client import app
    from onyx.regulatory.amendments.annexes import config

    delivery = approve(source_session, live_review)
    sent: list[dict[str, object]] = []

    def send_task(name: str, **kwargs: object) -> None:
        assert name == ANNEX_PUBLICATION_TASK
        sent.append(kwargs)

    monkeypatch.setattr(app, "send_task", send_task)
    for _ in range(2):
        count = recover_annex_publications(
            tenant_id=delivery.tenant_id,
            environment=delivery.environment,
            database_identity=delivery.database_identity,
        )
        assert count >= 1
    own = [item for item in sent if item["kwargs"] == delivery.model_dump(mode="json")]
    assert len(own) == 2
    assert all(
        item["expires"] == 3600 and item["queue"] == config.publication_queue_name()
        for item in own
    )
    assert live_review.review.publication_generation == 1
    for changed in (
        {"environment": "wrong"},
        {"database_identity": "wrong"},
        {"tenant_id": "tenant_wrong"},
        {"review_sha256": "0" * 64},
        {"publication_generation": 2},
    ):
        with pytest.raises(ValueError):
            invoke(delivery.model_copy(update=changed))
