from collections.abc import Generator
from dataclasses import dataclass
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from elasticsearch import Elasticsearch
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentBatch,
    AnnexChangeSet,
    AnnexPublicationIntent,
    DocumentSet,
    RegulatoryChunk,
    UserFile,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexChangeDraft,
)
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    es as es,
)


def test_review_revision_preserves_old_payload_and_rejects_stale_hash(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_annex_changes import (
        persist_annex_checkpoint,
        require_current_annex_review,
        revise_annex_review,
    )

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="EK-1",
        status="analyzing",
        lease_generation=1,
        instruction_count=1,
    )
    source_session.add(batch)
    source_session.flush()
    draft = AnnexChangeDraft(
        instruction_indices=[0],
        instruction_texts=["EK-1"],
        annex_label="EK-1",
        issues=["missing_original"],
    )
    first = persist_annex_checkpoint(
        source_session,
        batch_id=batch.id,
        lease_generation=1,
        draft=draft,
        environment="local-test",
    )
    assert first is not None
    second = revise_annex_review(
        source_session,
        change_set_id=first.id,
        expected_review_sha256=first.review_sha256,
        draft=draft.model_copy(update={"issues": ["ambiguous_original"]}),
        environment="local-test",
    )
    assert second.id != first.id and second.logical_group_id == first.logical_group_id
    assert second.review_revision == 2 and first.review_revision == 1
    assert first.review_payload["issues"] == ["missing_original"]
    assert batch.processed_instruction_count == 1
    with pytest.raises(ValueError, match="stale"):
        require_current_annex_review(
            source_session,
            change_set_id=first.id,
            expected_review_sha256=first.review_sha256,
            environment="local-test",
        )
    with pytest.raises(ValueError, match="stale"):
        revise_annex_review(
            source_session,
            change_set_id=second.id,
            expected_review_sha256=first.review_sha256,
            draft=draft,
            environment="local-test",
        )
    second.status = "preparing"
    source_session.commit()
    with pytest.raises(ValueError, match="state"):
        revise_annex_review(
            source_session,
            change_set_id=second.id,
            expected_review_sha256=second.review_sha256,
            draft=draft,
            environment="local-test",
        )


@dataclass
class LiveReview:
    batch: AmendmentBatch
    review: AnnexChangeSet
    file: UserFile
    outside: RegulatoryChunk
    before: list[AnnexCanonicalSnapshot]
    llm: MagicMock


@pytest.fixture
def live_review(
    source_session: Session,
    es: tuple[Elasticsearch, str],
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Generator[LiveReview, None, None]:
    from contextlib import contextmanager
    from hashlib import sha256
    from io import BytesIO
    from unittest.mock import MagicMock

    from onyx.configs.constants import FileOrigin
    from onyx.db.amendment_sources import create_source_package
    from onyx.db.models import RegulatorySourceAsset, SearchSettings
    from onyx.db.regulatory_annex_changes import (
        capture_canonical_scope,
        list_annex_changes,
    )
    from onyx.file_store.file_store import get_default_file_store
    from onyx.llm.interfaces import LLMConfig
    from onyx.natural_language_processing.utils import BaseTokenizer
    from onyx.regulatory.amendments import job
    from onyx.regulatory.amendments.annexes import analysis, config
    from tests.external_dependency_unit.regulatory.test_annex_baseline import (
        _chunk,
        _file,
    )

    class Tokenizer(BaseTokenizer):
        def encode(self, string: str) -> list[int]:
            return list(string.encode())

        def decode(self, tokens: list[int]) -> str:
            return bytes(tokens).decode(errors="ignore")

        def tokenize(self, string: str) -> list[str]:
            return list(string)

    @contextmanager
    def session_context() -> Generator[Session, None, None]:
        yield source_session

    monkeypatch.setattr(analysis, "get_session_with_current_tenant", session_context)
    monkeypatch.setattr(
        "onyx.db.engine.sql_engine.get_session_with_current_tenant", session_context
    )
    monkeypatch.setattr(job, "_session", session_context)
    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    monkeypatch.setattr(config, "REGULATORY_ANNEX_ENVIRONMENT", "local-test")
    monkeypatch.setattr(
        "onyx.regulatory.publication_reads.REGULATORY_ANNEX_ENVIRONMENT", "local-test"
    )
    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    file = _file(source_session, docset)
    old = _chunk(source_session, file, 0, "old")
    from datetime import date

    old.validity_start_date = (
        None if getattr(request, "param", None) == "open-start" else date(2020, 1, 1)
    )
    outside = _chunk(source_session, file, 1, "outside")
    outside.heading_path, outside.chunk_metadata = ["Madde 2"], {}
    outside.validity_start_date = old.validity_start_date
    expired = None
    if getattr(request, "param", None) == "expired":
        expired = _chunk(source_session, file, 2, "expired")
        expired.position = 0
        expired.validity_start_date, expired.validity_end_date = (
            date(2010, 1, 1),
            date(2020, 1, 1),
        )
        expired.status = "superseded"
        expired.superseded_by_chunk_id = old.id
        old.supersedes_chunk_id = expired.id
    store = get_default_file_store()
    old_bytes = b'<h1>EK-1</h1><p id="rate">old</p>'
    source_only = getattr(request, "param", None) in (
        "source-only",
        "layout",
        "source-only-verified",
    )
    multipart = getattr(request, "param", None) == "multipart"
    new_bytes = (
        b'<h1>EK-1</h1><p id="rate">old</p><!-- new source bytes -->'
        if source_only
        else b'<h1>EK-1</h1><p id="rate">new</p>'
    )
    if getattr(request, "param", None) == "layout":
        new_bytes = b'<h1>EK-1</h1><div id="rate">old</div>'
    file.file_id = store.save_file(
        BytesIO(old_bytes), "old.html", FileOrigin.OTHER, "text/html"
    )
    file.file_type = "text/html"
    package, _ = create_source_package(
        source_session,
        document_set_id=docset.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="request",
        input_spec={},
        created_by=file.user_id,
    )
    asset = RegulatorySourceAsset(
        package_id=package.id,
        sha256=sha256(new_bytes).hexdigest(),
        file_id=store.save_file(
            BytesIO(new_bytes), "new.html", FileOrigin.OTHER, "text/html"
        ),
        mime_type="text/html",
        display_name="new.html",
        byte_count=len(new_bytes),
    )
    source_session.add(asset)
    source_session.flush()
    extra_assets: list[RegulatorySourceAsset] = []
    links: list[dict[str, str]] = []
    if multipart:
        for name, content in [
            ("directory", b"<h1>Instructions</h1><a>EK-1</a><a>EK-1</a>"),
            ("second", b'<h1>EK-1</h1><p id="extra">extra</p>'),
        ]:
            extra = RegulatorySourceAsset(
                package_id=package.id,
                sha256=sha256(content).hexdigest(),
                file_id=store.save_file(
                    BytesIO(content), name + ".html", FileOrigin.OTHER, "text/html"
                ),
                mime_type="text/html",
                display_name=name,
                byte_count=len(content),
            )
            source_session.add(extra)
            extra_assets.append(extra)
        source_session.flush()
        links = [
            dict(
                parent_asset_hash=extra_assets[0].sha256,
                target_asset_hash=target.sha256,
                source_field=f"html:{index}",
                label="EK-1",
                kind="url",
            )
            for index, target in enumerate([asset, extra_assets[1]])
        ]
    import json

    manifest = json.dumps(
        {
            "status": "ready",
            "issues": [],
            "links": links,
            "assets": [
                {"sha256": item.sha256, "mime_type": item.mime_type}
                for item in [asset, *extra_assets]
            ],
        }
    ).encode()
    package.manifest_file_id = store.save_file(
        BytesIO(manifest), "manifest.json", FileOrigin.OTHER, "application/json"
    )
    package.manifest_sha256 = sha256(manifest).hexdigest()
    package.status, package.asset_count = "ready", 1 + len(extra_assets)
    mode = getattr(request, "param", None)
    raw_text = "EK-1 replaced\nEK-1 second instruction"
    if mode == "temporary":
        raw_text = "EK-1 replaced\nEK-1 01.01.2027 tarihinde önceki hükümler yeniden uygulanır."
    elif mode == "cessation":
        raw_text = "EK-1 replaced\nEK-1 01.01.2027 tarihinde yürürlükten kalkar."
    from datetime import date

    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text=raw_text,
        user_file_ids=[str(file.id)],
        created_by=file.user_id,
        status="analyzing",
        lease_generation=1,
        instruction_count=2,
        reference_date=date(2026, 9, 10),
        source_package_id=package.id,
        source_text_sha256=sha256(raw_text.encode()).hexdigest(),
        segmented_instructions=[
            {"instruction_text": text} for text in raw_text.splitlines()
        ],
    )
    source_session.add(batch)
    source_session.flush()
    if mode == "historical-derived-verified":
        outside.chunk_metadata = {
            "chunk_variant": "hierarchical_aggregate",
            "source_regulatory_chunk_ids": [old.id],
            "hierarchy_root_path": ["Root"],
        }
        source_session.flush()
    before = capture_canonical_scope(source_session, file.id)
    from onyx.db.enums import IndexModelStatus

    settings = SearchSettings(
        id=1,
        status=IndexModelStatus.PRESENT,
        index_name=es[1],
        model_name="embedding",
        model_dim=3,
        normalize=True,
        enable_contextual_rag=False,
    )
    monkeypatch.setattr(
        "onyx.db.search_settings.get_current_search_settings",
        lambda *_args, **_kwargs: settings,
    )
    monkeypatch.setattr(
        "onyx.db.search_settings.get_active_search_settings_list",
        lambda *_args, **_kwargs: [settings],
    )
    monkeypatch.setattr(
        "onyx.indexing.contextual_settings.require_contextual_rag_llm",
        lambda _settings: llm,
    )
    embedder = MagicMock()
    model = embedder.embedding_model
    model.tokenizer = Tokenizer()
    for key, value in dict(
        provider_type=None,
        model_name="embedding",
        normalize=True,
        passage_prefix=None,
        retrim_content=True,
        reduced_dimension=None,
        api_url=None,
        api_version=None,
        deployment_name=None,
    ).items():
        setattr(model, key, value)
    monkeypatch.setattr(
        "onyx.indexing.embedder.DefaultIndexingEmbedder.from_db_search_settings",
        lambda **_kwargs: embedder,
    )
    future_name = None
    if mode == "future":
        from copy import deepcopy

        future_name = es[1] + "-future"
        es[0].indices.create(
            index=future_name,
            mappings=es[0].indices.get_mapping(index=es[1])[es[1]]["mappings"],
        )
        future = SearchSettings(
            id=2,
            status=IndexModelStatus.FUTURE,
            index_name=future_name,
            model_name="future-embedding",
            model_dim=6,
            reduced_dimension=3,
            normalize=True,
            enable_contextual_rag=False,
        )
        future_embedder = deepcopy(embedder)
        future_embedder.embedding_model.model_name = "future-embedding"
        future_embedder.embedding_model.reduced_dimension = 3
        monkeypatch.setattr(
            "onyx.db.search_settings.get_active_search_settings_list",
            lambda *_args, **_kwargs: [settings, future],
        )
        monkeypatch.setattr(
            "onyx.indexing.embedder.DefaultIndexingEmbedder.from_db_search_settings",
            lambda *, search_settings: (
                future_embedder if search_settings.id == 2 else embedder
            ),
        )
    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="analysis",
        temperature=0,
        max_input_tokens=10000,
    )
    monkeypatch.setattr(job, "get_default_llm", lambda: llm)
    monkeypatch.setattr("onyx.llm.factory.get_default_llm", lambda: llm)
    monkeypatch.setattr("onyx.llm.factory.get_default_llm_with_vision", lambda: None)
    monkeypatch.setattr(
        "onyx.regulatory.projection._get_contextual_tokenizer",
        lambda *_args, **_kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        "onyx.indexing.indexing_pipeline._invoke_contextual_llm_with_retry",
        lambda *_args, **_kwargs: "Same context",
    )
    retriever = MagicMock()
    monkeypatch.setattr(
        job, "build_amendment_search_retriever", lambda *_args, **_kwargs: retriever
    )
    from onyx.regulatory.amendments.models import DateResolution

    monkeypatch.setattr(
        analysis,
        "resolve_group_effective_date",
        lambda **_kwargs: DateResolution(
            effective_start_date="2026-09-10",
            effective_end_date="2027-01-01"
            if mode in ("temporary", "cessation")
            else None,
            rationale="explicit date",
        ),
    )
    import json
    from datetime import date

    from onyx.access.access import get_access_for_user_files
    from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
    from onyx.document_index.interfaces_new import TenantState
    from onyx.regulatory.amendments.annexes.models import AnnexProjectionAccess
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        _source_template,
    )
    from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
    from onyx.regulatory.projection import prepare_normal_context_view

    source_session.commit()
    baseline_rows = canonical_snapshot_rows(before)
    retained_image_id = None
    if mode in ("historical-source-verified", "historical-derived-verified"):
        from base64 import b64decode

        from onyx.regulatory.chunker import hierarchical_aggregate_text

        retained_image_id = store.save_file(
            BytesIO(
                b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aHosAAAAASUVORK5CYII="
                )
            ),
            "historical.png",
            FileOrigin.OTHER,
            "image/png",
        )
        retained = next(row for row in baseline_rows if row.id == outside.id)
        retained.chunk_metadata = {
            **retained.chunk_metadata,
            "source_links": {"0": file.file_id},
            "image_file_id": retained_image_id,
        }
        retained.position = 8
        if mode == "historical-derived-verified":
            retained.text = hierarchical_aggregate_text("Root", [old.text])
    indexed_view = prepare_normal_context_view(
        rows=baseline_rows,
        user_file=file,
        search_settings=settings,
        embedder=embedder,
        llm=llm,
        as_of_date=date(2020, 1, 1),
    )
    if mode in ("verified-compatible-receipt", "verified-partial-receipt"):
        from onyx.regulatory.amendments.annexes.context_dependencies import context_hash

        updated = []
        for item in indexed_view.projections:
            receipt = {**item.embedding_config, "transport": "synchronous_encoder"}
            if mode == "verified-partial-receipt":
                receipt.pop("normalize")
            updated.append(
                item.model_copy(
                    update={
                        "embedding_config": receipt,
                        "embedding_config_sha256": context_hash(receipt),
                    }
                )
            )
        indexed_view = indexed_view.model_copy(update={"projections": updated})
    access = AnnexProjectionAccess(
        access=get_access_for_user_files([str(file.id)], source_session)[str(file.id)],
        project_ids=[],
        persona_ids=[],
        document_sets=[docset.name],
    )
    verified = mode in (
        "verified",
        "source-only-verified",
        "historical-source-verified",
        "historical-derived-verified",
        "verified-compatible-receipt",
        "verified-partial-receipt",
    )
    from datetime import timedelta

    from onyx.db.regulatory_context_projections import activate_temporal_projection
    from onyx.db.regulatory_publication import PublicationStore
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.publication_models import (
        FrozenPublicationProjection,
        PublicationIndexSnapshot,
        PublicationScope,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection

    authority = PublicationStore(
        PublicationScope(
            tenant_id="public",
            environment="local-test",
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    index = PublicationIndexSnapshot(
        index_name=es[1],
        index_uuid=es[0].indices.get(index=es[1])[es[1]]["settings"]["index"]["uuid"],
        search_settings_id=1,
        model_provider="",
        model_name="embedding",
        vector_dimension=3,
        embedding_config_sha256=indexed_view.projections[0].embedding_config_sha256,
        multitenant=False,
    )
    adapter = FencedPublicationIndex(es[0], index)
    seeded: list[FrozenPublicationProjection] = []
    owner = reservations = None
    if verified:
        owner = authority.acquire(file.id, owner_id=uuid4(), ttl=timedelta(minutes=5))
        authority.close_gate(owner)
        reservations = authority.reservations(owner)
        adapter.seal(reservations)
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        effective_context_rows,
    )

    indexed_rows = effective_context_rows(baseline_rows, date(2020, 1, 1))
    indexed_pairs = list(zip(indexed_rows, indexed_view.projections, strict=True))
    if expired is not None:
        expired_view = prepare_normal_context_view(
            rows=baseline_rows,
            user_file=file,
            search_settings=settings,
            embedder=embedder,
            llm=llm,
            as_of_date=date(2010, 1, 1),
        )
        indexed_pairs.extend(
            zip(
                effective_context_rows(baseline_rows, date(2010, 1, 1)),
                expired_view.projections,
                strict=True,
            )
        )
    for canonical, context in indexed_pairs:
        raw = json.loads(
            _source_template(
                canonical,
                context,
                file_name=file.name,
                access=access,
                tenant_id="public",
                dimension=3,
            )
        )
        raw.update(content_vector=[0.1, 0.2, 0.3], title_vector=[0.1, 0.2, 0.3])
        if verified:
            assert reservations is not None
            identity = uuid4()
            projection = FrozenPublicationProjection(
                ordinal=canonical.projection_ordinal,
                context_projection_id=str(identity),
                source_json=json.dumps(raw),
                embedding_inputs=tuple(context.embedding_texts),
                embedding_config_json=json.dumps(context.embedding_config),
            )
            adapter.upsert(reservations, projection)
            seeded.append(projection)
            activate_temporal_projection(
                source_session,
                user_file_id=file.id,
                binding=AnnexTemporalProjection(
                    id=identity,
                    index=index,
                    projection=projection,
                    canonical_base_sha256=context_hash(
                        next(row.text for row in before if row.id == canonical.id)
                    ),
                    derived_role="hierarchical_aggregate"
                    if canonical.chunk_metadata.get("chunk_variant")
                    == "hierarchical_aggregate"
                    else "canonical",
                    dependency_ids=canonical.chunk_metadata.get(
                        "source_regulatory_chunk_ids", []
                    ),
                    representation_text=canonical.text,
                    representation_metadata=canonical.chunk_metadata,
                    context=context,
                    reference_date=date(2020, 1, 1),
                    effective_start=canonical.validity_start_date,
                    effective_end=canonical.validity_end_date,
                    semantic_position=canonical.position,
                ),
            )
        else:
            es[0].index(
                index=es[1],
                id=get_elasticsearch_doc_chunk_id(
                    TenantState(tenant_id="public", multitenant=False),
                    str(file.id),
                    canonical.projection_ordinal,
                ),
                document=raw,
            )
    if verified:
        assert owner is not None and reservations is not None
        authority.finalize(
            source_session, owner, adapter.verify(reservations, tuple(seeded))
        )
        source_session.commit()
        authority.release(owner)
    job.run_amendment_batch(batch_id=batch.id, lease_generation=1)
    groups = list_annex_changes(source_session, batch.id)
    assert len(groups) == 1
    assert groups[0].status == "pending", groups[0].review_payload["issues"]
    draft = AnnexChangeDraft.model_validate(groups[0].review_payload)
    assert draft.instruction_indices == [0, 1]
    assert draft.baseline_context is not None
    assert draft.impact is not None
    assert draft.impact_strategy == "source_dependencies_v1"
    expected_consumers = {
        chunk.id for item in draft.items for chunk in item.new_chunks
    } | set(draft.source_only_canonical_ids)
    if mode == "temporary":
        from onyx.regulatory.amendments.annexes.publication_preparation import (
            validate_frozen_publication_review,
        )

        expected_consumers.update(
            validate_frozen_publication_review(draft).legal.restoration_predecessors
        )
    if mode == "historical-derived-verified":
        expected_consumers.add(outside.id)
    assert {
        p.canonical_chunk_id for p in draft.impact.prepared.projections
    } == expected_consumers
    if not source_only:
        assert next(
            chunk for item in draft.items for chunk in item.new_chunks
        ).metadata["source_asset_ids"] == [str(asset.id)]
    assert batch.processed_instruction_count == 2 and batch.status == "analyzed"
    assert capture_canonical_scope(source_session, file.id) == before
    retriever.search.assert_not_called()
    original_file_id = file.file_id
    try:
        yield LiveReview(batch, groups[0], file, outside, before, llm)
    finally:
        if future_name:
            es[0].indices.delete(index=future_name)
        for file_id in [
            original_file_id,
            retained_image_id,
            asset.file_id,
            *[item.file_id for item in extra_assets],
            package.manifest_file_id,
            *[evidence.file_id for evidence in draft.evidence],
            draft.publication.artifact_file_id if draft.publication else None,
        ]:
            if file_id:
                store.delete_file(file_id, error_on_missing=False)


def test_live_batch_prepares_annex_once_without_legacy_or_canonical_writes(
    live_review: LiveReview, source_session: Session
) -> None:
    from onyx.db.regulatory_annex_changes import (
        capture_canonical_scope,
        queue_annex_publication,
    )

    file, outside, before = live_review.file, live_review.outside, live_review.before
    groups = [live_review.review]

    def queue() -> AnnexPublicationIntent:
        return queue_annex_publication(
            source_session,
            change_set_id=groups[0].id,
            expected_review_sha256=groups[0].review_sha256,
            environment="local-test",
            tenant_id="public",
            database_identity="local-db",
            decided_by=file.user_id,
        )

    outside.text = "changed outside annex"
    source_session.flush()
    with pytest.raises(ValueError, match="baseline"):
        queue()
    outside.text = "outside"
    source_session.flush()
    file.name = "Changed title"
    source_session.flush()
    with pytest.raises(ValueError, match="configuration"):
        queue()
    file.name = "Regulation"
    source_session.flush()
    intent = queue()
    assert intent.review_sha256 == groups[0].review_sha256
    assert (
        intent.logical_group_id == groups[0].logical_group_id
        and intent.review_revision == 1
    )
    assert intent.publication_generation == 1
    assert groups[0].status == "approving"
    assert capture_canonical_scope(source_session, file.id) == before
    repeated = queue()
    assert repeated.id == intent.id
    frozen = (
        groups[0].id,
        groups[0].review_revision,
        groups[0].review_sha256,
        groups[0].review_payload,
    )
    groups[0].status = "failed"
    source_session.commit()
    retried = queue_annex_publication(
        source_session,
        change_set_id=groups[0].id,
        expected_review_sha256=groups[0].review_sha256,
        environment="local-test",
        tenant_id="public",
        database_identity="local-db",
        decided_by=file.user_id,
        retry=True,
    )
    assert retried.id != intent.id and retried.publication_generation == 2
    assert (
        groups[0].id,
        groups[0].review_revision,
        groups[0].review_sha256,
        groups[0].review_payload,
    ) == frozen
    assert queue().id == retried.id


def test_source_text_edits_create_new_batch_and_invalidate_previous_review(
    source_session: Session,
) -> None:
    from hashlib import sha256

    from onyx.db.regulatory_annex_changes import create_source_text_revision

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    old = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="Original extraction",
        status="analyzed",
        source_text_sha256=sha256(b"Original extraction").hexdigest(),
    )
    source_session.add(old)
    source_session.flush()
    assert old.source_text_sha256 is not None
    from onyx.db.models import AmendmentProposal
    from onyx.db.regulatory_amendments import queue_amendment_proposal_approval

    proposal = AmendmentProposal(
        batch_id=old.id,
        instruction_index=0,
        instruction_text="old article edit",
        new_chunk_draft={},
        old_chunk_snapshot={},
        status="pending",
    )
    source_session.add(proposal)
    source_session.flush()
    new = create_source_text_revision(
        source_session,
        batch_id=old.id,
        raw_text="Edited instruction text",
        expected_source_text_sha256=old.source_text_sha256,
        environment="local-test",
        created_by=None,
    )
    with pytest.raises(ValueError, match="superseded source"):
        queue_amendment_proposal_approval(source_session, proposal, decided_by=None)
    assert new.source_parent_batch_id == old.id
    assert old.superseded_by_batch_id == new.id
    assert old.raw_text == "Original extraction"
    assert new.source_text_sha256 == sha256(b"Edited instruction text").hexdigest()
    assert old.source_text_sha256 is not None
    with pytest.raises(ValueError, match="stale"):
        create_source_text_revision(
            source_session,
            batch_id=old.id,
            raw_text="Racing edit",
            expected_source_text_sha256=old.source_text_sha256,
            environment="local-test",
            created_by=None,
        )


def test_group_api_enforces_owner_and_returns_exact_frozen_bytes(
    live_review: LiveReview, source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from onyx.auth.users import current_user
    from onyx.db.engine.sql_engine import get_session
    from onyx.db.enums import Permission
    from onyx.error_handling.exceptions import register_onyx_exception_handlers
    from onyx.server.features.regulatory import annex_api

    class Principal:
        def __init__(self, identifier: UUID, permissions: list[str]) -> None:
            self.id = identifier
            self.effective_permissions = permissions

    owner_id = live_review.file.user_id
    assert owner_id is not None
    owner = Principal(owner_id, [Permission.FULL_ADMIN_PANEL_ACCESS.value])
    monkeypatch.setattr(
        annex_api,
        "get_document_set_by_id_for_user",
        lambda **_kwargs: live_review.batch,
    )
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(annex_api.router)
    app.dependency_overrides[get_session] = lambda: source_session
    app.dependency_overrides[current_user] = lambda: owner
    client = TestClient(app)
    path = f"/amendments/batches/{live_review.batch.id}/annex-groups"
    response = client.get(path)
    assert response.status_code == 200
    assert len(response.json()) == 1 and response.json()[0]["review_revision"] == 1
    assert response.json()[0]["review_payload"]["items"] == []
    assert response.json()[0]["review_payload"]["new_extraction"] is None
    chunk_path = f"{path}/{live_review.review.id}/chunks"
    page = client.get(chunk_path, params={"limit": 1})
    assert page.status_code == 200 and page.json()["total"] == 1
    assert len(page.json()["items"]) == 1 and page.json()["items"][0]["old_chunks"]
    assert client.get(chunk_path, params={"limit": 51}).status_code == 422
    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    evidence = next(
        item
        for item in draft.evidence
        if item.side == "new" and item.kind == "original"
    )
    evidence_path = f"{path}/{live_review.review.id}/evidence/{evidence.id}"
    response = client.get(evidence_path)
    assert (
        response.status_code == 200
        and response.content == b'<h1>EK-1</h1><p id="rate">new</p>'
    )
    revalidate = MagicMock(side_effect=AssertionError("retry must not reprepare"))
    monkeypatch.setattr(
        "onyx.regulatory.amendments.annexes.corrections.revalidate_annex_review",
        revalidate,
    )
    from onyx.server.features.regulatory.models import AnnexReviewSnapshot

    identity = (
        live_review.review.id,
        live_review.review.review_sha256,
        AnnexReviewSnapshot.model_validate(live_review.review).model_dump(mode="json")[
            "review_payload"
        ],
    )
    for status in ("pending", "rejected", "failed", "blocked"):
        live_review.review.status = status
        source_session.commit()
        response = client.post(
            f"{path}/{live_review.review.id}/retry",
            json={"expected_review_sha256": identity[1]},
        )
        assert response.status_code == 200, response.text
        assert (
            response.json()["id"],
            response.json()["review_sha256"],
            response.json()["review_payload"],
        ) == (str(identity[0]), identity[1], identity[2])
        assert response.json()["review_revision"] == 1
    revalidate.assert_not_called()
    assert (
        client.post(
            f"{path}/{live_review.review.id}/retry",
            json={"expected_review_sha256": "0" * 64},
        ).status_code
        == 400
    )
    live_review.review.status = "pending"
    source_session.commit()
    monkeypatch.setattr(annex_api, "enqueue_annex_publication", MagicMock())
    decision = {"expected_review_sha256": identity[1]}
    response = client.post(f"{path}/{live_review.review.id}/approve", json=decision)
    assert response.status_code == 200, response.text
    response = client.post(f"{path}/{live_review.review.id}/retry", json=decision)
    assert (
        response.status_code == 200 and response.json()["publication_generation"] == 1
    )
    live_review.review.status = "failed"
    source_session.commit()
    response = client.post(f"{path}/{live_review.review.id}/retry", json=decision)
    assert (
        response.status_code == 200 and response.json()["publication_generation"] == 2
    )
    assert (
        response.json()["id"],
        response.json()["review_sha256"],
        response.json()["review_payload"],
    ) == (str(identity[0]), identity[1], identity[2])
    assert response.json()["review_revision"] == 1
    revalidate.assert_not_called()
    owner.id = uuid4()
    assert client.get(path).status_code == 404
    assert client.get(evidence_path).status_code == 404
    assert client.get(chunk_path).status_code == 404
    owner.effective_permissions = []
    assert client.get(path).status_code == 403


def test_unreconciled_correction_remains_blocked_with_raw_evidence_preserved(
    live_review: LiveReview, source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.db.regulatory_annex_changes import revise_annex_review
    from onyx.regulatory.amendments.annexes import corrections
    from onyx.regulatory.amendments.annexes.models import (
        AnnexCorrectionReconciliation,
        AnnexElementCorrection,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.raw_new_extraction is not None
    position = next(
        index
        for index, item in enumerate(draft.raw_new_extraction.elements)
        if item.text == "new"
    )
    monkeypatch.setattr(
        corrections,
        "reconcile_corrections",
        lambda **_kwargs: AnnexCorrectionReconciliation(
            supported=False, rationale="Not printed at the source location"
        ),
    )
    assert live_review.file.user_id is not None
    corrected = corrections.revalidate_annex_review(
        draft=draft,
        corrections=[
            AnnexElementCorrection(
                position=position,
                before_text="new",
                corrected_text="invented",
                reason="edit",
            )
        ],
        corrected_by=live_review.file.user_id,
        llm=live_review.llm,
        vision_llm=None,
    )
    review = revise_annex_review(
        source_session,
        change_set_id=live_review.review.id,
        expected_review_sha256=live_review.review.review_sha256,
        draft=corrected,
        environment="local-test",
    )
    assert review.status == "blocked" and review.review_revision == 2
    assert corrected.raw_new_extraction == draft.raw_new_extraction
    assert corrected.evidence == draft.evidence
    assert live_review.review.review_payload == draft.model_dump(mode="json")


def test_revalidation_creates_new_ready_revision_with_new_ids_and_selective_context(
    live_review: LiveReview, source_session: Session
) -> None:
    from onyx.db.regulatory_annex_changes import (
        require_current_annex_review,
        revise_annex_review,
    )
    from onyx.regulatory.amendments.annexes.corrections import revalidate_annex_review

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert live_review.file.user_id is not None
    revised = revalidate_annex_review(
        draft=draft,
        corrections=[],
        corrected_by=live_review.file.user_id,
        llm=live_review.llm,
        vision_llm=None,
    )
    review = revise_annex_review(
        source_session,
        change_set_id=live_review.review.id,
        expected_review_sha256=live_review.review.review_sha256,
        draft=revised,
        environment="local-test",
    )
    assert review.status == "pending" and review.review_revision == 2
    assert {chunk.id for item in revised.items for chunk in item.new_chunks}.isdisjoint(
        {chunk.id for item in draft.items for chunk in item.new_chunks}
    )
    assert revised.baseline_context == draft.baseline_context
    assert revised.evidence == draft.evidence
    assert revised.impact is not None
    assert {p.canonical_chunk_id for p in revised.impact.prepared.projections} == {
        chunk.id for item in revised.items for chunk in item.new_chunks
    }
    with pytest.raises(ValueError, match="stale"):
        require_current_annex_review(
            source_session,
            change_set_id=live_review.review.id,
            expected_review_sha256=live_review.review.review_sha256,
            environment="local-test",
        )


@pytest.mark.parametrize("batch_transport", [False, True])
def test_live_preparation_uses_actual_durable_transport(
    live_review: LiveReview, monkeypatch: pytest.MonkeyPatch, batch_transport: bool
) -> None:
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.llm.constants import LlmProviderNames
    from onyx.regulatory.amendments.annexes import analysis
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        validate_complete_context_view,
    )
    from onyx.regulatory.amendments.annexes.staging import prepare_staged_candidate_rows
    from onyx.regulatory.indexing_jobs.models import (
        OpenRouterBatchConfig,
        RegulatoryIndexingConfigSnapshot,
        RegulatoryInputHashVersion,
        VertexAuthenticationMode,
        VertexBatchConfig,
    )
    from shared_configs.enums import EmbeddingProvider

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    snapshot = RegulatoryIndexingConfigSnapshot(
        input_content_hash="a" * 64,
        input_hash_version=RegulatoryInputHashVersion.CANONICAL_V2,
        chunk_generation_hash="b" * 64,
        search_settings_id=1,
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name="embedding",
        model_dimension=3,
        effective_dimension=3,
        index_name="test-index",
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
        if batch_transport
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
        update={
            "model_provider": LlmProviderNames.VERTEX_AI.value,
            "model_name": snapshot.vertex.model_name,
        }
    )
    live_review.llm.invoke.return_value.choice.message.content = (
        "Durable contextual output"
    )
    monkeypatch.setattr(
        analysis,
        "resolve_review_context_llm",
        lambda *_args, **_kwargs: live_review.llm,
    )
    from onyx.db.models import SearchSettings

    embedder = DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=SearchSettings()
    )
    embedder.embedding_model.provider_type = EmbeddingProvider.OPENROUTER
    embedder.embedding_model.reduced_dimension = 3
    prepared = analysis.prepare_review_context(draft)
    assert prepared.impact is not None and prepared.effective_date is not None
    assert prepared.indexing_configuration == snapshot.model_dump(mode="json")
    assert all(
        item.generation_path == "durable"
        for item in prepared.impact.prepared.projections
    )
    assert all(
        item.embedding_config["transport"]
        == ("openrouter_batch" if batch_transport else "synchronous_encoder")
        for item in prepared.impact.prepared.projections
    )
    rows = prepare_staged_candidate_rows(
        baseline_scope=prepared.baseline_scope,
        items=prepared.items,
        effective_date=prepared.effective_date,
        evidence_remapping=prepared.new_evidence_remapping,
    )
    validate_complete_context_view(
        rows=[
            row
            for row in rows
            if row.id
            in {p.canonical_chunk_id for p in prepared.impact.prepared.projections}
        ],
        view=prepared.impact.prepared,
        as_of_date=prepared.effective_date,
    )


@pytest.mark.parametrize("live_review", ["source-only", "layout"], indirect=True)
def test_source_only_change_keeps_legal_identity_and_stages_new_source(
    live_review: LiveReview, source_session: Session
) -> None:
    from onyx.db.regulatory_annex_changes import capture_canonical_scope

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert live_review.review.status == "pending"
    assert draft.items == [] and draft.source_only_canonical_ids
    from onyx.regulatory.amendments.annexes.comparison import (
        _native_changes,
        validate_annex_comparison,
    )

    assert draft.comparison is not None and draft.comparison.schema_version == 2
    assert draft.old_extraction is not None and draft.new_extraction is not None
    legacy = draft.comparison.model_copy(
        update={
            "schema_version": 1,
            "changes": _native_changes(draft.old_extraction, draft.new_extraction),
            "coverage": draft.comparison.coverage.model_copy(
                update={
                    "old_positions": list(range(len(draft.old_extraction.elements))),
                    "new_positions": list(range(len(draft.new_extraction.elements))),
                }
            ),
        }
    )
    assert legacy.changes
    assert (
        validate_annex_comparison(
            legacy, old=draft.old_extraction, new=draft.new_extraction
        )
        == []
    )
    assert draft.impact is not None and draft.baseline_context is not None
    assert {row.canonical_chunk_id for row in draft.impact.prepared.projections} == set(
        draft.source_only_canonical_ids
    )
    assert draft.new_evidence_remapping is not None
    assert {item.parent_sha256 for item in draft.evidence if item.side == "new"} != {
        item.parent_sha256 for item in draft.evidence if item.side == "old"
    }
    assert (
        capture_canonical_scope(source_session, live_review.file.id)
        == live_review.before
    )


def test_prepared_original_text_and_graph_identity_reject_stale_input(
    live_review: LiveReview, source_session: Session
) -> None:
    from onyx.db.regulatory_annex_changes import validate_prepared_annex_change

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    for field in ("original_source_text_sha256", "source_graph_sha256"):
        with pytest.raises(ValueError, match="source graph or original text"):
            validate_prepared_annex_change(
                source_session,
                batch=live_review.batch,
                draft=draft.model_copy(update={field: "0" * 64}),
                environment="local-test",
            )


@pytest.mark.parametrize("live_review", ["multipart"], indirect=True)
def test_live_batch_combines_complete_native_new_parts(live_review: LiveReview) -> None:
    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert (
        draft.new_extraction is not None
        and draft.new_extraction.evidence_view is not None
    )
    view = draft.new_extraction.evidence_view
    assert (
        view.selection_method == "ordered_source_occurrences" and len(view.parents) == 2
    )
    assert draft.new_evidence_remapping is not None
    assert (
        len({item.source_asset_id for item in draft.new_evidence_remapping.elements})
        == 2
    )
    assert sum(len(item.new_chunks) for item in draft.items) == 2


@pytest.mark.parametrize(
    "instruction,legacy",
    [
        ("EK-1 içindeki old ibaresi new olarak değiştirilmiştir.", True),
        (
            "EK-1 ekteki şekilde değiştirilmiştir.\nEK-1\nNew complete annex content",
            True,
        ),
        ("EK-1 ekteki şekilde değiştirilmiştir.", False),
    ],
)
def test_live_single_chunk_text_annex_keeps_legacy_route(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    instruction: str,
    legacy: bool,
) -> None:
    from onyx.db.regulatory_annex_changes import list_annex_changes
    from onyx.regulatory.amendments import job

    batch = AmendmentBatch(
        document_set_id=live_review.batch.document_set_id,
        user_file_ids=live_review.batch.user_file_ids,
        raw_text=instruction,
        created_by=live_review.batch.created_by,
        status="analyzing",
        lease_generation=1,
        instruction_count=1,
        reference_date=live_review.batch.reference_date,
        segmented_instructions=[{"instruction_text": instruction}],
    )
    source_session.add(batch)
    source_session.commit()
    retrieve = MagicMock(return_value=([], None))
    monkeypatch.setattr(job, "retrieve_and_confirm_instruction", retrieve)
    job.run_amendment_batch(batch_id=batch.id, lease_generation=1)
    assert retrieve.call_count == int(legacy)
    assert bool(list_annex_changes(source_session, batch.id)) != legacy
    assert batch.processed_instruction_count == 1
