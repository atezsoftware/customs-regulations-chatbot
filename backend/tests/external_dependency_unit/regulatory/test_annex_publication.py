"""Annex application publication against isolated local PostgreSQL/ES/FileStore."""

from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from elasticsearch import Elasticsearch

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import FrozenContextProjection
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
    adapter_for,
    frozen_projection,
    store,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    es as es,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    owned_file as owned_file,
)


def test_actual_index_proof_reuses_only_exact_input_config_and_validated_lineage(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    from onyx.regulatory.amendments.annexes import publication

    assert hasattr(publication, "exact_reusable_projection"), (
        "actual annex evidence comparison is missing"
    )
    authority, adapter = store(), adapter_for(es)
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=60))
    authority.close_gate(owner)
    reserved = authority.reservations(owner)
    adapter.seal(reserved)
    old = frozen_projection(owned_file, 0)
    adapter.upsert(reserved, old)
    actual = adapter.read_evidence(reserved, 0)
    wanted = FrozenContextProjection(
        canonical_chunk_id="new-canonical",
        source_snapshot_sha256="source",
        generation_path="normal",
        request_hashes=[],
        embedding_input_sha256=context_hash(list(old.embedding_inputs)),
        embedding_config_sha256=context_hash({"model": "fixture"}),
        embedding_texts=list(old.embedding_inputs),
        embedding_config={"model": "fixture"},
        canonical_text_sha256="canonical",
        metadata_sha256="metadata",
    )
    assert (
        publication.exact_reusable_projection(
            wanted, actual, predecessor_id="same-canonical"
        )
        == actual.frozen_projection
    )
    assert (
        publication.exact_reusable_projection(wanted, actual, predecessor_id=None)
        is None
    )
    assert (
        publication.exact_reusable_projection(
            wanted.model_copy(update={"embedding_texts": ["changed"]}),
            actual,
            predecessor_id="same-canonical",
        )
        is None
    )
    assert (
        publication.exact_reusable_projection(
            wanted.model_copy(update={"embedding_config": {"model": "new"}}),
            actual,
            predecessor_id="same-canonical",
        )
        is None
    )
    legacy = actual.model_copy(
        update={"frozen_projection": None, "payload_sha256": None}
    )
    assert (
        publication.exact_reusable_projection(
            wanted, legacy, predecessor_id="same-canonical"
        )
        is None
    )
    authority.release(owner)


def test_one_index_accepts_explicit_compatible_formatter_receipts_but_not_other_models(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    import json

    from onyx.document_index import publication_models as models
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex

    assert hasattr(models, "PublicationEncoderAuthority"), (
        "compatible encoder authority is missing"
    )
    authority = models.PublicationEncoderAuthority(
        provider="fixture",
        model="fixture",
        effective_dimension=3,
        endpoint_sha256="endpoint",
        deployment_name=None,
        api_version=None,
        normalize=True,
        passage_prefix=None,
    )
    configs = [
        {
            "provider": "fixture",
            "model": "fixture",
            "dimension": 3,
            "endpoint_sha256": "endpoint",
            "normalize": True,
            "passage_prefix": None,
            "api_version": None,
            "deployment_name": None,
            "formatter": formatter,
        }
        for formatter in ("normal-v1", "durable-context-before-text-v1")
    ]
    receipts = tuple(
        models.PublicationEncoderReceipt(
            configuration_json=json.dumps(config),
            authority=authority,
            resolution_sha256="a" * 64,
        )
        for config in configs
    )
    adapter = adapter_for(es)
    snapshot = adapter.snapshot.model_copy(
        update={
            "embedding_config_sha256": models.publication_digest(configs[0]),
            "encoder_authority": authority,
            "encoder_receipts": receipts,
        }
    )
    adapter = FencedPublicationIndex(es[0], snapshot)
    repository = store()
    owner = repository.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(seconds=60))
    repository.close_gate(owner)
    reservations = repository.reservations(owner)
    adapter.seal(reservations)
    projections = tuple(
        frozen_projection(owned_file, ordinal).model_copy(
            update={"embedding_config_json": json.dumps(config)}
        )
        for ordinal, config in zip(reservations.ordinals, configs, strict=True)
    )
    for projection in projections:
        adapter.upsert(reservations, projection)
        actual = adapter.read_evidence(
            reservations, projection.ordinal
        ).frozen_projection
        assert actual is not None
        assert json.loads(actual.embedding_config_json) == json.loads(
            projection.embedding_config_json
        )
    assert (
        adapter.verify(reservations, projections).live_ordinals == reservations.ordinals
    )

    with pytest.raises(ValueError, match="authority"):
        models.PublicationEncoderReceipt(
            configuration_json=json.dumps({**configs[0], "model": "another-model"}),
            authority=authority,
            resolution_sha256="a" * 64,
        )
    for field, value in (
        ("endpoint_sha256", "other-endpoint"),
        ("normalize", False),
        ("dimension", 6),
        ("passage_prefix", "other-prefix"),
    ):
        with pytest.raises(ValueError, match="authority"):
            models.PublicationEncoderReceipt(
                configuration_json=json.dumps({**configs[0], field: value}),
                authority=authority,
                resolution_sha256="a" * 64,
            )
    with pytest.raises(ValueError, match="unresolved"):
        models.PublicationEncoderReceipt(
            configuration_json=json.dumps(
                {
                    "provider": "fixture",
                    "model": "fixture",
                    "dimension": 3,
                    "transport": "openrouter_batch",
                }
            ),
            authority=authority,
            resolution_sha256="a" * 64,
        )
    repository.release(owner)


def test_live_group_freezes_actual_index_evidence_and_complete_preapproval_scope(
    live_review: LiveReview,
) -> None:
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.publication is not None, (
        "live review has no final indexed publication preparation"
    )
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        read_publication_preparation,
    )

    frozen = read_publication_preparation(draft.publication)
    assert len(frozen.indexed_baseline) == 2
    assert draft.publication.counts.total_projections == len(frozen.projections)
    assert draft.publication.counts.embeddings == sum(
        item.reuse_from is None for item in frozen.projections
    )
    assert all(item.reuse_from is None for item in frozen.projections)
    assert "content_vector" not in draft.publication.model_dump_json()
    outside = [
        item for item in frozen.projections if item.row.id == live_review.outside.id
    ]
    assert len(outside) == 2
    assert (
        outside[0].context.source_snapshot_sha256
        != outside[1].context.source_snapshot_sha256
    )
    assert all(item.reference_date is not None for item in outside)
    assert outside[0].reference_date is not None
    assert outside[0].reference_date.isoformat() == "2020-01-01"
    assert outside[0].reason == "historical_context_rebuilt_from_dated_canonical_source"
    assert all(item.ordinal in frozen.reserved_ordinals for item in frozen.projections)
    assert all(item.reference_date is None for item in frozen.indexed_baseline)


@pytest.mark.parametrize(
    "live_review", ["verified", "source-only-verified"], indirect=True
)
def test_verified_preapproval_reuses_actual_vectors_and_preserves_outside_identity(
    live_review: LiveReview,
) -> None:
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        read_publication_preparation,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.publication is not None
    frozen = read_publication_preparation(draft.publication)
    outside = [
        item for item in frozen.projections if item.row.id == live_review.outside.id
    ]
    assert (
        len(outside) == 2
        and outside[0].ordinal == live_review.outside.projection_ordinal
    )
    assert outside[0].id != outside[1].id
    assert outside[0].reuse_from is not None, (
        "verified actual outside vector was not reused"
    )
    assert all(item.reference_date is not None for item in frozen.indexed_baseline)
    if draft.source_only_canonical_ids:
        assert not draft.items
        assert draft.publication.counts.embeddings == 0
        assert {row.id for row in frozen.legal.canonical_rows} == {
            row.id for row in draft.baseline_scope
        }


@pytest.mark.parametrize("live_review", ["open-start"], indirect=True)
def test_unknown_index_date_rebuilds_authoritative_open_start_without_guessing_old_context(
    live_review: LiveReview,
) -> None:
    from datetime import date

    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        read_publication_preparation,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert draft.publication is not None
    frozen = read_publication_preparation(draft.publication)
    historical = [item for item in frozen.projections if item.effective_start is None]
    assert len(historical) == 2
    assert all(
        item.effective_end == date(2026, 9, 10)
        and item.reference_date == date(2026, 9, 9)
        for item in historical
    )
    assert all(
        item.reuse_from is None and frozen.views[item.view_sha256].snapshots
        for item in historical
    )
    assert all(item.reference_date is None for item in frozen.indexed_baseline)


@pytest.mark.parametrize("live_review", ["temporary", "cessation"], indirect=True)
def test_actual_review_freezes_before_during_and_authorized_after_window(
    live_review: LiveReview,
) -> None:
    from datetime import date

    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        validate_frozen_publication_review,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    frozen = validate_frozen_publication_review(draft)
    assert draft.after_window_authority is not None
    legal_ids = {row.id for row in frozen.legal.canonical_rows}
    assert all(plan.row.id in legal_ids for plan in frozen.projections)
    outside = [
        plan for plan in frozen.projections if plan.row.id == live_review.outside.id
    ]
    assert len(outside) == 3
    assert outside[-1].effective_start == date(2027, 1, 1)
    assert (
        outside[0].context.source_snapshot_sha256
        != outside[1].context.source_snapshot_sha256
    )
    if draft.after_window_authority.kind == "restore_predecessor":
        assert len(frozen.legal.restoration_predecessors) == 1
        restored = next(
            plan
            for plan in frozen.projections
            if plan.row.id in frozen.legal.restoration_predecessors
        )
        assert restored.row.text == "old" and restored.effective_start == date(
            2027, 1, 1
        )
    else:
        assert not frozen.legal.restoration_predecessors
        assert {
            plan.row.id
            for plan in frozen.projections
            if plan.effective_start == date(2027, 1, 1)
        } == {live_review.outside.id}


@pytest.mark.parametrize("live_review", ["future"], indirect=True)
def test_present_future_prepare_full_history_using_actual_effective_dimensions(
    live_review: LiveReview,
) -> None:
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        validate_frozen_publication_review,
        validate_indexed_publication_baseline,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    frozen = validate_frozen_publication_review(draft)
    assert len(frozen.indexes) == 2
    future = next(index for index in frozen.indexes if index.search_settings_id == 2)
    assert future.encoder_authority is not None
    assert (
        future.vector_dimension == 3
        and future.encoder_authority.effective_dimension == 3
    )
    assert future.model_name == "future-embedding"
    current = [
        plan for plan in frozen.projections if plan.index.search_settings_id == 1
    ]
    upcoming = [plan for plan in frozen.projections if plan.index == future]
    assert {
        (plan.row.id, plan.effective_start, plan.effective_end) for plan in current
    } == {(plan.row.id, plan.effective_start, plan.effective_end) for plan in upcoming}
    assert not {plan.id for plan in current}.intersection(plan.id for plan in upcoming)
    validate_indexed_publication_baseline(draft)


def test_retry_validation_never_replaces_artifact_and_detects_actual_index_change(
    live_review: LiveReview, es: tuple[Elasticsearch, str]
) -> None:
    import json

    from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
    from onyx.document_index.interfaces_new import TenantState
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        validate_frozen_publication_review,
        validate_indexed_publication_baseline,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    original = draft.publication
    validate_indexed_publication_baseline(draft)
    assert draft.publication == original
    changed = draft.model_copy(update={"submitted_source_text": "changed"})
    with pytest.raises(ValueError, match="different review"):
        validate_frozen_publication_review(changed)
    frozen = validate_frozen_publication_review(draft)
    source = json.loads(frozen.indexed_baseline[0].evidence.source_json)
    identity = get_elasticsearch_doc_chunk_id(
        TenantState(tenant_id="public", multitenant=False),
        str(live_review.file.id),
        source["chunk_index"],
    )
    es[0].update(
        index=es[1], id=identity, doc={"chunk_context": "changed indexed context"}
    )
    with pytest.raises(ValueError, match="indexed publication baseline changed"):
        validate_indexed_publication_baseline(draft)
    assert draft.publication == original


@pytest.mark.parametrize("live_review", ["expired"], indirect=True)
def test_expired_indexed_history_is_frozen_and_reencoded_before_approval(
    live_review: LiveReview,
) -> None:
    from datetime import date

    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        validate_frozen_publication_review,
    )

    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    frozen = validate_frozen_publication_review(draft)
    assert len(frozen.indexed_baseline) == 3
    expired = next(row for row in draft.baseline_scope if row.text == "expired")
    plan = next(plan for plan in frozen.projections if plan.row.id == expired.id)
    assert (plan.effective_start, plan.effective_end, plan.reference_date) == (
        date(2010, 1, 1),
        date(2020, 1, 1),
        date(2010, 1, 1),
    )
    assert plan.reuse_from is None and frozen.views[plan.view_sha256].snapshots
    assert frozen.counts.historical_projections == 3
