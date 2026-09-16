import json

import pytest
from sqlalchemy.orm import Session

from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
from onyx.regulatory.amendments.annexes.publication_preparation import (
    validate_frozen_publication_review,
)
from onyx.regulatory.amendments.annexes.selective_impact import review_units
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


@pytest.mark.parametrize("live_review", ["selective"], indirect=True)
@pytest.mark.parametrize("selected", [False, True])
def test_real_index_preparation_freezes_only_the_changed_chunk_and_preserves_legacy_vectors(
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    selected: bool,
) -> None:
    if selected:
        from dataclasses import replace

        from onyx.background.celery.tasks.regulatory_amendments.annex_preparation import (
            prepare_annex_review,
        )
        from onyx.db.regulatory_annex_selection import (
            chunk_review_page,
            create_selection,
            selection_reviews,
        )
        from onyx.regulatory.amendments.annexes import config

        parent = live_review.review
        total, groups, _ = chunk_review_page(source_session, parent, offset=0, limit=10)
        assert total == 1  # One remove plus its linked inserts are one review unit.
        assert live_review.file.user_id is not None
        child = create_selection(
            source_session,
            parent_id=parent.id,
            expected_sha256=parent.review_sha256,
            item_ids=[groups[0][0].id],
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            user_id=live_review.file.user_id,
            tenant_id="public",
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
        live_review.llm.reset_mock()
        prepare_annex_review(
            review_id=str(child.id),
            tenant_id="public",
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
        live_review.llm.invoke.assert_not_called()
        source_session.expire_all()
        latest = selection_reviews(source_session, parent.id)[0]
        assert latest.id != child.id and latest.status == "pending"
        live_review = replace(live_review, review=latest)
    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    assert (
        draft.dependency_impact is not None and not draft.dependency_impact.unresolved
    )
    prepared = validate_frozen_publication_review(draft)
    assert {p.row.id for p in prepared.projections} == {
        row.id for item in draft.items for row in item.new_chunks
    }
    assert len(prepared.retained) == 2
    assert sum(item.close_interval for item in prepared.retained) == 1
    outside = next(
        item
        for item in prepared.retained
        if json.loads(item.evidence.source_json)["regulatory_chunk_id"]
        == live_review.outside.id
    )
    assert not outside.close_interval
    assert outside.evidence.frozen_projection is None
    assert prepared.counts.canonical_changes == len(review_units(draft.items))
    assert prepared.counts.embeddings == sum(
        len(p.context.embedding_texts) for p in prepared.projections
    )
    assert prepared.counts.preserved_vectors == 2
    assert live_review.outside.id in draft.dependency_impact.unchanged_ids
    from onyx.db.regulatory_annex_publication import load_annex_publication_inputs
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from tests.external_dependency_unit.regulatory.test_annex_publisher_execution import (
        approve,
        invoke,
    )

    _file, settings, _access = load_annex_publication_inputs(source_session, draft)
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
    delivery = approve(source_session, live_review)
    assert invoke(delivery) == "approved"
    source_session.expire_all()
    assert live_review.review.status == "approved"
    assert sum(len(call) for call in calls) == prepared.counts.embeddings
    assert invoke(delivery) == "approved"
    assert sum(len(call) for call in calls) == prepared.counts.embeddings
