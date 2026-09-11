"""Provider properties remain usable after the short annex read session closes."""

from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session, attributes

from onyx.db.enums import IndexModelStatus
from onyx.db.models import (
    AmendmentBatch,
    CloudEmbeddingProvider,
    DocumentSet,
    SearchSettings,
)
from onyx.db.regulatory_annex_publication import (
    load_annex_context_settings,
    load_annex_publication_inputs,
)
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
from shared_configs.enums import EmbeddingProvider
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _file


@pytest.mark.parametrize("cloud", [False, True])
@pytest.mark.parametrize("boundary", ["context", "publication"])
def test_detached_annex_embedding_provider_properties(
    cloud: bool,
    boundary: str,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_type = None
    if cloud:
        existing = set(
            source_session.scalars(select(CloudEmbeddingProvider.provider_type))
        )
        provider_type = next(
            (item for item in EmbeddingProvider if item not in existing), None
        )
        assert provider_type is not None, "fixture needs an unused provider enum"
        source_session.add(
            CloudEmbeddingProvider(
                provider_type=provider_type,
                api_key="annex-test-only-key",
                api_url="https://fixture.invalid/v1",
                api_version="fixture-version",
                deployment_name="fixture-deployment",
            )
        )
    settings = SearchSettings(
        status=IndexModelStatus.PAST,
        index_name="annex-detached-" + uuid4().hex,
        model_name="fixture-embedding",
        model_dim=3,
        normalize=True,
        query_prefix="",
        passage_prefix="",
        provider_type=provider_type,
    )
    source_session.add(settings)
    scope = DocumentSet(
        name="annex-detached-" + uuid4().hex, description="", is_up_to_date=True
    )
    source_session.add(scope)
    source_session.flush()
    file = _file(source_session, scope)
    batch = AmendmentBatch(
        document_set_id=scope.id,
        user_file_ids=[str(file.id)],
        raw_text="Fictional source",
        status="analyzed",
    )
    source_session.add(batch)
    source_session.flush()
    identifier = settings.id
    draft = AnnexChangeDraft(
        batch_id=batch.id,
        user_file_id=file.id,
        instruction_indices=[0],
        instruction_texts=["EK-1"],
        annex_label="EK-1",
    )

    def owned_settings(session: Session, **_kwargs: object) -> SearchSettings:
        row = session.get(SearchSettings, identifier)
        assert row is not None
        # Select a test-only row without altering the configured PRESENT index.
        attributes.set_committed_value(row, "status", IndexModelStatus.PRESENT)
        assert "cloud_provider" in inspect(row).unloaded
        return row

    monkeypatch.setattr(
        "onyx.db.search_settings.get_current_search_settings", owned_settings
    )
    monkeypatch.setattr(
        "onyx.db.search_settings.get_active_search_settings_list",
        lambda session: [owned_settings(session)],
    )
    monkeypatch.setattr(
        "onyx.regulatory.amendments.annexes.config.REGULATORY_ANNEX_ENVIRONMENT",
        "local-test",
    )
    with Session(
        source_session.connection(), join_transaction_mode="create_savepoint"
    ) as short_read:
        if boundary == "context":
            detached = load_annex_context_settings(short_read)
        else:
            _, loaded, _ = load_annex_publication_inputs(short_read, draft)
            detached = loaded[0]
    assert inspect(detached).detached

    class CapturingEmbedder(DefaultIndexingEmbedder):
        def __init__(self, **kwargs: Any) -> None:
            self.arguments = kwargs

    embedder = CapturingEmbedder.from_db_search_settings(detached)
    assert isinstance(embedder, CapturingEmbedder)
    assert embedder.arguments["api_key"] == ("annex-test-only-key" if cloud else None)
    assert embedder.arguments["api_url"] == (
        "https://fixture.invalid/v1" if cloud else None
    )
    assert embedder.arguments["api_version"] == ("fixture-version" if cloud else None)
    assert embedder.arguments["deployment_name"] == (
        "fixture-deployment" if cloud else None
    )
    assert "cloud_provider" not in inspect(detached).unloaded
