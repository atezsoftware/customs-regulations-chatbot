from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.enums import EmbeddingPrecision, IndexModelStatus, SwitchoverType
from onyx.db.models import SearchSettings
from onyx.db.reranking import (
    delete_reranker_configuration,
    get_reranker_configuration,
    transfer_reranker_configuration__no_commit,
    upsert_reranker_configuration,
)
from onyx.utils.sensitive import SensitiveValue
from onyx.utils.variable_functionality import fetch_versioned_implementation
from shared_configs.enums import RerankerProvider

TEST_ENCRYPTION_KEY = "reranker-test-key-material-32bytes"


def _new_search_settings(status: IndexModelStatus) -> SearchSettings:
    unique_suffix = uuid4().hex[:8]
    return SearchSettings(
        model_name=f"reranker-test-model-{unique_suffix}",
        model_dim=768,
        normalize=True,
        query_prefix="",
        passage_prefix="",
        status=status,
        index_name=f"reranker_test_{unique_suffix}",
        provider_type=None,
        switchover_type=SwitchoverType.REINDEX,
        use_port_flow=False,
        embedding_precision=EmbeddingPrecision.FLOAT,
        multipass_indexing=False,
        enable_contextual_rag=True,
    )


@pytest.fixture(autouse=True)
def _strict_encryption_environment(
    enable_ee: None,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    monkeypatch.setenv("ENCRYPTION_KEY_SECRET", TEST_ENCRYPTION_KEY)
    fetch_versioned_implementation.cache_clear()
    yield
    fetch_versioned_implementation.cache_clear()


@pytest.fixture
def present_settings(db_session: Session) -> Generator[SearchSettings, None, None]:
    settings = db_session.scalar(
        select(SearchSettings)
        .where(SearchSettings.status == IndexModelStatus.PRESENT)
        .order_by(SearchSettings.id.desc())
    )
    created = settings is None
    if settings is None:
        settings = _new_search_settings(IndexModelStatus.PRESENT)
        db_session.add(settings)
        db_session.commit()

    delete_reranker_configuration(db_session)
    try:
        yield settings
    finally:
        delete_reranker_configuration(db_session)
        if created:
            db_session.delete(settings)
            db_session.commit()


def test_upsert_and_get_reranker_configuration(
    db_session: Session,
    present_settings: SearchSettings,  # noqa: ARG001
) -> None:
    upsert_reranker_configuration(
        db_session,
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/legal-reranker",
        api_key="openrouter-secret",
    )

    config = get_reranker_configuration(db_session)

    assert config.enabled is True
    assert config.provider_type == RerankerProvider.OPENROUTER
    assert config.model_name == "openrouter/legal-reranker"
    assert isinstance(config.api_key, SensitiveValue)
    assert config.api_key.get_value(apply_mask=False) == "openrouter-secret"


def test_upsert_without_key_preserves_existing_secret(
    db_session: Session,
    present_settings: SearchSettings,  # noqa: ARG001
) -> None:
    upsert_reranker_configuration(
        db_session,
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/first-reranker",
        api_key="openrouter-secret",
    )

    upsert_reranker_configuration(
        db_session,
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/second-reranker",
        api_key=None,
    )

    config = get_reranker_configuration(db_session)
    assert config.model_name == "openrouter/second-reranker"
    assert config.api_key is not None
    assert config.api_key.get_value(apply_mask=False) == "openrouter-secret"


def test_transfer_copies_live_reranker_configuration_to_promotion_target(
    db_session: Session,
    present_settings: SearchSettings,
) -> None:
    upsert_reranker_configuration(
        db_session,
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/legal-reranker",
        api_key="openrouter-secret",
    )
    target = _new_search_settings(IndexModelStatus.PAST)
    db_session.add(target)
    db_session.commit()

    try:
        transfer_reranker_configuration__no_commit(
            db_session,
            source_search_settings_id=present_settings.id,
            target_search_settings_id=target.id,
        )
        db_session.commit()
        db_session.expire(target)

        assert target.rerank_enabled is True
        assert target.rerank_provider_type == RerankerProvider.OPENROUTER
        assert target.rerank_model_name == "openrouter/legal-reranker"
        assert target.rerank_api_key is not None
        assert target.rerank_api_key.get_value(apply_mask=False) == "openrouter-secret"
    finally:
        db_session.delete(target)
        db_session.commit()


def test_delete_clears_reranker_secret_from_every_settings_row(
    db_session: Session,
    present_settings: SearchSettings,  # noqa: ARG001
) -> None:
    historical = _new_search_settings(IndexModelStatus.PAST)
    historical.rerank_enabled = True
    historical.rerank_provider_type = RerankerProvider.OPENROUTER
    historical.rerank_model_name = "openrouter/historical-reranker"
    historical.rerank_api_key = "historical-secret"  # ty: ignore[invalid-assignment]
    db_session.add(historical)
    db_session.commit()

    upsert_reranker_configuration(
        db_session,
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/live-reranker",
        api_key="live-secret",
    )

    try:
        delete_reranker_configuration(db_session)

        assert all(
            row.rerank_api_key is None
            for row in db_session.scalars(select(SearchSettings))
        )
        assert all(
            row.rerank_enabled is False
            for row in db_session.scalars(select(SearchSettings))
        )
    finally:
        db_session.delete(historical)
        db_session.commit()
