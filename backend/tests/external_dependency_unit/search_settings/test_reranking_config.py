from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import select, text
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
    if settings is None:
        settings = _new_search_settings(IndexModelStatus.PRESENT)
        db_session.add(settings)
        db_session.flush()

    delete_reranker_configuration(db_session, commit=False)
    try:
        yield settings
    finally:
        db_session.rollback()


def test_upsert_can_defer_commit_for_caller_owned_transaction(
    db_session: Session,
) -> None:
    settings_id = db_session.scalar(
        select(SearchSettings.id)
        .where(SearchSettings.status == IndexModelStatus.PRESENT)
        .order_by(SearchSettings.id.desc())
    )
    assert settings_id is not None
    persisted_row = text(
        "SELECT rerank_enabled, rerank_provider_type, rerank_model_name, "
        "rerank_api_key, rerank_updated_at, rerank_updated_by_user_id "
        "FROM search_settings WHERE id = :settings_id"
    )
    before = db_session.execute(persisted_row, {"settings_id": settings_id}).one()

    try:
        upsert_reranker_configuration(
            db_session,
            enabled=True,
            provider_type=RerankerProvider.OPENROUTER,
            model_name="openrouter/rollback-reranker",
            api_key="rollback-secret",
            commit=False,
        )
        db_session.rollback()

        after = db_session.execute(persisted_row, {"settings_id": settings_id}).one()
        assert after == before
    finally:
        db_session.rollback()


def test_delete_can_defer_commit_for_caller_owned_transaction(
    db_session: Session,
) -> None:
    persisted_rows = text(
        "SELECT id, rerank_enabled, rerank_provider_type, rerank_model_name, "
        "rerank_api_key, rerank_updated_at, rerank_updated_by_user_id "
        "FROM search_settings ORDER BY id"
    )
    before = list(db_session.execute(persisted_rows).all())

    try:
        delete_reranker_configuration(db_session, commit=False)
        db_session.rollback()

        after = list(db_session.execute(persisted_rows).all())
        assert after == before
    finally:
        db_session.rollback()


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
        commit=False,
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
        commit=False,
    )

    upsert_reranker_configuration(
        db_session,
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/second-reranker",
        api_key=None,
        commit=False,
    )

    config = get_reranker_configuration(db_session)
    assert config.model_name == "openrouter/second-reranker"
    assert config.api_key is not None
    assert config.api_key.get_value(apply_mask=False) == "openrouter-secret"


def test_disabling_retains_provider_model_and_exact_ciphertext(
    db_session: Session,
    present_settings: SearchSettings,
) -> None:
    upsert_reranker_configuration(
        db_session,
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/first-reranker",
        api_key="openrouter-secret",
        commit=False,
    )
    before = db_session.execute(
        text("SELECT rerank_api_key FROM search_settings WHERE id = :settings_id"),
        {"settings_id": present_settings.id},
    ).scalar_one()

    upsert_reranker_configuration(
        db_session,
        enabled=False,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/second-reranker",
        api_key=None,
        commit=False,
    )

    config = get_reranker_configuration(db_session)
    after = db_session.execute(
        text("SELECT rerank_api_key FROM search_settings WHERE id = :settings_id"),
        {"settings_id": present_settings.id},
    ).scalar_one()
    assert config.enabled is False
    assert config.provider_type == RerankerProvider.OPENROUTER
    assert config.model_name == "openrouter/second-reranker"
    assert after == before


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
        commit=False,
    )
    target = _new_search_settings(IndexModelStatus.PAST)
    db_session.add(target)
    db_session.flush()

    transfer_reranker_configuration__no_commit(
        db_session,
        source_search_settings_id=present_settings.id,
        target_search_settings_id=target.id,
    )
    db_session.flush()
    db_session.expire(target)

    assert target.rerank_enabled is True
    assert target.rerank_provider_type == RerankerProvider.OPENROUTER
    assert target.rerank_model_name == "openrouter/legal-reranker"
    assert target.rerank_api_key is not None
    assert target.rerank_api_key.get_value(apply_mask=False) == "openrouter-secret"


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
    db_session.flush()

    upsert_reranker_configuration(
        db_session,
        enabled=True,
        provider_type=RerankerProvider.OPENROUTER,
        model_name="openrouter/live-reranker",
        api_key="live-secret",
        commit=False,
    )

    delete_reranker_configuration(db_session, commit=False)

    assert all(
        row.rerank_api_key is None for row in db_session.scalars(select(SearchSettings))
    )
    assert all(
        row.rerank_enabled is False
        for row in db_session.scalars(select(SearchSettings))
    )
