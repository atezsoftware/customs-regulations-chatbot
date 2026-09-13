from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.llm import (
    fetch_existing_llm_provider_by_id,
    get_gemini_batch_api_key,
    remove_llm_provider,
    upsert_llm_provider,
)
from onyx.db.models import EncryptedString
from onyx.db.models import LLMProvider as LLMProviderModel
from onyx.llm.constants import LlmProviderNames
from onyx.server.manage.llm.models import LLMProviderUpsertRequest, LLMProviderView

_BATCH_KEY = "gemini-batch-secret-for-storage-test"
_CHAT_KEY = "existing-chat-key"
_CUSTOM_CONFIG = {"existing_auth_setting": "unchanged"}


def _request(
    *,
    provider_id: int | None = None,
    batch_key: object = ...,
) -> LLMProviderUpsertRequest:
    values: dict[str, object] = {
        "id": provider_id,
        "name": f"gemini-batch-key-{uuid4().hex}" if provider_id is None else None,
        "provider": LlmProviderNames.VERTEX_AI,
        "api_key": _CHAT_KEY,
        "api_key_changed": True,
        "custom_config": _CUSTOM_CONFIG,
        "custom_config_changed": True,
    }
    if batch_key is not ...:
        values["gemini_batch_api_key"] = batch_key
    return LLMProviderUpsertRequest.model_validate(values)


def _stored_batch_key(db_session: Session, provider_id: int) -> str | None:
    provider = fetch_existing_llm_provider_by_id(provider_id, db_session)
    assert provider is not None
    return (
        provider.gemini_batch_api_key.get_value(apply_mask=False)
        if provider.gemini_batch_api_key
        else None
    )


def test_batch_key_request_is_write_only_and_storage_is_encrypted(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = _request(batch_key=_BATCH_KEY)
    assert "gemini_batch_api_key" in request.model_fields_set
    assert _BATCH_KEY not in repr(request)
    assert "gemini_batch_api_key" not in request.model_dump()

    provider = upsert_llm_provider(request, db_session)
    try:
        assert provider.has_gemini_batch_api_key is True
        assert "gemini_batch_api_key" not in provider.model_dump()
        assert _BATCH_KEY not in provider.model_dump_json()
        assert _stored_batch_key(db_session, provider.id) == _BATCH_KEY

        assert isinstance(
            LLMProviderModel.__table__.c.gemini_batch_api_key.type, EncryptedString
        )
        assert _BATCH_KEY not in caplog.text
    finally:
        remove_llm_provider(db_session, provider.id)


def test_omitted_batch_key_preserves_it_without_changing_other_auth(
    db_session: Session,
) -> None:
    created = upsert_llm_provider(_request(batch_key=_BATCH_KEY), db_session)
    try:
        updated = upsert_llm_provider(_request(provider_id=created.id), db_session)

        assert updated.has_gemini_batch_api_key is True
        assert _stored_batch_key(db_session, created.id) == _BATCH_KEY
        stored = fetch_existing_llm_provider_by_id(created.id, db_session)
        assert stored is not None
        assert stored.api_key is not None
        assert stored.api_key.get_value(apply_mask=False) == _CHAT_KEY
        assert stored.custom_config == _CUSTOM_CONFIG
    finally:
        remove_llm_provider(db_session, created.id)


def test_presence_projection_does_not_audit_or_expose_until_explicit_use(
    db_session: Session,
) -> None:
    created = upsert_llm_provider(_request(batch_key=_BATCH_KEY), db_session)
    try:
        stored = fetch_existing_llm_provider_by_id(created.id, db_session)
        assert stored is not None

        with patch("onyx.utils.credential_audit.emit_credential_access") as emit_access:
            view = LLMProviderView.from_model(stored, include_api_key=False)
            assert view.has_gemini_batch_api_key is True
            emit_access.assert_not_called()

            assert (
                get_gemini_batch_api_key(stored, user_id="credential-reader")
                == _BATCH_KEY
            )
            emit_access.assert_called_once_with(
                credential_type="gemini_batch_api_key",
                provider=LlmProviderNames.VERTEX_AI,
                row_id=created.id,
                user_id="credential-reader",
            )
    finally:
        remove_llm_provider(db_session, created.id)


def test_batch_key_is_redacted_from_persistence_errors(
    db_session: Session,
) -> None:
    created = upsert_llm_provider(_request(batch_key=None), db_session)
    try:
        request = _request(provider_id=created.id, batch_key=_BATCH_KEY)
        with patch.object(
            db_session,
            "commit",
            side_effect=RuntimeError(f"database rejected {_BATCH_KEY}"),
        ):
            with pytest.raises(ValueError) as exc_info:
                upsert_llm_provider(request, db_session)

        assert _BATCH_KEY not in str(exc_info.value)
        db_session.rollback()
    finally:
        remove_llm_provider(db_session, created.id)


@pytest.mark.parametrize("clear_value", [None, "", "   "])
def test_explicit_empty_batch_key_clears_only_batch_auth(
    db_session: Session,
    clear_value: str | None,
) -> None:
    created = upsert_llm_provider(_request(batch_key=_BATCH_KEY), db_session)
    try:
        updated = upsert_llm_provider(
            _request(provider_id=created.id, batch_key=clear_value), db_session
        )

        assert updated.has_gemini_batch_api_key is False
        assert _stored_batch_key(db_session, created.id) is None
        stored = fetch_existing_llm_provider_by_id(created.id, db_session)
        assert stored is not None
        assert stored.api_key is not None
        assert stored.api_key.get_value(apply_mask=False) == _CHAT_KEY
        assert stored.custom_config == _CUSTOM_CONFIG
    finally:
        remove_llm_provider(db_session, created.id)
