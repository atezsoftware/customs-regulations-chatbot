import json
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.db import labeling_configuration as configuration
from onyx.db.models import LLMProvider, ModelConfiguration, User
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import VERTEX_CREDENTIALS_FILE_KWARG


def _model() -> ModelConfiguration:
    provider = LLMProvider(
        id=1,
        provider=LlmProviderNames.VERTEX_AI,
        custom_config={
            VERTEX_CREDENTIALS_FILE_KWARG: json.dumps(
                {
                    "project_id": "intake-test-project",
                    "client_email": "intake@intake-test-project.iam.gserviceaccount.com",
                    "private_key": "test-only-not-a-real-credential",
                }
            )
        },
    )
    return ModelConfiguration(id=7, llm_provider=provider, is_visible=True)


@pytest.mark.parametrize("batch_key", [None, "test-only-batch-key"])
def test_intake_resolves_existing_credentials_without_gcs_configuration(
    monkeypatch: pytest.MonkeyPatch, batch_key: str | None
) -> None:
    monkeypatch.delenv("REGULATORY_LABELING_VERTEX_GCS_URI", raising=False)
    model = _model()
    authorize = MagicMock(return_value=model)
    monkeypatch.setattr(configuration, "_authorized_model", authorize)
    monkeypatch.setattr(
        configuration, "get_gemini_batch_api_key", lambda *_args, **_kwargs: batch_key
    )
    binding = configuration.resolve_intake_labeling_provider_binding(
        cast(Session, object()), model.id
    )
    assert binding.staging_uri is None
    assert binding.transport == "vertex_direct_v1"
    assert binding.api_key_hash is None
    assert binding.model == "gemini-3.5-flash"
    assert not authorize.call_args.kwargs.get("include_batch_key", False)
    if batch_key:
        assert batch_key not in binding.model_dump_json()
    with pytest.raises(
        configuration.LabelingStorageConfigurationError, match="Cloud Storage"
    ):
        configuration.resolve_labeling_provider_binding(
            cast(Session, object()), model.id
        )


@pytest.mark.parametrize("frozen_model", ["gemini-3.5-flash", "gemini-3.8-flash"])
@pytest.mark.parametrize("direct_timeout", [None, 240])
def test_direct_binding_selects_the_frozen_model_and_existing_credentials_without_storage(
    monkeypatch: pytest.MonkeyPatch,
    frozen_model: str,
    direct_timeout: float | None,
) -> None:
    model = _model()
    monkeypatch.setattr(
        configuration, "_authorized_model", lambda *_args, **_kwargs: model
    )
    monkeypatch.setattr(
        configuration, "get_gemini_batch_api_key", lambda *_args, **_kwargs: None
    )
    binding = configuration._binding(
        model, transport="vertex_direct_v1", model_name=frozen_model
    )
    with patch(
        "onyx.regulatory.labeling.vertex_direct.LabelingVertexDirectGateway",
        autospec=True,
    ) as gateway:
        if direct_timeout is None:
            result = configuration.resolve_labeling_gateway(
                cast(Session, object()), model.id, expected_binding=binding
            )
        else:
            result = configuration.resolve_labeling_gateway(
                cast(Session, object()),
                model.id,
                expected_binding=binding,
                direct_request_timeout_seconds=direct_timeout,
            )
    assert result is gateway.return_value
    assert "staging_uri" not in gateway.call_args.kwargs
    assert gateway.call_args.kwargs["config"].model_name == frozen_model
    assert gateway.call_args.kwargs["binding_sha256"] == binding.fingerprint
    assert gateway.call_args.kwargs["request_timeout_seconds"] == (
        direct_timeout or 120
    )


def test_intake_reports_missing_service_credentials_without_a_gcs_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    model.llm_provider.custom_config = {}
    monkeypatch.setattr(
        configuration, "_authorized_model", lambda *_args, **_kwargs: model
    )
    monkeypatch.setattr(
        configuration, "get_gemini_batch_api_key", lambda *_args, **_kwargs: None
    )
    with pytest.raises(ValueError, match="service-account configuration"):
        configuration.resolve_intake_labeling_provider_binding(
            cast(Session, object()), model.id
        )


def test_intake_catalog_checks_credentials_and_permission_without_batch_or_gcs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = _model()
    invalid = _model()
    invalid.id = 8
    invalid.llm_provider.id = 2
    invalid.llm_provider.custom_config = {}
    denied = _model()
    denied.id = 9
    denied.llm_provider.id = 3
    session = MagicMock(spec=Session)
    session.scalars.return_value = [valid, invalid, denied]
    monkeypatch.setattr(
        configuration,
        "get_labeling_provider_options",
        lambda *_args, **_kwargs: pytest.fail("Intake read normal GCS setup"),
    )
    monkeypatch.setattr(configuration, "fetch_user_group_ids", lambda *_args: [])
    monkeypatch.setattr(
        configuration,
        "can_user_access_llm_provider",
        lambda provider, *_args, **_kwargs: provider.id != 3,
    )
    monkeypatch.setattr(
        configuration,
        "get_gemini_batch_api_key",
        lambda *_args, **_kwargs: pytest.fail("Intake read a Developer Batch key"),
    )
    result = configuration.get_intake_labeling_provider_options(
        session, user=User(role=UserRole.ADMIN)
    )
    assert result[0]["configuration_error"] is None
    assert (
        result[1]["configuration_error"]
        == "The Google service-account configuration is invalid"
    )
    assert [option["id"] for option in result] == [7, 8]


def test_intake_automatically_uses_first_valid_authorized_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: list[configuration.LabelingProviderOption] = [
        {"id": 6, "name": "Broken", "configuration_error": "Missing credentials"},
        {"id": 7, "name": "Google", "configuration_error": None},
        {"id": 8, "name": "Other Google", "configuration_error": None},
    ]
    monkeypatch.setattr(
        configuration,
        "get_intake_labeling_provider_options",
        lambda *_args, **_kwargs: options,
    )
    binding = configuration._binding(
        _model(), transport="vertex_direct_v1", model_name="gemini-3.5-flash"
    )
    resolve = MagicMock(return_value=binding)
    monkeypatch.setattr(
        configuration, "resolve_intake_labeling_provider_binding", resolve
    )
    session, user = cast(Session, object()), cast(User, object())
    assert (
        configuration.select_intake_labeling_provider_binding(session, user=user)
        is binding
    )
    resolve.assert_called_once_with(session, 7, user=user)


def test_intake_automatic_selection_reports_no_valid_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        configuration,
        "get_intake_labeling_provider_options",
        lambda *_args, **_kwargs: [],
    )
    with pytest.raises(ValueError, match="accessible Google Vertex connection"):
        configuration.select_intake_labeling_provider_binding(
            cast(Session, object()), user=cast(User, object())
        )


def test_intake_provider_authorization_error_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorize = MagicMock(side_effect=ValueError("Provider unavailable to this user"))
    monkeypatch.setattr(configuration, "_authorized_model", authorize)
    with pytest.raises(ValueError, match="unavailable to this user"):
        configuration.resolve_intake_labeling_provider_binding(
            cast(Session, object()), 7
        )
