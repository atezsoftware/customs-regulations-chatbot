"""Transport-specific result limits resolved without a database or cloud call."""

import json
from typing import cast
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from onyx.db import labeling_configuration
from onyx.db.labeling_configuration import LabelingProviderTransport
from onyx.db.models import LLMProvider, ModelConfiguration
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import VERTEX_CREDENTIALS_FILE_KWARG


@pytest.mark.parametrize(
    "transport,constructor,expected_bytes",
    [
        (
            "vertex_gcs_v1",
            "onyx.regulatory.labeling.vertex_batch.LabelingVertexBatchGateway",
            8 * 1024**3,
        ),
        (
            "gemini_files_v1",
            "onyx.db.labeling_configuration.GoogleGeminiFilesBatchGateway",
            64 * 1024 * 1024,
        ),
        (
            "gemini_inline_api_key_v1",
            "onyx.regulatory.labeling.gemini_inline_batch.LabelingGeminiInlineBatchGateway",
            64 * 1024 * 1024,
        ),
    ],
)
def test_only_native_storage_transport_uses_large_result_limit(
    monkeypatch: pytest.MonkeyPatch,
    transport: LabelingProviderTransport,
    constructor: str,
    expected_bytes: int,
) -> None:
    staging_uri = "gs://labeling-test-bucket/staging"
    monkeypatch.setenv("REGULATORY_LABELING_VERTEX_GCS_URI", staging_uri)
    provider = LLMProvider(
        id=1,
        provider=LlmProviderNames.VERTEX_AI,
        custom_config={
            VERTEX_CREDENTIALS_FILE_KWARG: json.dumps(
                {
                    "project_id": "labeling-project",
                    "client_email": "labeler@labeling-project.iam.gserviceaccount.com",
                    "private_key": "test-only-not-a-real-credential",
                }
            )
        },
    )
    model = ModelConfiguration(id=7, llm_provider=provider, is_visible=True)
    monkeypatch.setattr(
        labeling_configuration, "_authorized_model", lambda *_args, **_kwargs: model
    )
    monkeypatch.setattr(
        labeling_configuration,
        "get_gemini_batch_api_key",
        lambda *_args, **_kwargs: "test-only-batch-key",
    )
    if transport == "gemini_inline_api_key_v1":
        binding, _ = labeling_configuration._inline_binding(model, None)
    else:
        binding = labeling_configuration._binding(
            model,
            transport=transport,
            staging_uri=staging_uri if transport == "vertex_gcs_v1" else None,
        )

    with patch(constructor, autospec=True) as gateway_constructor:
        labeling_configuration.resolve_labeling_gateway(
            cast(Session, object()), model.id, expected_binding=binding
        )

    arguments = gateway_constructor.call_args.kwargs
    assert arguments["max_result_bytes"] == expected_bytes
    assert arguments["request_timeout_seconds"] == 20
