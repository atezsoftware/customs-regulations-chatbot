"""Whole-table requests preserve exact text through freezing, encoding, and replay."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google import genai
from google.auth.credentials import AnonymousCredentials
from google.genai import types as genai_types

from onyx.natural_language_processing.exceptions import EmbeddingProviderResponseError
from onyx.natural_language_processing.search_nlp_models import (
    CloudEmbedding,
)
from onyx.regulatory.amendments.annexes.context_dependencies import (
    freeze_encoder_inputs,
)
from onyx.regulatory.document_intake.encoder_budget import frozen_encoder_options
from shared_configs.enums import EmbeddingProvider, EmbedTextType
from tests.unit.onyx.regulatory.whole_table_encoder_helpers import _model, _table


def test_complete_freezing_and_encode_preserve_tail_above_legacy_limit() -> None:
    model = _model()
    text = _table()
    assert 512 < len(model.tokenizer.encode(text)) < 1848
    texts, config = freeze_encoder_inputs(
        [text],
        model,
        model_dim=3,
        formatter="normal-v1",
        max_seq_length=2048,
        strict_full_input=True,
    )
    assert texts == [text]
    assert config["max_sequence_length"] == 2048 and config["strict_full_input"] is True
    repeated, repeated_config = freeze_encoder_inputs(
        texts,
        model,
        model_dim=3,
        formatter="normal-v1",
        **frozen_encoder_options(config),
    )
    assert repeated == texts and repeated_config == config
    with patch.object(
        model, "_batch_encode_texts", return_value=[[1.0, 0.0, 0.0]]
    ) as send:
        model.encode(
            texts, text_type=EmbedTextType.PASSAGE, **frozen_encoder_options(config)
        )
    assert send.call_args.kwargs["texts"] == [text]
    assert send.call_args.kwargs["strict_full_input"] is True
    legacy, legacy_config = freeze_encoder_inputs(
        [text], model, model_dim=3, formatter="normal-v1"
    )
    assert legacy[0] != text and "SON_SATIR_KORUNDU" not in legacy[0]
    assert (
        "strict_full_input" not in legacy_config
        and legacy_config["max_sequence_length"] == 512
    )


@pytest.mark.parametrize("kind", ["overflow", "unicode", "model"])
def test_complete_encoder_rejects_loss_or_unsupported_model_before_any_call(
    kind: str,
) -> None:
    model = _model()
    text = _table()
    if kind == "overflow":
        text *= 5
    if kind == "unicode":
        text += "\ud800"
    if kind == "model":
        model.model_name = "different-model"
    with patch.object(model, "_batch_encode_texts") as send:
        with pytest.raises(ValueError):
            model.encode(
                [text],
                text_type=EmbedTextType.PASSAGE,
                max_seq_length=2048,
                strict_full_input=True,
            )
        send.assert_not_called()
    with pytest.raises(ValueError):
        freeze_encoder_inputs(
            [text],
            model,
            model_dim=3,
            formatter="normal-v1",
            max_seq_length=2048,
            strict_full_input=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statistics,accepted",
    [
        (dict(token_count=1200, truncated=False), True),
        (dict(token_count=1200), True),
        (dict(token_count=9000, truncated=False), False),
        (dict(token_count=1200, truncated=True), False),
        (None, False),
    ],
)
async def test_google_complete_transport_disables_truncation_and_checks_native_evidence(
    statistics: dict[str, object] | None, accepted: bool
) -> None:
    response = genai_types.EmbedContentResponse(
        embeddings=[
            genai_types.ContentEmbedding(
                values=[1.0, 0.0, 0.0],
                statistics=genai_types.ContentEmbeddingStatistics.model_validate(
                    statistics
                )
                if statistics is not None
                else None,
            )
        ]
    )
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(return_value=response)
    client.aio.aclose = AsyncMock()
    with (
        patch(
            "google.oauth2.service_account.Credentials.from_service_account_info",
            return_value=MagicMock(),
        ),
        patch("google.genai.Client", return_value=client),
    ):
        async with CloudEmbedding(
            '{"project_id":"test-project"}', EmbeddingProvider.GOOGLE
        ) as embedding:
            if accepted:
                result = await embedding._embed_vertex(
                    [_table()],
                    "gemini-embedding-2",
                    "RETRIEVAL_DOCUMENT",
                    3,
                    strict_full_input=True,
                )
                assert result == [[1.0, 0.0, 0.0]]
            else:
                with pytest.raises(
                    EmbeddingProviderResponseError, match="completeness"
                ):
                    await embedding._embed_vertex(
                        [_table()],
                        "gemini-embedding-2",
                        "RETRIEVAL_DOCUMENT",
                        3,
                        strict_full_input=True,
                    )
    assert client.aio.models.embed_content.await_args is not None
    request = client.aio.models.embed_content.await_args.kwargs
    assert request["config"].auto_truncate is False
    assert request["contents"][0].parts[0].text == "title: none | text: " + _table()


def test_unscoped_legacy_request_dump_does_not_gain_a_new_flag() -> None:
    from shared_configs.model_server_models import EmbedRequest

    request = EmbedRequest(
        texts=["text"],
        max_context_length=512,
        normalize_embeddings=True,
        text_type=EmbedTextType.PASSAGE,
    )
    assert "strict_full_input" not in request.model_dump()


@pytest.mark.asyncio
async def test_real_sdk_decodes_enterprise_native_usage_without_inventing_truncated_false() -> (
    None
):
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "embedding": {"values": [1.0, 0.0, 0.0]},
                "usageMetadata": {"promptTokenCount": 1421},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        credentials = AnonymousCredentials()
        credentials.token = "unit-test-token"
        client = genai.Client(
            enterprise=True,
            project="test-project",
            location="global",
            credentials=credentials,
            http_options=genai_types.HttpOptions(
                api_version="v1", httpx_async_client=http_client
            ),
        )
        try:
            result = await client.aio.models.embed_content(
                model="gemini-embedding-2",
                contents="title: none | text: " + _table(),
                config=genai_types.EmbedContentConfig(
                    auto_truncate=False, output_dimensionality=3
                ),
            )
        finally:
            await client.aio.aclose()
    assert result.embeddings is not None and result.embeddings[0].statistics is not None
    assert result.embeddings[0].statistics.token_count == 1421
    assert result.embeddings[0].statistics.truncated is None
    assert requests[0]["embedContentConfig"]["autoTruncate"] is False
