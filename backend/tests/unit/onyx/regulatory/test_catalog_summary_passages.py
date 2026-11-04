"""Lossless server-owned passages prevent quotation and source-copying failures."""

import json
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from pydantic import ValidationError

from onyx.llm.model_response import Choice, Message, ModelResponse, Usage
from onyx.llm.multi_llm import LitellmLLM, LLMRateLimitError
from onyx.regulatory import catalog_summary as summary
from onyx.regulatory.catalog_summary_usage import CatalogSummaryUsage
from onyx.server.features.regulatory.catalog_models import (
    CatalogFile,
    CatalogSearchHit,
    CatalogSearchPage,
)
from onyx.server.features.regulatory.models import RegulatoryChunkSnapshot
from onyx.server.manage.llm.models import LLMProviderView, ModelConfigurationView


def _page(texts: list[str]) -> CatalogSearchPage:
    now = datetime.now(timezone.utc)
    hits = []
    for index, text in enumerate(texts):
        file_id = UUID(int=index // 8 + 1)
        file = CatalogFile(
            id=file_id,
            name=f"Örnekler/{file_id}.md",
            folder_path="Örnekler",
            user_id=None,
            file_id=str(file_id),
            status="COMPLETED",
            file_type="text/markdown",
            created_at=now,
            chunk_count=8,
            document_sets=[],
            pending_review=False,
        )
        chunk = RegulatoryChunkSnapshot(
            id=f"canonical-{index}",
            user_file_id=str(file_id),
            text=text,
            position=index,
            chunk_type="paragraph",
            heading_path=["Örnek", "Koşullar"],
            chunk_metadata={},
            validity_start_date=None,
            validity_end_date=None,
            status="active",
            source="original",
            supersedes_chunk_id=None,
            superseded_by_chunk_id=None,
            created_at=now,
            updated_at=now,
        )
        hits.append(CatalogSearchHit(file=file, chunk=chunk, score=1 / (index + 1)))
    return CatalogSearchPage(
        items=hits, total=len(hits), offset=0, limit=20, has_more=False
    )


def test_twenty_source_fixture_is_lossless_without_duplicate_text_on_wire() -> None:
    # Match the failed search's 20-source structure, not its private corpus text.
    texts = [
        f"**Örnek {i}**\n\n  Bu paragraf {i} numaralı kaydın koşullarını açıklar.\nİkinci cümle kapsamı korur.  "
        for i in range(20)
    ]
    texts[4] = " ".join(["Uzun cümle ayrı bir koşulu eksiksiz açıklıyor."] * 25)
    page = _page(texts)
    payload = json.loads(
        str(summary.catalog_summary_messages("örnek", page)[1].content)
    )
    assert len(payload["sources"]) == 20
    assert "passages" not in payload
    for source, hit in zip(payload["sources"], page.items):
        assert "text" not in source
        assert (
            "".join(segment["text"] for segment in source["segments"]) == hit.chunk.text
        )
        assert source["chunk_id"] == hit.chunk.id
        assert source["file_id"] == str(hit.file.id)
    assert summary.build_catalog_summary_passages(
        page
    ) == summary.build_catalog_summary_passages(page)
    for passage in summary.build_catalog_summary_passages(page):
        owner = page.items[passage["reference"] - 1].chunk
        assert owner.id == passage["chunk_id"]
        assert owner.text[passage["start"] : passage["end"]] == passage["text"]
        assert len(passage["passage_id"]) <= 12


def test_complete_paragraph_preferred_and_long_context_is_never_cut() -> None:
    text = "Bir koşul açıklanıyor. Ancak ikinci koşul da korunuyor."
    page = _page([text, " ".join(["uzun"] * 118)])
    passages = summary.build_catalog_summary_passages(page)
    assert len(passages) == 1 and passages[0]["text"] == text
    payload = json.loads(
        str(summary.catalog_summary_messages("örnek", page)[1].content)
    )
    assert payload["sources"][1]["segments"] == [
        {"text": page.items[1].chunk.text, "eligible": False}
    ]


def test_long_paragraph_uses_complete_sentences_and_preserves_numeric_abbreviations() -> (
    None
):
    sentence = "Örnek 1. bölüm 12.10.2023 tarihinde Dr. Örnek tarafından açıklanıyor."
    page = _page([" ".join([sentence] * 20)])
    passages = summary.build_catalog_summary_passages(page)
    assert passages and all(passage["text"] == sentence for passage in passages)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("not json", summary.CatalogSummaryRejectionCode.INVALID_JSON),
        (
            '{"passage_ids":[],"passage_ids":[]}',
            summary.CatalogSummaryRejectionCode.INVALID_JSON,
        ),
        ("[]", summary.CatalogSummaryRejectionCode.INVALID_SHAPE),
        ('{"passage_ids":[]}', summary.CatalogSummaryRejectionCode.INVALID_SHAPE),
        ('{"passage_ids":[true]}', summary.CatalogSummaryRejectionCode.INVALID_SHAPE),
        ('{"passage_ids":[1]}', summary.CatalogSummaryRejectionCode.INVALID_SHAPE),
        (
            '{"passage_ids":["s2p1"]}',
            summary.CatalogSummaryRejectionCode.UNKNOWN_PASSAGE,
        ),
        (
            '{"passage_ids":["s1p1","s1p1"]}',
            summary.CatalogSummaryRejectionCode.DUPLICATE_PASSAGE,
        ),
        (
            '{"passage_ids":["s1p1"],"quote":"invented"}',
            summary.CatalogSummaryRejectionCode.INVALID_SHAPE,
        ),
    ],
)
def test_invalid_selection_has_safe_typed_reason(
    content: str, code: summary.CatalogSummaryRejectionCode
) -> None:
    with pytest.raises(summary.CatalogSummaryUnavailable) as error:
        summary.render_catalog_passage_summary(
            content, _page(["Tam kaynak içeriği korunuyor."])
        )
    assert error.value.code is code


def test_budget_keeps_whole_selected_units_and_validates_even_skipped_ids() -> None:
    texts = [
        " ".join(["birinci"] * 100),
        " ".join(["ikinci"] * 100),
        "Üçüncü kaynak tam kalır.",
    ]
    page = _page(texts)
    rendered = summary.render_catalog_passage_summary(
        '{"passage_ids":["s1p1","s2p1","s3p1"]}', page
    )
    assert (
        rendered == f"[1] numaralı parça: “{texts[0]}” [3] numaralı parça: “{texts[2]}”"
    )
    assert len(rendered.split()) <= 120
    with pytest.raises(summary.CatalogSummaryUnavailable) as error:
        summary.render_catalog_passage_summary(
            '{"passage_ids":["s1p1","s2p1","unknown"]}', page
        )
    assert error.value.code is summary.CatalogSummaryRejectionCode.UNKNOWN_PASSAGE


def test_model_cannot_override_passage_reference_ownership() -> None:
    page = _page(
        [f"Özgün kaynak içeriği {i} olarak belirlenmiştir." for i in range(20)]
    )
    rendered = summary.render_catalog_passage_summary('{"passage_ids":["s16p1"]}', page)
    assert rendered == f"[16] numaralı parça: “{page.items[15].chunk.text}”"


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_observer_records_actual_usage_before_invalid_selection_or_truncation(
    finish_reason: str,
) -> None:
    llm = MagicMock()
    llm.config.max_input_tokens = 100_000
    llm.invoke.return_value = ModelResponse(
        id="gen-safe-id",
        created="now",
        choice=Choice(
            message=Message(content='{"passage_ids":["unknown"]}'),
            finish_reason=finish_reason,
        ),
        usage=Usage(
            prompt_tokens=1000,
            completion_tokens=25,
            total_tokens=1025,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            cost=0.00056,
        ),
    )
    events: list[CatalogSummaryUsage] = []
    with (
        patch.object(summary, "get_catalog_summary_llm", return_value=llm),
        patch.object(summary, "get_llm_token_counter", return_value=lambda _: 1),
        patch.object(summary, "llm_generation_span"),
        patch.object(summary, "record_llm_response"),
        pytest.raises(summary.CatalogSummaryUnavailable),
    ):
        summary.generate_catalog_summary(
            "örnek",
            _page(["Tam kaynak içeriği korunuyor."]),
            MagicMock(),
            usage_observer=events.append,
        )
    assert len(events) == 1
    assert events[0].model_dump() == {
        "response_id": "gen-safe-id",
        "input_tokens": 1000,
        "output_tokens": 25,
        "provider_cost_usd": 0.00056,
    }
    llm.invoke.assert_called_once()
    assert llm.invoke.call_args.kwargs["provider_compatibility_attempts"] == 1


@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf"), -0.1])
def test_usage_type_rejects_nonfinite_or_negative_provider_cost(
    bad_cost: float,
) -> None:
    with pytest.raises(ValidationError):
        CatalogSummaryUsage(
            response_id=None,
            input_tokens=None,
            output_tokens=None,
            provider_cost_usd=bad_cost,
        )


def test_unselectable_context_skips_paid_call() -> None:
    with (
        patch.object(summary, "get_catalog_summary_llm") as get_llm,
        pytest.raises(summary.CatalogSummaryUnavailable) as error,
    ):
        summary.generate_catalog_summary(
            "örnek", _page([" ".join(["uzun"] * 118)]), MagicMock()
        )
    assert error.value.code is summary.CatalogSummaryRejectionCode.NO_PASSAGES
    get_llm.assert_not_called()


@pytest.mark.parametrize("status", [200, 429])
def test_real_litellm_wire_selects_ids_once_with_reasoning_and_retries_off(
    status: int,
) -> None:
    page = _page(["Tam kaynak içeriği korunuyor."])
    bodies: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if status == 429:
            return httpx.Response(
                status, json={"error": {"message": "rate limited", "code": 429}}
            )
        return httpx.Response(
            status,
            json={
                "id": "gen-safe-id",
                "object": "chat.completion",
                "created": 0,
                "model": summary.CATALOG_SUMMARY_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"passage_ids":["s1p1"]}',
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 12,
                    "total_tokens": 512,
                    "cost": 0.00029,
                },
            },
        )

    provider = MagicMock(id=1, provider="openrouter")
    model = MagicMock(is_visible=True)
    model.name = summary.CATALOG_SUMMARY_MODEL
    provider.model_configurations = [model]
    view = LLMProviderView(
        id=1,
        provider="openrouter",
        api_key="test-key",
        model_configurations=[
            ModelConfigurationView(
                name=model.name,
                is_visible=True,
                supports_image_input=False,
                max_input_tokens=200_000,
            )
        ],
    )
    with (
        patch.object(summary, "fetch_existing_llm_providers", return_value=[provider]),
        patch.object(summary, "can_user_access_llm_provider", return_value=True),
        patch.object(summary.LLMProviderView, "from_model", return_value=view),
    ):
        llm = summary.get_catalog_summary_llm(MagicMock())
    assert isinstance(llm, LitellmLLM)
    original = llm._completion
    events: list[CatalogSummaryUsage] = []
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:

        def with_client(*args: Any, **kwargs: Any) -> object:
            kwargs["client"] = HTTPHandler(client=http_client)
            return original(*args, **kwargs)

        with (
            patch.object(llm, "_completion", side_effect=with_client) as invoke,
            patch.object(summary, "get_catalog_summary_llm", return_value=llm),
            patch.object(summary, "get_llm_token_counter", return_value=lambda _: 1),
            patch.object(summary, "llm_generation_span"),
            patch.object(summary, "record_llm_response"),
            pytest.raises(LLMRateLimitError) if status == 429 else nullcontext(),
        ):
            result = summary.generate_catalog_summary(
                "örnek", page, MagicMock(), usage_observer=events.append
            )
            assert result == "[1] numaralı parça: “Tam kaynak içeriği korunuyor.”"
            assert events[0].provider_cost_usd == 0.00029
    assert len(bodies) == 1 and invoke.call_count == 1
    assert bodies[0]["reasoning"] == {"enabled": False}
    assert bodies[0]["max_tokens"] == 160
    assert "num_retries" not in bodies[0] and "max_retries" not in bodies[0]
