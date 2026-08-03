from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.indexing.indexing_pipeline import (
    ContextualEnrichmentError,
    _contextual_prompt_content_budget,
    _invoke_contextual_llm_with_retry,
    _trim_contextual_document,
)
from onyx.llm.interfaces import LLM
from onyx.llm.models import UserMessage
from onyx.llm.multi_llm import LLMTimeoutError
from onyx.llm.utils import MAX_CONTEXT_TOKENS
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.prompts.contextual_retrieval import DOCUMENT_SUMMARY_PROMPT
from onyx.tracing.flows import LLMFlow


class _CharacterTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self.decode_calls = 0

    def encode(self, string: str) -> list[int]:
        return [ord(character) for character in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        self.decode_calls += 1
        return "".join(chr(token) for token in tokens)


def _llm_with_input_limit(max_input_tokens: int) -> LLM:
    llm = MagicMock(spec=LLM)
    llm.config.max_input_tokens = max_input_tokens
    return cast(LLM, llm)


def test_contextual_prompt_budget_reserves_margin_prompt_output_and_chunk() -> None:
    tokenizer = _CharacterTokenizer()
    llm = _llm_with_input_limit(200_000)
    static_prompt = DOCUMENT_SUMMARY_PROMPT.format(document="")

    with patch(
        "onyx.indexing.indexing_pipeline.GEN_AI_INPUT_TOKEN_SAFETY_MARGIN", 0.05
    ):
        budget = _contextual_prompt_content_budget(
            llm=llm,
            tokenizer=tokenizer,
            prompt_without_content=static_prompt,
            reserved_dynamic_tokens=2_048,
        )

    assert budget == (
        190_000 - len(tokenizer.encode(static_prompt)) - 2_048 - MAX_CONTEXT_TOKENS
    )


def test_oversized_legal_document_preserves_ends_within_budget() -> None:
    tokenizer = _CharacterTokenizer()
    llm = _llm_with_input_limit(200_000)
    static_prompt = DOCUMENT_SUMMARY_PROMPT.format(document="")
    document = (
        "BAŞLANGIÇ — BİRİNCİ KISIM\n"
        + ("MADDE 123 — Genel düzenleyici hüküm ve yükümlülükler.\n" * 5_000)
        + "SON — GEÇİCİ MADDE VE EKLER"
    )

    with patch(
        "onyx.indexing.indexing_pipeline.GEN_AI_INPUT_TOKEN_SAFETY_MARGIN", 0.05
    ):
        budget = _contextual_prompt_content_budget(
            llm=llm,
            tokenizer=tokenizer,
            prompt_without_content=static_prompt,
        )
    trimmed = _trim_contextual_document(
        document_content=document,
        document_tokens=tokenizer.encode(document),
        token_budget=budget,
        tokenizer=tokenizer,
    )
    complete_prompt = DOCUMENT_SUMMARY_PROMPT.format(document=trimmed)

    assert trimmed.startswith("BAŞLANGIÇ — BİRİNCİ KISIM")
    assert trimmed.endswith("SON — GEÇİCİ MADDE VE EKLER")
    assert "tokens removed" in trimmed
    assert len(tokenizer.encode(complete_prompt)) <= 190_000 - MAX_CONTEXT_TOKENS


def test_within_budget_contextual_document_is_returned_without_decode() -> None:
    tokenizer = _CharacterTokenizer()
    document = "BİRİNCİ BÖLÜM\nMADDE 1 — Amaç ve kapsam."
    document_tokens = tokenizer.encode(document)

    result = _trim_contextual_document(
        document_content=document,
        document_tokens=document_tokens,
        token_budget=len(document_tokens),
        tokenizer=tokenizer,
    )

    assert result == document
    assert tokenizer.decode_calls == 0


def test_contextual_budget_fails_before_an_impossible_provider_request() -> None:
    tokenizer = _CharacterTokenizer()
    llm = _llm_with_input_limit(MAX_CONTEXT_TOKENS)

    with pytest.raises(ContextualEnrichmentError, match="input window is too small"):
        _contextual_prompt_content_budget(
            llm=llm,
            tokenizer=tokenizer,
            prompt_without_content=DOCUMENT_SUMMARY_PROMPT.format(document=""),
        )


def test_contextual_timeout_retries_are_bounded() -> None:
    llm = _llm_with_input_limit(4_096)
    cast(MagicMock, llm.invoke).side_effect = LLMTimeoutError("provider timeout")

    with (
        patch("onyx.indexing.indexing_pipeline.CONTEXTUAL_RAG_MAX_RETRIES", 2),
        patch("onyx.indexing.indexing_pipeline.time.sleep") as sleep,
        patch("onyx.indexing.indexing_pipeline.llm_generation_span") as span,
        pytest.raises(LLMTimeoutError, match="provider timeout"),
    ):
        _invoke_contextual_llm_with_retry(
            llm=llm,
            prompt=UserMessage(content="summarize"),
            flow=LLMFlow.CONTEXTUAL_RAG_DOC_SUMMARY,
        )

    assert cast(MagicMock, llm.invoke).call_count == 3
    assert sleep.call_count == 2
    assert span.call_count == 3
