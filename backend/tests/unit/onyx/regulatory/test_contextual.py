import datetime
from typing import cast

from onyx.configs.constants import RETURN_SEPARATOR
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.contextual import (
    context_reference_date,
    contextual_reserve_for_embedding_text,
    fit_context_fields_to_embedding_budget,
    validity_window_contains,
)


class _CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


_TOKENIZER = cast(BaseTokenizer, _CharacterTokenizer())


def test_context_reference_date_stays_inside_target_version() -> None:
    today = datetime.date(2026, 8, 1)

    assert context_reference_date(
        datetime.date(2020, 1, 1), datetime.date(2021, 1, 1), today=today
    ) == datetime.date(2020, 1, 1)
    assert context_reference_date(
        None, datetime.date(2021, 1, 1), today=today
    ) == datetime.date(2020, 12, 31)
    assert context_reference_date(None, None, today=today) == today


def test_validity_window_has_exclusive_end_boundary() -> None:
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2021, 1, 1)

    assert validity_window_contains(start, end, start)
    assert not validity_window_contains(start, end, end)
    assert not validity_window_contains(start, end, datetime.date(2021, 1, 2))


def test_context_reserve_uses_remaining_capacity_without_displacing_text() -> None:
    assert (
        contextual_reserve_for_embedding_text(
            "x" * 312,
            tokenizer=_TOKENIZER,
            embedding_token_limit=512,
            requested_reserve=200,
        )
        == 200
    )
    assert (
        contextual_reserve_for_embedding_text(
            "x" * 313,
            tokenizer=_TOKENIZER,
            embedding_token_limit=512,
            requested_reserve=200,
        )
        == 199
    )
    assert (
        contextual_reserve_for_embedding_text(
            "x" * 424,
            tokenizer=_TOKENIZER,
            embedding_token_limit=512,
            requested_reserve=200,
        )
        == 88
    )
    assert (
        contextual_reserve_for_embedding_text(
            "x" * 480,
            tokenizer=_TOKENIZER,
            embedding_token_limit=512,
            requested_reserve=200,
        )
        == 32
    )
    assert (
        contextual_reserve_for_embedding_text(
            "x" * 481,
            tokenizer=_TOKENIZER,
            embedding_token_limit=512,
            requested_reserve=200,
        )
        == 0
    )
    assert (
        contextual_reserve_for_embedding_text(
            "x" * 512,
            tokenizer=_TOKENIZER,
            embedding_token_limit=512,
            requested_reserve=200,
        )
        == 0
    )


def test_generated_context_has_explicit_legal_text_boundaries() -> None:
    summary, context = fit_context_fields_to_embedding_budget(
        title_prefix="",
        content="LEGAL",
        metadata_suffix="",
        doc_summary="Document summary",
        chunk_context="Chunk context",
        tokenizer=_TOKENIZER,
        embedding_token_limit=64,
    )

    assert summary == f"Document summary{RETURN_SEPARATOR}"
    assert context == f"{RETURN_SEPARATOR}Chunk context"
    assert f"{summary}LEGAL{context}" == (
        f"Document summary{RETURN_SEPARATOR}LEGAL{RETURN_SEPARATOR}Chunk context"
    )


def test_generated_context_is_trimmed_readably_before_legal_text() -> None:
    summary, context = fit_context_fields_to_embedding_budget(
        title_prefix="",
        content="LEGAL",
        metadata_suffix="",
        doc_summary="alpha beta gamma",
        chunk_context="delta epsilon",
        tokenizer=_TOKENIZER,
        embedding_token_limit=32,
    )

    assert summary == f"alpha …{RETURN_SEPARATOR}"
    assert context == f"{RETURN_SEPARATOR}delta epsilon"
    assert len(f"{summary}LEGAL{context}") <= 32
    assert "…LEGAL" not in f"{summary}LEGAL{context}"


def test_partial_context_budget_preserves_the_complete_legal_text() -> None:
    legal_text = "L" * 424
    summary, context = fit_context_fields_to_embedding_budget(
        title_prefix="",
        content=legal_text,
        metadata_suffix="",
        doc_summary="document summary " * 20,
        chunk_context="specific chunk context " * 20,
        tokenizer=_TOKENIZER,
        embedding_token_limit=512,
    )

    embedded_text = f"{summary}{legal_text}{context}"
    assert summary or context
    assert len(_TOKENIZER.encode(embedded_text)) <= 512
    assert embedded_text.removeprefix(summary).removesuffix(context) == legal_text
