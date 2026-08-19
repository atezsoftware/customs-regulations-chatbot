"""Shared safeguards for contextual enrichment of regulatory chunks."""

import datetime
from collections.abc import Sequence
from typing import Protocol, TypeVar

from onyx.configs.constants import RETURN_SEPARATOR
from onyx.natural_language_processing.utils import BaseTokenizer

_TRUNCATION_MARKER = "…"
MIN_CONTEXTUAL_RAG_RESERVED_TOKENS = 32


class RegulatoryValidityRow(Protocol):
    id: str
    position: int
    validity_start_date: datetime.date | datetime.datetime | None
    validity_end_date: datetime.date | datetime.datetime | None


_RegulatoryValidityRowT = TypeVar(
    "_RegulatoryValidityRowT", bound=RegulatoryValidityRow
)


def as_calendar_date(
    value: datetime.date | datetime.datetime | None,
) -> datetime.date | None:
    if isinstance(value, datetime.datetime):
        return value.date()
    return value


def context_reference_date(
    validity_start_date: datetime.date | datetime.datetime | None,
    validity_end_date: datetime.date | datetime.datetime | None,
    *,
    today: datetime.date | None = None,
) -> datetime.date:
    """Choose a date on which the target chunk itself is legally visible."""

    start = as_calendar_date(validity_start_date)
    end = as_calendar_date(validity_end_date)
    if start is not None:
        return start
    if end is not None:
        # Validity windows are half-open: an end boundary is the first day on
        # which the version no longer applies. Anchor an end-only historical
        # chunk on the preceding day so its context cannot see its successor.
        return end - datetime.timedelta(days=1) if end > datetime.date.min else end
    return today or datetime.date.today()


def validity_window_contains(
    validity_start_date: datetime.date | datetime.datetime | None,
    validity_end_date: datetime.date | datetime.datetime | None,
    reference_date: datetime.date,
) -> bool:
    """Return whether a regulatory version is visible at ``reference_date``."""

    start = as_calendar_date(validity_start_date)
    end = as_calendar_date(validity_end_date)
    return (start is None or start <= reference_date) and (
        end is None or end > reference_date
    )


def visible_regulatory_snapshot_for_target(
    rows: Sequence[_RegulatoryValidityRowT],
    target: _RegulatoryValidityRowT,
    *,
    today: datetime.date | None = None,
) -> list[_RegulatoryValidityRowT]:
    """Select one unambiguous legal version per position for a target row."""

    reference_date = context_reference_date(
        target.validity_start_date,
        target.validity_end_date,
        today=today,
    )
    candidates_by_position: dict[int, list[_RegulatoryValidityRowT]] = {}
    for candidate in rows:
        if validity_window_contains(
            candidate.validity_start_date,
            candidate.validity_end_date,
            reference_date,
        ):
            candidates_by_position.setdefault(candidate.position, []).append(candidate)

    visible_rows: list[_RegulatoryValidityRowT] = []
    for position in sorted(set(candidates_by_position) | {target.position}):
        candidates = candidates_by_position.get(position, [])
        if position == target.position:
            visible_rows.append(target)
        elif len(candidates) == 1:
            visible_rows.append(candidates[0])
        # Ambiguous overlapping versions are safer to omit than to mix.
    return sorted(
        visible_rows, key=lambda candidate: (candidate.position, candidate.id)
    )


def contextual_reserve_for_embedding_text(
    text: str,
    *,
    tokenizer: BaseTokenizer,
    embedding_token_limit: int,
    requested_reserve: int,
) -> int:
    """Use available embedding capacity without displacing the legal text."""

    if requested_reserve <= 0 or embedding_token_limit <= 0:
        return 0
    content_tokens = len(tokenizer.encode(text))
    remaining_capacity = max(0, embedding_token_limit - content_tokens)
    reserve = min(
        requested_reserve,
        remaining_capacity,
    )
    # Contextual enrichment invokes the LLM before the generated text is fit
    # into this reserve. Avoid paying for output that is too small to retain a
    # useful, readable source/provision relationship.
    return reserve if reserve >= MIN_CONTEXTUAL_RAG_RESERVED_TOKENS else 0


def fit_context_fields_to_embedding_budget(
    *,
    title_prefix: str,
    content: str,
    metadata_suffix: str,
    doc_summary: str,
    chunk_context: str,
    tokenizer: BaseTokenizer,
    embedding_token_limit: int,
) -> tuple[str, str]:
    """Format and trim generated context without changing the legal text.

    The generic enrichment serializer concatenates these fields directly around
    ``content``. Keeping the separators in the contextual fields therefore makes
    both the stored text and the embedding input unambiguous while preserving the
    existing cleanup inverse.
    """

    base_embedding_text = f"{title_prefix}{content}{metadata_suffix}"
    base_tokens = len(tokenizer.encode(base_embedding_text))
    if base_tokens >= embedding_token_limit:
        return "", ""

    clean_summary = doc_summary.strip()
    clean_context = chunk_context.strip()
    summary_token_budget = len(tokenizer.encode(clean_summary))
    context_token_budget = len(tokenizer.encode(clean_context))

    def readable_prefix(value: str, token_budget: int) -> str:
        if not value or token_budget <= 0:
            return ""
        value_tokens = tokenizer.encode(value)
        if len(value_tokens) <= token_budget:
            return value

        decoded_prefix = tokenizer.decode(value_tokens[:token_budget]).strip()
        if not decoded_prefix:
            return ""

        decoded_end = len(decoded_prefix)
        ended_at_boundary = (
            value.startswith(decoded_prefix)
            and decoded_end < len(value)
            and value[decoded_end].isspace()
        )
        if not ended_at_boundary:
            last_boundary = max(
                decoded_prefix.rfind(" "),
                decoded_prefix.rfind("\n"),
                decoded_prefix.rfind("\t"),
            )
            if last_boundary <= 0:
                return ""
            decoded_prefix = decoded_prefix[:last_boundary].rstrip()

        while decoded_prefix:
            candidate = f"{decoded_prefix} {_TRUNCATION_MARKER}"
            if len(tokenizer.encode(candidate)) <= token_budget:
                return candidate
            last_boundary = max(
                decoded_prefix.rfind(" "),
                decoded_prefix.rfind("\n"),
                decoded_prefix.rfind("\t"),
            )
            if last_boundary <= 0:
                return ""
            decoded_prefix = decoded_prefix[:last_boundary].rstrip()
        return ""

    while True:
        summary = readable_prefix(clean_summary, summary_token_budget)
        context = readable_prefix(clean_context, context_token_budget)
        formatted_summary = f"{summary}{RETURN_SEPARATOR}" if summary else ""
        formatted_context = f"{RETURN_SEPARATOR}{context}" if context else ""
        full_text = (
            f"{title_prefix}{formatted_summary}{content}"
            f"{formatted_context}{metadata_suffix}"
        )
        overflow = len(tokenizer.encode(full_text)) - embedding_token_limit
        if overflow <= 0:
            return formatted_summary, formatted_context

        # Chunk-specific context is usually more discriminative than a document
        # summary, so the summary yields budget first.
        if summary_token_budget > 0:
            summary_token_budget = max(0, summary_token_budget - overflow)
        elif context_token_budget > 0:
            context_token_budget = max(0, context_token_budget - overflow)
        else:
            return "", ""
