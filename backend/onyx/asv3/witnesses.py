"""Address bounded contiguous original passages without storing duplicate text."""

from __future__ import annotations

import hashlib
from typing import TypedDict


class WitnessSpan(TypedDict):
    witness_id: str
    start_char: int
    end_char: int


def original_witness_spans(citation: int, text: str) -> list[WitnessSpan]:
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    spans: list[WitnessSpan] = []
    start = 0
    while start < len(text):
        end = min(start + 800, len(text))
        if end < len(text):
            # Keep nearby sentence/line boundaries without losing intervening text.
            boundary = max(
                text.rfind(separator, start + 400, end) + len(separator)
                for separator in ("\n", ". ", "; ")
            )
            if boundary > start + 400:
                end = boundary
        spans.append(
            {
                "witness_id": f"w{citation}-{digest}-{start}-{end}",
                "start_char": start,
                "end_char": end,
            }
        )
        start = end
    return spans


def original_witness_text(citation: int, text: str, witness_id: str) -> str | None:
    """Accept only selectors in this exact original's host-generated catalogue."""
    for span in original_witness_spans(citation, text):
        if witness_id == span["witness_id"]:
            return text[span["start_char"] : span["end_char"]]
    return None
