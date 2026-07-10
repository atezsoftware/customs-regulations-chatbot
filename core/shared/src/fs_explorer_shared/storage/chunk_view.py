"""Human/LLM-facing chunk view helpers.

Stored chunk rows carry `start_char`/`end_char` — raw parse offsets used only
by the indexer's own id derivation (`make_chunk_id`) — which are meaningless
to a human reviewing an amendment proposal or to an LLM drafting one. This
module is the single place that trims them, so the amendment review API
(what the frontend renders) and the LLM prompt construction (what the model
sees) can't independently drift on what "the chunk" looks like.
"""

from __future__ import annotations

from typing import Any

_REVIEW_DROP_FIELDS = ("start_char", "end_char")


def chunk_to_review_dict(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a stored chunk dict trimmed for review/editing.

    Only drops internal bookkeeping fields — never touches what's persisted.
    """
    return {
        key: value for key, value in chunk.items() if key not in _REVIEW_DROP_FIELDS
    }
