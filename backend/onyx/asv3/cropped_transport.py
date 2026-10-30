"""Lossless metadata transport and pre-provider input limits for Cropped."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast

from pydantic import JsonValue

from onyx.asv3.cheap_metadata_transport import serialize_cheap_payload
from onyx.asv3.cropped_catalogue_aliases import (
    CATALOGUE_ALIAS_HEADER,
    encode_cropped_catalogue_aliases,
)
from onyx.asv3.cropped_handoff_transport import (
    canonical_cropped_handoff_payload,
    encode_cropped_handoff_transport,
    expand_cropped_handoff_transport,
)
from onyx.asv3.models import HarnessView, RunContext, RunStopped

# Leave room for provider message framing and native schema normalization.
CROPPED_INPUT_TOKEN_LIMIT = 240_000
CROPPED_PROVIDER_TIER_BOUNDARY = 272_000


class CroppedInputOverflow(RunStopped):
    def __init__(self, counted_tokens: int) -> None:
        self.counted_tokens = counted_tokens
        self.limit = CROPPED_INPUT_TOKEN_LIMIT
        super().__init__(
            f"Cropped input packet counts {counted_tokens} tokens; "
            f"the pre-provider limit is {self.limit}. Required originals were retained."
        )


def serialize_cropped_payload(
    payload: dict[str, JsonValue],
    token_counter: Callable[[str], int],
    *,
    validate_roots_once: bool = False,
) -> str:
    """Reuse only the reversible same-message codec, without Cheap policies."""
    original = serialize_cheap_payload(
        payload,
        token_counter,
        extra_fields=(
            "research_tasks",
            "repair_research_observations",
            "cropped_completion",
            "session_research",
        ),
        validate_roots_once=validate_roots_once,
    )
    canonical = canonical_cropped_handoff_payload(payload)
    alternatives = [encode_cropped_handoff_transport(payload)]
    aliased = encode_cropped_catalogue_aliases(canonical)
    if CATALOGUE_ALIAS_HEADER in aliased:
        alternatives.append(encode_cropped_handoff_transport(aliased))
    selected = original
    selected_tokens = token_counter(original)
    for encoded in alternatives:
        if not {
            "cropped_canonical_fields_transport",
            CATALOGUE_ALIAS_HEADER,
        }.intersection(encoded):
            continue
        candidate = serialize_cheap_payload(
            encoded,
            token_counter,
            extra_fields=(
                "research_tasks",
                "repair_research_observations",
                "cropped_completion",
                "session_research",
            ),
            validate_roots_once=validate_roots_once,
        )
        restored = expand_cropped_handoff_transport(json.loads(candidate))
        if json.dumps(
            restored, ensure_ascii=False, sort_keys=True, allow_nan=False
        ) != json.dumps(canonical, ensure_ascii=False, sort_keys=True, allow_nan=False):
            raise ValueError("Cropped handoff transport changed its actual payload")
        candidate_tokens = token_counter(candidate)
        if candidate_tokens < selected_tokens:
            selected, selected_tokens = candidate, candidate_tokens
    return selected


def enforce_cropped_input_budget(
    input_tokens: int, *, response_format_tokens: int = 0
) -> dict[str, int]:
    """Count the whole request before a physical call, including output schema."""
    for count in (input_tokens, response_format_tokens):
        if type(count) is not int or count < 0:
            raise ValueError("Cropped input token counts must be nonnegative integers")
    counted = input_tokens + response_format_tokens
    if counted > CROPPED_INPUT_TOKEN_LIMIT:
        raise CroppedInputOverflow(counted)
    return {
        "counted_input_tokens": counted,
        "response_format_tokens": response_format_tokens,
        "pre_provider_limit": CROPPED_INPUT_TOKEN_LIMIT,
        "provider_tier_boundary": CROPPED_PROVIDER_TIER_BOUNDARY,
        "headroom_tokens": CROPPED_PROVIDER_TIER_BOUNDARY - counted,
    }


def prepare_cropped_writer_view(
    view: HarnessView,
    context: RunContext,
    historical_originals: list[dict[str, JsonValue]],
) -> HarnessView:
    """Deliver the complete retained source union afresh, without native history."""
    from onyx.asv3.citation_numbers import extract_citation_numbers
    from onyx.asv3.cropped_task_packets import restore_cropped_topic_context
    from onyx.asv3.evidence import EvidenceLedger
    from onyx.asv3.research_state import ResearchState

    view = restore_cropped_topic_context(view, context)
    ledger = context.services.get("evidence")
    if not isinstance(ledger, EvidenceLedger):
        raise RunStopped("Cropped final writing requires its original evidence ledger")
    required = set(view.required_evidence_numbers)
    if context.depth == 0:
        # Completed delegates return citation IDs; their bodies live in the shared ledger.
        from onyx.asv3.cropped_session_reuse import active_cropped_session_numbers

        required.update(active_cropped_session_numbers(ledger, context))
    required.update(extract_citation_numbers(view.draft_to_repair or ""))
    originals = [*historical_originals, *view.original_evidence]
    for row in originals:
        number, start, text = (
            row.get("citation"),
            row.get("start_char", 0),
            row.get("text"),
        )
        if (
            type(number) is not int
            or type(start) is not int
            or not isinstance(text, str)
        ):
            raise RunStopped("Cropped final writing received an invalid original range")
        item = ledger.get(number)
        if (
            item is None
            or start < 0
            or item.text[start : start + len(text)] != text
            or any(
                row.get(key) != getattr(item, key)
                for key in ("source_id", "chunk_id", "text_hash")
            )
        ):
            raise RunStopped("Cropped final writing original identity or text changed")
        required.add(number)
    if any(
        type(number) is not int or ledger.get(number) is None for number in required
    ):
        raise RunStopped("Cropped final writing lacks a required recorded original")
    full = cast(
        list[dict[str, JsonValue]],
        json.loads(
            ledger.serialize_records(
                sorted(required),
                required=sorted(required),
                max_chars=None,
                include_witness_spans=any(
                    row.get("witness_spans") for row in originals
                ),
            )
        ),
    )
    research_state = context.services.get("research_state")
    full_research_state = (
        research_state.export()
        if isinstance(research_state, ResearchState)
        else view.research_state
    )
    full_research_state = {
        **full_research_state,
        **{
            key: value
            for key, value in view.research_state.items()
            if key not in full_research_state
        },
    }
    if isinstance(research_state, ResearchState):
        for field in ("needs", "findings", "source_conditions"):
            full_research_state[field + "_omitted"] = 0
    return view.model_copy(
        deep=True,
        update={
            "turns": [],
            "original_evidence": full,
            "required_evidence_numbers": sorted(required),
            "research_state": full_research_state,
        },
    )
