"""Bounded fail-open Decisions advisory for the guarded ASv3 workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import httpx

from onyx.configs import app_configs
from onyx.context.search.models import InferenceChunk
from onyx.tracing.flows import LLMFlow
from onyx.tracing.llm_utils import traced_llm_call
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MODEL = "gpt-6-luna"
_TIMEOUT_SECONDS = 1.5
_DECISIONS_URL = "https://api.openai.com/v1/decisions"
_MAX_BOUNDARY_CANDIDATES = 12
_MAX_PROMOTIONS = 4
_MIN_RELEVANCE_PROBABILITY = 0.75
_MAX_SNIPPET_CHARS = 600


def guarded_decisions_enabled() -> bool:
    return (
        app_configs.ASV3_GUARDED_DECISIONS_ENABLED
        and app_configs.OPENAI_DEFAULT_API_KEY is not None
    )


def _create_client() -> Any:
    return httpx.Client(timeout=_TIMEOUT_SECONDS)


def _answer_field(answer: object, field: str) -> object | None:
    if isinstance(answer, Mapping):
        return cast(Mapping[str, object], answer).get(field)
    return getattr(answer, field, None)


def _input(query: str, candidates: list[InferenceChunk]) -> str:
    records = [f"Question:\n{query.strip()}", "Candidate originals:"]
    for index, chunk in enumerate(candidates):
        records.append(
            f"[candidate_{index}]\n{chunk.content[:_MAX_SNIPPET_CHARS]}"
        )
    return "\n\n".join(records)


def _questions(candidates: list[InferenceChunk]) -> list[dict[str, str]]:
    return [
        {
            "type": "predicate",
            "name": f"candidate_{index}",
            "instructions": (
                f"Does candidate_{index} contain an original passage directly "
                "relevant to answering the question? Require an operative rule, "
                "condition, exception, procedure, or authoritative factual basis."
            ),
        }
        for index in range(len(candidates))
    ]


def _qualifying_boundary_indexes(response: object, candidate_count: int) -> list[int]:
    answers = (
        cast(Mapping[str, object], response).get("answers")
        if isinstance(response, Mapping)
        else None
    )
    if not isinstance(answers, list):
        return []
    probabilities: dict[int, float] = {}
    for answer in answers:
        if _answer_field(answer, "type") != "predicate":
            continue
        name = _answer_field(answer, "name")
        probability = _answer_field(answer, "probability")
        if not isinstance(name, str) or not name.startswith("candidate_"):
            continue
        if not isinstance(probability, (float, int)):
            continue
        try:
            index = int(name.removeprefix("candidate_"))
        except ValueError:
            continue
        if (
            0 <= index < candidate_count
            and probability >= _MIN_RELEVANCE_PROBABILITY
        ):
            probabilities[index] = float(probability)
    return [
        index
        for index, _ in sorted(
            probabilities.items(), key=lambda item: (-item[1], item[0])
        )[:_MAX_PROMOTIONS]
    ]


def promote_guarded_boundary_candidates(
    *,
    query: str,
    ordered_chunks: list[InferenceChunk],
    baseline_limit: int,
) -> list[InferenceChunk]:
    """Promote Decisions-qualified boundary candidates without removing baseline hits."""
    if not guarded_decisions_enabled() or baseline_limit < 1:
        return ordered_chunks
    baseline = ordered_chunks[:baseline_limit]
    boundary = ordered_chunks[
        baseline_limit : baseline_limit + _MAX_BOUNDARY_CANDIDATES
    ]
    if not boundary:
        return ordered_chunks
    try:
        with traced_llm_call(
            flow=LLMFlow.ASV3_GUARDED_DECISIONS,
            model=_MODEL,
            provider="openai",
            extra_config={"candidate_count": str(len(boundary))},
        ):
            with _create_client() as client:
                response = client.post(
                    _DECISIONS_URL,
                    headers={
                        "Authorization": f"Bearer {app_configs.OPENAI_DEFAULT_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": _MODEL,
                        "input": _input(query, boundary),
                        "questions": _questions(boundary),
                    },
                )
                response.raise_for_status()
                payload = response.json()
        indexes = _qualifying_boundary_indexes(payload, len(boundary))
    except Exception:
        logger.info("Guarded Decisions advisory unavailable; retaining deterministic ranking")
        return ordered_chunks
    if not indexes:
        return ordered_chunks
    promoted = [boundary[index] for index in indexes]
    promoted_identities = {(chunk.document_id, chunk.chunk_id) for chunk in promoted}
    return [
        *baseline,
        *promoted,
        *[
            chunk
            for chunk in ordered_chunks[baseline_limit:]
            if (chunk.document_id, chunk.chunk_id) not in promoted_identities
        ],
    ]
