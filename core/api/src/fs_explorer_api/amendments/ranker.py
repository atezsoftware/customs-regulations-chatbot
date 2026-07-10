"""Ranking helpers for merging amendment candidate-chunk search results.

Mirrors `fs_explorer_api.search.ranker`'s `RankedDocument`/`rank_documents`
split, but keyed for chunk-level amendment matching rather than
document-level chat retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateChunk:
    """Merged hybrid-search candidate for one amendment instruction."""

    chunk_id: str
    doc_id: str
    relative_path: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    semantic_score: float = 0.0
    text_trgm_score: float = 0.0
    heading_trgm_score: float = 0.0
    structured_match: bool = False

    @property
    def combined_score(self) -> float:
        # A structured (exact article/document number) match is the
        # strongest possible signal. Semantic similarity is the next most
        # reliable cross-check, then fuzzy text, then fuzzy heading —
        # heading_path itself is only a best-effort reconstruction from
        # document formatting (see RegulatoryChunker), so it's weighted
        # lowest among the fuzzy signals.
        return (
            (100.0 if self.structured_match else 0.0)
            + self.semantic_score * 10.0
            + self.text_trgm_score * 5.0
            + self.heading_trgm_score * 2.0
        )


def rank_candidates(
    candidates: list[CandidateChunk], *, limit: int
) -> list[CandidateChunk]:
    ordered = sorted(candidates, key=lambda c: -c.combined_score)
    return ordered[: max(limit, 1)]
