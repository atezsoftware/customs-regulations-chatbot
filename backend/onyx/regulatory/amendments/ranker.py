"""Ranking helpers for merging amendment candidate-chunk search results."""

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class CandidateChunk:
    """Merged hybrid-search candidate for one amendment instruction."""

    chunk_id: str
    user_file_id: str
    text: str
    source_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    text_trgm_score: float = 0.0
    heading_trgm_score: float = 0.0
    structured_match: bool = False
    source_score: float = 0.0

    @property
    def combined_score(self) -> float:
        # A structured (exact article number) match is the strongest possible
        # signal — the amendment text names the article directly. Fuzzy text
        # similarity is the next most reliable cross-check, then fuzzy
        # heading — heading_path is only a best-effort reconstruction from
        # document formatting (see RegulatoryChunker), so it's weighted
        # lowest.
        return (
            (100.0 if self.structured_match else 0.0)
            + self.text_trgm_score * 10.0
            + self.heading_trgm_score * 2.0
            + self.source_score * 8.0
        )


def with_score(
    candidate: CandidateChunk, field_name: str, score: float
) -> CandidateChunk:
    current = getattr(candidate, field_name)
    if score <= current:
        return candidate
    return replace(candidate, **{field_name: score})


def rank_candidates(
    candidates: list[CandidateChunk], *, limit: int
) -> list[CandidateChunk]:
    ordered = sorted(candidates, key=lambda c: -c.combined_score)
    return ordered[: max(limit, 1)]
