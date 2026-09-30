"""Immutable, query-time label metadata; never part of embedding inputs."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from onyx.regulatory.labeling.provider import TaxonomyDefinition

LabelSearchMode = Literal["off", "rerank", "hybrid"]
LabelFacet = Literal["subject", "effect", "annex", "sector", "untyped"]


class LabelSearchSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    run_ids: tuple[UUID, ...] = Field(min_length=1, max_length=32)
    taxonomy: TaxonomyDefinition
    mode: LabelSearchMode


class LabelSearchHint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label_ids: tuple[str, ...] = Field(default=(), max_length=12)

    def candidate_label_ids(self, taxonomy: TaxonomyDefinition) -> tuple[str, ...]:
        from onyx.regulatory.labeling.search_ranking import resolve_label_facets

        facets = resolve_label_facets(taxonomy)
        return tuple(
            label
            for label in self.label_ids
            if facets.get(label) in ("subject", "untyped")
        )


class SearchLabelEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    source_chunk_id: str
    label_id: str
    evidence_quote: str = Field(min_length=1, max_length=1024)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: UUID


class LabelSearchOverlay(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_by_chunk: dict[str, tuple[SearchLabelEvidence, ...]] = Field(
        default_factory=dict
    )
    candidate_ids: tuple[str, ...] = ()
    source_texts: dict[str, str] = Field(default_factory=dict)


class LabelFusionScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    baseline_rank: int | None
    label_rank: int | None
    baseline_score: float
    label_score: float
    combined_score: float


class LabelQueryHint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=240)
    label_ids: list[str] = Field(default_factory=list, max_length=12)
