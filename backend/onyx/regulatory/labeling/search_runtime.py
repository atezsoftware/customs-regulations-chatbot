"""Fail-open read-only integration between label snapshots and normal retrieval."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError

from onyx.context.search.models import InferenceChunk
from onyx.db.regulatory_label_search import (
    label_read_session,
    load_document_set_search_snapshot,
    load_label_overlay,
    load_search_snapshot,
)
from onyx.regulatory.labeling.search_hints import (
    explicit_subject_hint,
    query_subject_hint,
)
from onyx.regulatory.labeling.search_models import (
    LabelFusionScore,
    LabelSearchHint,
    LabelSearchMode,
    LabelSearchSnapshot,
    SearchLabelEvidence,
)
from onyx.regulatory.labeling.search_ranking import (
    fuse_label_candidate_scores,
    merge_label_candidates,
    validate_search_hint,
)
from onyx.tracing.answer_graph import graph_step
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

logger = setup_logger()


class LabelSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: list[InferenceChunk]
    hint: LabelSearchHint = Field(default_factory=LabelSearchHint)
    evidence_by_chunk: dict[str, tuple[SearchLabelEvidence, ...]] = Field(
        default_factory=dict
    )
    fusion_scores: tuple[LabelFusionScore, ...] = ()
    status: Literal[
        "disabled",
        "snapshot_unavailable",
        "ineligible",
        "snapshot_mismatch",
        "no_hint",
        "failed",
        "no_verified_match",
        "scored",
    ] = "disabled"


@dataclass(frozen=True)
class LabelDiscoveryAcquisition:
    chunks: tuple[InferenceChunk, ...] = ()
    error: Exception | None = None

    @classmethod
    def capture(
        cls,
        discover: Callable[[LabelSearchHint], list[InferenceChunk]],
        hint: LabelSearchHint,
    ) -> "LabelDiscoveryAcquisition":
        try:
            return cls(chunks=tuple(discover(hint)))
        except Exception as error:
            return cls(error=error)

    def result(self) -> list[InferenceChunk]:
        if self.error is not None:
            raise self.error
        return list(self.chunks)


def prepare_label_search_hint(
    snapshot: LabelSearchSnapshot, raw_hint: object, query: str
) -> LabelSearchHint:
    hint = validate_search_hint(raw_hint, snapshot)
    explicit = explicit_subject_hint(snapshot, query)
    vocabulary = query_subject_hint(snapshot, query)
    return LabelSearchHint(
        label_ids=tuple(
            dict.fromkeys((*explicit.label_ids, *vocabulary.label_ids, *hint.label_ids))
        )[:12]
    )


def search_snapshot_for_run_ids(
    run_ids: tuple[UUID, ...],
    *,
    mode: LabelSearchMode = "hybrid",
    document_set_id: int | None = None,
) -> LabelSearchSnapshot | None:
    if not run_ids or len(run_ids) > 32 or mode == "off":
        return None
    try:
        with label_read_session() as session:
            return load_search_snapshot(
                session,
                tenant_id=get_current_tenant_id(),
                run_ids=run_ids,
                mode=mode,
                document_set_id=document_set_id,
            )
    except SQLAlchemyError:
        logger.warning(
            "Label search snapshot unavailable; preserving baseline retrieval",
            exc_info=True,
        )
        return None


def search_snapshot_for_document_set(
    document_set_id: int,
) -> LabelSearchSnapshot | None:
    try:
        with label_read_session() as session:
            return load_document_set_search_snapshot(
                session,
                tenant_id=get_current_tenant_id(),
                document_set_id=document_set_id,
            )
    except SQLAlchemyError:
        logger.warning(
            "Native label snapshot unavailable; preserving baseline retrieval",
            exc_info=True,
        )
        return None


def search_with_labels(
    baseline: Sequence[InferenceChunk],
    *,
    snapshot: LabelSearchSnapshot | None,
    raw_hint: object,
    as_of_date: date,
    limit: int,
    retrieve: Callable[[tuple[str, ...]], list[InferenceChunk]],
    query: str = "",
    discover: Callable[[LabelSearchHint], list[InferenceChunk]] | None = None,
    discovery_acquisition: LabelDiscoveryAcquisition | None = None,
) -> LabelSearchResult:
    fallback = LabelSearchResult(candidates=list(baseline))
    if snapshot is None or snapshot.tenant_id != get_current_tenant_id():
        return fallback.model_copy(update={"status": "snapshot_mismatch"})
    hint = prepare_label_search_hint(snapshot, raw_hint, query)
    if not hint.label_ids:
        return fallback.model_copy(update={"status": "no_hint"})
    try:
        proposals: list[InferenceChunk] | None = None
        discovered: list[InferenceChunk] = []
        candidate_labels = (
            hint.candidate_label_ids(snapshot.taxonomy)
            if snapshot.mode == "hybrid"
            else ()
        )
        if discover is not None and candidate_labels:
            baseline_ids = {chunk.regulatory_chunk_id for chunk in baseline}
            seen = set(baseline_ids)
            proposals = []
            discovered = (
                discovery_acquisition.result()
                if discovery_acquisition is not None
                else discover(hint)
            )
            for chunk in discovered:
                identifier = chunk.regulatory_chunk_id
                if identifier and identifier not in seen:
                    proposals.append(chunk)
                    seen.add(identifier)
                if len(proposals) >= 32:
                    break
        logger.info(
            "Label discovery proposed=%d",
            len(proposals) if proposals is not None else 0,
        )
        with label_read_session() as session:
            overlay = load_label_overlay(
                session,
                snapshot=snapshot,
                chunk_ids=tuple(
                    chunk.regulatory_chunk_id
                    for chunk in baseline[:limit]
                    if chunk.regulatory_chunk_id
                ),
                candidate_label_ids=candidate_labels,
                candidate_chunk_ids=(
                    tuple(
                        chunk.regulatory_chunk_id
                        for chunk in proposals
                        if chunk.regulatory_chunk_id
                    )
                    if proposals is not None
                    else None
                ),
                as_of_date=as_of_date,
            )
        extra = (
            proposals
            if proposals is not None
            else (retrieve(overlay.candidate_ids) if overlay.candidate_ids else [])
        )
    except Exception:
        # This optional lane cannot change the success/failure of the baseline.
        logger.warning(
            "Label search lane failed; preserving baseline retrieval", exc_info=True
        )
        return fallback.model_copy(update={"status": "failed"})
    extra = [
        chunk
        for chunk in extra
        if chunk.regulatory_chunk_id in overlay.candidate_ids
        and chunk.regulatory_chunk_id in overlay.evidence_by_chunk
        and overlay.source_texts.get(chunk.regulatory_chunk_id, "")
        and overlay.source_texts[chunk.regulatory_chunk_id] in chunk.content
    ]
    candidates = merge_label_candidates(baseline, extra, limit=limit)
    evidence = {
        chunk.regulatory_chunk_id: overlay.evidence_by_chunk[chunk.regulatory_chunk_id]
        for chunk in candidates
        if chunk.regulatory_chunk_id in overlay.evidence_by_chunk
        and overlay.source_texts.get(chunk.regulatory_chunk_id, "")
        and overlay.source_texts[chunk.regulatory_chunk_id] in chunk.content
    }
    if not extra and not any(
        entry.label_id in hint.label_ids
        for entries in evidence.values()
        for entry in entries
    ):
        logger.info("Label search no verified match; preserving baseline retrieval")
        return fallback.model_copy(update={"status": "no_verified_match"})
    requested = set(hint.label_ids)
    label_ranked_ids = list(
        dict.fromkeys(
            chunk.regulatory_chunk_id
            for chunk in [*discovered, *baseline]
            if chunk.regulatory_chunk_id
            and any(
                entry.label_id in requested
                for entry in evidence.get(chunk.regulatory_chunk_id, ())
            )
        )
    )
    with graph_step(
        "search.label_score_fusion",
        {
            "mode": snapshot.mode,
            "hint_label_ids": hint.label_ids,
            "baseline_ids": [chunk.regulatory_chunk_id for chunk in baseline[:limit]],
            "validated_label_lane_ids": label_ranked_ids,
            "validated_extra_ids": [chunk.regulatory_chunk_id for chunk in extra],
        },
        summary=(f"equal RRF; {len(label_ranked_ids)} verified; {len(extra)} added"),
    ) as fusion_step:
        ranked, fusion_scores = fuse_label_candidate_scores(
            baseline,
            extra,
            label_ranked_ids=label_ranked_ids,
            limit=limit,
        )
        fusion_step.output_value = {
            "ranked_ids": [chunk.regulatory_chunk_id for chunk in ranked],
            "scores": [score.model_dump() for score in fusion_scores],
        }
    logger.info(
        "Label search mode=%s taxonomy=%s baseline=%d added=%d matched=%d",
        snapshot.mode,
        snapshot.taxonomy.version_hash[:12],
        len(baseline),
        len(extra),
        len(evidence),
    )
    return LabelSearchResult(
        candidates=ranked,
        evidence_by_chunk=evidence,
        hint=hint,
        fusion_scores=fusion_scores,
        status="scored",
    )
