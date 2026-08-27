"""Orchestrates the amendment analysis pipeline: segment -> find candidates ->
confirm match -> draft new chunk, for every instruction in a pasted text.

Pure orchestration — never writes to `regulatory_chunk`. The caller persists
the resulting `AnalysisResult` as an `amendment_batch` + `amendment_proposal`
rows, and nothing lands in `regulatory_chunk` until an admin approves a
specific proposal (see onyx/db/regulatory_amendments.py).
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk
from onyx.db.regulatory_chunks import get_chunk_by_id, get_next_chunk_position
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.candidate_finder import find_candidates
from onyx.regulatory.amendments.drafter import draft_new_chunk
from onyx.regulatory.amendments.matcher import confirm_match
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    AnalysisResult,
    MatchResult,
    ProposalDraft,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.segmenter import segment_amendment_text
from onyx.utils.logger import setup_logger

logger = setup_logger()


def _chunk_to_review_dict(chunk: RegulatoryChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "user_file_id": str(chunk.user_file_id),
        "position": chunk.position,
        "text": chunk.text,
        "chunk_type": chunk.chunk_type,
        "heading_path": list(chunk.heading_path),
        "metadata": dict(chunk.chunk_metadata),
    }


@dataclass(frozen=True)
class InstructionDraftContext:
    match: MatchResult
    old_chunk_snapshot: dict[str, Any]
    target_user_file_id: UUID
    target_position: int
    sibling_reference: dict[str, Any] | None
    base_metadata: dict[str, Any]
    base_heading_path: list[str]


def confirm_instruction_match(
    llm: LLM,
    *,
    instruction: AmendmentInstruction,
    candidates: list[CandidateChunk],
) -> MatchResult | None:
    match = confirm_match(llm, instruction=instruction, candidates=candidates)
    candidate_ids = {candidate.chunk_id for candidate in candidates}
    if match.old_chunk_id is not None and match.old_chunk_id not in candidate_ids:
        logger.warning(
            "Amendment matcher returned candidate id outside the supplied set: %s",
            match.old_chunk_id,
        )
        return None
    if match.old_chunk_id is None and not instruction.is_new_provision:
        logger.warning(
            "Amendment matcher found no existing candidate for a non-new provision"
        )
        return None
    return match


def load_instruction_draft_context(
    db_session: Session,
    *,
    candidates: list[CandidateChunk],
    match: MatchResult,
) -> InstructionDraftContext:
    old_chunk: RegulatoryChunk | None = None
    if match.old_chunk_id:
        old_chunk = get_chunk_by_id(db_session, match.old_chunk_id)
        if old_chunk is None:
            match = MatchResult(
                old_chunk_id=None,
                confidence=match.confidence,
                rationale=(
                    f"{match.rationale} "
                    "(named chunk id no longer exists, treated as a new provision)"
                ),
            )

    sibling_reference: dict[str, Any] | None = None
    if old_chunk is not None:
        target_user_file_id = old_chunk.user_file_id
        target_position = old_chunk.position
    else:
        best_candidate = candidates[0]
        target_user_file_id = UUID(best_candidate.user_file_id)
        sibling_reference = {
            "text": best_candidate.text,
            "metadata": best_candidate.metadata,
            "heading_path": best_candidate.metadata.get("heading_path"),
        }
        target_position = get_next_chunk_position(db_session, target_user_file_id)

    return InstructionDraftContext(
        match=match,
        old_chunk_snapshot=_chunk_to_review_dict(old_chunk) if old_chunk else {},
        target_user_file_id=target_user_file_id,
        target_position=target_position,
        sibling_reference=sibling_reference,
        base_metadata=dict(old_chunk.chunk_metadata) if old_chunk else {},
        base_heading_path=list(old_chunk.heading_path) if old_chunk else [],
    )


def draft_instruction_proposal(
    llm: LLM,
    *,
    instruction_index: int,
    instruction: AmendmentInstruction,
    reference_date: str | None,
    context: InstructionDraftContext,
) -> ProposalDraft:
    draft = draft_new_chunk(
        llm,
        instruction=instruction,
        old_chunk=context.old_chunk_snapshot or None,
        sibling_reference=context.sibling_reference,
        reference_date=reference_date,
    )
    merged_metadata = {
        **context.base_metadata,
        **draft.new_chunk.metadata_changes,
    }
    heading_path = draft.new_chunk.heading_path or context.base_heading_path
    new_chunk_draft: dict[str, Any] = {
        "user_file_id": str(context.target_user_file_id),
        "position": context.target_position,
        "text": draft.new_chunk.text,
        "chunk_type": draft.new_chunk.chunk_type,
        "heading_path": heading_path,
        "metadata": merged_metadata,
        "effective_start_date": draft.dates.effective_start_date,
        "effective_end_date": draft.dates.effective_end_date,
    }
    return ProposalDraft(
        instruction_index=instruction_index,
        instruction_text=instruction.instruction_text,
        old_chunk_id=context.match.old_chunk_id,
        old_chunk_snapshot=context.old_chunk_snapshot,
        new_chunk_draft=new_chunk_draft,
        match_confidence=context.match.confidence,
        match_rationale=context.match.rationale,
        date_rationale=draft.dates.rationale,
    )


def analyze_instruction(
    db_session: Session,
    *,
    llm: LLM,
    user_file_ids: list[UUID],
    instruction_index: int,
    instruction: AmendmentInstruction,
    reference_date: str | None,
) -> ProposalDraft | None:
    candidates = find_candidates(
        db_session, user_file_ids=user_file_ids, instruction=instruction
    )
    if not candidates:
        return None

    match = confirm_instruction_match(
        llm, instruction=instruction, candidates=candidates
    )
    if match is None:
        return None
    context = load_instruction_draft_context(
        db_session, candidates=candidates, match=match
    )
    return draft_instruction_proposal(
        llm,
        instruction_index=instruction_index,
        instruction=instruction,
        reference_date=reference_date,
        context=context,
    )


def analyze_amendment(
    db_session: Session,
    *,
    llm: LLM,
    user_file_ids: list[UUID],
    raw_text: str,
) -> AnalysisResult:
    segmentation = segment_amendment_text(llm, raw_text)

    proposals: list[ProposalDraft] = []
    unmatched: list[AmendmentInstruction] = []

    for index, instruction in enumerate(segmentation.instructions):
        proposal = analyze_instruction(
            db_session,
            llm=llm,
            user_file_ids=user_file_ids,
            instruction_index=index,
            instruction=instruction,
            reference_date=segmentation.reference_date,
        )
        if proposal is None:
            unmatched.append(instruction)
        else:
            proposals.append(proposal)

    return AnalysisResult(
        reference_date=segmentation.reference_date,
        proposals=proposals,
        unmatched_instructions=unmatched,
    )
