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
from onyx.db.regulatory_chunks import (
    get_chunk_by_id,
    get_next_chunk_position,
    has_active_structural_descendants,
)
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.candidate_finder import find_candidates
from onyx.regulatory.amendments.draft_integrity import (
    reconcile_existing_heading_path,
    reject_unsupported_descendant_replacement,
    validate_explicit_replacements,
)
from onyx.regulatory.amendments.drafter import draft_combined_chunk, draft_new_chunk
from onyx.regulatory.amendments.matcher import confirm_match
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    AnalysisResult,
    DraftResult,
    MatchResult,
    ProposalDraft,
)
from onyx.regulatory.amendments.new_provision_policy import (
    explicitly_adds_top_level_provision,
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
        "validity_start_date": (
            chunk.validity_start_date.isoformat()
            if chunk.validity_start_date is not None
            else None
        ),
        "validity_end_date": (
            chunk.validity_end_date.isoformat()
            if chunk.validity_end_date is not None
            else None
        ),
        "status": chunk.status,
        "source": chunk.source,
        "supersedes_chunk_id": chunk.supersedes_chunk_id,
        "superseded_by_chunk_id": chunk.superseded_by_chunk_id,
        "created_at": chunk.created_at.isoformat(),
        "updated_at": chunk.updated_at.isoformat(),
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
    has_active_descendants: bool = False


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
    if match.old_chunk_id is None and not explicitly_adds_top_level_provision(
        instruction.instruction_text
    ):
        logger.warning(
            "Amendment matcher declined all candidates for an instruction that "
            "does not explicitly add a top-level provision; marking unmatched"
        )
        return None
    return match


def load_instruction_draft_context(
    db_session: Session,
    *,
    candidates: list[CandidateChunk],
    match: MatchResult,
) -> InstructionDraftContext | None:
    old_chunk: RegulatoryChunk | None = None
    if match.old_chunk_id:
        old_chunk = get_chunk_by_id(db_session, match.old_chunk_id)
        if old_chunk is None:
            logger.warning(
                "Matched amendment chunk %s no longer exists; marking instruction unmatched",
                match.old_chunk_id,
            )
            return None

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
        has_active_descendants=(
            has_active_structural_descendants(db_session, old_chunk)
            if old_chunk is not None
            else False
        ),
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
    return _build_proposal_draft(
        instruction_indices=[instruction_index],
        instructions=[instruction],
        matches=[context.match],
        context=context,
        draft=draft,
    )


def _build_proposal_draft(
    *,
    instruction_indices: list[int],
    instructions: list[AmendmentInstruction],
    matches: list[MatchResult],
    context: InstructionDraftContext,
    draft: DraftResult,
) -> ProposalDraft:
    validate_explicit_replacements(instructions, draft.new_chunk.text)
    canonical_chunk_type = context.old_chunk_snapshot.get("chunk_type")
    chunk_type = (
        canonical_chunk_type
        if context.old_chunk_snapshot
        else draft.new_chunk.chunk_type
    )
    merged_metadata = {
        **context.base_metadata,
        **draft.new_chunk.metadata_changes,
    }
    if context.old_chunk_snapshot:
        canonical_metadata = dict(context.old_chunk_snapshot.get("metadata") or {})
        for key in (
            "article_no",
            "paragraph_no",
            "clause_label",
            "subclause_label",
        ):
            if canonical_metadata.get(key) is None:
                merged_metadata.pop(key, None)
            else:
                merged_metadata[key] = canonical_metadata[key]
    heading_path = (
        list(context.base_heading_path)
        if context.old_chunk_snapshot
        else list(draft.new_chunk.heading_path or [])
    )
    if context.old_chunk_snapshot:
        heading_path = reconcile_existing_heading_path(
            heading_path,
            amended_text=draft.new_chunk.text,
            chunk_type=chunk_type,
            article_no=(
                str(merged_metadata["article_no"])
                if merged_metadata.get("article_no") is not None
                else None
            ),
            article_title=(
                str(merged_metadata["article_title"])
                if merged_metadata.get("article_title") is not None
                else None
            ),
            paragraph_no=(
                str(merged_metadata["paragraph_no"])
                if merged_metadata.get("paragraph_no") is not None
                else None
            ),
            clause_label=(
                str(merged_metadata["clause_label"])
                if merged_metadata.get("clause_label") is not None
                else None
            ),
            subclause_label=(
                str(merged_metadata["subclause_label"])
                if merged_metadata.get("subclause_label") is not None
                else None
            ),
        )
    merged_metadata["heading_path"] = list(heading_path)
    new_chunk_draft: dict[str, Any] = {
        "user_file_id": str(context.target_user_file_id),
        "position": context.target_position,
        "text": draft.new_chunk.text,
        "chunk_type": chunk_type,
        "heading_path": heading_path,
        "metadata": merged_metadata,
        "effective_start_date": draft.dates.effective_start_date,
        "effective_end_date": draft.dates.effective_end_date,
    }
    combined_match_rationale = (
        matches[0].rationale
        if len(matches) == 1
        else "\n".join(
            f"Instruction {instruction_index}: {match.rationale}"
            for instruction_index, match in zip(instruction_indices, matches)
        )
    )
    return ProposalDraft(
        instruction_index=instruction_indices[0],
        instruction_text=instructions[0].instruction_text,
        instruction_indices=instruction_indices,
        instruction_texts=[
            instruction.instruction_text for instruction in instructions
        ],
        old_chunk_id=matches[0].old_chunk_id,
        old_chunk_snapshot=context.old_chunk_snapshot,
        new_chunk_draft=new_chunk_draft,
        match_confidence=min(match.confidence for match in matches),
        match_rationale=combined_match_rationale,
        date_rationale=draft.dates.rationale,
    )


def draft_instruction_group_proposal(
    llm: LLM,
    *,
    instruction_indices: list[int],
    instructions: list[AmendmentInstruction],
    matches: list[MatchResult],
    reference_date: str | None,
    context: InstructionDraftContext,
) -> ProposalDraft:
    if not instructions or len(instruction_indices) != len(instructions):
        raise ValueError(
            "Grouped amendment drafting requires one index per instruction"
        )
    if len(matches) != len(instructions):
        raise ValueError(
            "Grouped amendment drafting requires one match per instruction"
        )
    reject_unsupported_descendant_replacement(
        instructions,
        has_active_descendants=context.has_active_descendants,
    )
    old_chunk_ids = {match.old_chunk_id for match in matches}
    if len(old_chunk_ids) != 1:
        raise ValueError("Grouped amendment instructions must share one target chunk")

    draft = draft_combined_chunk(
        llm,
        instructions=instructions,
        old_chunk=context.old_chunk_snapshot or None,
        sibling_reference=context.sibling_reference,
        reference_date=reference_date,
    )
    return _build_proposal_draft(
        instruction_indices=instruction_indices,
        instructions=instructions,
        matches=matches,
        context=context,
        draft=draft,
    )


def analyze_instruction(
    db_session: Session,
    *,
    llm: LLM,
    user_file_ids: list[UUID],
    instruction_index: int,
    instruction: AmendmentInstruction,
    reference_date: str | None,
    source_scope_cache: dict[str, list[UUID]] | None = None,
) -> ProposalDraft | None:
    candidates = find_candidates(
        db_session,
        user_file_ids=user_file_ids,
        instruction=instruction,
        source_scope_cache=source_scope_cache,
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
    if context is None:
        return None
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
    source_scope_cache: dict[str, list[UUID]] = {}

    for index, instruction in enumerate(segmentation.instructions):
        proposal = analyze_instruction(
            db_session,
            llm=llm,
            user_file_ids=user_file_ids,
            instruction_index=index,
            instruction=instruction,
            reference_date=segmentation.reference_date,
            source_scope_cache=source_scope_cache,
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
