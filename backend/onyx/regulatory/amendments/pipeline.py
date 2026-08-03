"""Orchestrates the amendment analysis pipeline: segment -> find candidates ->
confirm match -> draft new chunk, for every instruction in a pasted text.

Pure orchestration — never writes to `regulatory_chunk`. The caller persists
the resulting `AnalysisResult` as an `amendment_batch` + `amendment_proposal`
rows, and nothing lands in `regulatory_chunk` until an admin approves a
specific proposal (see onyx/db/regulatory_amendments.py).
"""

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk
from onyx.db.regulatory_chunks import get_chunk_by_id, get_chunks_for_file
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
        candidates = find_candidates(
            db_session, user_file_ids=user_file_ids, instruction=instruction
        )
        if not candidates:
            # Nothing in the directory resembles this instruction at all — no
            # chunk to attach a new one to, so surface it as unmatched rather
            # than guessing.
            unmatched.append(instruction)
            continue

        match = confirm_match(llm, instruction=instruction, candidates=candidates)

        old_chunk: RegulatoryChunk | None = None
        if match.old_chunk_id:
            old_chunk = get_chunk_by_id(db_session, match.old_chunk_id)
            if old_chunk is None:
                # The LLM named a chunk id outside the candidate set — don't
                # trust it blindly, fall back to "no match".
                match = MatchResult(
                    old_chunk_id=None,
                    confidence=match.confidence,
                    rationale=(
                        f"{match.rationale} "
                        "(named chunk id not found among candidates, treated as unmatched)"
                    ),
                )

        sibling_reference: dict[str, Any] | None = None
        if old_chunk is not None:
            target_user_file_id = old_chunk.user_file_id
            target_position = old_chunk.position
        else:
            # New provision, no existing chunk to replace — attach it to the
            # same file as the strongest candidate (virtually always the one
            # this amendment is targeting) and append it after that file's
            # existing chunks. Give the LLM that candidate's own metadata as
            # a reference so it can build a heading_path/article_no
            # consistent with the rest of the file instead of leaving them
            # empty — citations and search both depend on these being
            # populated, for amendment-created chunks same as indexed ones.
            best_candidate = candidates[0]
            target_user_file_id = UUID(best_candidate.user_file_id)
            sibling_reference = {
                "text": best_candidate.text,
                "metadata": best_candidate.metadata,
                "heading_path": best_candidate.metadata.get("heading_path"),
            }
            siblings = get_chunks_for_file(db_session, target_user_file_id)
            target_position = max((s.position for s in siblings), default=-1) + 1

        draft = draft_new_chunk(
            llm,
            instruction=instruction,
            old_chunk=_chunk_to_review_dict(old_chunk) if old_chunk else None,
            sibling_reference=sibling_reference,
            reference_date=segmentation.reference_date,
        )

        # Merge, don't replace: the LLM only returns the metadata fields it
        # actually changed (`metadata_changes`), so fields it wasn't asked to
        # touch survive by construction instead of depending on the model
        # faithfully reproducing every unrelated key.
        base_metadata = dict(old_chunk.chunk_metadata) if old_chunk else {}
        merged_metadata = {**base_metadata, **draft.new_chunk.metadata_changes}

        base_heading_path = list(old_chunk.heading_path) if old_chunk else []
        heading_path = draft.new_chunk.heading_path or base_heading_path

        new_chunk_draft: dict[str, Any] = {
            "user_file_id": str(target_user_file_id),
            "position": target_position,
            "text": draft.new_chunk.text,
            "chunk_type": draft.new_chunk.chunk_type,
            "heading_path": heading_path,
            "metadata": merged_metadata,
            "effective_start_date": draft.dates.effective_start_date,
            "effective_end_date": draft.dates.effective_end_date,
        }

        proposals.append(
            ProposalDraft(
                instruction_index=index,
                instruction_text=instruction.instruction_text,
                old_chunk_id=match.old_chunk_id,
                old_chunk_snapshot=(
                    _chunk_to_review_dict(old_chunk) if old_chunk else {}
                ),
                new_chunk_draft=new_chunk_draft,
                match_confidence=match.confidence,
                match_rationale=match.rationale,
                date_rationale=draft.dates.rationale,
            )
        )

    return AnalysisResult(
        reference_date=segmentation.reference_date,
        proposals=proposals,
        unmatched_instructions=unmatched,
    )
