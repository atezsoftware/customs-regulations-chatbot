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
    get_chunk_snapshot_by_id,
    get_next_chunk_position,
    load_active_structural_descendants,
)
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.amendment_context import (
    AmendmentContext,
    build_amendment_context,
)
from onyx.regulatory.amendments.candidate_finder import find_candidates
from onyx.regulatory.amendments.draft_integrity import (
    explicit_replacement_body,
    reconcile_existing_heading_path,
    reject_unsupported_descendant_replacement,
    validate_explicit_replacements,
)
from onyx.regulatory.amendments.drafter import (
    draft_combined_chunk,
    draft_multi_chunk_scope,
    draft_new_chunk,
)
from onyx.regulatory.amendments.matcher import confirm_match
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    AnalysisResult,
    DraftResult,
    MatchResult,
    ProposalChunkChange,
    ProposalDraft,
)
from onyx.regulatory.amendments.new_provision_policy import (
    added_subordinate_unit_kind,
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.pdf_vision import PdfBatchSource
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
    amendment_context: AmendmentContext | None = None,
) -> MatchResult | None:
    match = confirm_match(
        llm,
        instruction=instruction,
        candidates=candidates,
        amendment_context=amendment_context,
    )
    candidate_ids = {candidate.chunk_id for candidate in candidates}
    if match.old_chunk_id is not None and match.old_chunk_id not in candidate_ids:
        logger.warning(
            "Amendment matcher returned candidate id outside the supplied set: %s",
            match.old_chunk_id,
        )
        return None
    if (
        match.old_chunk_id is None
        and not explicitly_adds_top_level_provision(instruction.instruction_text)
        and added_subordinate_unit_kind(instruction.instruction_text) is None
    ):
        logger.warning(
            "Amendment matcher declined all candidates for an instruction that "
            "adds no provision of its own; marking unmatched"
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
        old_chunk = get_chunk_snapshot_by_id(db_session, match.old_chunk_id)
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
        if not candidates:
            logger.warning(
                "Amendment addition has no candidate to anchor its file and "
                "position; marking instruction unmatched"
            )
            return None
        # An exact structural match names the amended article outright, so it
        # anchors a new unit far more reliably than the best ranked search hit,
        # which may sit in a neighbouring provision.
        best_candidate = next(
            (candidate for candidate in candidates if candidate.structured_match),
            candidates[0],
        )
        target_user_file_id = UUID(best_candidate.user_file_id)
        sibling_reference = {
            "text": best_candidate.text,
            "metadata": best_candidate.metadata,
            "heading_path": best_candidate.metadata.get("heading_path"),
        }
        target_position = get_next_chunk_position(db_session, target_user_file_id)

    descendants = (
        load_active_structural_descendants(db_session, old_chunk)
        if old_chunk is not None
        else []
    )
    snapshot = _chunk_to_review_dict(old_chunk) if old_chunk else {}
    if descendants:
        snapshot["descendant_snapshots"] = [
            _chunk_to_review_dict(row) for row in descendants
        ]
    return InstructionDraftContext(
        match=match,
        old_chunk_snapshot=snapshot,
        target_user_file_id=target_user_file_id,
        target_position=target_position,
        sibling_reference=sibling_reference,
        base_metadata=dict(old_chunk.chunk_metadata) if old_chunk else {},
        base_heading_path=list(old_chunk.heading_path) if old_chunk else [],
        has_active_descendants=bool(descendants),
    )


def draft_instruction_proposal(
    llm: LLM,
    *,
    instruction_index: int,
    instruction: AmendmentInstruction,
    reference_date: str | None,
    context: InstructionDraftContext,
    amendment_context: AmendmentContext | None = None,
) -> ProposalDraft:
    draft = draft_new_chunk(
        llm,
        instruction=instruction,
        old_chunk=context.old_chunk_snapshot or None,
        sibling_reference=context.sibling_reference,
        reference_date=reference_date,
        amendment_context=amendment_context,
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
    pdf_source: PdfBatchSource | None = None,
    amendment_context: AmendmentContext | None = None,
) -> ProposalDraft:
    if not instructions or len(instruction_indices) != len(instructions):
        raise ValueError(
            "Grouped amendment drafting requires one index per instruction"
        )
    if len(matches) != len(instructions):
        raise ValueError(
            "Grouped amendment drafting requires one match per instruction"
        )
    full_replacement = any(
        explicit_replacement_body(item.instruction_text) for item in instructions
    )
    descendants = context.old_chunk_snapshot.get("descendant_snapshots") or []
    if context.has_active_descendants and full_replacement and not descendants:
        reject_unsupported_descendant_replacement(
            instructions, has_active_descendants=True
        )
    if descendants and full_replacement:
        from onyx.regulatory.amendments.draft_integrity import (
            validate_complete_scope_replacement,
        )

        validate_complete_scope_replacement(
            [item.instruction_text for item in instructions]
        )
    old_chunk_ids = {match.old_chunk_id for match in matches}
    if len(old_chunk_ids) != 1:
        raise ValueError("Grouped amendment instructions must share one target chunk")

    evidence = None
    if pdf_source is not None:
        from onyx.file_store.file_store import get_default_file_store
        from onyx.regulatory.amendments.analysis_llm import (
            get_amendment_analysis_llm,
        )
        from onyx.regulatory.amendments.pdf_vision import prepare_pdf_draft_evidence

        evidence = prepare_pdf_draft_evidence(
            pdf_source, instructions, get_default_file_store()
        )
        if evidence is not None:
            if (
                context.target_user_file_id not in pdf_source.user_file_ids
                or context.old_chunk_snapshot.get("user_file_id")
                != str(context.target_user_file_id)
                or context.match.old_chunk_id != context.old_chunk_snapshot.get("id")
            ):
                raise ValueError("pdf_draft_target_scope_mismatch")
            # The pinned analysis model reads the original pages itself, so
            # no separate vision provider is selected for this pipeline.
            llm = get_amendment_analysis_llm()
    draft = draft_combined_chunk(
        llm,
        instructions=instructions,
        old_chunk=context.old_chunk_snapshot or None,
        sibling_reference=context.sibling_reference,
        reference_date=reference_date,
        pdf_evidence=evidence,
        amendment_context=amendment_context,
    )
    proposal = _build_proposal_draft(
        instruction_indices=instruction_indices,
        instructions=instructions,
        matches=matches,
        context=context,
        draft=draft,
    )
    if evidence is not None:
        from onyx.regulatory.amendments.pdf_vision import (
            PDF_EVIDENCE_KEY,
            verify_pdf_draft,
        )

        receipt = verify_pdf_draft(
            llm,
            evidence=evidence,
            instructions=instructions,
            old_chunk=context.old_chunk_snapshot,
            draft_text=proposal.new_chunk_draft["text"],
        )
        proposal.old_chunk_snapshot = {
            **proposal.old_chunk_snapshot,
            PDF_EVIDENCE_KEY: receipt.model_dump(mode="json"),
        }
    return proposal


def draft_multi_chunk_group_proposal(
    llm: LLM,
    *,
    instruction_indices: list[int],
    instructions: list[AmendmentInstruction],
    matches: list[MatchResult],
    contexts: list[InstructionDraftContext],
    reference_date: str | None,
    amendment_context: AmendmentContext | None = None,
) -> ProposalDraft:
    """Create one atomically reviewed proposal spanning multiple old chunks."""

    if len(instruction_indices) != len(instructions) or len(matches) != len(
        instructions
    ):
        raise ValueError("Multi-chunk group has inconsistent instruction coverage")
    context_by_id = {
        context.match.old_chunk_id: context
        for context in contexts
        if context.match.old_chunk_id is not None
    }
    if len(context_by_id) < 2:
        raise ValueError("Multi-chunk group requires at least two existing chunks")
    file_ids = {context.target_user_file_id for context in context_by_id.values()}
    if len(file_ids) != 1:
        raise ValueError("Atomic multi-chunk changes must stay within one source file")

    result = draft_multi_chunk_scope(
        llm,
        instructions=instructions,
        old_chunks=[context.old_chunk_snapshot for context in contexts],
        reference_date=reference_date,
        amendment_context=amendment_context,
    )
    changed_ids = [change.old_chunk_id for change in result.changes]
    if any(chunk_id not in context_by_id for chunk_id in changed_ids):
        raise ValueError("Multi-chunk draft escaped its frozen candidate scope")

    covered_instruction_indexes: set[int] = set()
    changes: list[ProposalChunkChange] = []
    built_proposals: list[ProposalDraft] = []
    for change in result.changes:
        local_indexes = sorted(set(change.instruction_indexes))
        if any(index < 0 or index >= len(instructions) for index in local_indexes):
            raise ValueError("Multi-chunk draft returned an invalid instruction index")
        covered_instruction_indexes.update(local_indexes)
        local_instructions = [instructions[index] for index in local_indexes]
        local_global_indices = [instruction_indices[index] for index in local_indexes]
        context = context_by_id[change.old_chunk_id]
        local_matches = [
            MatchResult(
                old_chunk_id=change.old_chunk_id,
                confidence=matches[index].confidence,
                rationale=matches[index].rationale,
            )
            for index in local_indexes
        ]
        built = _build_proposal_draft(
            instruction_indices=local_global_indices,
            instructions=local_instructions,
            matches=local_matches,
            context=context,
            draft=DraftResult(new_chunk=change.new_chunk, dates=result.dates),
        )
        built_proposals.append(built)
        changes.append(
            ProposalChunkChange(
                old_chunk_id=built.old_chunk_id,
                old_chunk_snapshot=built.old_chunk_snapshot,
                new_chunk_draft=built.new_chunk_draft,
                instruction_indices=built.instruction_indices,
                instruction_texts=built.instruction_texts,
                match_confidence=built.match_confidence,
                match_rationale=built.match_rationale,
                date_rationale=built.date_rationale,
            )
        )
    if covered_instruction_indexes != set(range(len(instructions))):
        raise ValueError("Multi-chunk draft did not apply every instruction")

    primary = built_proposals[0]
    return ProposalDraft(
        instruction_index=instruction_indices[0],
        instruction_text=instructions[0].instruction_text,
        instruction_indices=instruction_indices,
        instruction_texts=[
            instruction.instruction_text for instruction in instructions
        ],
        old_chunk_id=primary.old_chunk_id,
        old_chunk_snapshot=primary.old_chunk_snapshot,
        new_chunk_draft=primary.new_chunk_draft,
        chunk_changes=changes,
        match_confidence=min(match.confidence for match in matches),
        match_rationale="Atomic multi-chunk structural amendment",
        date_rationale=result.dates.rationale,
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
    amendment_context: AmendmentContext | None = None,
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
        llm,
        instruction=instruction,
        candidates=candidates,
        amendment_context=amendment_context,
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
        amendment_context=amendment_context,
    )


def analyze_amendment(
    db_session: Session,
    *,
    llm: LLM,
    user_file_ids: list[UUID],
    raw_text: str,
) -> AnalysisResult:
    segmentation = segment_amendment_text(llm, raw_text)
    amendment_context = build_amendment_context(raw_text)

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
            amendment_context=amendment_context,
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
