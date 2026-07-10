"""Orchestrates the amendment analysis pipeline: segment -> find candidates ->
confirm match -> draft new chunk, for every instruction in a pasted text.

Pure orchestration — never writes to storage. The caller (the REST endpoint
in `fs_explorer_api.server`) persists the resulting `AnalysisResult` via
`PostgresStorage`'s amendment batch/proposal methods, and nothing lands in
`core_chunks` until an admin approves a specific proposal.
"""

from __future__ import annotations

from typing import Any

from fs_explorer_shared.embeddings import EmbeddingProvider
from fs_explorer_shared.storage import StorageBackend, chunk_to_review_dict

from ..llm import LLMClient
from .candidate_finder import find_candidates
from .drafter import draft_new_chunk
from .matcher import confirm_match
from .models import AmendmentInstruction, AnalysisResult, MatchResult, ProposalDraft
from .segmenter import segment_amendment_text


async def analyze_amendment(
    *,
    storage: StorageBackend,
    embedding_provider: EmbeddingProvider | None,
    llm: LLMClient,
    corpus_id: str,
    raw_text: str,
) -> AnalysisResult:
    segmentation, _usage = await segment_amendment_text(llm, raw_text)

    proposals: list[ProposalDraft] = []
    unmatched: list[AmendmentInstruction] = []

    for index, instruction in enumerate(segmentation.instructions):
        candidates = find_candidates(
            storage,
            embedding_provider,
            corpus_id=corpus_id,
            instruction=instruction,
        )
        if not candidates:
            # Nothing in the corpus resembles this instruction at all — no
            # document to attach a new chunk to, so surface it as unmatched
            # rather than guessing.
            unmatched.append(instruction)
            continue

        match, _usage = await confirm_match(
            llm, instruction=instruction, candidates=candidates
        )

        old_chunk: dict[str, Any] | None = None
        if match.old_chunk_id:
            old_chunk = storage.get_chunk(chunk_id=match.old_chunk_id)
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
            target_document_id = str(old_chunk["document_id"])
            target_position = int(old_chunk["position"])
        else:
            # New provision, no existing chunk to replace — attach it to the
            # same document as the strongest candidate (virtually always the
            # one document this gazette text is amending) and append it
            # after that document's existing chunks. Give the LLM that
            # candidate's own metadata as a reference so it can build a
            # heading_path/article_no consistent with the rest of the
            # document instead of leaving them empty — search
            # (search_chunks_by_heading_trigram) and chat citation labels
            # (backend's locatorForHit) both depend on these fields being
            # populated, for amendment-created chunks same as indexed ones.
            target_document_id = candidates[0].doc_id
            sibling_reference = candidates[0].metadata
            siblings = storage.list_document_chunks(doc_id=target_document_id)
            target_position = max((c["position"] for c in siblings), default=-1) + 1

        draft, _usage = await draft_new_chunk(
            llm,
            instruction=instruction,
            old_chunk=old_chunk,
            sibling_reference=sibling_reference,
            reference_date=segmentation.reference_date,
        )

        # Merge, don't replace: the LLM only returns the metadata fields it
        # actually changed (`metadata_changes`), so fields it wasn't asked
        # to touch — heading_path, document_date, etc. — survive by
        # construction instead of depending on the model faithfully
        # reproducing every unrelated key.
        base_metadata = (old_chunk.get("metadata") or {}) if old_chunk else {}
        merged_metadata = {**base_metadata, **draft.new_chunk.metadata_changes}

        new_chunk_draft: dict[str, Any] = {
            "document_id": target_document_id,
            "position": target_position,
            "text": draft.new_chunk.text,
            "chunk_type": draft.new_chunk.chunk_type,
            "metadata": merged_metadata,
            "effective_start_date": draft.dates.effective_start_date,
            "effective_end_date": draft.dates.effective_end_date,
        }

        proposals.append(
            ProposalDraft(
                instruction_index=index,
                instruction_text=instruction.instruction_text,
                old_chunk_id=match.old_chunk_id,
                old_chunk_snapshot=chunk_to_review_dict(old_chunk) if old_chunk else {},
                new_chunk_draft=new_chunk_draft,
                match_confidence=match.confidence,
                match_rationale=match.rationale,
                date_rationale=draft.dates.rationale,
            )
        )

    _flag_duplicate_targets(proposals)

    return AnalysisResult(
        reference_date=segmentation.reference_date,
        proposals=proposals,
        unmatched_instructions=unmatched,
    )


def _flag_duplicate_targets(proposals: list[ProposalDraft]) -> None:
    """Mark proposals from the same analysis run that target the same old
    chunk — approving both would race on the DB's `status='active'` guard
    (`PostgresStorage.approve_amendment_proposal`), so it's better to warn
    the admin up front than let the second approval fail silently later."""
    seen: set[str] = set()
    for proposal in proposals:
        if proposal.old_chunk_id:
            if proposal.old_chunk_id in seen:
                proposal.duplicate_target = True
            seen.add(proposal.old_chunk_id)


def flag_duplicate_targets_in_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Same idea as `_flag_duplicate_targets`, but for already-persisted
    proposal records (dicts read back from storage) rather than in-memory
    drafts. Used when serving analyze/list responses, since
    `duplicate_target` isn't a stored column — it's a derived hint,
    recomputed from whichever proposals are being read together."""
    counts: dict[str, int] = {}
    for record in records:
        old_chunk_id = record.get("old_chunk_id")
        if old_chunk_id and record.get("status") == "pending":
            counts[old_chunk_id] = counts.get(old_chunk_id, 0) + 1
    return [
        {
            **record,
            "duplicate_target": counts.get(record.get("old_chunk_id") or "", 0) > 1,
        }
        for record in records
    ]
