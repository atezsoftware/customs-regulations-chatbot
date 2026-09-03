"""Durable execution of one checkpointed amendment-analysis batch."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_amendments import (
    get_batch,
    mark_batch_analyzed,
    persist_proposal_checkpoint,
    persist_segmentation_checkpoint,
    persist_unmatched_checkpoint,
    touch_batch_heartbeat,
)
from onyx.llm.factory import get_default_llm
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.draft_integrity import DraftIntegrityError
from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.pipeline import (
    confirm_instruction_match,
    draft_instruction_group_proposal,
    load_instruction_draft_context,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.search_retriever import (
    AmendmentSearchRetriever,
    build_amendment_search_retriever,
)
from onyx.regulatory.amendments.segmenter import (
    propagate_target_sources,
    segment_amendment_text,
)
from onyx.regulatory.amendments.structural_target import (
    appendix_replacement_attention_message,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()


@dataclass(frozen=True)
class _MatchedInstruction:
    instruction_index: int
    instruction: AmendmentInstruction
    candidates: list[CandidateChunk]
    match: MatchResult


@contextmanager
def _session() -> Generator[Session, None, None]:
    with get_session_with_current_tenant() as db_session:
        yield db_session


def _merge_candidates(
    initial: list[CandidateChunk], recovered: list[CandidateChunk]
) -> list[CandidateChunk]:
    merged: list[CandidateChunk] = []
    seen_ids: set[str] = set()
    for candidate in [*initial, *recovered]:
        if candidate.chunk_id in seen_ids:
            continue
        seen_ids.add(candidate.chunk_id)
        merged.append(candidate)
    return merged


def retrieve_and_confirm_instruction(
    *,
    retriever: AmendmentSearchRetriever,
    llm: LLM,
    instruction: AmendmentInstruction,
) -> tuple[list[CandidateChunk], MatchResult | None]:
    """Search, confirm, then make at most one focused recovery attempt."""

    candidates = retriever.search(instruction=instruction, recovery=False)
    if appendix_replacement_attention_message(instruction, candidates) is not None:
        return candidates, None
    if candidates:
        match = confirm_instruction_match(
            llm,
            instruction=instruction,
            candidates=candidates,
        )
        if match is not None:
            return candidates, match

    recovered = retriever.search(instruction=instruction, recovery=True)
    if not recovered:
        return candidates, None
    candidates = _merge_candidates(candidates, recovered)
    if appendix_replacement_attention_message(instruction, candidates) is not None:
        return candidates, None
    match = confirm_instruction_match(
        llm,
        instruction=instruction,
        candidates=candidates,
    )
    return candidates, match


def run_amendment_batch(*, batch_id: int, lease_generation: int) -> None:
    llm = get_default_llm()

    with _session() as db_session:
        batch = get_batch(db_session, batch_id)
        if batch is None:
            raise RuntimeError(f"Amendment batch {batch_id} no longer exists")
        user_file_ids = [UUID(value) for value in batch.user_file_ids]
        document_set_id = batch.document_set_id
        created_by = batch.created_by
        raw_text = batch.raw_text
        exact_processed_indices = getattr(batch, "processed_instruction_indices", None)
        processed_instruction_indices = (
            set(range(batch.processed_instruction_count))
            if not exact_processed_indices
            else set(exact_processed_indices)
        )
        if batch.segmented_instructions:
            instruction_payloads = list(batch.segmented_instructions)
            reference_date = (
                batch.reference_date.isoformat()
                if hasattr(batch.reference_date, "isoformat")
                else batch.reference_date
            )
        else:
            instruction_payloads = []
            reference_date = None

        retriever = build_amendment_search_retriever(
            db_session,
            document_set_id=document_set_id,
            created_by=created_by,
            user_file_ids=user_file_ids,
            llm=llm,
        )

    if not instruction_payloads:
        # No database session is held while the provider performs segmentation.
        segmentation = segment_amendment_text(llm, raw_text)
        if not segmentation.instructions:
            raise RuntimeError("Amendment segmentation returned no instructions")
        instruction_payloads = [
            instruction.model_dump() for instruction in segmentation.instructions
        ]
        with _session() as db_session:
            if not persist_segmentation_checkpoint(
                db_session,
                batch_id=batch_id,
                lease_generation=lease_generation,
                reference_date=segmentation.reference_date,
                instructions=instruction_payloads,
            ):
                raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
        reference_date = segmentation.reference_date

    instructions = propagate_target_sources(
        [
            AmendmentInstruction.model_validate(payload)
            for payload in instruction_payloads
        ]
    )
    matched_instructions: list[_MatchedInstruction] = []
    for instruction_index, instruction in enumerate(instructions):
        if instruction_index in processed_instruction_indices:
            continue
        candidates, match = retrieve_and_confirm_instruction(
            retriever=retriever,
            llm=llm,
            instruction=instruction,
        )

        if match is None:
            unresolved_text = (
                appendix_replacement_attention_message(instruction, candidates)
                or instruction.instruction_text
            )
            with _session() as db_session:
                persisted = persist_unmatched_checkpoint(
                    db_session,
                    batch_id=batch_id,
                    lease_generation=lease_generation,
                    instruction_index=instruction_index,
                    instruction_text=unresolved_text,
                )
            if not persisted:
                raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
        else:
            with _session() as db_session:
                heartbeat_refreshed = touch_batch_heartbeat(
                    db_session,
                    batch_id=batch_id,
                    lease_generation=lease_generation,
                )
            if not heartbeat_refreshed:
                raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
            matched_instructions.append(
                _MatchedInstruction(
                    instruction_index=instruction_index,
                    instruction=instruction,
                    candidates=candidates,
                    match=match,
                )
            )

        logger.info(
            "Amendment batch=%s collected instruction=%s/%s lease=%s candidates=%s",
            batch_id,
            instruction_index + 1,
            len(instruction_payloads),
            lease_generation,
            len(candidates),
        )

    groups: dict[tuple[str, str | int], list[_MatchedInstruction]] = {}
    for matched_instruction in matched_instructions:
        old_chunk_id = matched_instruction.match.old_chunk_id
        group_key: tuple[str, str | int] = (
            ("existing", old_chunk_id)
            if old_chunk_id is not None
            else ("new", matched_instruction.instruction_index)
        )
        groups.setdefault(group_key, []).append(matched_instruction)

    ordered_groups = sorted(
        groups.values(),
        key=lambda group: min(item.instruction_index for item in group),
    )
    for group in ordered_groups:
        ordered_group = sorted(group, key=lambda item: item.instruction_index)
        instruction_indices = [item.instruction_index for item in ordered_group]
        group_candidates: list[CandidateChunk] = []
        for item in ordered_group:
            group_candidates = _merge_candidates(group_candidates, item.candidates)

        with _session() as db_session:
            context = load_instruction_draft_context(
                db_session,
                candidates=group_candidates,
                match=ordered_group[0].match,
            )
        if context is None:
            for item in ordered_group:
                with _session() as db_session:
                    persisted = persist_unmatched_checkpoint(
                        db_session,
                        batch_id=batch_id,
                        lease_generation=lease_generation,
                        instruction_index=item.instruction_index,
                        instruction_text=item.instruction.instruction_text,
                    )
                if not persisted:
                    raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
            continue

        try:
            proposal = draft_instruction_group_proposal(
                llm,
                instruction_indices=instruction_indices,
                instructions=[item.instruction for item in ordered_group],
                matches=[item.match for item in ordered_group],
                reference_date=reference_date,
                context=context,
            )
        except DraftIntegrityError as error:
            for item in ordered_group:
                with _session() as db_session:
                    persisted = persist_unmatched_checkpoint(
                        db_session,
                        batch_id=batch_id,
                        lease_generation=lease_generation,
                        instruction_index=item.instruction_index,
                        instruction_text=(
                            f"{item.instruction.instruction_text}\n\nAttention: {error}"
                        ),
                    )
                if not persisted:
                    raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
            continue
        with _session() as db_session:
            persisted = persist_proposal_checkpoint(
                db_session,
                batch_id=batch_id,
                lease_generation=lease_generation,
                proposal=proposal,
            )
        if not persisted:
            raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
        logger.info(
            "Amendment batch=%s processed instruction group=%s lease=%s candidates=%s",
            batch_id,
            instruction_indices,
            lease_generation,
            len(group_candidates),
        )

    with _session() as db_session:
        if not mark_batch_analyzed(
            db_session,
            batch_id=batch_id,
            lease_generation=lease_generation,
        ):
            raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
