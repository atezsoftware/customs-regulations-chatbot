"""Durable execution of one checkpointed amendment-analysis batch."""

from collections.abc import Generator
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_amendments import (
    get_batch,
    mark_batch_analyzed,
    persist_proposal_checkpoint,
    persist_segmentation_checkpoint,
    persist_unmatched_checkpoint,
)
from onyx.llm.factory import get_default_llm
from onyx.regulatory.amendments.candidate_finder import find_candidates
from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.pipeline import (
    confirm_instruction_match,
    draft_instruction_proposal,
    load_instruction_draft_context,
)
from onyx.regulatory.amendments.segmenter import segment_amendment_text
from onyx.utils.logger import setup_logger

logger = setup_logger()


@contextmanager
def _session() -> Generator[Session, None, None]:
    with get_session_with_current_tenant() as db_session:
        yield db_session


def run_amendment_batch(*, batch_id: int, lease_generation: int) -> None:
    llm = get_default_llm()

    with _session() as db_session:
        batch = get_batch(db_session, batch_id)
        if batch is None:
            raise RuntimeError(f"Amendment batch {batch_id} no longer exists")
        user_file_ids = [UUID(value) for value in batch.user_file_ids]
        raw_text = batch.raw_text
        start_index = batch.processed_instruction_count
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

    for instruction_index in range(start_index, len(instruction_payloads)):
        instruction = AmendmentInstruction.model_validate(
            instruction_payloads[instruction_index]
        )
        with _session() as db_session:
            candidates = find_candidates(
                db_session,
                user_file_ids=user_file_ids,
                instruction=instruction,
            )

        if not candidates:
            with _session() as db_session:
                persisted = persist_unmatched_checkpoint(
                    db_session,
                    batch_id=batch_id,
                    lease_generation=lease_generation,
                    instruction_index=instruction_index,
                    instruction_text=instruction.instruction_text,
                )
        else:
            # Match and drafting are provider calls; keep them outside database
            # sessions so slow LLM responses cannot leave idle transactions.
            match = confirm_instruction_match(
                llm,
                instruction=instruction,
                candidates=candidates,
            )
            if match is None:
                with _session() as db_session:
                    persisted = persist_unmatched_checkpoint(
                        db_session,
                        batch_id=batch_id,
                        lease_generation=lease_generation,
                        instruction_index=instruction_index,
                        instruction_text=instruction.instruction_text,
                    )
            else:
                with _session() as db_session:
                    context = load_instruction_draft_context(
                        db_session,
                        candidates=candidates,
                        match=match,
                    )
                proposal = draft_instruction_proposal(
                    llm,
                    instruction_index=instruction_index,
                    instruction=instruction,
                    reference_date=reference_date,
                    context=context,
                )
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
            "Amendment batch=%s processed instruction=%s/%s lease=%s candidates=%s",
            batch_id,
            instruction_index + 1,
            len(instruction_payloads),
            lease_generation,
            len(candidates),
        )

    with _session() as db_session:
        if not mark_batch_analyzed(
            db_session,
            batch_id=batch_id,
            lease_generation=lease_generation,
        ):
            raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
