"""Restore only batch 299's already-matched 6/A proposal in DEV."""

import hashlib
import os
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.amendment_match_checkpoints import load_match_checkpoint
from onyx.db.amendment_pdf_evidence import load_batch_pdf_source
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.models import AmendmentBatch, AmendmentMatchCheckpoint, AmendmentProposal
from onyx.regulatory.amendments.amendment_context import build_amendment_context
from onyx.regulatory.amendments.analysis_llm import get_amendment_analysis_llm
from onyx.regulatory.amendments.draft_integrity import (
    validate_added_article_draft,
    validate_explicit_replacement_texts,
)
from onyx.regulatory.amendments.pipeline import (
    draft_instruction_group_proposal,
    load_instruction_draft_context,
)

BATCH_ID = 299
INDEX = 5
SOURCE_SHA256 = "30f7be2484b800594679c3e9f81eac71fb8f4fa18e7d193bcbd9fbe00be2c70b"
TARGET_FILE_ID = "014fbc6a-2da2-4845-9df9-f4afc625587a"


def checked_batch(session: Session) -> AmendmentBatch:
    batch = session.scalar(
        select(AmendmentBatch).where(AmendmentBatch.id == BATCH_ID).with_for_update()
    )
    if batch is None or batch.status != "analyzed" or batch.superseded_by_batch_id:
        raise RuntimeError("Batch 299 is no longer the analyzed active batch")
    if hashlib.sha256(batch.raw_text.encode()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Batch source text changed")
    if (
        batch.instruction_count <= INDEX
        or INDEX not in batch.processed_instruction_indices
    ):
        raise RuntimeError("Frozen instruction state changed")
    return batch


def main() -> None:
    from onyx.regulatory.amendments.annexes import config as annex_config

    if os.environ.get("POSTGRES_DB") != "customs-regulations-dev":
        raise RuntimeError("This repair is restricted to the DEV database")
    if annex_config.REGULATORY_ANNEX_ENVIRONMENT != "dev":
        raise RuntimeError("This repair is restricted to DEV publication")
    SqlEngine.init_engine(pool_size=1, max_overflow=0)

    with get_session_with_current_tenant() as session:
        batch = checked_batch(session)
        instruction_text = batch.segmented_instructions[INDEX]["instruction_text"]
        attention = [
            item
            for item in batch.unmatched_instructions
            if item == instruction_text or item.startswith(instruction_text + "\n\n")
        ]
        if (
            len(attention) != 1
            or "Insertion article identity already exists" not in attention[0]
        ):
            raise RuntimeError("Batch 299 attention record changed")
        if (
            session.scalar(
                select(AmendmentProposal.id).where(
                    AmendmentProposal.batch_id == BATCH_ID,
                    AmendmentProposal.instruction_index == INDEX,
                )
            )
            is not None
        ):
            raise RuntimeError("Instruction 5 already has a proposal")
        row = session.get(AmendmentMatchCheckpoint, (BATCH_ID, INDEX))
        if row is None:
            raise RuntimeError("Frozen match checkpoint missing")
        checkpoint = load_match_checkpoint(
            session,
            batch_id=BATCH_ID,
            instruction_index=INDEX,
            input_sha256=row.input_sha256,
        )
        if (
            checkpoint is None
            or checkpoint.instruction.instruction_text != instruction_text
        ):
            raise RuntimeError("Frozen match checkpoint or source fingerprints changed")
        if checkpoint.match.old_chunk_id is not None:
            raise RuntimeError("Instruction is no longer a new-article addition")
        context = load_instruction_draft_context(
            session,
            candidates=checkpoint.candidates,
            match=checkpoint.match,
            instruction=checkpoint.instruction,
        )
        if context is None or str(context.target_user_file_id) != TARGET_FILE_ID:
            raise RuntimeError("Target file could not be verified")
        if context.expected_new_article_no != "6/A":
            raise RuntimeError("Added article identity is not 6/A")
        pdf_source = load_batch_pdf_source(session, batch)
        reference_date = (
            batch.reference_date.isoformat() if batch.reference_date else None
        )
        amendment_context = build_amendment_context(batch.raw_text)
        instruction = checkpoint.instruction
        match = checkpoint.match
    llm = get_amendment_analysis_llm()
    proposal = draft_instruction_group_proposal(
        llm,
        instruction_indices=[INDEX],
        instructions=[instruction],
        matches=[match],
        reference_date=reference_date,
        context=context,
        pdf_source=pdf_source,
        amendment_context=amendment_context,
    )
    draft = proposal.new_chunk_draft
    if (
        proposal.instruction_indices != [INDEX]
        or proposal.instruction_text != instruction_text
    ):
        raise RuntimeError("Draft instruction identity changed")
    if (
        proposal.old_chunk_id is not None
        or str(draft.get("user_file_id")) != TARGET_FILE_ID
    ):
        raise RuntimeError("Draft source identity changed")
    validate_explicit_replacement_texts([instruction_text], str(draft.get("text", "")))
    validate_added_article_draft(
        [instruction_text],
        metadata=draft.get("metadata") or {},
        heading_path=draft.get("heading_path") or [],
        old_chunk_id=proposal.old_chunk_id,
        insertion_order=draft.get("insertion_order"),
    )
    if draft.get("metadata", {}).get("article_no") != "6/A":
        raise RuntimeError("Draft article identity changed")

    with get_session_with_current_tenant() as session:
        batch = checked_batch(session)
        if (
            session.scalar(
                select(AmendmentProposal.id).where(
                    AmendmentProposal.batch_id == BATCH_ID,
                    AmendmentProposal.instruction_index == INDEX,
                )
            )
            is not None
        ):
            raise RuntimeError("Instruction 5 was already repaired")
        row = session.get(AmendmentMatchCheckpoint, (BATCH_ID, INDEX))
        if (
            row is None
            or load_match_checkpoint(
                session,
                batch_id=BATCH_ID,
                instruction_index=INDEX,
                input_sha256=row.input_sha256,
            )
            is None
        ):
            raise RuntimeError("Frozen source evidence changed while drafting")
        attention = [
            item
            for item in batch.unmatched_instructions
            if item == instruction_text or item.startswith(instruction_text + "\n\n")
        ]
        if len(attention) != 1:
            raise RuntimeError("Attention state changed while drafting")
        saved = AmendmentProposal(
            batch_id=BATCH_ID,
            instruction_index=INDEX,
            instruction_text=proposal.instruction_text,
            instruction_indices=proposal.instruction_indices,
            instruction_texts=proposal.instruction_texts,
            old_chunk_id=proposal.old_chunk_id,
            old_chunk_snapshot=proposal.old_chunk_snapshot,
            new_chunk_draft=proposal.new_chunk_draft,
            chunk_changes=[
                item.model_dump(mode="json") for item in proposal.chunk_changes
            ],
            match_confidence=proposal.match_confidence,
            match_rationale=proposal.match_rationale,
            date_rationale=proposal.date_rationale,
            status="pending",
        )
        session.add(saved)
        session.flush()
        batch.unmatched_instructions = [
            item for item in batch.unmatched_instructions if item != attention[0]
        ]
        batch.analysis_log = [
            *batch.analysis_log,
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "step": "direct_unmatched_proposal_repaired",
                "index": INDEX,
                "proposal_id": saved.id,
                "source_checkpoint": row.input_sha256,
            },
        ][-2000:]
        session.commit()
        print(
            "REPAIRED",
            BATCH_ID,
            INDEX,
            saved.id,
            hashlib.sha256(str(draft["text"]).encode()).hexdigest(),
            draft["metadata"].get("article_no"),
        )


if __name__ == "__main__":
    main()
