"""Durable execution of one checkpointed amendment-analysis batch."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_amendments import (
    append_batch_log,
    get_batch,
    mark_batch_analyzed,
    persist_proposal_checkpoint,
    persist_segmentation_checkpoint,
    persist_unmatched_checkpoint,
    touch_batch_heartbeat,
)
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.amendment_context import (
    AmendmentContext,
    build_amendment_context,
)
from onyx.regulatory.amendments.analysis_llm import get_amendment_analysis_llm
from onyx.regulatory.amendments.draft_integrity import DraftIntegrityError
from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.pipeline import (
    confirm_instruction_match,
    draft_instruction_group_proposal,
    draft_multi_chunk_group_proposal,
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
    deterministic_structural_candidate,
    normalize_appendix_label,
    parse_amendment_structural_target,
)
from onyx.regulatory.structured_llm import StructuredOutputValidationError
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


@dataclass
class _InstructionTrace:
    """What actually happened while resolving one instruction.

    An unresolved instruction is otherwise indistinguishable from one that was
    never searched at all, which makes the difference between "retrieval found
    nothing" and "the model declined" invisible to whoever has to fix it.
    """

    searched: int = 0
    candidates: int = 0
    confirmations: int = 0
    declined: bool = False
    note: str | None = None
    queries: list[dict[str, object]] = field(default_factory=list)

    def describe(self) -> str:
        if self.note:
            return self.note
        if self.searched == 0:
            return "No search was run for this instruction."
        if any(query.get("lexical_only") for query in self.queries):
            return (
                "The embedding service was unreachable, so retrieval ran without "
                "its semantic half and matched nothing. This is a failed search, "
                "not a missing provision."
            )
        if self.candidates == 0:
            return (
                f"{self.searched} search(es) returned no candidate chunk in this "
                "Document Set, so no comparison was possible."
            )
        if self.confirmations == 0:
            return f"{self.candidates} candidate(s) found but none was confirmed."
        if self.declined:
            return (
                f"{self.candidates} candidate(s) found; the model declined all of "
                "them after " + f"{self.confirmations} check(s)."
            )
        return f"{self.candidates} candidate(s) found; no match was recorded."


def retrieve_and_confirm_instruction(
    *,
    retriever: AmendmentSearchRetriever,
    llm: LLM,
    instruction: AmendmentInstruction,
    amendment_context: AmendmentContext | None = None,
    trace: "_InstructionTrace | None" = None,
) -> tuple[list[CandidateChunk], MatchResult | None]:
    """Search, confirm, then make at most one focused recovery attempt."""

    trace = trace if trace is not None else _InstructionTrace()
    candidates = retriever.search(instruction=instruction, recovery=False)
    trace.searched += 1
    trace.candidates = len(candidates)
    trace.queries.extend(retriever.query_stats)
    appendix_note = appendix_replacement_attention_message(instruction, candidates)
    if appendix_note is not None:
        trace.note = "This annex target needs its replacement body supplied."
        return candidates, None
    if candidates:
        structural_candidate = deterministic_structural_candidate(
            instruction, candidates
        )
        if structural_candidate is not None:
            trace.confirmations += 1
            return candidates, MatchResult(
                old_chunk_id=structural_candidate.chunk_id,
                confidence=1.0,
                rationale=(
                    "Exact canonical source and structural metadata match the "
                    "named amendment target."
                ),
            )
        trace.confirmations += 1
        match = confirm_instruction_match(
            llm,
            instruction=instruction,
            candidates=candidates,
            amendment_context=amendment_context,
        )
        if match is not None:
            return candidates, match
        trace.declined = True

    recovered = retriever.search(instruction=instruction, recovery=True)
    trace.searched += 1
    trace.queries.extend(retriever.query_stats)
    if not recovered:
        return candidates, None
    candidates = _merge_candidates(candidates, recovered)
    trace.candidates = len(candidates)
    if appendix_replacement_attention_message(instruction, candidates) is not None:
        trace.note = "This annex target needs its replacement body supplied."
        return candidates, None
    trace.confirmations += 1
    match = confirm_instruction_match(
        llm,
        instruction=instruction,
        candidates=candidates,
        amendment_context=amendment_context,
    )
    trace.declined = match is None
    return candidates, match


def run_amendment_batch(*, batch_id: int, lease_generation: int) -> None:
    def log(step: str, **fields: object) -> None:
        with _session() as log_session:
            append_batch_log(
                log_session, batch_id=batch_id, entries=[{"step": step, **fields}]
            )

    log("batch_started", lease_generation=lease_generation)
    llm = get_amendment_analysis_llm()
    log(
        "analysis_model_resolved",
        provider=llm.config.model_provider,
        model=llm.config.model_name,
    )

    from onyx.db.amendment_pdf_evidence import load_batch_pdf_source

    with _session() as db_session:
        batch = get_batch(db_session, batch_id)
        if batch is None:
            raise RuntimeError(f"Amendment batch {batch_id} no longer exists")
        pdf_source = load_batch_pdf_source(db_session, batch)
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
        # An empty result is a legitimate terminal outcome, not a failure: the
        # segmenter is instructed to return no instructions whenever the pasted
        # text, together with its surrounding context, does not express update
        # intent (e.g. an unrelated notice, or a full replacement document with
        # no "this is the new version of X" framing). The batch still needs a
        # checkpoint and a normal analyzed/finalized lifecycle so the admin
        # sees "no update instructions detected" instead of a crash they can
        # never resolve by retrying the identical text.
        log("segmentation_started", raw_text_chars=len(raw_text))
        segmentation = segment_amendment_text(llm, raw_text)
        log(
            "segmentation_finished",
            instructions=len(segmentation.instructions),
            reference_date=segmentation.reference_date,
        )
        if not segmentation.instructions:
            logger.warning(
                "Amendment batch=%s segmentation found no update instructions; "
                "marking analyzed with nothing to review. This is expected for "
                "genuinely non-amendment text, but silently drops the batch if "
                "the segmenter misjudged real amendment text — check raw_text "
                "if that's suspected. raw_text_prefix=%r",
                batch_id,
                raw_text[:200],
            )
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
    # Every instruction is matched and drafted alone, so without this it never
    # sees the article stating when the amendment enters into force, nor the
    # ones defining the terms it uses.
    amendment_context = build_amendment_context(raw_text)
    log(
        "amendment_context_built",
        background_chars=len(amendment_context.background) if amendment_context else 0,
        commencement=list(amendment_context.commencement) if amendment_context else [],
    )
    from onyx.db.regulatory_annex_changes import legacy_text_annex_is_complete
    from onyx.regulatory.amendments.annexes import config as annex_config
    from onyx.regulatory.amendments.annexes.analysis import (
        group_annex_instructions,
        run_annex_groups,
    )

    annex_indices: set[int] = set()
    groups = group_annex_instructions(instructions)
    if annex_config.REGULATORY_ANNEX_UPDATES_ENABLED and groups:
        with _session() as db_session:
            batch = get_batch(db_session, batch_id)
            if batch is None:
                raise RuntimeError(f"Amendment batch {batch_id} no longer exists")
            # The annex review compares a frozen original document against the
            # indexed annex. Without one there is nothing to freeze, and the
            # indexed chunks already carry the annex text, so the instruction is
            # an ordinary amendment against those chunks rather than a blocked
            # review waiting for a document the amendment never referenced.
            if batch.source_package_id is not None:
                for group in groups:
                    if not legacy_text_annex_is_complete(
                        db_session,
                        batch=batch,
                        group=group,
                        reference_date=date.fromisoformat(reference_date)
                        if reference_date
                        else date.today(),
                    ):
                        annex_indices.update(group.instruction_indices)
    log(
        "instruction_loop_started",
        instructions=len(instructions),
        already_processed=sorted(processed_instruction_indices),
        annex_handled=sorted(annex_indices),
        annex_groups=len(groups),
    )
    first_output_error: StructuredOutputValidationError | TimeoutError | None = None
    matched_instructions: list[_MatchedInstruction] = []
    for instruction_index, instruction in enumerate(instructions):
        if instruction_index in processed_instruction_indices:
            log(
                "instruction_skipped",
                index=instruction_index,
                reason="already processed",
            )
            continue
        if instruction_index in annex_indices:
            log("instruction_skipped", index=instruction_index, reason="annex review")
            continue
        log(
            "instruction_started",
            index=instruction_index,
            text=instruction.instruction_text[:300],
            search_query=instruction.search_query,
            target_source=instruction.target_source,
            article_reference=instruction.article_reference,
        )
        trace = _InstructionTrace()
        try:
            candidates, match = retrieve_and_confirm_instruction(
                retriever=retriever,
                llm=llm,
                instruction=instruction,
                amendment_context=amendment_context,
                trace=trace,
            )
        except (StructuredOutputValidationError, TimeoutError) as error:
            # Leave this index unfinished so Retry resumes it, and keep going:
            # one instruction the confirming model could not answer must not
            # decide the outcome of every other instruction in the batch.
            first_output_error = first_output_error or error
            log(
                "instruction_failed",
                index=instruction_index,
                error=type(error).__name__,
                searches=trace.searched,
                candidates=trace.candidates,
                detail=str(error)[:300],
            )
            continue
        except Exception as error:
            # Still fatal, but no longer silent: the batch's own record names
            # what stopped it instead of only saying that it stopped.
            log(
                "instruction_error",
                index=instruction_index,
                error=type(error).__name__,
                searches=trace.searched,
                candidates=trace.candidates,
                detail=str(error)[:300],
            )
            raise

        log(
            "instruction_finished",
            index=instruction_index,
            searches=trace.searched,
            candidates=trace.candidates,
            confirmations=trace.confirmations,
            declined=trace.declined,
            queries=trace.queries,
            matched_chunk_id=match.old_chunk_id if match else None,
            outcome="matched" if match else "unmatched",
            detail=trace.describe() if match is None else None,
        )
        if match is None:
            unresolved_text = (
                appendix_replacement_attention_message(instruction, candidates)
                or f"{instruction.instruction_text}\n\nAttention: {trace.describe()}"
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
        structural_target = parse_amendment_structural_target(
            matched_instruction.instruction
        )
        group_key: tuple[str, str | int] = (
            (
                "appendix",
                normalize_appendix_label(structural_target.appendix_label),
            )
            if structural_target is not None
            and structural_target.appendix_label is not None
            else ("existing", old_chunk_id)
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

        appendix_target = parse_amendment_structural_target(
            ordered_group[0].instruction
        )
        appendix_candidate_ids = (
            [
                candidate.chunk_id
                for candidate in group_candidates
                if isinstance(candidate.metadata.get("appendix_label"), str)
                and appendix_target is not None
                and appendix_target.appendix_label is not None
                and normalize_appendix_label(str(candidate.metadata["appendix_label"]))
                == normalize_appendix_label(appendix_target.appendix_label)
            ]
            if appendix_target is not None
            and appendix_target.appendix_label is not None
            else []
        )
        appendix_candidate_ids = list(dict.fromkeys(appendix_candidate_ids))
        contexts = []
        with _session() as db_session:
            representative = next(
                (
                    item.match
                    for item in ordered_group
                    if item.match.old_chunk_id is not None
                ),
                ordered_group[0].match,
            )
            representative_context = load_instruction_draft_context(
                db_session,
                candidates=group_candidates,
                match=representative,
            )
            if (
                representative_context is not None
                and appendix_target is not None
                and appendix_target.appendix_label is not None
                and representative.old_chunk_id is not None
            ):
                from onyx.db.regulatory_annexes import load_legacy_annex_chunks

                appendix_rows = load_legacy_annex_chunks(
                    db_session,
                    document_set_id=document_set_id,
                    user_file_id=representative_context.target_user_file_id,
                    annex_label=appendix_target.appendix_label,
                    as_of_date=(
                        date.fromisoformat(reference_date)
                        if reference_date
                        else date.today()
                    ),
                )
                # Image companions are evidence for their bound canonical chunk,
                # not independent legal units to supersede.
                appendix_candidate_ids = [
                    row.id
                    for row in appendix_rows
                    if not row.chunk_metadata.get("bound_to_regulatory_chunk_id")
                ]
            if len(appendix_candidate_ids) > 1:
                for candidate_id in appendix_candidate_ids:
                    context = load_instruction_draft_context(
                        db_session,
                        candidates=group_candidates,
                        match=MatchResult(
                            old_chunk_id=candidate_id,
                            confidence=representative.confidence,
                            rationale=("Canonical member of the same appendix scope"),
                        ),
                    )
                    if context is not None:
                        contexts.append(context)
            elif representative_context is not None:
                contexts.append(representative_context)
        log(
            "draft_group_started",
            indices=instruction_indices,
            candidates=len(group_candidates),
            old_chunk_id=ordered_group[0].match.old_chunk_id,
        )
        if not contexts:
            log("draft_group_context_missing", indices=instruction_indices)
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
            if len(contexts) > 1:
                proposal = draft_multi_chunk_group_proposal(
                    llm,
                    instruction_indices=instruction_indices,
                    instructions=[item.instruction for item in ordered_group],
                    matches=[item.match for item in ordered_group],
                    contexts=contexts,
                    reference_date=reference_date,
                    amendment_context=amendment_context,
                )
            else:
                proposal = draft_instruction_group_proposal(
                    llm,
                    instruction_indices=instruction_indices,
                    instructions=[item.instruction for item in ordered_group],
                    matches=[item.match for item in ordered_group],
                    reference_date=reference_date,
                    context=contexts[0],
                    pdf_source=pdf_source,
                    amendment_context=amendment_context,
                )
        except (StructuredOutputValidationError, TimeoutError) as error:
            log(
                "draft_group_failed",
                indices=instruction_indices,
                error=type(error).__name__,
                detail=str(error)[:300],
            )
            # Leave these indices unfinished so Retry resumes them, while other
            # independent groups can still produce durable review proposals.
            first_output_error = first_output_error or error
            logger.warning(
                "Amendment batch=%s drafting group=%s failed: %s",
                batch_id,
                instruction_indices,
                type(error).__name__,
            )
            continue
        except DraftIntegrityError as error:
            log(
                "draft_group_rejected",
                indices=instruction_indices,
                error=str(error)[:300],
            )
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

    log("proposals_finished", groups=len(ordered_groups))
    run_annex_groups(
        batch_id=batch_id,
        lease_generation=lease_generation,
        instructions=instructions,
        processed_indices=processed_instruction_indices,
        reference_date=reference_date,
        llm=llm,
    )
    if first_output_error is not None:
        log("batch_failed", error=type(first_output_error).__name__)
        raise first_output_error

    with _session() as db_session:
        log("batch_analyzed")
        if not mark_batch_analyzed(
            db_session,
            batch_id=batch_id,
            lease_generation=lease_generation,
        ):
            raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
