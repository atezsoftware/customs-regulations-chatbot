"""Durable execution of one checkpointed amendment-analysis batch."""

from collections.abc import Callable, Generator, Hashable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date
from time import monotonic
from traceback import extract_tb
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.amendment_match_checkpoints import (
    capture_match_evidence,
    load_match_checkpoint,
    match_scope_fingerprint,
    persist_match_checkpoint,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_amendments import (
    append_batch_log,
    get_batch,
    mark_batch_analyzed,
    persist_proposal_checkpoint,
    persist_segmentation_checkpoint,
    persist_unmatched_checkpoint,
)
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.amendment_context import (
    AmendmentContext,
    build_amendment_context,
)
from onyx.regulatory.amendments.analysis_llm import get_amendment_analysis_llm
from onyx.regulatory.amendments.draft_integrity import DraftIntegrityError
from onyx.regulatory.amendments.drafter import AmendmentDateConflict
from onyx.regulatory.amendments.insertion_order import InsertionOrderError
from onyx.regulatory.amendments.match_checkpoint import (
    MatchedInstruction,
    MatchEvidence,
    match_input_sha256,
)
from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.new_provision_policy import (
    added_subordinate_unit_kind,
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.pipeline import (
    confirm_instruction_match,
    draft_article_heading_group_proposal,
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


@contextmanager
def _session() -> Generator[Session, None, None]:
    with get_session_with_current_tenant() as db_session:
        yield db_session


def _merge_candidates(
    initial: list[CandidateChunk], recovered: list[CandidateChunk]
) -> list[CandidateChunk]:
    merged: dict[str, CandidateChunk] = {}
    for candidate in [*initial, *recovered]:
        previous = merged.get(candidate.chunk_id)
        if previous is None:
            merged[candidate.chunk_id] = candidate
        else:
            merged[candidate.chunk_id] = replace(
                previous,
                structured_match=previous.structured_match
                or candidate.structured_match,
                source_verified=previous.source_verified or candidate.source_verified,
                source_ambiguous=previous.source_ambiguous
                or candidate.source_ambiguous,
                structure_conflict=previous.structure_conflict
                or candidate.structure_conflict,
                resolved_article_no=candidate.resolved_article_no
                or previous.resolved_article_no,
                scope_evidence=candidate.scope_evidence or previous.scope_evidence,
            )
    return list(merged.values())


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
    decisions: list[dict[str, object]] = field(default_factory=list)

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
                "them after "
                + f"{self.confirmations} check(s)."
                + (
                    f" Reason: {self.decisions[-1].get('rationale', '')}"
                    if self.decisions
                    else ""
                )
            )
        return f"{self.candidates} candidate(s) found; no match was recorded."


def retrieve_and_confirm_instruction(
    *,
    retriever: AmendmentSearchRetriever,
    llm: LLM,
    instruction: AmendmentInstruction,
    amendment_context: AmendmentContext | None = None,
    trace: "_InstructionTrace | None" = None,
    capture_evidence: Callable[[list[CandidateChunk]], None] | None = None,
) -> tuple[list[CandidateChunk], MatchResult | None]:
    """Search, confirm, then make at most one focused recovery attempt."""

    trace = trace if trace is not None else _InstructionTrace()
    candidates = retriever.search(instruction=instruction, recovery=False)
    if capture_evidence is not None:
        capture_evidence(candidates)
    trace.searched += 1
    trace.candidates = len(candidates)
    trace.queries.extend(retriever.query_stats)
    if isinstance(retriever.last_attention, str):
        trace.note = retriever.last_attention
        return candidates, None
    appendix_note = appendix_replacement_attention_message(instruction, candidates)
    if appendix_note is not None:
        trace.note = "This annex target needs its replacement body supplied."
        return candidates, None

    def exact_match(items: list[CandidateChunk]) -> MatchResult | None:
        structural_candidate = (
            deterministic_structural_candidate(instruction, items)
            if not (
                explicitly_adds_top_level_provision(instruction.instruction_text)
                or added_subordinate_unit_kind(instruction.instruction_text)
            )
            else None
        )
        if structural_candidate is not None:
            return MatchResult(
                old_chunk_id=structural_candidate.chunk_id,
                confidence=1.0,
                rationale=(
                    "Exact canonical source and structural metadata match the "
                    "named amendment target."
                ),
            )
        return None

    if candidates:
        structural_match = exact_match(candidates)
        if structural_match is not None:
            trace.confirmations += 1
            return candidates, structural_match
        trace.confirmations += 1
        match = confirm_instruction_match(
            llm,
            instruction=instruction,
            candidates=candidates,
            amendment_context=amendment_context,
            decisions=trace.decisions,
        )
        if match is not None:
            return candidates, match
        trace.declined = True

    recovered = retriever.search(instruction=instruction, recovery=True)
    trace.searched += 1
    trace.queries.extend(retriever.query_stats)
    if not recovered:
        return candidates, None
    merged = _merge_candidates(candidates, recovered)
    if merged == candidates:
        trace.note = (
            "Recovery returned the same canonical evidence; no repeated model check."
        )
        return candidates, None
    candidates = merged
    if capture_evidence is not None:
        capture_evidence(candidates)
    trace.candidates = len(candidates)
    if appendix_replacement_attention_message(instruction, candidates) is not None:
        trace.note = "This annex target needs its replacement body supplied."
        return candidates, None
    trace.confirmations += 1
    match = exact_match(candidates) or confirm_instruction_match(
        llm,
        instruction=instruction,
        candidates=candidates,
        amendment_context=amendment_context,
        decisions=trace.decisions,
    )
    trace.declined = match is None
    return candidates, match


MatchGroupKey = tuple[str, str | int]
MatchOutcome = tuple[int, MatchGroupKey] | None


class InstructionRunner(Protocol):
    def __call__(
        self,
        function: Callable[[int], MatchOutcome],
        items: list[int],
        /,
        *,
        concurrency_key: Callable[[int], Hashable] | None = None,
        work_class: Callable[[int], Hashable] | None = None,
    ) -> Iterable[MatchOutcome]: ...


def run_amendment_batch(
    *,
    batch_id: int,
    lease_generation: int,
    instruction_runner: InstructionRunner | None = None,
    check_resources: Callable[[], None] | None = None,
    before_work: Callable[[], None] | None = None,
) -> None:
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
            stored_date = batch.reference_date
            reference_date: str | None = (
                stored_date.isoformat()
                if isinstance(stored_date, date)
                else stored_date
            )
        else:
            instruction_payloads = []
            reference_date = None

    del batch
    if not instruction_payloads:
        if before_work is not None:
            before_work()
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
        segmented_date = segmentation.reference_date
        reference_date = (
            segmented_date.isoformat()
            if isinstance(segmented_date, date)
            else segmented_date
        )
        del segmentation

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
    from onyx.db.regulatory_annex_changes import can_draft_annex_from_instructions
    from onyx.regulatory.amendments.annexes import config as annex_config
    from onyx.regulatory.amendments.annexes.analysis import (
        group_annex_instructions,
        run_annex_groups,
    )

    annex_indices: set[int] = set()
    annex_groups = group_annex_instructions(instructions)
    review_groups = []
    if annex_config.REGULATORY_ANNEX_UPDATES_ENABLED and annex_groups:
        with _session() as db_session:
            batch = get_batch(db_session, batch_id)
            if batch is None:
                raise RuntimeError(f"Amendment batch {batch_id} no longer exists")
            # A package supplies evidence, not the operation type. Explicit
            # edits use indexed chunks; only replacement evidence needs review.
            if batch.source_package_id is not None:
                for group in annex_groups:
                    if not can_draft_annex_from_instructions(
                        db_session,
                        batch=batch,
                        group=group,
                        reference_date=date.fromisoformat(reference_date)
                        if reference_date
                        else date.today(),
                    ):
                        annex_indices.update(group.instruction_indices)
                        review_groups.append(group)
        del batch
    for group in annex_groups:
        log(
            "annex_route_selected",
            indices=group.instruction_indices,
            annex_label=group.annex_label,
            route="document_comparison"
            if group in review_groups
            else "chunk_amendment",
        )
    log(
        "instruction_loop_started",
        instructions=len(instructions),
        already_processed=sorted(processed_instruction_indices),
        annex_handled=sorted(annex_indices),
        annex_groups=len(annex_groups),
    )
    input_sha256 = match_input_sha256(
        raw_text=raw_text,
        instructions=[item.model_dump(mode="json") for item in instructions],
        document_set_id=document_set_id,
        user_file_ids=sorted(str(value) for value in user_file_ids),
        created_by=created_by,
        reference_date=reference_date,
        retrieval_date=date.today().isoformat(),
        model=str(llm.config.model_name),
        provider=str(llm.config.model_provider),
    )
    del raw_text, instruction_payloads
    groups: dict[tuple[str, str | int], list[int]] = {}
    from onyx.regulatory.amendments.compound_heading import article_heading_title

    heading_articles: set[str] = set()
    for instruction in instructions:
        if article_heading_title(instruction.instruction_text):
            target = parse_amendment_structural_target(instruction)
            if target is not None and target.article_no is not None:
                heading_articles.add(target.article_no)
    pending_indices: list[int] = []
    # Only real retrieval work can calibrate the scheduler's memory budget.
    for instruction_index in range(len(instructions)):
        if check_resources is not None:
            check_resources()
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
        with _session() as db_session:
            checkpoint = load_match_checkpoint(
                db_session,
                batch_id=batch_id,
                instruction_index=instruction_index,
                input_sha256=input_sha256,
            )
        if checkpoint is not None:
            groups.setdefault(checkpoint.draft_group_key(heading_articles), []).append(
                instruction_index
            )
            log("instruction_match_restored", index=instruction_index)
            del checkpoint
            continue
        pending_indices.append(instruction_index)

    def match_one(instruction_index: int) -> MatchOutcome:
        if check_resources is not None:
            check_resources()
        instruction = instructions[instruction_index]
        instruction_llm = (
            get_amendment_analysis_llm() if instruction_runner is not None else llm
        )
        log(
            "instruction_started",
            index=instruction_index,
            text=instruction.instruction_text[:300],
            search_query=instruction.search_query,
            target_source=instruction.target_source,
            article_reference=instruction.article_reference,
        )
        with _session() as db_session:
            source_scope = match_scope_fingerprint(
                db_session, [str(value) for value in user_file_ids]
            )
            retriever = build_amendment_search_retriever(
                db_session,
                document_set_id=document_set_id,
                created_by=created_by,
                user_file_ids=user_file_ids,
                llm=instruction_llm,
            )
        evidence: MatchEvidence | None = None

        def capture(candidates: list[CandidateChunk]) -> None:
            nonlocal evidence
            with _session() as db_session:
                evidence = capture_match_evidence(
                    db_session, [str(value) for value in user_file_ids], candidates
                )
            if evidence.scope_sha256 != source_scope:
                raise ValueError("Amendment source identity changed during retrieval")

        trace = _InstructionTrace()
        try:
            candidates, match = retrieve_and_confirm_instruction(
                retriever=retriever,
                llm=instruction_llm,
                instruction=instruction,
                amendment_context=amendment_context,
                trace=trace,
                capture_evidence=capture,
            )
        except (StructuredOutputValidationError, TimeoutError) as error:
            log(
                "instruction_failed",
                index=instruction_index,
                error=type(error).__name__,
                searches=trace.searched,
                candidates=trace.candidates,
                detail=str(error)[:300],
            )
            with _session() as db_session:
                persisted = persist_unmatched_checkpoint(
                    db_session,
                    batch_id=batch_id,
                    lease_generation=lease_generation,
                    instruction_index=instruction_index,
                    expected_evidence=evidence
                    or MatchEvidence(scope_sha256=source_scope, candidates={}),
                    instruction_text=(
                        f"{instruction.instruction_text}\n\nAttention: Model output "
                        f"could not be validated ({type(error).__name__})."
                    ),
                )
            if not persisted:
                raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
            return None
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
                frames=[
                    {
                        "file": frame.filename,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in extract_tb(error.__traceback__)[-20:]
                ],
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
            decisions=trace.decisions,
            matched_chunk_id=match.old_chunk_id if match else None,
            outcome="matched" if match else "unmatched",
            detail=trace.describe() if match is None else None,
        )
        if check_resources is not None:
            check_resources()
        outcome: MatchOutcome = None
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
                    expected_evidence=evidence
                    or MatchEvidence(scope_sha256=source_scope, candidates={}),
                    instruction_text=unresolved_text,
                )
            if not persisted:
                raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
        else:
            checkpoint = MatchedInstruction(
                instruction_index=instruction_index,
                instruction=instruction,
                candidates=candidates,
                match=match,
                evidence=evidence,
            )
            with _session() as db_session:
                persisted = persist_match_checkpoint(
                    db_session,
                    batch_id=batch_id,
                    lease_generation=lease_generation,
                    input_sha256=input_sha256,
                    checkpoint=checkpoint,
                )
            if not persisted:
                raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
            outcome = instruction_index, checkpoint.draft_group_key(heading_articles)
            log("instruction_match_checkpointed", index=instruction_index)
            del checkpoint

        logger.info(
            "Amendment batch=%s collected instruction=%s/%s lease=%s candidates=%s",
            batch_id,
            instruction_index + 1,
            len(instructions),
            lease_generation,
            len(candidates),
        )

        # Only checkpoint references survive the next retrieval/model call.
        return outcome

    def instruction_work_class(index: int) -> str:
        target = parse_amendment_structural_target(instructions[index])
        if target is not None and target.appendix_label:
            return "annex"
        if article_heading_title(instructions[index].instruction_text):
            return "heading"
        return "provision" if target is not None else "general"

    outcomes = (
        instruction_runner(
            match_one, pending_indices, work_class=instruction_work_class
        )
        if instruction_runner is not None
        else map(match_one, pending_indices)
    )
    for outcome in outcomes:
        if outcome is not None:
            index, group_key = outcome
            groups.setdefault(group_key, []).append(index)

    with _session() as db_session:
        batch = get_batch(db_session, batch_id)
        if batch is None:
            raise RuntimeError(f"Amendment batch {batch_id} no longer exists")
        pdf_source = load_batch_pdf_source(db_session, batch)
    del batch
    draft_groups = []
    for key, indices in groups.items():
        if key[0] != "article_heading" or any(
            article_heading_title(instructions[index].instruction_text)
            for index in indices
        ):
            draft_groups.append(indices)
            continue
        # A heading in another file (or an unmatched heading) does not merge
        # otherwise independent operations in this source.
        separate: dict[MatchGroupKey, list[int]] = {}
        for index in indices:
            with _session() as db_session:
                checkpoint = load_match_checkpoint(
                    db_session,
                    batch_id=batch_id,
                    instruction_index=index,
                    input_sha256=input_sha256,
                )
            if checkpoint is None:
                raise RuntimeError("Amendment match source changed before grouping")
            separate.setdefault(checkpoint.group_key, []).append(index)
        draft_groups.extend(separate.values())
    ordered_groups = sorted(draft_groups, key=lambda indices: min(indices))

    def draft_group(group: list[int]) -> None:
        draft_started = monotonic()
        if before_work is not None:
            before_work()
        if check_resources is not None:
            check_resources()
        instruction_indices = sorted(group)
        ordered_group: list[MatchedInstruction] = []
        group_candidates: list[CandidateChunk] = []
        with _session() as db_session:
            for instruction_index in instruction_indices:
                checkpoint = load_match_checkpoint(
                    db_session,
                    batch_id=batch_id,
                    instruction_index=instruction_index,
                    input_sha256=input_sha256,
                )
                if checkpoint is None:
                    raise RuntimeError(
                        "Amendment match source changed before drafting; resume to revalidate"
                    )
                group_candidates = _merge_candidates(
                    group_candidates, checkpoint.candidates
                )
                # Drafting needs each operation and one shared candidate set.
                ordered_group.append(checkpoint.model_copy(update={"candidates": []}))
                del checkpoint

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
        heading_group = any(
            article_heading_title(item.instruction.instruction_text)
            for item in ordered_group
        )
        heading_contexts = []
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
                instruction=ordered_group[0].instruction,
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
                del appendix_rows
            if len(appendix_candidate_ids) > 1:
                for candidate_id in appendix_candidate_ids:
                    context = load_instruction_draft_context(
                        db_session,
                        candidates=group_candidates,
                        instruction=ordered_group[0].instruction,
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
            if heading_group:
                for item in ordered_group:
                    context = load_instruction_draft_context(
                        db_session,
                        candidates=group_candidates,
                        match=item.match,
                        instruction=item.instruction,
                    )
                    if context is None:
                        heading_contexts = []
                        break
                    heading_contexts.append(
                        (item.instruction_index, item.instruction, item.match, context)
                    )
                contexts = [item[3] for item in heading_contexts]
        log(
            "draft_group_started",
            indices=instruction_indices,
            candidates=len(group_candidates),
            old_chunk_id=ordered_group[0].match.old_chunk_id,
            canonical_scope=[
                {
                    "file_id": str(getattr(context, "target_user_file_id", "")),
                    "chunk_id": (getattr(context, "old_chunk_snapshot", {}) or {}).get(
                        "id"
                    ),
                    "target_evidence": getattr(context, "target_evidence", None),
                    "expected_new_article_no": getattr(
                        context, "expected_new_article_no", None
                    ),
                }
                for context in contexts
            ],
            operations=[
                {
                    "instruction_index": item.instruction_index,
                    "kind": (
                        added_subordinate_unit_kind(item.instruction.instruction_text)
                        or "article"
                    )
                    if item.match.old_chunk_id is None
                    else "replace_existing",
                    "target": str(parse_amendment_structural_target(item.instruction)),
                }
                for item in ordered_group
            ],
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
                        instruction_text=f"{item.instruction.instruction_text}\n\nAttention: The target source or parent provision could not be verified unambiguously.",
                    )
                if not persisted:
                    raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
            return

        try:
            draft_llm = (
                get_amendment_analysis_llm() if instruction_runner is not None else llm
            )
            if heading_group:
                proposal = draft_article_heading_group_proposal(
                    draft_llm,
                    items=heading_contexts,
                    reference_date=reference_date,
                    amendment_context=amendment_context,
                    pdf_source=pdf_source,
                )
            elif len(contexts) > 1:
                proposal = draft_multi_chunk_group_proposal(
                    draft_llm,
                    instruction_indices=instruction_indices,
                    instructions=[item.instruction for item in ordered_group],
                    matches=[item.match for item in ordered_group],
                    contexts=contexts,
                    reference_date=reference_date,
                    amendment_context=amendment_context,
                )
            else:
                proposal = draft_instruction_group_proposal(
                    draft_llm,
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
            logger.warning(
                "Amendment batch=%s drafting group=%s failed: %s",
                batch_id,
                instruction_indices,
                type(error).__name__,
            )
            for item in ordered_group:
                with _session() as db_session:
                    persisted = persist_unmatched_checkpoint(
                        db_session,
                        batch_id=batch_id,
                        lease_generation=lease_generation,
                        instruction_index=item.instruction_index,
                        instruction_text=(
                            f"{item.instruction.instruction_text}\n\nAttention: Draft "
                            f"output could not be validated ({type(error).__name__})."
                        ),
                    )
                if not persisted:
                    raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
            return
        except (
            DraftIntegrityError,
            AmendmentDateConflict,
            InsertionOrderError,
        ) as error:
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
            return
        if check_resources is not None:
            check_resources()
        with _session() as db_session:
            persisted = persist_proposal_checkpoint(
                db_session,
                batch_id=batch_id,
                lease_generation=lease_generation,
                proposal=proposal,
            )
        if not persisted:
            raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
        log(
            "draft_group_checkpointed",
            indices=instruction_indices,
            elapsed_seconds=round(monotonic() - draft_started, 3),
        )
        logger.info(
            "Amendment batch=%s processed instruction group=%s lease=%s candidates=%s",
            batch_id,
            instruction_indices,
            lease_generation,
            len(group_candidates),
        )

    if instruction_runner is None:
        for group in ordered_groups:
            draft_group(group)
    else:
        lane_groups: dict[tuple[str, str], list[list[int]]] = {}
        for group in ordered_groups:
            with _session() as db_session:
                checkpoint = load_match_checkpoint(
                    db_session,
                    batch_id=batch_id,
                    instruction_index=group[0],
                    input_sha256=input_sha256,
                )
            if checkpoint is None:
                raise RuntimeError("Amendment match source changed before scheduling")
            lane_groups.setdefault(checkpoint.draft_lane_key(), []).append(group)
        # An unqualified source scope may overlap any article in that file.
        wildcard_files = {
            file_id for file_id, provision in lane_groups if provision == "source"
        }
        lanes: dict[tuple[str, str], list[list[int]]] = {}
        for key, members in lane_groups.items():
            lane = (
                ("unverified", "source")
                if "unverified" in wildcard_files
                else ((key[0], "source") if key[0] in wildcard_files else key)
            )
            lanes.setdefault(lane, []).extend(members)
        work = sorted(
            [(group, lane) for lane, members in lanes.items() for group in members],
            key=lambda item: min(item[0]),
        )

        def draft_item(index: int) -> MatchOutcome:
            draft_group(work[index][0])
            return None

        for _ in instruction_runner(
            draft_item,
            list(range(len(work))),
            concurrency_key=lambda index: work[index][1],
            work_class=lambda index: tuple(
                sorted({instruction_work_class(item) for item in work[index][0]})
            ),
        ):
            pass
    log("proposals_finished", groups=len(ordered_groups))
    if check_resources is not None:
        check_resources()

    if review_groups and before_work is not None:
        before_work()
    run_annex_groups(
        batch_id=batch_id,
        lease_generation=lease_generation,
        groups=review_groups,
        processed_indices=processed_instruction_indices,
        reference_date=reference_date,
        llm=llm,
    )
    with _session() as db_session:
        log("batch_analyzed")
        if not mark_batch_analyzed(
            db_session,
            batch_id=batch_id,
            lease_generation=lease_generation,
        ):
            raise RuntimeError(f"Amendment batch {batch_id} lost its lease")
