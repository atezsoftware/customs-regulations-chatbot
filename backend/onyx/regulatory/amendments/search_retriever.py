"""Retrieve amendment targets through Onyx's production SearchTool pipeline."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from difflib import SequenceMatcher
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.chat.emitter import NullEmitter
from onyx.configs.constants import MessageType
from onyx.context.search.models import (
    BaseFilters,
    PersonaSearchInfo,
    SearchDoc,
    SearchDocsResponse,
)
from onyx.context.search.retrieval.search_runner import (
    regulatory_embedding_unavailable,
)
from onyx.db.document_set import get_document_set_by_id
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_amendment_targets import (
    load_amendment_source_chunks,
    load_amendment_source_identities,
)
from onyx.db.regulatory_chunks import (
    RegulatoryChunkStructuralMatch,
    get_active_chunks_by_structural_reference,
    get_current_chunks_by_ids,
)
from onyx.db.search_settings import get_current_search_settings
from onyx.db.tools import get_tools
from onyx.db.users import fetch_user_by_id
from onyx.document_index.factory import get_default_document_index
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.new_provision_policy import (
    added_subordinate_unit_kind,
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    AmendmentStructuralTarget,
    named_law_number,
    parse_amendment_structural_target,
    source_identity_distinguishing_tokens,
    source_identity_matches,
)
from onyx.regulatory.amendments.target_scope import article_scope_candidates
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS, SEARCH_TOOL_ID
from onyx.tools.models import ChatMinimalTextMessage, SearchToolOverrideKwargs
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MAX_AMENDMENT_CANDIDATES = 12
_MAX_SEARCH_CANDIDATES = 8
_MAX_STRUCTURAL_CANDIDATES = 6


def _bounded_query(value: str) -> str:
    return " ".join(value.split())[:REGULATORY_MAX_SEARCH_QUERY_CHARS].strip()


SearchToolFactory = Callable[[], SearchTool]
CanonicalCandidateLoader = Callable[[Sequence[str]], Mapping[str, CandidateChunk]]
StructuralCandidateLoader = Callable[[AmendmentInstruction], Sequence[CandidateChunk]]
SourceFileLoader = Callable[[AmendmentInstruction], Sequence[str] | None]


def _structural_lookup_scopes(
    target: AmendmentStructuralTarget,
) -> tuple[tuple[str | None, str | None], ...]:
    if target.appendix_label is not None:
        return ((None, None),)
    narrow = (target.clause_label, target.paragraph_no)
    return (narrow,) if narrow == (None, None) else (narrow, (None, None))


class AmendmentSearchRetriever:
    """A fresh SearchTool cycle for each focused amendment query."""

    def __init__(
        self,
        *,
        search_tool_factory: SearchToolFactory,
        canonical_candidate_loader: CanonicalCandidateLoader,
        structural_candidate_loader: StructuralCandidateLoader | None = None,
        source_file_loader: SourceFileLoader | None = None,
        allowed_user_file_ids: Sequence[UUID],
    ) -> None:
        self._search_tool_factory = search_tool_factory
        self.last_query_stats: dict[str, object] = {}
        self.query_stats: list[dict[str, object]] = []
        self._canonical_candidate_loader = canonical_candidate_loader
        self._structural_candidate_loader = structural_candidate_loader
        self._source_file_loader = source_file_loader
        self.last_attention: str | None = None
        self._allowed_user_file_ids = {
            str(user_file_id) for user_file_id in allowed_user_file_ids
        }

    def _run_query(
        self,
        instruction: AmendmentInstruction,
        query: str,
        *,
        skip_query_expansion: bool,
    ) -> list[CandidateChunk]:
        """Run one focused query and return its in-scope candidates in rank order.

        Each stage that can empty the result set is counted separately: an index
        that returned nothing, rows without a canonical id, ids that no longer
        resolve, and rows outside the batch are four different faults that look
        identical from a candidate count alone.
        """

        source_anchors = (
            [instruction.target_source.strip()]
            if instruction.target_source and instruction.target_source.strip()
            else []
        )
        response = self._search_tool_factory().run(
            placement=Placement(turn_index=0),
            override_kwargs=SearchToolOverrideKwargs(
                starting_citation_num=1,
                original_query=query,
                message_history=[
                    ChatMinimalTextMessage(
                        message=query,
                        message_type=MessageType.USER,
                    )
                ],
                skip_query_expansion=skip_query_expansion,
                num_hits=_MAX_SEARCH_CANDIDATES,
                max_llm_chunks=_MAX_SEARCH_CANDIDATES,
            ),
            queries=[query],
            search_mode="hybrid",
            source_anchors=source_anchors,
        )
        rich_response = response.rich_response
        if not isinstance(rich_response, SearchDocsResponse):
            logger.warning("Amendment SearchTool returned no document response")
            return []
        ranked_docs: Sequence[SearchDoc] = (
            rich_response.displayed_docs or rich_response.search_docs
        )
        docs_by_chunk_id: dict[str, SearchDoc] = {}
        for search_doc in ranked_docs:
            raw_chunk_id = search_doc.metadata.get("regulatory_chunk_id")
            chunk_id = (
                raw_chunk_id.strip()
                if isinstance(raw_chunk_id, str) and raw_chunk_id.strip()
                else None
            )
            if chunk_id is None or chunk_id in docs_by_chunk_id:
                continue
            docs_by_chunk_id[chunk_id] = search_doc
        canonical_candidates = self._canonical_candidate_loader(list(docs_by_chunk_id))
        candidates: list[CandidateChunk] = []
        out_of_scope = 0
        for chunk_id, search_doc in docs_by_chunk_id.items():
            candidate = canonical_candidates.get(chunk_id)
            if candidate is None:
                continue
            if candidate.user_file_id not in self._allowed_user_file_ids:
                out_of_scope += 1
                continue
            candidates.append(
                replace(candidate, source_name=search_doc.semantic_identifier)
            )
        logger.info(
            "Amendment query docs=%s with_chunk_id=%s resolved=%s "
            "out_of_scope=%s kept=%s query=%r",
            len(ranked_docs),
            len(docs_by_chunk_id),
            len(canonical_candidates),
            out_of_scope,
            len(candidates),
            query[:120],
        )
        self.last_query_stats = {
            "docs": len(ranked_docs),
            "lexical_only": regulatory_embedding_unavailable(),
            "with_chunk_id": len(docs_by_chunk_id),
            "resolved": len(canonical_candidates),
            "out_of_scope": out_of_scope,
            "kept": len(candidates),
        }
        return candidates

    def search(
        self,
        instruction: AmendmentInstruction,
        *,
        recovery: bool = False,
    ) -> list[CandidateChunk]:
        """Search once, then add the provision the instruction names outright."""

        initial_query = _bounded_query(
            instruction.search_query or instruction.instruction_text
        )
        if recovery:
            query = _bounded_query(instruction.recovery_query or "")
            if not query or query.casefold() == initial_query.casefold():
                return []
        else:
            query = initial_query
        if not query:
            return []

        self.last_attention = None
        source_files = (
            self._source_file_loader(instruction) if self._source_file_loader else None
        )
        if source_files is not None and not source_files:
            self.query_stats = []
            self.last_attention = (
                f"Target source '{instruction.target_source}' could not be verified "
                "in this batch's Document Set. References in other documents "
                "cannot replace the source itself."
            )
            return []

        if (
            source_files is not None
            and len(set(source_files)) > 1
            and (
                explicitly_adds_top_level_provision(instruction.instruction_text)
                or added_subordinate_unit_kind(instruction.instruction_text)
            )
        ):
            self.query_stats = []
            self.last_attention = (
                "Multiple target source files or versions were verified. "
                "A new provision requires one unambiguous source; narrow the Document Set."
            )
            return []

        ranked = self._run_query(instruction, query, skip_query_expansion=recovery)
        self.query_stats = [dict(self.last_query_stats)]

        # An amendment describes the text it introduces, not the text it
        # replaces, so the provision it names by article/paragraph/clause is
        # regularly unreachable by wording alone. Expansion still requires an
        # instrument-specific source identity: without one, "article 3" names a
        # provision in every instrument the batch covers.
        structural_source_tokens = source_identity_distinguishing_tokens(
            instruction.target_source
        )
        structural = (
            list(self._structural_candidate_loader(instruction))[
                :_MAX_STRUCTURAL_CANDIDATES
            ]
            if self._structural_candidate_loader is not None
            and structural_source_tokens
            else []
        )

        # The confirming model reads every candidate in full, so the merged list
        # stays bounded: a prompt it cannot read inside its deadline loses every
        # match, not just the weak ones.
        candidates: list[CandidateChunk] = []
        candidate_indexes: dict[str, int] = {}
        for candidate in [
            *[
                item
                for item in structural
                if item.structured_match or item.scope_evidence
            ],
            *ranked,
            *structural,
        ]:
            if candidate.user_file_id not in self._allowed_user_file_ids:
                continue
            if source_files is not None and candidate.user_file_id not in source_files:
                continue
            existing_index = candidate_indexes.get(candidate.chunk_id)
            if existing_index is not None:
                existing = candidates[existing_index]
                if candidate.structured_match or candidate.scope_evidence:
                    candidates[existing_index] = replace(
                        existing,
                        structured_match=existing.structured_match
                        or candidate.structured_match,
                        resolved_article_no=candidate.resolved_article_no
                        or existing.resolved_article_no,
                        scope_evidence=candidate.scope_evidence
                        or existing.scope_evidence,
                        source_score=max(existing.source_score, candidate.source_score),
                        source_name=candidate.source_name or existing.source_name,
                        metadata={**existing.metadata, **candidate.metadata},
                    )
                continue
            candidates.append(
                replace(candidate, source_verified=True)
                if source_files is not None
                else candidate
            )
            candidate_indexes[candidate.chunk_id] = len(candidates) - 1

        logger.info(
            "Amendment retrieval phase=%s ranked=%s structural=%s candidates=%s query=%r",
            "recovery" if recovery else "initial",
            len(ranked),
            len(structural),
            len(candidates),
            query[:120],
        )
        return candidates[:_MAX_AMENDMENT_CANDIDATES]


def build_amendment_search_retriever(
    db_session: Session,
    *,
    document_set_id: int,
    created_by: UUID | None,
    user_file_ids: Sequence[UUID],
    llm: LLM,
) -> AmendmentSearchRetriever:
    """Build a Document Set-scoped SearchTool factory for a durable batch."""

    if created_by is None:
        raise RuntimeError("Amendment batch has no creator for retrieval ACLs")
    user = fetch_user_by_id(db_session, created_by)
    if user is None:
        raise RuntimeError(f"Amendment batch creator {created_by} no longer exists")
    document_set = get_document_set_by_id(db_session, document_set_id)
    if document_set is None:
        raise RuntimeError(f"Document Set {document_set_id} no longer exists")
    search_settings = get_current_search_settings(db_session)
    if search_settings is None:
        raise RuntimeError("No search settings configured for amendment retrieval")
    document_index = get_default_document_index(search_settings, None, db_session)
    tool_id = next(
        (
            tool.id
            for tool in get_tools(db_session)
            if tool.in_code_tool_id == SEARCH_TOOL_ID
        ),
        None,
    )
    if tool_id is None:
        raise RuntimeError("Search tool not found for amendment retrieval")

    persona_search_info = PersonaSearchInfo(
        document_set_names=[],
        search_start_date=None,
        attached_document_ids=[],
        hierarchy_node_ids=[],
    )
    # What Atez Search V2 queries this index with. The Document Set tag is not
    # queried: it is written from the membership a file had when it was last
    # published, so a file republished through another writer can carry a stale
    # tag and vanish from a set-filtered query while staying fully searchable.
    # The update's scope is the batch's own file list — which is this Document
    # Set — and every candidate is checked against it after retrieval, where no
    # publication path can weaken it.
    filters = BaseFilters(
        regulatory_chunks_only=True,
    )

    def search_tool_factory() -> SearchTool:
        return SearchTool(
            tool_id=tool_id,
            emitter=NullEmitter(),
            user=user,
            persona_search_info=persona_search_info,
            llm=llm,
            document_index=document_index,
            user_selected_filters=filters,
            project_id_filter=None,
            persona_id_filter=None,
            bypass_acl=False,
            slack_context=None,
            enable_slack_search=False,
            auto_detect_filters=False,
        )

    def canonical_candidate_loader(
        chunk_ids: Sequence[str],
    ) -> Mapping[str, CandidateChunk]:
        with get_session_with_current_tenant() as canonical_session:
            rows = get_current_chunks_by_ids(canonical_session, chunk_ids)
            return {
                chunk_id: CandidateChunk(
                    chunk_id=row.id,
                    user_file_id=str(row.user_file_id),
                    text=row.text,
                    metadata={
                        **row.chunk_metadata,
                        "heading_path": list(row.heading_path),
                    },
                )
                for chunk_id, row in rows.items()
            }

    source_identities = load_amendment_source_identities(db_session, user_file_ids)
    source_chunks: dict[str, list[CandidateChunk]] = {}

    def source_file_loader(instruction: AmendmentInstruction) -> Sequence[str] | None:
        number = named_law_number(instruction.target_source or "")
        if number is None:
            return None
        files = []
        for source in source_identities:
            filename_number = named_law_number(source.name)
            root_number = named_law_number(source.root_heading)
            if (
                filename_number is not None
                and root_number is not None
                and filename_number != root_number
            ):
                continue
            if (filename_number or root_number) == number:
                files.append(str(source.user_file_id))
        return files

    def structural_candidate_loader(
        instruction: AmendmentInstruction,
    ) -> Sequence[CandidateChunk]:
        target = parse_amendment_structural_target(instruction)
        if target is None:
            return []
        normalized_source = " ".join(
            (instruction.target_source or "").casefold().split()
        )
        source_tokens = source_identity_distinguishing_tokens(instruction.target_source)

        def as_candidate(
            match: RegulatoryChunkStructuralMatch, *, exact: bool = False
        ) -> CandidateChunk:
            return CandidateChunk(
                chunk_id=match.chunk.id,
                user_file_id=str(match.chunk.user_file_id),
                text=match.chunk.text,
                source_name=match.source_name,
                metadata={
                    **match.chunk.chunk_metadata,
                    "heading_path": list(match.chunk.heading_path),
                },
                structured_match=exact,
                source_score=(
                    SequenceMatcher(
                        None,
                        normalized_source,
                        " ".join(match.source_name.casefold().split()),
                    ).ratio()
                    if normalized_source
                    else 0.0
                ),
            )

        verified_files = source_file_loader(instruction)
        scoped_ids = (
            [UUID(value) for value in verified_files]
            if verified_files is not None
            else user_file_ids
        )
        candidates: list[CandidateChunk] = []
        seen_chunk_ids: set[str] = set()
        with get_session_with_current_tenant() as structural_session:
            for clause_label, paragraph_no in _structural_lookup_scopes(target):
                for match in get_active_chunks_by_structural_reference(
                    structural_session,
                    user_file_ids=scoped_ids,
                    article_no=target.article_no,
                    clause_label=clause_label,
                    appendix_label=target.appendix_label,
                    source_name_hint=instruction.target_source,
                    source_name_tokens=()
                    if verified_files is not None
                    else source_tokens,
                    paragraph_no=paragraph_no,
                    limit=32 if target.appendix_label is not None else 8,
                ):
                    if match.chunk.id in seen_chunk_ids:
                        continue
                    if verified_files is None and not source_identity_matches(
                        instruction.target_source, match.source_name
                    ):
                        continue
                    seen_chunk_ids.add(match.chunk.id)
                    candidates.append(
                        as_candidate(
                            match,
                            exact=(clause_label, paragraph_no)
                            == (target.clause_label, target.paragraph_no),
                        )
                    )
            # Legacy documents can carry stale article metadata. Resolve only
            # within one verified source, with explicit heading boundaries.
            if (
                verified_files is not None
                and len(verified_files) == 1
                and target.article_no is not None
            ):
                file_id = verified_files[0]
                if file_id not in source_chunks:
                    # Keep only the current source; a batch can touch thousands of files.
                    source_chunks.clear()
                    source_chunks[file_id] = [
                        replace(as_candidate(match), source_verified=True)
                        for match in load_amendment_source_chunks(
                            structural_session, UUID(file_id)
                        )
                    ]
                rows = source_chunks[file_id]
                resolved = article_scope_candidates(rows, target.article_no)
                resolved_by_id = {
                    candidate.chunk_id: candidate for candidate in resolved
                }
                candidates = [
                    replace(
                        candidate,
                        resolved_article_no=target.article_no,
                        scope_evidence=resolved_by_id[
                            candidate.chunk_id
                        ].scope_evidence,
                    )
                    if candidate.chunk_id in resolved_by_id
                    else candidate
                    for candidate in candidates
                ]
                candidates.extend(
                    candidate
                    for candidate in resolved
                    if candidate.chunk_id not in seen_chunk_ids
                )
                if not candidates and explicitly_adds_top_level_provision(
                    instruction.instruction_text
                ):
                    candidates = rows[:1]
        # Stable order keeps the narrow named unit ahead of article siblings.
        return candidates

    return AmendmentSearchRetriever(
        search_tool_factory=search_tool_factory,
        canonical_candidate_loader=canonical_candidate_loader,
        structural_candidate_loader=structural_candidate_loader,
        source_file_loader=source_file_loader,
        allowed_user_file_ids=user_file_ids,
    )
