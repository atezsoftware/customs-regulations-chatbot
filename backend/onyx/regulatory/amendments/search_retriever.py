"""Retrieve amendment targets through Onyx's production SearchTool pipeline."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
from onyx.db.document_set import get_document_set_by_id
from onyx.db.engine.sql_engine import get_session_with_current_tenant
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
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    AmendmentStructuralTarget,
    amended_body,
    canonical_structural_query_anchor,
    parse_amendment_structural_target,
    source_identity_distinguishing_tokens,
    source_identity_matches,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS, SEARCH_TOOL_ID
from onyx.tools.models import ChatMinimalTextMessage, SearchToolOverrideKwargs
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.search.search_utils import (
    weighted_reciprocal_rank_fusion,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MAX_AMENDMENT_CANDIDATES = 16
_MAX_LANE_HITS = 10
# An amendment quotes the wording it replaces; that literal is the strongest
# retrieval anchor available when the rest of the sentence describes new text.
_QUOTED_SOURCE_WORDING_RE = re.compile(
    r"[\"“]([^\"”]{3,160})[\"”]\s*"
    r"(?:ibaresi|ibareleri|ifadesi|ifadeleri|kelimesi|hükmü|gtip|gt[ıi]p)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _RetrievalLane:
    """One independent retrieval attempt for a single amendment instruction."""

    query: str
    search_mode: str
    weight: float


def _bounded_query(value: str) -> str:
    return " ".join(value.split())[:REGULATORY_MAX_SEARCH_QUERY_CHARS].strip()


def _instruction_lanes(
    instruction: AmendmentInstruction,
    target: AmendmentStructuralTarget | None,
) -> list[_RetrievalLane]:
    """Describe the amended provision from several independent angles.

    A single query built from the raw instruction is dominated by the *new*
    wording, which by definition is absent from the indexed corpus. Splitting
    the instruction into a structural anchor, the quoted wording it replaces,
    and the planned semantic question gives each signal its own ranked list, so
    a target that only one of them can reach still surfaces.
    """

    source = " ".join((instruction.target_source or "").split())
    lanes: list[_RetrievalLane] = []
    seen: set[str] = set()

    def add(query: str, *, search_mode: str, weight: float) -> None:
        bounded = _bounded_query(query)
        if not bounded or bounded.casefold() in seen:
            return
        seen.add(bounded.casefold())
        lanes.append(
            _RetrievalLane(query=bounded, search_mode=search_mode, weight=weight)
        )

    anchor = canonical_structural_query_anchor(target)
    if anchor:
        add(
            f"{source} {anchor}" if source else anchor,
            search_mode="keyword",
            weight=1.6,
        )
    for quoted in list(
        dict.fromkeys(_QUOTED_SOURCE_WORDING_RE.findall(instruction.instruction_text))
    )[:2]:
        add(quoted, search_mode="full_text", weight=1.4)
    planned = (instruction.search_query or "").strip()
    add(
        planned or amended_body(instruction.instruction_text),
        search_mode="hybrid",
        weight=1.0,
    )
    return lanes


SearchToolFactory = Callable[[], SearchTool]
CanonicalCandidateLoader = Callable[[Sequence[str]], Mapping[str, CandidateChunk]]
StructuralCandidateLoader = Callable[[AmendmentInstruction], Sequence[CandidateChunk]]


class AmendmentSearchRetriever:
    """A fresh SearchTool cycle for each focused amendment query."""

    def __init__(
        self,
        *,
        search_tool_factory: SearchToolFactory,
        canonical_candidate_loader: CanonicalCandidateLoader,
        structural_candidate_loader: StructuralCandidateLoader | None = None,
        allowed_user_file_ids: Sequence[UUID],
    ) -> None:
        self._search_tool_factory = search_tool_factory
        self._canonical_candidate_loader = canonical_candidate_loader
        self._structural_candidate_loader = structural_candidate_loader
        self._allowed_user_file_ids = {
            str(user_file_id) for user_file_id in allowed_user_file_ids
        }

    def _run_lane(
        self,
        instruction: AmendmentInstruction,
        lane: _RetrievalLane,
        *,
        skip_query_expansion: bool,
    ) -> list[CandidateChunk]:
        """Execute one lane and return its in-scope candidates in rank order."""

        source_anchors = (
            [instruction.target_source.strip()]
            if instruction.target_source and instruction.target_source.strip()
            else []
        )
        response = self._search_tool_factory().run(
            placement=Placement(turn_index=0),
            override_kwargs=SearchToolOverrideKwargs(
                starting_citation_num=1,
                original_query=lane.query,
                message_history=[
                    ChatMinimalTextMessage(
                        message=lane.query,
                        message_type=MessageType.USER,
                    )
                ],
                skip_query_expansion=skip_query_expansion,
                num_hits=_MAX_LANE_HITS,
                max_llm_chunks=_MAX_LANE_HITS,
            ),
            queries=[lane.query],
            search_mode=lane.search_mode,
            source_anchors=source_anchors,
        )
        rich_response = response.rich_response
        if not isinstance(rich_response, SearchDocsResponse):
            logger.warning(
                "Amendment SearchTool lane returned no document response mode=%s",
                lane.search_mode,
            )
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
        for chunk_id, search_doc in docs_by_chunk_id.items():
            candidate = canonical_candidates.get(chunk_id)
            if (
                candidate is None
                or candidate.user_file_id not in self._allowed_user_file_ids
            ):
                continue
            candidates.append(
                replace(candidate, source_name=search_doc.semantic_identifier)
            )
        return candidates

    def search(
        self,
        instruction: AmendmentInstruction,
        *,
        recovery: bool = False,
    ) -> list[CandidateChunk]:
        """Fuse several independent lanes into one bounded candidate list."""

        target = parse_amendment_structural_target(instruction)
        if recovery:
            recovery_query = _bounded_query(instruction.recovery_query or "")
            planned = _bounded_query(
                instruction.search_query or instruction.instruction_text
            )
            if not recovery_query or recovery_query.casefold() == planned.casefold():
                return []
            lanes = [
                _RetrievalLane(query=recovery_query, search_mode="hybrid", weight=1.0)
            ]
        else:
            lanes = _instruction_lanes(instruction, target)
        if not lanes:
            return []

        ranked_lanes: list[list[CandidateChunk]] = []
        weights: list[float] = []
        merged: dict[str, CandidateChunk] = {}
        for lane in lanes:
            lane_candidates = self._run_lane(
                instruction, lane, skip_query_expansion=recovery
            )
            if not lane_candidates:
                continue
            for candidate in lane_candidates:
                merged.setdefault(candidate.chunk_id, candidate)
            ranked_lanes.append(lane_candidates)
            weights.append(lane.weight)

        fused = (
            weighted_reciprocal_rank_fusion(
                ranked_lanes, weights, lambda candidate: candidate.chunk_id
            )
            if ranked_lanes
            else []
        )
        candidates = [merged[candidate.chunk_id] for candidate in fused][
            :_MAX_AMENDMENT_CANDIDATES
        ]
        seen_candidate_ids = {candidate.chunk_id for candidate in candidates}

        # The exact structural target is authoritative even when no lexical or
        # semantic lane reached it, which is the norm for an instruction whose
        # body is entirely new text. Expansion still requires an instrument-
        # specific source identity: without one, "article 3" names a provision
        # in every instrument the batch covers.
        structural_source_tokens = source_identity_distinguishing_tokens(
            instruction.target_source
        )
        if self._structural_candidate_loader is not None and structural_source_tokens:
            for candidate in self._structural_candidate_loader(instruction):
                if (
                    candidate.chunk_id in seen_candidate_ids
                    or candidate.user_file_id not in self._allowed_user_file_ids
                ):
                    continue
                candidates.append(candidate)
                seen_candidate_ids.add(candidate.chunk_id)

        logger.info(
            "Amendment retrieval phase=%s lanes=%s target=%s candidates=%s",
            "recovery" if recovery else "initial",
            len(ranked_lanes),
            target,
            len(candidates),
        )
        return candidates


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
    filters = BaseFilters(
        document_set=[document_set.name],
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

        def as_candidate(match: RegulatoryChunkStructuralMatch) -> CandidateChunk:
            return CandidateChunk(
                chunk_id=match.chunk.id,
                user_file_id=str(match.chunk.user_file_id),
                text=match.chunk.text,
                source_name=match.source_name,
                metadata={
                    **match.chunk.chunk_metadata,
                    "heading_path": list(match.chunk.heading_path),
                },
                structured_match=True,
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

        with get_session_with_current_tenant() as structural_session:
            matches: list[RegulatoryChunkStructuralMatch] = []
            seen_chunk_ids: set[str] = set()
            # The narrow reference identifies the amended unit; the article-level
            # pass supplies its siblings, which is the only way an instruction
            # that adds a new paragraph or clause can be positioned at all.
            for clause_label, paragraph_no in (
                (target.clause_label, target.paragraph_no),
                (None, None),
            ):
                if target.article_no is None and (clause_label, paragraph_no) == (
                    None,
                    None,
                ):
                    continue
                for match in get_active_chunks_by_structural_reference(
                    structural_session,
                    user_file_ids=user_file_ids,
                    article_no=target.article_no,
                    clause_label=clause_label,
                    appendix_label=target.appendix_label,
                    source_name_hint=instruction.target_source,
                    source_name_tokens=source_tokens,
                    paragraph_no=paragraph_no,
                    limit=32 if target.appendix_label is not None else 16,
                ):
                    if match.chunk.id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(match.chunk.id)
                    matches.append(match)
                if (clause_label, paragraph_no) == (None, None):
                    break
            candidates = [
                as_candidate(match)
                for match in matches
                if source_identity_matches(instruction.target_source, match.source_name)
            ]
        return sorted(
            candidates,
            key=lambda candidate: (-candidate.source_score, candidate.chunk_id),
        )

    return AmendmentSearchRetriever(
        search_tool_factory=search_tool_factory,
        canonical_candidate_loader=canonical_candidate_loader,
        structural_candidate_loader=structural_candidate_loader,
        allowed_user_file_ids=user_file_ids,
    )
