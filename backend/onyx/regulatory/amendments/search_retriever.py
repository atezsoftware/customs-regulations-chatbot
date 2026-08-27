"""Retrieve amendment targets through Onyx's production SearchTool pipeline."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
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
from onyx.db.regulatory_chunks import get_active_chunks_by_ids
from onyx.db.search_settings import get_current_search_settings
from onyx.db.tools import get_tools
from onyx.db.users import fetch_user_by_id
from onyx.document_index.factory import get_default_document_index
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS, SEARCH_TOOL_ID
from onyx.tools.models import ChatMinimalTextMessage, SearchToolOverrideKwargs
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MAX_AMENDMENT_CANDIDATES = 8
SearchToolFactory = Callable[[], SearchTool]
CanonicalCandidateLoader = Callable[[Sequence[str]], Mapping[str, CandidateChunk]]


class AmendmentSearchRetriever:
    """A fresh SearchTool cycle for each focused amendment query."""

    def __init__(
        self,
        *,
        search_tool_factory: SearchToolFactory,
        canonical_candidate_loader: CanonicalCandidateLoader,
        allowed_user_file_ids: Sequence[UUID],
    ) -> None:
        self._search_tool_factory = search_tool_factory
        self._canonical_candidate_loader = canonical_candidate_loader
        self._allowed_user_file_ids = {
            str(user_file_id) for user_file_id in allowed_user_file_ids
        }

    def search(
        self,
        instruction: AmendmentInstruction,
        *,
        recovery: bool = False,
    ) -> list[CandidateChunk]:
        initial_query = (
            instruction.search_query or instruction.instruction_text
        ).strip()[:REGULATORY_MAX_SEARCH_QUERY_CHARS]
        recovery_query = (instruction.recovery_query or "").strip()
        if recovery:
            if (
                not recovery_query
                or recovery_query.casefold() == initial_query.casefold()
            ):
                return []
            query = recovery_query
        else:
            query = initial_query

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
                skip_query_expansion=recovery,
                num_hits=_MAX_AMENDMENT_CANDIDATES,
                max_llm_chunks=_MAX_AMENDMENT_CANDIDATES,
            ),
            queries=[query],
            search_mode="hybrid",
            source_anchors=source_anchors,
        )
        rich_response = response.rich_response
        if not isinstance(rich_response, SearchDocsResponse):
            logger.warning(
                "Amendment SearchTool retrieval returned no document response phase=%s",
                "recovery" if recovery else "initial",
            )
            return []

        ranked_docs = rich_response.displayed_docs or rich_response.search_docs
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
            if len(candidates) == _MAX_AMENDMENT_CANDIDATES:
                break

        logger.info(
            "Amendment SearchTool retrieval phase=%s returned=%s in_scope=%s",
            "recovery" if recovery else "initial",
            len(ranked_docs),
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
            rows = get_active_chunks_by_ids(canonical_session, chunk_ids)
            return {
                chunk_id: CandidateChunk(
                    chunk_id=chunk_id,
                    user_file_id=str(row.user_file_id),
                    text=row.text,
                    metadata={
                        **row.chunk_metadata,
                        "heading_path": list(row.heading_path),
                    },
                )
                for chunk_id, row in rows.items()
            }

    return AmendmentSearchRetriever(
        search_tool_factory=search_tool_factory,
        canonical_candidate_loader=canonical_candidate_loader,
        allowed_user_file_ids=user_file_ids,
    )
