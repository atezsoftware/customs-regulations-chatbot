"""Current-access-fenced citation presentation for owned ASv3 history."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy.orm import Session

from onyx.asv3.citation_identity import saved_item_from_checkpoint
from onyx.asv3.citation_presentation import (
    citation_presentation_enabled,
    evidence_citation_presentation,
)
from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import EvidenceItem, RunContext, RunStopped
from onyx.asv3.workflow_variant import ASV3_STANDARD_VARIANT
from onyx.chat.citation_utils import extract_citation_order_from_text
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusScopeUnavailable, find_sources
from onyx.db.asv3_runs import ASV3_CHECKPOINT_TOOL_ID, decode_asv3_checkpoint
from onyx.db.models import ChatMessage, User
from onyx.db.regulatory_chat_reads import message_publication_available
from onyx.error_handling.exceptions import OnyxError
from onyx.server.query_and_chat.streaming_models import CitationInfo
from onyx.utils.logger import setup_logger

logger = setup_logger()


@dataclass
class CitationHistoryGroup:
    filters: IndexFilters
    messages: list[tuple[ChatMessage, dict[int, EvidenceItem]]] = field(
        default_factory=list
    )


def _eligible_checkpoint(message: ChatMessage) -> dict[str, JsonValue] | None:
    for tool in message.tool_calls or []:
        if tool.tool_id != ASV3_CHECKPOINT_TOOL_ID or not tool.tool_call_response:
            continue
        checkpoint = decode_asv3_checkpoint(tool.tool_call_response)
        variant = checkpoint.get("asv3_workflow_variant", ASV3_STANDARD_VARIANT)
        profile = checkpoint.get("research_profile")
        parallel = checkpoint.get("parallel_research")
        if (
            isinstance(variant, str)
            and isinstance(profile, str)
            and isinstance(parallel, bool)
            and citation_presentation_enabled(variant, profile, parallel)
        ):
            return checkpoint
    return None


def _retained_citations(
    message: ChatMessage, checkpoint: dict[str, JsonValue]
) -> tuple[dict[int, EvidenceItem], IndexFilters]:
    docs = {doc.id: doc for doc in message.search_docs}
    items: dict[int, EvidenceItem] = {}
    filters = IndexFilters.model_validate(checkpoint.get("scope"))
    for raw_number, doc_id in (message.citations or {}).items():
        # JSON object keys are strings after a database round trip.
        number = int(raw_number)
        if number in items:
            raise ValueError("Duplicate saved citation number")
        document = docs.get(doc_id)
        if document is None:
            raise ValueError("Saved citation has no matching search document")
        item, _ = saved_item_from_checkpoint(checkpoint, number)
        assert item.search_doc is not None
        if (
            item.source_id != document.document_id
            or item.search_doc.chunk_ind != document.chunk_ind
            or (
                item.chunk_id is not None
                and (document.doc_metadata or {}).get("regulatory_chunk_id")
                != item.chunk_id
            )
        ):
            raise ValueError("Saved citation differs from its original evidence")
        items[number] = item
    return items, filters


def _current_titles(
    items: Sequence[EvidenceItem],
    *,
    user: User,
    filters: IndexFilters,
    db_session: Session,
) -> dict[str, dict[str, JsonValue] | None]:
    source_ids = sorted({UUID(item.source_id) for item in items})
    annotations: dict[str, dict[str, JsonValue] | None] = {}
    for offset in range(0, len(source_ids), 100):
        selected = tuple(source_ids[offset : offset + 100])
        sources, _ = find_sources(
            db_session,
            user=user,
            filters=filters,
            source_ids=selected,
            limit=len(selected),
        )
        if {source.id for source in sources} != set(selected):
            raise PermissionError("Citation source is no longer accessible")
        annotations.update(
            {
                str(source.id): source.legistation_metadata.model_dump(mode="json")
                if source.legistation_metadata is not None
                else None
                for source in sources
            }
        )
    return annotations


def _present_citations(
    message: ChatMessage,
    items: dict[int, EvidenceItem],
    annotations: dict[str, dict[str, JsonValue] | None],
) -> list[CitationInfo]:
    citations: list[CitationInfo] = []
    for number, item in items.items():
        presented = item.model_copy(
            update={
                "metadata": {
                    **item.metadata,
                    "legistation_metadata": annotations[item.source_id],
                }
            }
        )
        presentation = evidence_citation_presentation(presented)
        document = item.search_doc
        assert document is not None
        citations.append(
            CitationInfo(
                citation_number=number,
                document_id=item.source_id,
                chunk_ind=document.chunk_ind,
                semantic_identifier=document.semantic_identifier,
                source_type=document.source_type,
                preview_url=f"/api/asv3/citation/{message.id}/{number}",
                display_title=presentation.display_title,
                chunk_id=presentation.chunk_id,
                text=presentation.text,
            )
        )
    order = {
        number: index
        for index, number in enumerate(
            extract_citation_order_from_text(message.message or "")
        )
    }
    citations.sort(key=lambda item: order.get(item.citation_number, float("inf")))
    return citations


def owned_asv3_citation_info(
    messages: Sequence[ChatMessage],
    *,
    user: User,
    session_owner_id: UUID | None,
    db_session: Session,
) -> dict[int, list[CitationInfo]]:
    """Never add retained text to a shared/anonymous viewer's legacy response."""
    if user.id is None or user.id != session_owner_id:
        return {}
    result: dict[int, list[CitationInfo]] = {}
    groups: dict[str, CitationHistoryGroup] = {}
    for message in messages:
        if not message.citations or not message_publication_available(message):
            continue
        try:
            checkpoint = _eligible_checkpoint(message)
            if checkpoint is None:
                continue
            items, filters = _retained_citations(message, checkpoint)
            group = groups.setdefault(
                filters.model_dump_json(), CitationHistoryGroup(filters)
            )
            group.messages.append((message, items))
        except (PermissionError, ValueError):
            logger.warning(
                "ASv3 inline citation unavailable for message %s", message.id
            )

    for group in groups.values():
        filters = group.filters
        all_items = [item for _, items in group.messages for item in items.values()]
        try:
            context = RunContext(
                scope=filters.model_dump(mode="json"), timeout_seconds=30
            )
            # Select the existing batched canonical validator, not a research mode.
            context.services["asv3_exp_cropped"] = True
            CorpusBroker(user, filters).revalidate_evidence(all_items, context)
            annotations = _current_titles(
                all_items, user=user, filters=filters, db_session=db_session
            )
            for message, items in group.messages:
                result[message.id] = _present_citations(message, items, annotations)
        except (
            PermissionError,
            ValueError,
            CorpusScopeUnavailable,
            TimeoutError,
            RunStopped,
            OnyxError,
        ):
            # Optional enrichment must neither expose unfenced text nor break history.
            logger.warning(
                "ASv3 inline citations unavailable for messages %s",
                [message.id for message, _ in group.messages],
            )
    return result
