"""Read saved citation identities without treating retrieved documents as citations."""

from collections.abc import Mapping, Sequence
from uuid import UUID
from zlib import error as ZlibError

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from onyx.asv3.citation_identity import saved_item_from_checkpoint
from onyx.asv3.models import EvidenceItem
from onyx.chat.citation_utils import extract_citation_order_from_text
from onyx.configs.constants import MessageType
from onyx.db.asv3_runs import ASV3_CHECKPOINT_TOOL_ID, decode_asv3_checkpoint
from onyx.db.chat import load_saved_citation_documents
from onyx.db.models import ChatMessage, SearchDoc
from onyx.db.regulatory_chat_reads import message_publication_available
from onyx.server.query_and_chat.streaming_models import CitationInfo
from onyx.tools.tool_implementations.web_search.models import WEB_SEARCH_PREFIX


def citation_documents(
    session: Session, messages: Sequence[ChatMessage]
) -> dict[int, SearchDoc]:
    return load_saved_citation_documents(session, messages)


def citation_checkpoint(message: ChatMessage) -> dict[str, JsonValue] | None:
    for tool in message.tool_calls or []:
        if tool.tool_id == ASV3_CHECKPOINT_TOOL_ID and tool.tool_call_response:
            try:
                return decode_asv3_checkpoint(tool.tool_call_response)
            except ZlibError as error:
                raise ValueError("Invalid compressed citation checkpoint") from error
    return None


def retained_citation(
    message: ChatMessage,
    number: int,
    document: SearchDoc,
    *,
    checkpoint: dict[str, JsonValue] | None = None,
) -> tuple[EvidenceItem, dict[str, JsonValue]]:
    checkpoint = checkpoint if checkpoint is not None else citation_checkpoint(message)
    if checkpoint is None:
        raise ValueError("Citation has no retained original")
    evidence = checkpoint.get("evidence")
    records = evidence.get("records") if isinstance(evidence, dict) else None
    if (
        not isinstance(records, list)
        or sum(
            isinstance(row, dict) and row.get("citation") == number for row in records
        )
        != 1
    ):
        raise ValueError("Citation has no unique retained original")
    item, _ = saved_item_from_checkpoint(checkpoint, number)
    assert item.search_doc is not None
    if (
        item.source_id.startswith(("notice:", "catalog:"))
        or item.metadata.get("metadata_only") is True
        or item.source_id != document.document_id
        or item.search_doc.chunk_ind != document.chunk_ind
        or (
            item.chunk_id is not None
            and (document.doc_metadata or {}).get("regulatory_chunk_id")
            != item.chunk_id
        )
    ):
        raise ValueError("Citation differs from its retained original")
    return item, checkpoint


def history_document_metadata(document: SearchDoc) -> dict[str, str | list[str]]:
    metadata = document.doc_metadata or {}
    result: dict[str, str | list[str]] = {}
    title = metadata.get("asv3_source_display_name")
    heading = metadata.get("regulatory_heading_path")
    chunk_id = metadata.get("regulatory_chunk_id")
    if isinstance(title, str):
        result["asv3_source_display_name"] = title
    if isinstance(chunk_id, str):
        result["regulatory_chunk_id"] = chunk_id
    if isinstance(heading, str) or (
        isinstance(heading, list) and all(isinstance(part, str) for part in heading)
    ):
        result["regulatory_heading_path"] = heading
    return result


def is_external_citation(document: SearchDoc) -> bool:
    return bool(document.is_internet) or document.document_id.startswith(
        WEB_SEARCH_PREFIX
    )


def history_citation_info(
    message: ChatMessage, documents: Mapping[int, SearchDoc]
) -> tuple[list[CitationInfo], list[int]]:
    if message.message_type != MessageType.ASSISTANT:
        return [], []
    markers = extract_citation_order_from_text(message.message or "")
    saved = {
        int(number): doc_id
        for number, doc_id in (message.citations or {}).items()
        if str(number).isdigit() and int(number) > 0
    }
    numbers = list(dict.fromkeys([*markers, *saved]))
    if not numbers:
        return [], []
    if not message_publication_available(message):
        return [], numbers
    citations: list[CitationInfo] = []
    unresolved: list[int] = []
    asv3 = any(
        tool.tool_id == ASV3_CHECKPOINT_TOOL_ID for tool in message.tool_calls or []
    )
    try:
        checkpoint = citation_checkpoint(message) if asv3 else None
    except ValueError:
        checkpoint = None
    evidence = checkpoint.get("evidence") if checkpoint else None
    records = evidence.get("records") if isinstance(evidence, dict) else None
    indexed: dict[int, JsonValue] = {}
    duplicates: set[int] = set()
    for record in records if isinstance(records, list) else []:
        number = record.get("citation") if isinstance(record, dict) else None
        if type(number) is int:
            if number in indexed:
                duplicates.add(number)
            indexed[number] = record
    for number in numbers:
        document = documents.get(saved.get(number, -1))
        if document is None or document.document_id.startswith(("notice:", "catalog:")):
            unresolved.append(number)
            continue
        metadata = document.doc_metadata or {}
        if metadata.get("metadata_only") is True:
            unresolved.append(number)
            continue
        heading = history_document_metadata(document).get("regulatory_heading_path")
        reference = (
            heading
            if isinstance(heading, str)
            else " > ".join(heading)
            if isinstance(heading, list)
            else None
        )
        title = metadata.get("asv3_source_display_name")
        chunk_id = metadata.get("regulatory_chunk_id")
        target = CitationInfo(
            citation_number=number,
            document_id=document.document_id,
            chunk_ind=document.chunk_ind,
            semantic_identifier=document.semantic_id,
            source_type=document.source_type,
            display_title=title if isinstance(title, str) else None,
            chunk_id=chunk_id if isinstance(chunk_id, str) else None,
            link=document.link,
            reference=reference,
            resolution_status="unavailable",
        )
        external = is_external_citation(document)
        if asv3:
            try:
                if (
                    checkpoint is None
                    or not isinstance(evidence, dict)
                    or number not in indexed
                    or number in duplicates
                ):
                    raise ValueError("Citation has no unique retained original")
                retained_citation(
                    message,
                    number,
                    document,
                    checkpoint={
                        **checkpoint,
                        "evidence": {**evidence, "records": [indexed[number]]},
                    },
                )
            except ValueError:
                if external:
                    target.resolution_status = "external"
            else:
                target.resolution_status = "retained"
        elif external:
            target.resolution_status = "external"
        elif document.chunk_ind >= 0:
            target.resolution_status = "current"
        if (
            target.resolution_status in {"retained", "current"}
            and message.chat_session_id is not None
        ):
            target.preview_url = (
                f"/api/admin/chat-session-history/{message.chat_session_id}"
                f"/citation/{message.id}/{number}"
            )
        citations.append(target)
    return citations, unresolved


def history_citation_message(
    session: Session, session_id: UUID, message_id: int
) -> ChatMessage:
    message = session.scalar(
        select(ChatMessage)
        .where(ChatMessage.id == message_id, ChatMessage.chat_session_id == session_id)
        .options(
            selectinload(ChatMessage.search_docs), selectinload(ChatMessage.tool_calls)
        )
    )
    if message is None or not message_publication_available(message):
        raise ValueError("Citation message is unavailable")
    return message
