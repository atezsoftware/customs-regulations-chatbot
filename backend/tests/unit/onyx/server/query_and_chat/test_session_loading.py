from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource, MessageType
from onyx.db.models import ChatMessage, SearchDoc
from onyx.server.query_and_chat.session_loading import (
    translate_assistant_message_to_packets,
)
from onyx.server.query_and_chat.streaming_models import CitationInfo


def test_reloaded_citation_packet_preserves_exact_source_identity() -> None:
    chat_message = cast(
        ChatMessage,
        SimpleNamespace(
            id=10,
            message_type=MessageType.ASSISTANT,
            tool_calls=[],
            citations={1: 77},
            search_docs=[],
            reasoning_tokens=None,
            message="Answer [[1]]()",
        ),
    )
    saved_search_doc = cast(
        SearchDoc,
        SimpleNamespace(
            document_id="customs-law",
            chunk_ind=46,
            semantic_id="Gümrük Kanunu — MADDE 46",
            source_type=DocumentSource.USER_FILE,
        ),
    )

    with patch(
        "onyx.server.query_and_chat.session_loading.get_db_search_doc_by_id",
        return_value=saved_search_doc,
    ):
        packets = translate_assistant_message_to_packets(
            chat_message,
            cast(Session, MagicMock()),
        )

    citation_packets = [
        packet.obj for packet in packets if isinstance(packet.obj, CitationInfo)
    ]
    assert citation_packets == [
        CitationInfo(
            citation_number=1,
            document_id="customs-law",
            chunk_ind=46,
            semantic_identifier="Gümrük Kanunu — MADDE 46",
            source_type=DocumentSource.USER_FILE,
        )
    ]
