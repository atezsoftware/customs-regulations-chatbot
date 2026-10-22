"""External API JSON and replay share the same saved citation identities."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from onyx.db.chat import (
    load_saved_citation_documents,
    saved_message_citation_info,
    translate_db_message_to_chat_message_detail,
)
from onyx.db.models import ChatMessage
from onyx.server.query_and_chat.session_loading import (
    translate_assistant_message_to_packets,
)
from onyx.server.query_and_chat.streaming_models import CitationInfo
from tests.unit.onyx.db.test_asv3_citation_details import fixture_message


def test_json_history_and_replay_keep_241_and_242_without_rich_enrichment() -> None:
    message, _ = fixture_message("asv3_tuned")
    message.message = "First [5], then [1]."
    for document, article in zip(message.search_docs, ("241", "242"), strict=True):
        document.doc_metadata = {
            **document.doc_metadata,
            "regulatory_heading_path": ["Gümrük Kanunu", "MADDE " + article],
        }
    detail = translate_db_message_to_chat_message_detail(message)
    replay = [
        part.obj
        for part in translate_assistant_message_to_packets(
            message, cast(Session, MagicMock())
        )
        if isinstance(part.obj, CitationInfo)
    ]
    assert detail.citation_info == replay
    assert [row.citation_number for row in detail.citation_info] == [5, 1]
    assert [row.chunk_ind for row in detail.citation_info] == [8, 7]
    assert [row.reference for row in detail.citation_info] == [
        "Gümrük Kanunu > MADDE 242",
        "Gümrük Kanunu > MADDE 241",
    ]
    assert len({row.document_id for row in detail.citation_info}) == 1
    assert all(
        row.text is None and row.display_title is None for row in detail.citation_info
    )
    assert (
        translate_db_message_to_chat_message_detail(
            message, remove_doc_content=True
        ).citation_info
        == []
    )


def test_all_68_saved_targets_and_two_numbers_for_one_chunk_survive() -> None:
    template, _ = fixture_message()
    document = template.search_docs[0]
    documents = {
        number: SimpleNamespace(
            id=number,
            document_id=document.document_id,
            chunk_ind=number,
            semantic_id=document.semantic_id,
            source_type=document.source_type,
            doc_metadata={"regulatory_heading_path": [f"MADDE {number}"]},
            link=None,
        )
        for number in range(1, 69)
    }
    message = cast(
        ChatMessage,
        SimpleNamespace(
            id=template.id,
            publication_read=None,
            citations={**{str(number): number for number in documents}, "70": 1},
            search_docs=list(documents.values()),
            tool_calls=[],
            message=" ".join(f"[{number}]" for number in range(68, 0, -1)) + " [70]",
        ),
    )
    result = saved_message_citation_info(message)
    assert [row.citation_number for row in result] == [*range(68, 0, -1), 70]
    assert result[-1].document_id == result[-2].document_id
    assert result[-1].chunk_ind == result[-2].chunk_ind == 1
    overrides = [
        CitationInfo(
            citation_number=number,
            document_id=document.document_id,
            chunk_ind=number,
            chunk_id=f"rc-{number}",
            display_title="One official title",
            text=f"Exact original {number}\n",
        )
        for number in documents
    ]
    enriched = saved_message_citation_info(message, overrides=overrides)
    assert len(enriched) == 69
    assert all(item.reference == f"MADDE {item.chunk_ind}" for item in enriched)
    assert [item.text for item in enriched[:-1]] == [
        f"Exact original {number}\n" for number in range(68, 0, -1)
    ]


def test_missing_associations_use_one_read_and_match_json_replay_targets() -> None:
    message, _ = fixture_message("asv3_tuned")
    detached_document = message.search_docs.pop()
    session = MagicMock()
    session.scalars.return_value = [detached_document]
    documents = load_saved_citation_documents(session, [message])
    detail = translate_db_message_to_chat_message_detail(
        message, citation_documents=documents
    )
    replay = [
        part.obj
        for part in translate_assistant_message_to_packets(
            message, session, citation_documents=documents
        )
        if isinstance(part.obj, CitationInfo)
    ]
    assert detail.citation_info == replay
    assert [item.citation_number for item in detail.citation_info] == [1, 5]
    assert detail.citations == {
        1: message.search_docs[0].document_id,
        5: detached_document.document_id,
    }
    assert len(message.search_docs) == 1
    session.scalars.assert_called_once()
    session.commit.assert_not_called()


@pytest.mark.parametrize("text", ["", " \n\t ", "Exact original\n"])
def test_rich_override_preserves_supplied_text_and_adds_only_recorded_location(
    text: str,
) -> None:
    message, _ = fixture_message()
    document = message.search_docs[0]
    document.doc_metadata["regulatory_heading_path"] = "MADDE 241"
    override = CitationInfo(
        citation_number=1,
        document_id=document.document_id,
        chunk_ind=document.chunk_ind,
        chunk_id="original-id",
        text=text,
        display_title="Official title",
    )
    result = saved_message_citation_info(message, overrides=[override])
    assert result[0].text == text and result[0].chunk_id == "original-id"
    assert (
        result[0].display_title == "Official title"
        and result[0].reference == "MADDE 241"
    )
    wrong = override.model_copy(update={"chunk_ind": 999})
    assert saved_message_citation_info(message, overrides=[wrong])[0].text is None


@pytest.mark.parametrize("prefix", ["notice:", "catalog:"])
def test_source_notices_do_not_become_saved_citations(prefix: str) -> None:
    message, _ = fixture_message()
    message.search_docs[0].document_id = prefix + "metadata-only"
    assert [row.citation_number for row in saved_message_citation_info(message)] == [5]
