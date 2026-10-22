"""Owned history retains exact targets without weakening the preview fence."""

import copy
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import JsonValue
from sqlalchemy.orm import Session

from onyx.asv3.models import EvidenceItem, RunStopped
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db import asv3_citation_details as details
from onyx.db.asv3_corpus import CorpusSource
from onyx.db.asv3_runs import ASV3_CHECKPOINT_TOOL_ID
from onyx.db.chat import translate_db_message_to_chat_message_detail
from onyx.db.models import ChatMessage, ToolCall, User
from onyx.db.models import SearchDoc as DBSearchDoc
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.legistation_metadata import LegistationMetadata
from onyx.server.query_and_chat.streaming_models import CitationInfo
from tests.unit.onyx.server.query_and_chat.test_pending_chat_session import (
    _USER_ID,
    _get_session,
    _message,
    _source,
)

SOURCE_ID = UUID("a572ced3-13dc-44f9-8742-8d92e85a7913")
TITLE = "2024/9366 Sayılı Karar / Tam Resmî Başlık"


def fixture_message(
    variant: str | None = None,
) -> tuple[ChatMessage, list[EvidenceItem]]:
    items: list[EvidenceItem] = []
    docs: list[DBSearchDoc] = []
    for index, (number, text) in enumerate(
        ((1, "  Birinci özgün hüküm.\n"), (5, "İkinci özgün hüküm.\n\n"))
    ):
        chunk_id = str(UUID(int=100 + index))
        document = SearchDoc(
            document_id=str(SOURCE_ID),
            chunk_ind=7 + index,
            semantic_identifier="KANUNLAR/stored_name.md",
            link=None,
            blurb=text,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": chunk_id},
            match_highlights=[],
        )
        items.append(
            EvidenceItem(
                source_id=str(SOURCE_ID),
                chunk_id=chunk_id,
                text=text,
                search_doc=document,
            )
        )
        saved = _source()
        saved.id = 70 + index
        saved.document_id = str(SOURCE_ID)
        saved.chunk_ind = 7 + index
        saved.semantic_id = document.semantic_identifier
        setattr(saved, "doc_metadata", {"regulatory_chunk_id": chunk_id})
        docs.append(saved)
    checkpoint: dict[str, JsonValue] = {
        "version": 1,
        "research_profile": "normal",
        "parallel_research": False,
        "scope": IndexFilters(access_control_list=None).model_dump(mode="json"),
        "evidence": {
            "version": 1,
            "records": [
                {"citation": number, "item": item.model_dump(mode="json")}
                for number, item in zip((1, 5), items, strict=True)
            ],
        },
        "progress": [],
    }
    if variant is not None:
        checkpoint["asv3_workflow_variant"] = variant
    tool = cast(
        ToolCall,
        SimpleNamespace(
            tool_id=ASV3_CHECKPOINT_TOOL_ID,
            tool_call_response=json.dumps(checkpoint),
            turn_number=0,
        ),
    )
    message = _message(
        900, message="Birinci [1]. İkinci [5].", tool_calls=[tool], search_docs=docs
    )
    message.citations = cast(dict[int, int], {"1": 70, "5": 71})
    return message, items


@pytest.fixture
def fences(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock, MagicMock]:
    validator = MagicMock()
    broker = MagicMock(return_value=SimpleNamespace(revalidate_evidence=validator))
    lookup = MagicMock(
        return_value=(
            [
                CorpusSource(
                    SOURCE_ID,
                    "KANUNLAR/stored_name.md",
                    "file-id",
                    LegistationMetadata(official_title=TITLE),
                )
            ],
            False,
        )
    )
    monkeypatch.setattr(details, "CorpusBroker", broker)
    monkeypatch.setattr(details, "find_sources", lookup)
    return broker, validator, lookup


@pytest.mark.parametrize("variant", [None, "asv3_exp_cropped"])
def test_owner_history_json_and_packets_share_exact_inline_targets(
    monkeypatch: pytest.MonkeyPatch,
    fences: tuple[MagicMock, MagicMock, MagicMock],
    variant: str | None,
) -> None:
    message, originals = fixture_message(variant)
    assert message.tool_calls
    checkpoint_before = message.tool_calls[0].tool_call_response
    docs_before = [copy.deepcopy(doc.doc_metadata) for doc in message.search_docs]
    response = _get_session(monkeypatch, [message], run_id=None)
    result = response.messages[0]
    assert result.citations == {1: str(SOURCE_ID), 5: str(SOURCE_ID)}
    assert [row.citation_number for row in result.citation_info] == [1, 5]
    assert [row.text for row in result.citation_info] == [
        item.text for item in originals
    ]
    assert [row.chunk_id for row in result.citation_info] == [
        item.chunk_id for item in originals
    ]
    assert [row.chunk_ind for row in result.citation_info] == [7, 8]
    assert all(row.display_title == TITLE for row in result.citation_info)
    assert all(
        row.semantic_identifier == "KANUNLAR/stored_name.md"
        for row in result.citation_info
    )
    assert result.context_docs is not None
    assert all(
        doc.metadata["asv3_source_display_name"] == TITLE for doc in result.context_docs
    )
    replayed = [
        packet.obj
        for group in response.packets
        for packet in group
        if isinstance(packet.obj, CitationInfo)
    ]
    assert replayed == result.citation_info
    assert message.tool_calls
    assert message.tool_calls[0].tool_call_response == checkpoint_before
    assert [doc.doc_metadata for doc in message.search_docs] == docs_before
    broker, validator, lookup = fences
    broker.assert_called_once()
    validator.assert_called_once()
    assert [item.text for item in validator.call_args.args[0]] == [
        item.text for item in originals
    ]
    lookup.assert_called_once()


@pytest.mark.parametrize("viewer", [None, UUID(int=200)])
def test_shared_and_anonymous_viewers_do_not_receive_new_saved_text(
    fences: tuple[MagicMock, MagicMock, MagicMock], viewer: UUID | None
) -> None:
    message, _ = fixture_message()
    assert (
        details.owned_asv3_citation_info(
            [message],
            user=cast(User, SimpleNamespace(id=viewer)),
            session_owner_id=_USER_ID,
            db_session=cast(Session, MagicMock()),
        )
        == {}
    )
    for mock in fences:
        mock.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("Access revoked"),
        RunStopped("Read deadline expired"),
        OnyxError(OnyxErrorCode.SERVICE_UNAVAILABLE, "Publication unavailable"),
    ],
)
def test_revoked_access_preserves_legacy_response_without_inline_text(
    monkeypatch: pytest.MonkeyPatch,
    fences: tuple[MagicMock, MagicMock, MagicMock],
    error: Exception,
) -> None:
    message, _ = fixture_message()
    fences[1].side_effect = error
    response = _get_session(monkeypatch, [message], run_id=None)
    assert response.messages[0].citation_info == []
    assert response.messages[0].citations == {1: str(SOURCE_ID), 5: str(SOURCE_ID)}
    assert all(
        packet.obj.text is None
        for group in response.packets
        for packet in group
        if isinstance(packet.obj, CitationInfo)
    )
    fences[2].assert_not_called()


def test_wrong_persisted_chunk_never_uses_a_same_file_substitute(
    fences: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    message, _ = fixture_message()
    message.search_docs[0].chunk_ind = 99
    assert (
        details.owned_asv3_citation_info(
            [message],
            user=cast(User, SimpleNamespace(id=_USER_ID)),
            session_owner_id=_USER_ID,
            db_session=cast(Session, MagicMock()),
        )
        == {}
    )
    for mock in fences:
        mock.assert_not_called()


@pytest.mark.parametrize(
    "variant", ["asv3_tuned", "asv3_exp_cheap", "guardrails_v2", "supersearch"]
)
def test_other_workflows_remain_unenriched(
    fences: tuple[MagicMock, MagicMock, MagicMock], variant: str
) -> None:
    message, _ = fixture_message(variant)
    assert (
        details.owned_asv3_citation_info(
            [message],
            user=cast(User, SimpleNamespace(id=_USER_ID)),
            session_owner_id=_USER_ID,
            db_session=cast(Session, MagicMock()),
        )
        == {}
    )
    for mock in fences:
        mock.assert_not_called()


def test_unavailable_publication_clears_new_dto_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import regulatory_chat_reads

    message, _ = fixture_message()
    info = CitationInfo(
        citation_number=1,
        document_id=str(SOURCE_ID),
        display_title=TITLE,
        text="Retained private text",
    )
    monkeypatch.setattr(
        regulatory_chat_reads, "message_publication_available", lambda _message: False
    )
    result = translate_db_message_to_chat_message_detail(message, citation_info=[info])
    assert (
        result.citation_info == []
        and result.context_docs == []
        and result.citations is None
    )


def test_corrupt_checkpoint_does_not_break_old_history(
    monkeypatch: pytest.MonkeyPatch, fences: tuple[MagicMock, MagicMock, MagicMock]
) -> None:
    message, _ = fixture_message()
    assert message.tool_calls
    message.tool_calls[0].tool_call_response = "invalid checkpoint"
    response = _get_session(monkeypatch, [message], run_id=None)
    assert response.messages[0].citation_info == []
    for mock in fences:
        mock.assert_not_called()


def test_multi_turn_history_batches_shared_scope_and_keeps_message_numbers(
    fences: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    first, _ = fixture_message()
    second, _ = fixture_message("asv3_exp_cropped")
    second.id = 901
    second.message = "Önce [5], sonra [1]."
    result = details.owned_asv3_citation_info(
        [first, second],
        user=cast(User, SimpleNamespace(id=_USER_ID)),
        session_owner_id=_USER_ID,
        db_session=cast(Session, MagicMock()),
    )
    assert [row.citation_number for row in result[900]] == [1, 5]
    assert [row.citation_number for row in result[901]] == [5, 1]
    assert result[901][0].preview_url == "/api/asv3/citation/901/5"
    for mock in fences:
        mock.assert_called_once()
    assert len(fences[1].call_args.args[0]) == 4


def test_uncited_ledger_original_never_enters_inline_response(
    fences: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    message, originals = fixture_message()
    assert message.tool_calls and originals[0].search_doc
    checkpoint = json.loads(message.tool_calls[0].tool_call_response)
    uncited = EvidenceItem(
        source_id=str(SOURCE_ID),
        chunk_id=str(UUID(int=103)),
        text="Bu okunan parçaya cevapta atıf yapılmadı.",
        search_doc=originals[0].search_doc.model_copy(update={"chunk_ind": 9}),
    )
    checkpoint["evidence"]["records"].append(
        {"citation": 33, "item": uncited.model_dump(mode="json")}
    )
    message.tool_calls[0].tool_call_response = json.dumps(checkpoint)
    result = details.owned_asv3_citation_info(
        [message],
        user=cast(User, SimpleNamespace(id=_USER_ID)),
        session_owner_id=_USER_ID,
        db_session=cast(Session, MagicMock()),
    )
    assert len(fences[1].call_args.args[0]) == 2
    assert [row.citation_number for row in result[900]] == [1, 5]
    assert uncited.text not in "".join(row.text or "" for row in result[900])
