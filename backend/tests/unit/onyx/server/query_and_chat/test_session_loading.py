from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import JsonValue
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
            publication_read=None,
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


@pytest.mark.parametrize("asv3", [True, False])
def test_reloaded_canonical_preview_is_scoped_to_saved_asv3_runs(asv3: bool) -> None:
    from onyx.db.asv3_runs import ASV3_CHECKPOINT_TOOL_ID

    saved_search_doc = cast(
        SearchDoc,
        SimpleNamespace(
            id=77,
            document_id="corpus-original",
            chunk_ind=46,
            semantic_id="Source · Article 46",
            source_type=DocumentSource.FILE,
            doc_metadata={"regulatory_chunk_id": "rc-original"},
        ),
    )
    checkpoint_tool = SimpleNamespace(
        tool_id=ASV3_CHECKPOINT_TOOL_ID, tool_call_response="", turn_number=0
    )
    chat_message = cast(
        ChatMessage,
        SimpleNamespace(
            id=101,
            publication_read=None,
            message_type=MessageType.ASSISTANT,
            tool_calls=[checkpoint_tool] if asv3 else [],
            citations={2: 77},
            search_docs=[],
            reasoning_tokens=None,
            message="Answer [[2]]()",
        ),
    )
    with (
        patch(
            "onyx.server.query_and_chat.session_loading.create_asv3_progress_packets",
            return_value=[],
        ),
        patch(
            "onyx.server.query_and_chat.session_loading.get_db_search_doc_by_id",
            return_value=saved_search_doc,
        ),
    ):
        packets = translate_assistant_message_to_packets(
            chat_message, cast(Session, MagicMock())
        )
    citations = [
        packet.obj for packet in packets if isinstance(packet.obj, CitationInfo)
    ]
    assert len(citations) == 1
    assert citations[0].chunk_ind == 46
    assert citations[0].preview_url == ("/api/asv3/citation/101/2" if asv3 else None)


def test_reloaded_provision_tool_retains_its_label_and_query() -> None:
    from onyx.server.query_and_chat.session_loading import create_search_packets
    from onyx.server.query_and_chat.streaming_models import (
        SearchToolQueriesDelta,
        SearchToolStart,
    )

    packets = create_search_packets(
        search_queries=["5434 / MADDE 72"],
        search_docs=[],
        is_internet_search=False,
        turn_index=1,
        display_name="Mevzuat Maddesi Bul",
    )
    assert isinstance(packets[0].obj, SearchToolStart)
    assert packets[0].obj.display_name == "Mevzuat Maddesi Bul"
    assert isinstance(packets[1].obj, SearchToolQueriesDelta)
    assert packets[1].obj.queries == ["5434 / MADDE 72"]


def test_asv3_checkpoint_replays_public_updates_without_generic_tool_lookup() -> None:
    import json

    from onyx.db.asv3_runs import ASV3_CHECKPOINT_TOOL_ID
    from onyx.server.query_and_chat.streaming_models import ASv3Progress

    event = {
        "run_id": "run-1",
        "event_id": "event-1",
        "sequence": 1,
        "language": "tr",
        "phase": "completed",
        "status": "completed",
        "title": "Garanti kapsamındaki iki ihtimal değerlendirildi",
        "message": "Bedelsiz tamir ve yeni makine gönderilmesi ayrı incelendi.",
    }
    message = cast(
        ChatMessage,
        SimpleNamespace(
            id=10,
            publication_read=None,
            message_type=MessageType.ASSISTANT,
            tool_calls=[
                SimpleNamespace(
                    tool_id=ASV3_CHECKPOINT_TOOL_ID,
                    turn_number=0,
                    tab_index=0,
                    tool_call_response=json.dumps({"version": 1, "progress": [event]}),
                )
            ],
            citations={},
            search_docs=[],
            reasoning_tokens=None,
            message="Answer",
        ),
    )
    with patch(
        "onyx.server.query_and_chat.session_loading.get_tool_by_id"
    ) as tool_lookup:
        packets = translate_assistant_message_to_packets(
            message, cast(Session, MagicMock())
        )
    tool_lookup.assert_not_called()
    progress = [
        packet.obj for packet in packets if isinstance(packet.obj, ASv3Progress)
    ]
    assert len(progress) == 1
    assert progress[0].language == "tr"
    assert progress[0].title == event["title"]
    assert any(packet.obj.type == "message_delta" for packet in packets)


def test_asv3_replay_remains_behind_publication_fence() -> None:
    message = cast(ChatMessage, SimpleNamespace(id=10, publication_read={}))
    with patch(
        "onyx.db.regulatory_chat_reads.message_publication_available",
        return_value=False,
    ):
        packets = translate_assistant_message_to_packets(
            message, cast(Session, MagicMock())
        )
    assert packets == []


def test_invalid_asv3_progress_does_not_break_valid_public_updates() -> None:
    import json

    from onyx.server.query_and_chat.session_loading import create_asv3_progress_packets

    payload = json.dumps(
        {
            "version": 1,
            "progress": [
                {"sequence": -1},
                {
                    "run_id": "r",
                    "event_id": "e",
                    "sequence": 1,
                    "language": "en",
                    "phase": "completed",
                    "status": "completed",
                    "title": "Warranty scenarios reviewed",
                },
            ],
        }
    )
    packets = create_asv3_progress_packets(payload)
    assert len(packets) == 1
    assert packets[0].obj.type == "asv3_progress"


def test_invalid_asv3_checkpoint_shape_does_not_break_answer_replay() -> None:
    from onyx.server.query_and_chat.session_loading import create_asv3_progress_packets

    for invalid in [
        "[]",
        "null",
        "broken-json",
        '{"version": 1, "progress": "broken"}',
    ]:
        assert create_asv3_progress_packets(invalid) == []


def test_native_citation_replay_uses_owned_message_preview_endpoint() -> None:
    chat_message = cast(
        ChatMessage,
        SimpleNamespace(
            id=101,
            publication_read=None,
            message_type=MessageType.ASSISTANT,
            tool_calls=[],
            citations={2: 77},
            search_docs=[],
            reasoning_tokens=None,
            message="Answer [[2]]()",
        ),
    )
    saved_search_doc = cast(
        SearchDoc,
        SimpleNamespace(
            document_id="original-pdf",
            chunk_ind=-2,
            semantic_id="Original PDF · Page 4",
            source_type=DocumentSource.USER_FILE,
            doc_metadata={"asv3_native_locator": '{"derived":true,"locator":"page 4"}'},
        ),
    )
    with patch(
        "onyx.server.query_and_chat.session_loading.get_db_search_doc_by_id",
        return_value=saved_search_doc,
    ):
        packets = translate_assistant_message_to_packets(
            chat_message, cast(Session, MagicMock())
        )
    citations = [
        packet.obj for packet in packets if isinstance(packet.obj, CitationInfo)
    ]
    assert len(citations) == 1
    assert citations[0].preview_url == "/api/asv3/citation/101/2"
    assert citations[0].chunk_ind == -2


def test_idle_incomplete_asv3_replay_gets_localized_resume_only_from_saved_profile() -> (
    None
):
    import json

    from onyx.server.query_and_chat.session_loading import create_asv3_progress_packets
    from onyx.server.query_and_chat.streaming_models import ASv3Progress

    payload = json.dumps(
        {
            "version": 1,
            "run_id": "run-1",
            "public_profile": {
                "language": "tr",
                "notifications": {
                    "interrupted": [
                        "Araştırma yarıda kaldı",
                        "Doğrulanan kaynaklarla devam edebilirsiniz",
                    ],
                    "resume": ["Araştırmaya devam et", ""],
                },
            },
            "progress": [
                {
                    "run_id": "run-1",
                    "event_id": "event-1",
                    "sequence": 7,
                    "language": "tr",
                    "phase": "research",
                    "status": "running",
                    "title": "Tamir koşullarını inceliyorum",
                }
            ],
        }
    )
    active = create_asv3_progress_packets(payload)
    assert len(active) == 1
    idle = create_asv3_progress_packets(payload, interrupted=True)
    assert len(idle) == 2
    notice = idle[-1].obj
    assert isinstance(notice, ASv3Progress)
    assert notice.phase == "interrupted"
    assert notice.status == "failed"
    assert notice.sequence == 8
    assert notice.resume_label == "Araştırmaya devam et"
    assert notice.language == "tr"

    checkpoint = json.loads(payload)
    external_item: dict[str, JsonValue] = {
        "metadata": {"external": True},
        "search_doc": {"document_id": "web-1"},
    }
    checkpoint["evidence"] = {
        "records": [
            {
                "item": external_item,
            }
        ]
    }
    external = create_asv3_progress_packets(json.dumps(checkpoint), interrupted=True)
    external_notice = external[-1].obj
    assert isinstance(external_notice, ASv3Progress)
    assert external_notice.phase == "interrupted"
    assert external_notice.resume_label is None

    external_item["search_doc"] = None
    noncitable = create_asv3_progress_packets(json.dumps(checkpoint), interrupted=True)
    assert isinstance(noncitable[-1].obj, ASv3Progress)
    assert noncitable[-1].obj.resume_label == "Araştırmaya devam et"

    checkpoint["evidence"]["records"][0]["item"] = {
        "metadata": {},
        "search_doc": {"document_id": "canonical-1"},
    }
    canonical = create_asv3_progress_packets(json.dumps(checkpoint), interrupted=True)
    assert isinstance(canonical[-1].obj, ASv3Progress)
    assert canonical[-1].obj.resume_label == "Araştırmaya devam et"


def test_actual_terminal_asv3_replay_never_becomes_resumable() -> None:
    import json

    from onyx.server.query_and_chat.session_loading import create_asv3_progress_packets
    from onyx.server.query_and_chat.streaming_models import ASv3Progress

    for status in ["completed", "failed", "cancelled"]:
        payload = json.dumps(
            {
                "version": 1,
                "run_id": "r",
                "progress": [
                    {
                        "run_id": "r",
                        "event_id": "e",
                        "sequence": 1,
                        "language": "en",
                        "phase": "research",
                        "status": status,
                        "title": "Finished",
                    }
                ],
                "public_profile": {
                    "language": "en",
                    "notifications": {
                        "interrupted": ["Interrupted", "Resume"],
                        "resume": ["Resume", ""],
                    },
                },
            }
        )
        packets = create_asv3_progress_packets(payload, interrupted=True)
        assert len(packets) == 1
        assert isinstance(packets[0].obj, ASv3Progress)
        assert packets[0].obj.phase != "interrupted"


def test_incomplete_legacy_asv3_profile_does_not_invent_english_resume() -> None:
    import json

    from onyx.server.query_and_chat.session_loading import create_asv3_progress_packets

    assert (
        create_asv3_progress_packets(
            json.dumps({"version": 1, "run_id": "r", "progress": []}), interrupted=True
        )
        == []
    )


def test_cancelled_asv3_saved_stop_notice_never_offers_checkpoint_resume() -> None:
    import json

    from onyx.db.asv3_runs import ASV3_CHECKPOINT_TOOL_ID
    from onyx.server.query_and_chat.streaming_models import ASv3Progress

    checkpoint = json.dumps(
        {
            "version": 1,
            "run_id": "r",
            "progress": [
                {
                    "run_id": "r",
                    "event_id": "e",
                    "sequence": 1,
                    "language": "tr",
                    "phase": "research",
                    "status": "running",
                    "title": "Kaynakları inceliyorum",
                }
            ],
            "public_profile": {
                "language": "tr",
                "notifications": {
                    "interrupted": ["Araştırma yarıda kaldı", "Devam edebilirsiniz"],
                    "resume": ["Devam et", ""],
                },
            },
        }
    )
    for saved_message in ["Araştırma durduruldu.", "Tamamlanmış yanıt [[1]]()"]:
        message = cast(
            ChatMessage,
            SimpleNamespace(
                id=10,
                publication_read=None,
                message_type=MessageType.ASSISTANT,
                tool_calls=[
                    SimpleNamespace(
                        tool_id=ASV3_CHECKPOINT_TOOL_ID,
                        turn_number=0,
                        tab_index=0,
                        tool_call_response=checkpoint,
                    )
                ],
                citations={},
                search_docs=[],
                reasoning_tokens=None,
                message=saved_message,
            ),
        )
        packets = translate_assistant_message_to_packets(
            message, cast(Session, MagicMock()), asv3_interrupted=True
        )
        progress = [
            packet.obj for packet in packets if isinstance(packet.obj, ASv3Progress)
        ]
        assert len(progress) == 1
        assert all(event.resume_label is None for event in progress)
        assert all(event.phase != "interrupted" for event in progress)
