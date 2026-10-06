"""Load pooled checkpoints through owned memory, history and citation readers."""

import base64
import hashlib
import json
import zlib
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy.orm import Session

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.workers import WorkerPool
from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db import asv3_runs
from onyx.db.models import ChatMessage
from onyx.server.asv3_citations import (
    native_item_from_checkpoint,
    saved_item_from_checkpoint,
)
from onyx.server.query_and_chat.session_loading import (
    create_asv3_progress_packets,
    translate_assistant_message_to_packets,
)
from onyx.server.query_and_chat.streaming_models import ASv3Progress, CitationInfo
from tests.unit.onyx.db.test_asv3_runs import MemoryStore
from tests.unit.onyx.db.test_asv3_runs import store as store

ORIGINAL = (
    "Koşul A ve B birlikte sağlanır; istisna yalnız C hâlinde uygulanır.\nÖzgün devamı."
)
CHILD_BODY = "Koşullu kaynak sonucu [1].\nÖzgün devamı korunmuştur."


def _digest(value: dict[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _wire(payload: str) -> dict[str, JsonValue]:
    envelope = json.loads(payload)
    return json.loads(zlib.decompress(base64.b64decode(envelope["data"])))


def _snapshot() -> dict[str, JsonValue]:
    scope = IndexFilters(
        document_set=["Frozen corpus"], access_control_list=[]
    ).model_dump(mode="json")
    context = RunContext(run_id="parallel-run", scope=scope)
    ledger = EvidenceLedger()
    child_evidence: dict[str, JsonValue] = {}
    for number in (1, 2):
        native = number == 2
        source = f"source-{number}"
        item = EvidenceItem(
            source_id=source,
            chunk_id=None if native else "rc-original",
            text=ORIGINAL if not native else "Tablonun özgün ikinci satırı.",
            metadata={
                "version_unknown": True,
                **({"derived": True, "source_sha256": "a" * 64} if native else {}),
            },
            search_doc=SearchDoc(
                document_id=source,
                chunk_ind=-number if native else 7,
                semantic_identifier=f"Özgün kaynak {number}",
                link=None,
                blurb="Özgün kaynak",
                source_type=DocumentSource.USER_FILE,
                boost=0,
                hidden=False,
                metadata={} if native else {"regulatory_chunk_id": "rc-original"},
                match_highlights=[],
            ),
        )
        ledger.add([item], context)
        if number == 1:
            ledger.record_delivery(
                "partial-call",
                "asv3_researcher",
                [{"citation": 1, "text": ORIGINAL[9:26], "start_char": 9}],
            )
            ledger.pin_delivery("partial-call")
            ledger.include([1])
            child_evidence = ledger.export()
    ledger.record_delivery(
        "full-call",
        "asv3_researcher",
        [{"citation": 1, "text": ORIGINAL}],
    )
    ledger.pin_delivery("full-call")
    child_snapshot: dict[str, JsonValue] = {
        "version": 1,
        "run_id": context.run_id,
        "request": "Assigned issue",
        "evidence": child_evidence,
        "last_draft": CHILD_BODY,
        "turns": [],
    }
    binding: dict[str, JsonValue] = {
        "run_id": context.run_id,
        "scope_hash": _digest(scope),
        "task_id": "child-task",
        "task": "Assigned issue",
        "parent_task_id": None,
        "independent_question": True,
        "assignment_id": "q0",
        "outcome_ids": ["result"],
    }
    wrapped: dict[str, JsonValue] = {
        "version": 1,
        "run_id": context.run_id,
        "task_id": "child-task",
        "scope_hash": _digest(scope),
        "request_hash": hashlib.sha256(
            json.dumps("Assigned issue", ensure_ascii=False).encode()
        ).hexdigest(),
        "assignment_id": "q0",
        "outcome_ids": ["result"],
        "snapshot": child_snapshot,
    }
    event: dict[str, JsonValue] = {
        "run_id": context.run_id,
        "event_id": "event-1",
        "sequence": 1,
        "language": "tr",
        "phase": "completed",
        "status": "completed",
        "title": "Özgün koşullar incelendi",
        "message": "Koşul ve istisna ayrı değerlendirildi.",
    }
    return {
        "version": 1,
        "run_id": context.run_id,
        "sequence": 10,
        "request": "Full scenario and alternatives",
        "research_profile": "experimental",
        "parallel_research": True,
        "scope": scope,
        "evidence": ledger.export(),
        "progress": [event],
        "last_draft": CHILD_BODY,
        "session_research": {"requests": ["Full scenario and alternatives"]},
        "workers": {
            "version": 1,
            "run_id": context.run_id,
            "tasks": [
                {
                    "task_id": "child-task",
                    "task": "Assigned issue",
                    "status": "completed",
                    "independent_question": True,
                    "assignment_id": "q0",
                    "outcome_ids": ["result"],
                    "local_budget": {
                        "limits": {"tools": None, "decisions": None},
                        "used": {"tools": 0, "decisions": 0},
                    },
                    "child_checkpoint": {**wrapped, "integrity": _digest(wrapped)},
                }
            ],
            "task_binding_integrity": {"child-task": _digest(binding)},
        },
    }


def _insert_checkpoint(
    store: MemoryStore, message_id: int, snapshot: dict[str, JsonValue]
) -> str:
    payload = asv3_runs.encode_asv3_checkpoint(snapshot)
    store.session.execute(
        store.calls.insert().values(
            parent_chat_message_id=message_id,
            chat_session_id=store.chat_id,
            tool_id=asv3_runs.ASV3_CHECKPOINT_TOOL_ID,
            tool_call_response=payload,
        )
    )
    return payload


def test_owned_ancestor_hydrates_exact_child_and_partial_delivery(
    store: MemoryStore,
) -> None:
    store.message(1, None, MessageType.ASSISTANT)
    store.checkpoint(1, "older legacy checkpoint")
    store.message(2, 1, MessageType.USER)
    store.message(3, 2, MessageType.ASSISTANT)
    expected = _snapshot()
    payload = _insert_checkpoint(store, 3, expected)
    assert "parallel_checkpoint_storage" in _wire(payload)
    store.message(4, 2, MessageType.ASSISTANT)
    _insert_checkpoint(store, 4, {**_snapshot(), "request": "unselected sibling"})
    store.message(5, 3, MessageType.USER)
    store.message(6, 5, MessageType.ASSISTANT)
    _insert_checkpoint(store, 6, {**_snapshot(), "request": "current turn"})

    loaded = store.load(5)
    assert loaded == expected
    assert loaded is not None
    assert "parallel_checkpoint_storage" not in loaded
    context = RunContext(
        run_id="parallel-run", scope=cast(dict[str, JsonValue], loaded["scope"])
    )
    runner = MagicMock(
        return_value=ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Unexpected research"
        )
    )
    pool = WorkerPool(context, runner)
    try:
        pool.restore(cast(dict[str, JsonValue], loaded["workers"]))
        child = pool.checkpoint("child-task")
        assert child is not None and child["last_draft"] == CHILD_BODY
        ledger = EvidenceLedger()
        ledger.restore(cast(dict[str, JsonValue], child["evidence"]), context)
        item = ledger.get(1)
        assert item is not None and item.text == ORIGINAL
        assert item.source_id == "source-1" and item.chunk_id == "rc-original"
        assert item.text_hash == hashlib.sha256(ORIGINAL.encode()).hexdigest()
        assert ledger.get(2) is None
        assert ledger.completely_delivered("partial-call") == set()
        assert ledger.export() == child["evidence"]
        assert ledger.export()["pinned_delivery_calls"] == ["partial-call"]
        runner.assert_not_called()
    finally:
        pool.close()


@pytest.mark.parametrize("failure", ["owner", "chat", "deleted"])
def test_pooled_memory_stays_behind_current_owner_and_chat_fence(
    store: MemoryStore, failure: str
) -> None:
    store.message(1, None, MessageType.ASSISTANT)
    _insert_checkpoint(store, 1, _snapshot())
    store.message(2, 1, MessageType.USER)
    if failure == "deleted":
        store.session.execute(store.chats.update().values(deleted=True))
    with pytest.raises(PermissionError):
        asv3_runs.load_asv3_session_checkpoint(
            chat_session_id=uuid4() if failure == "chat" else store.chat_id,
            user_message_id=2,
            user_id=uuid4() if failure == "owner" else store.owner_id,
        )


def test_pooled_history_replays_progress_and_canonical_preview_path() -> None:
    snapshot = _snapshot()
    payload = asv3_runs.encode_asv3_checkpoint(snapshot)
    assert "parallel_checkpoint_storage" in _wire(payload)
    assert asv3_runs.checkpoint_progress_packets(payload) == snapshot["progress"]
    progress = create_asv3_progress_packets(payload)
    assert len(progress) == 1 and isinstance(progress[0].obj, ASv3Progress)
    assert progress[0].obj.title == "Özgün koşullar incelendi"
    saved_doc = SimpleNamespace(
        id=77,
        document_id="source-1",
        chunk_ind=7,
        semantic_id="Özgün kaynak 1",
        source_type=DocumentSource.USER_FILE,
        doc_metadata={"regulatory_chunk_id": "rc-original"},
    )
    message = cast(
        ChatMessage,
        SimpleNamespace(
            id=101,
            publication_read=None,
            message_type=MessageType.ASSISTANT,
            tool_calls=[
                SimpleNamespace(
                    tool_id=asv3_runs.ASV3_CHECKPOINT_TOOL_ID,
                    tool_call_response=payload,
                    turn_number=0,
                    tab_index=0,
                )
            ],
            citations={1: 77},
            search_docs=[],
            reasoning_tokens=None,
            message="Koşullu cevap [[1]]()",
        ),
    )
    with (
        patch(
            "onyx.server.query_and_chat.session_loading.get_db_search_doc_by_id",
            return_value=saved_doc,
        ),
        patch("onyx.server.query_and_chat.session_loading.get_tool_by_id") as lookup,
    ):
        packets = translate_assistant_message_to_packets(
            message, cast(Session, MagicMock())
        )
    lookup.assert_not_called()
    assert any(isinstance(packet.obj, ASv3Progress) for packet in packets)
    citations = [
        packet.obj for packet in packets if isinstance(packet.obj, CitationInfo)
    ]
    assert len(citations) == 1
    assert citations[0].document_id == "source-1" and citations[0].chunk_ind == 7
    assert citations[0].preview_url == "/api/asv3/citation/101/1"


@pytest.mark.parametrize("foreign_identity", [False, True])
def test_pooled_citation_reader_keeps_original_identity(
    store: MemoryStore, foreign_identity: bool
) -> None:
    snapshot = _snapshot()
    if foreign_identity:
        evidence = cast(dict[str, JsonValue], snapshot["evidence"])
        records = cast(list[JsonValue], evidence["records"])
        first = cast(dict[str, JsonValue], records[0])
        item = cast(dict[str, JsonValue], first["item"])
        search_doc = cast(dict[str, JsonValue], item["search_doc"])
        search_doc["document_id"] = "foreign-source"
    store.message(1, None, MessageType.ASSISTANT)
    _insert_checkpoint(store, 1, snapshot)
    store.message(2, 1, MessageType.USER)
    loaded = store.load(2)
    assert loaded is not None
    if foreign_identity:
        with pytest.raises(ValueError, match="source identity"):
            saved_item_from_checkpoint(loaded, 1)
        return
    item, filters = saved_item_from_checkpoint(loaded, 1)
    assert item.text == ORIGINAL and item.chunk_id == "rc-original"
    assert filters.document_set == ["Frozen corpus"]
    native, _ = native_item_from_checkpoint(loaded, 2)
    assert native.text == "Tablonun özgün ikinci satırı."
    assert native.metadata["source_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="not found"):
        saved_item_from_checkpoint(loaded, 3)


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_nonparallel_profile_retains_legacy_bytes_and_owned_memory(
    store: MemoryStore, profile: Literal["normal", "deep", "experimental"]
) -> None:
    snapshot = {**_snapshot(), "research_profile": profile, "parallel_research": False}
    store.message(1, None, MessageType.ASSISTANT)
    payload = _insert_checkpoint(store, 1, snapshot)
    legacy = json.dumps(
        {
            "version": 1,
            "encoding": "zlib-base64",
            "data": base64.b64encode(
                zlib.compress(
                    json.dumps(snapshot, ensure_ascii=False).encode(), level=3
                )
            ).decode("ascii"),
        }
    )
    assert payload == legacy
    assert "parallel_checkpoint_storage" not in _wire(payload)
    store.message(2, 1, MessageType.USER)
    assert store.load(2) == snapshot
    assert asv3_runs.checkpoint_progress_packets(payload) == snapshot["progress"]
