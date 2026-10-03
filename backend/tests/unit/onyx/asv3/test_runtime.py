"""Exercise the native harness and publication path with external boundaries faked."""

import json
import threading
from dataclasses import replace
from datetime import date
from queue import Queue
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import BaseFilters, IndexFilters
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.models import User
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import (
    ChatCompletionMessageToolCall,
    Choice,
    FunctionCall,
    Message,
    ModelResponse,
)
from onyx.llm.models import (
    AssistantMessage,
    ReasoningEffort,
    TextContentPart,
    ToolMessage,
    UserMessage,
)
from onyx.prompts.asv3.research import COORDINATOR_PROMPT
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
    CitationInfo,
    Packet,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def response(
    text: str | None = None, calls: list[tuple[str, dict[str, Any]]] | None = None
) -> ModelResponse:
    return ModelResponse(
        id=str(uuid4()),
        created="0",
        choice=Choice(
            message=Message(
                content=text,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=str(uuid4()),
                        function=FunctionCall(name=name, arguments=json.dumps(args)),
                    )
                    for name, args in calls
                ]
                if calls
                else None,
            )
        ),
    )


def packets(queue: Queue[Any]) -> list[Packet]:
    result = []
    while not queue.empty():
        _, packet = queue.get_nowait()
        result.append(packet)
    return result


def user_payload(message: UserMessage) -> dict[str, Any]:
    content = message.content
    if isinstance(content, list):
        assert isinstance(content[0], TextContentPart)
        content = content[0].text
    assert isinstance(content, str)
    return cast(dict[str, Any], json.loads(content))


def delivered_originals(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for message in arguments["prompt"]:
        if isinstance(message, ToolMessage):
            result.extend(json.loads(message.content).get("original_evidence", []))
        elif isinstance(message, UserMessage):
            result.extend(user_payload(message).get("original_evidence", []))
    return result


class CorpusBoundary:
    def __init__(self) -> None:
        self.sources = [
            CorpusSource(uuid4(), f"Tamir mevzuatı {n}", f"file-{n}") for n in range(2)
        ]
        self.chunks = {
            str(source.id): CorpusChunk(
                id=f"chunk-{n}",
                source_id=source.id,
                text=("Tamir şartları. " * 900) + f"ORIGINAL_TAIL_{n}",
                position=0,
                projection_ordinal=0,
                heading_path=(f"Madde {142 + n}",),
                metadata={"source_links": {"0": f"https://example.test/law-{n}"}},
                validity_start=date(2020, 1, 1),
                validity_end=None,
                status="published",
            )
            for n, source in enumerate(self.sources)
        }
        self.barrier = threading.Barrier(2)
        self.entered, self.release, self.cancelled = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        self.block = False
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.revalidated: list[EvidenceItem] = []
        self.search_adapter: Any = None
        self.scope: IndexFilters | None = None

    def page(
        self, source_id: str, _context: RunContext, **_kwargs: Any
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        with self.lock:
            self.active += 1
            self.peak = max(self.active, self.peak)
        self.barrier.wait(3)
        self.entered.set()
        if self.block:
            assert self.release.wait(3)
        with self.lock:
            self.active -= 1
        source = next(source for source in self.sources if str(source.id) == source_id)
        return source, [self.chunks[source_id]], False

    def revalidate_evidence(
        self, evidence: list[EvidenceItem], context: RunContext
    ) -> None:
        context.check_active()
        for item in evidence:
            assert item.text == self.chunks[item.source_id].text
        self.revalidated.extend(evidence)


def setup_run(
    monkeypatch: pytest.MonkeyPatch, *, language: str = "tr"
) -> tuple[dict[str, Any], CorpusBoundary, MagicMock, list[dict[str, Any]], Queue[Any]]:
    broker = CorpusBoundary()

    def bind_scope(*, user: User, filters: IndexFilters) -> IndexFilters:
        assert user.id is not None
        assert filters.source_type == [DocumentSource.USER_FILE]
        return filters.model_copy(
            deep=True,
            update={
                "tenant_id": "test-tenant",
                "asv3_document_set_id": 91,
                "forced_document_set": ["PC Külliyatı"],
            },
        )

    def create_broker(
        _user: User, scope: IndexFilters, **_kwargs: Any
    ) -> CorpusBoundary:
        broker.scope = scope
        return broker

    monkeypatch.setattr(runtime, "bind_pc_corpus_scope", bind_scope)
    monkeypatch.setattr(runtime, "CorpusBroker", create_broker)
    checkpoints: list[dict[str, Any]] = []
    monkeypatch.setattr(
        runtime,
        "save_asv3_checkpoint",
        lambda **kwargs: checkpoints.append(kwargs["snapshot"]),
    )
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: None)
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="scripted",
        temperature=0,
        max_input_tokens=500000,
    )
    title, description = (
        (
            "Tamir ve değişim",
            "Tamir ile yeni makine gönderilmesine ilişkin hükümleri karşılaştırıyorum.",
        )
        if language == "tr"
        else (
            "Repair and replacement",
            "I am comparing the original rules for repair and replacement.",
        )
    )
    initial = response(
        calls=[
            (
                "read_source_range",
                {
                    "source_id": str(source.id),
                    "_language": language,
                    "_public_update": [title, description],
                },
            )
            for source in broker.sources
        ]
    )
    invocation = 0

    def answer_from_originals(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            return initial
        assert invocation == 2
        originals = delivered_originals(arguments)
        assert {row["text"] for row in originals} == {
            chunk.text for chunk in broker.chunks.values()
        }
        numbers = {row["chunk_id"]: row["citation"] for row in originals}
        return response(
            f"Tamir sonucu [{numbers['chunk-0']}]; değiştirme sonucu [{numbers['chunk-1']}]."
        )

    llm.invoke.side_effect = answer_from_originals
    cache = MagicMock(spec=CacheBackend)
    cache.exists.side_effect = lambda _key: broker.cancelled.is_set()
    user = MagicMock(spec=User)
    user.id = uuid4()
    queue: Queue[Any] = Queue()
    kwargs = dict(
        emitter=Emitter(queue),
        state_container=ChatStateContainer(),
        simple_chat_history=[
            ChatMessageSimple(
                message="Garanti kapsamındaki tamir ve yeni makine aynı sonucu verir mi?",
                token_count=30,
                message_type=MessageType.USER,
            )
        ],
        tools=[],
        llm=llm,
        token_counter=len,
        user=user,
        filters=BaseFilters(
            regulatory_chunks_only=True, source_type=[DocumentSource.USER_FILE]
        ),
        document_set_names_override=None,
        user_identity=None,
        chat_session_id=uuid4(),
        user_message_id=1,
        assistant_message_id=2,
        reasoning_effort=ReasoningEffort.AUTO,
        include_citations=True,
        cache=cache,
    )
    return kwargs, broker, llm, checkpoints, queue


@pytest.mark.parametrize(
    "provider,model_name",
    [
        ("vertex_ai", "gemini-3.8-flash"),
        ("openai", "scripted"),
        ("anthropic", "scripted"),
    ],
)
def test_native_parallel_originals_preserve_selected_provider_and_publish_without_extra_model_phases(
    monkeypatch: pytest.MonkeyPatch, provider: str, model_name: str
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    llm.config.model_provider, llm.config.model_name = provider, model_name
    kwargs["custom_agent_prompt"] = "Her soruya ayrı ve kısa uygulama sonucu ver."
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 2
    assert broker.peak == 2
    for call in llm.invoke.call_args_list:
        assert call.kwargs["prompt"][0].content == COORDINATOR_PROMPT
        assert call.kwargs["structured_response_format"] is None
        assert call.kwargs["timeout_override"] is None
        assert (
            user_payload(call.kwargs["prompt"][1])["assistant_instructions"]
            == kwargs["custom_agent_prompt"]
        )
    assert llm.config.model_provider == provider and llm.config.model_name == model_name
    saved = checkpoints[-1]
    assert saved["execution_mode"] == "native"
    assert saved["publication_status"] == "found"
    assert saved["publication_stop_reason"] == "native_answer_published"
    assert saved["final_publication_gap"] is None
    assert saved["evidence"]["included"] == [1, 2]
    assert {item.chunk_id for item in broker.revalidated} == {"chunk-0", "chunk-1"}
    turn = saved["turns"][0]
    assert [result["tool_call_id"] for result in turn["results"]] == [
        call["id"] for call in turn["assistant"]["tool_calls"]
    ]
    assert [call["function"]["name"] for call in turn["assistant"]["tool_calls"]] == [
        "read_source_range",
        "read_source_range",
    ]
    assert len(kwargs["state_container"].citation_to_doc) == 2
    output = packets(queue)
    for packet in output:
        if isinstance(packet.obj, CitationInfo):
            assert (
                packet.obj.preview_url
                == f"/api/asv3/citation/{kwargs['assistant_message_id']}/{packet.obj.citation_number}"
            )
    completed = [
        i
        for i, packet in enumerate(output)
        if isinstance(packet.obj, ASv3Progress)
        and packet.obj.status == "completed"
        and packet.obj.task_id is None
    ]
    assert len(completed) == 1
    assert completed[0] > max(
        i
        for i, packet in enumerate(output)
        if isinstance(packet.obj, (AgentResponseDelta, CitationInfo))
    )


def test_one_search_delivery_reports_each_original_source_without_extra_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    names = ["4458 SAYILI GÜMRÜK KANUNU", "KATMA DEĞER VERGİSİ KANUNU"]
    articles = ["168", "16"]
    for n, (identity, chunk) in enumerate(broker.chunks.items()):
        broker.chunks[identity] = replace(
            chunk, heading_path=(names[n], f"MADDE {articles[n]}")
        )
    searches = 0

    def search(arguments: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        nonlocal searches
        context.check_active()
        searches += 1
        assert arguments == {"query": "Relevant relief and procedure"}
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual original passages returned",
            evidence=[
                evidence_for_chunk(source, broker.chunks[str(source.id)])
                for source in broker.sources
            ],
        )

    monkeypatch.setattr(
        runtime,
        "build_corpus_specs",
        lambda _broker, **_options: [
            ToolSpec(
                name="search_corpus",
                description="Search original provisions",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=search,
            )
        ],
    )
    invocation = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            return response(
                calls=[
                    (
                        "search_corpus",
                        {
                            "query": "Relevant relief and procedure",
                            "_language": "tr",
                            "_public_update": [
                                "İade koşulları",
                                "İlgili hükümleri arıyorum.",
                            ],
                        },
                    )
                ]
            )
        assert invocation == 2
        assert {row["text"] for row in delivered_originals(arguments)} == {
            chunk.text for chunk in broker.chunks.values()
        }
        assert [
            doc.metadata["asv3_source_display_name"]
            for doc in kwargs["state_container"].get_all_search_docs().values()
        ] == names
        return response("Tamir [1], değiştirme [2].")

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 2 and searches == 1
    assert len(checkpoints[-1]["receipts"]) == 1
    output = packets(queue)
    updates = [packet.obj for packet in output if isinstance(packet.obj, ASv3Progress)]
    sources = [
        event
        for event in updates
        if event.task_id and event.task_id.startswith("action:source:")
    ]
    assert [event.title for event in sources] == [
        f"{name} — Madde {article}" for name, article in zip(names, articles)
    ]
    assert all(
        event.status == "completed" and event.language == "tr" for event in sources
    )
    assert all(
        event.message == "Bu kaynaktaki ilgili özgün hükümler inceleniyor."
        for event in sources
    )
    assert not any(
        "Relevant relief" in event.title + (event.message or "") for event in sources
    )
    assert (
        len(
            [
                event
                for event in updates
                if event.phase == "completed" and event.task_id is None
            ]
        )
        == 1
    )
    final = next(
        packet.obj for packet in output if isinstance(packet.obj, AgentResponseStart)
    )
    assert final.final_documents is not None
    canonical = {
        row["item"]["source_id"]: row["item"]["search_doc"]
        for row in checkpoints[-1]["evidence"]["records"]
    }
    for document in final.final_documents:
        rendered = json.loads(document.model_dump_json())
        assert rendered["metadata"].pop("asv3_source_display_name") in names
        assert rendered == canonical[document.document_id]
    assert all(
        item.search_doc is not None
        and "asv3_source_display_name" not in item.search_doc.metadata
        for item in broker.revalidated
    )


@pytest.mark.parametrize("language", ["tr", "en"])
def test_first_useful_action_localizes_existing_progress_without_a_profile_call(
    monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch, language=language)
    runtime.run_asv3_loop(**kwargs)
    updates = [
        packet.obj for packet in packets(queue) if isinstance(packet.obj, ASv3Progress)
    ]
    assert llm.invoke.call_count == 2
    assert all(
        event.language == language for event in updates if event.language != "und"
    )
    assert updates[-1].language == language
    assert updates[-1].title == ("Yanıt hazır" if language == "tr" else "Answer ready")
    assert any(
        event.title
        == ("Tamir ve değişim" if language == "tr" else "Repair and replacement")
        for event in updates
    )
    assert not any("read_source_range" in (event.message or "") for event in updates)
    assert checkpoints[-1]["public_profile"]["language"] == language


def test_available_crossreference_is_model_navigation_and_adds_no_forced_audit_or_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    for identity, chunk in broker.chunks.items():
        broker.chunks[identity] = replace(
            chunk,
            text=chunk.text
            + " 4458 sayılı Gümrük Kanununun 168 inci maddesine göre uygulanır.",
        )
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 2
    assert len(checkpoints[-1]["receipts"]) == 2
    assert checkpoints[-1]["publication_status"] == "found"


def test_citation_guard_repairs_unknown_number_in_native_loop_without_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    baseline = llm.invoke.side_effect
    invocation = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            return baseline(**arguments)
        if invocation == 2:
            return response("Tamir sonucu [999].")
        assert invocation == 3
        footer = user_payload(arguments["prompt"][-1])
        assert footer["draft_to_repair"] == "Tamir sonucu [999]."
        assert footer["publication_gap"]["unknown_citations"] == [999]
        return response("Tamir [1], değiştirme [2].")

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 3
    assert checkpoints[-1]["publication_status"] == "found"
    assert "999" not in kwargs["state_container"].answer_tokens
    assert any(isinstance(packet.obj, CitationInfo) for packet in packets(queue))


def test_named_law_uses_its_local_original_without_a_routine_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kinds = ["tebliğ", "kanun"]
    titles = ["UYGULAMA TEBLİĞİ", "8917 SAYILI FAALİYET KANUNU"]
    for n, (identity, chunk) in enumerate(broker.chunks.items()):
        broker.chunks[identity] = replace(
            chunk,
            heading_path=(titles[n], "MADDE 27"),
            metadata={
                **chunk.metadata,
                "document_type": kinds[n],
                "title": titles[n],
                "article_no": "27",
            },
        )
    baseline = llm.invoke.side_effect
    invocation = 0
    direct_number = 0
    rejected = ""

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation, direct_number, rejected
        invocation += 1
        if invocation == 1:
            return baseline(**arguments)
        if invocation == 2:
            originals = delivered_originals(arguments)
            numbers = {row["chunk_id"]: row["citation"] for row in originals}
            direct_number = numbers["chunk-1"]
            rejected = (
                "8917 sayılı Faaliyet Kanunu'nun 27 inci maddesi uygulanır "
                f"[{numbers['chunk-0']}]."
            )
            return response(rejected)
        assert invocation == 3
        footer = user_payload(arguments["prompt"][-1])
        assert footer["draft_to_repair"] == rejected
        assert footer["publication_gap"]["named_authority_gaps"]
        assert arguments["structured_response_format"] is None
        return response(
            "8917 sayılı Faaliyet Kanunu'nun 27 inci maddesi uygulanır "
            f"[{direct_number}]."
        )

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 3
    assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"
    assert checkpoints[-1]["evidence"]["included"] == [direct_number]


def test_native_scope_preserves_excluded_label_snapshot_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, _llm, _checkpoints, _queue = setup_run(monkeypatch)
    run_id = uuid4()
    kwargs["filters"] = kwargs["filters"].model_copy(
        update={
            "regulatory_label_search_enabled": True,
            "regulatory_label_run_ids": (run_id,),
        }
    )
    runtime.run_asv3_loop(**kwargs)
    assert broker.scope is not None
    assert broker.scope.regulatory_workflow_mode == "standard"
    assert broker.scope.regulatory_label_search_enabled is True
    assert broker.scope.regulatory_label_run_ids == (run_id,)


def test_targeted_verification_is_only_invoked_when_model_selects_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, _checkpoints, _queue = setup_run(monkeypatch)
    baseline = llm.invoke.side_effect
    invocation = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            return baseline(**arguments)
        if invocation == 2:
            return response(
                calls=[("verify_claim", {"claim": "Tamir [1].", "citations": [1]})]
            )
        if invocation == 3:
            assert arguments["structured_response_format"] is not None
            originals = json.loads(user_payload(arguments["prompt"][-1])["evidence"])
            assert [row["citation"] for row in originals] == [1]
            return response(
                json.dumps(
                    {
                        "status": "supported",
                        "explanation": "Özgün hüküm bu iddiayı destekliyor.",
                        "required_conditions": [],
                        "missing_conditions": [],
                        "evidence_numbers": [1],
                        "safe_to_publish": True,
                    }
                )
            )
        assert invocation == 4
        assert arguments["structured_response_format"] is None
        results = [
            message
            for message in arguments["prompt"]
            if isinstance(message, ToolMessage)
        ]
        assert any(
            json.loads(message.content)["outcome"]["data"].get("status") == "supported"
            for message in results
        )
        return response("Tamir [1], değiştirme [2].")

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 4


def test_citation_guard_rejects_only_partial_delivery_before_exact_native_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    baseline = llm.invoke.side_effect
    invocation = 0
    record_delivery = EvidenceLedger.record_delivery

    def deliver(
        ledger: EvidenceLedger,
        call_id: str,
        flow: str,
        records: Any,
    ) -> None:
        selected = list(records)
        if invocation == 2:
            selected = [
                {**record, "text": record["text"][:10]}
                if isinstance(record.get("text"), str)
                else record
                for record in selected
            ]
        record_delivery(ledger, call_id, flow, selected)

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            return baseline(**arguments)
        if invocation == 3:
            footer = user_payload(arguments["prompt"][-1])
            assert footer["draft_to_repair"] == "Tamir [1], değiştirme [2]."
            assert footer["publication_gap"]["undelivered_citations"] == [1, 2]
        return response("Tamir [1], değiştirme [2].")

    monkeypatch.setattr(EvidenceLedger, "record_delivery", deliver)
    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 3
    assert checkpoints[-1]["publication_status"] == "found"


@pytest.mark.parametrize(
    "consent,intent", [(False, False), (False, True), (True, False), (True, True)]
)
def test_external_capability_requires_host_permission_and_first_action_explicit_intent(
    monkeypatch: pytest.MonkeyPatch, consent: bool, intent: bool
) -> None:
    kwargs, _broker, llm, _checkpoints, _queue = setup_run(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "build_external_specs",
        lambda *_args, **_kwargs: [
            ToolSpec(
                name="external_read_public_source",
                description="Authorized external read",
                parameters={"type": "object", "properties": {}},
                external=True,
                handler=lambda _args, _ctx: ToolOutcome(
                    status=OutcomeStatus.FOUND, summary="Read"
                ),
            )
        ],
    )
    invocation = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        names = {tool["function"]["name"] for tool in arguments["tools"]}
        if invocation == 1:
            assert "external_read_public_source" not in names
            return response(
                calls=[
                    (
                        "report_progress",
                        {
                            "title": "Kaynak tercihi",
                            "message": "İstenen kaynak kapsamını değerlendiriyorum.",
                            "_language": "tr",
                            "_external_requested": intent,
                        },
                    )
                ]
            )
        assert invocation == 2
        assert ("external_read_public_source" in names) is (consent and intent)
        return response(
            calls=[
                (
                    "submit_partial_answer",
                    {"answer": "İstenen hükmün özgün metni henüz incelenmedi."},
                )
            ]
        )

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs, allow_external=consent)
    assert llm.invoke.call_count == 2


def test_resume_retains_native_history_and_revalidates_originals_without_profile_or_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    runtime.run_asv3_loop(**kwargs)
    previous = checkpoints[-1]
    packets(queue)
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: previous)
    llm.reset_mock()

    def resume(**arguments: Any) -> ModelResponse:
        assert {row["text"] for row in delivered_originals(arguments)} == {
            chunk.text for chunk in broker.chunks.values()
        }
        return response("Tamir [1], değiştirme [2].")

    llm.invoke.side_effect = resume
    runtime.run_asv3_loop(**kwargs, resume_message_id=2)
    assert llm.invoke.call_count == 1
    assert checkpoints[-1]["evidence"]["records"] == previous["evidence"]["records"]
    assert {item.chunk_id for item in broker.revalidated} == {"chunk-0", "chunk-1"}
    assert all(
        packet.obj.language == "tr"
        for packet in packets(queue)
        if isinstance(packet.obj, ASv3Progress)
    )


def test_runtime_has_no_legacy_execution_limit_and_retains_more_than_six_native_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    invocation = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation <= 70:
            return response(
                calls=[
                    (
                        "report_progress",
                        {
                            "title": "Kaynak kapsamı",
                            "message": f"İlgili bağımsız hususu değerlendiriyorum: {invocation}.",
                            "_language": "tr",
                        },
                    )
                ]
            )
        assert (
            len(
                [
                    message
                    for message in arguments["prompt"]
                    if isinstance(message, AssistantMessage)
                ]
            )
            > 6
        )
        return response(
            calls=[
                (
                    "submit_partial_answer",
                    {"answer": "İstenen özgün hüküm henüz incelenmedi."},
                )
            ]
        )

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 71
    assert checkpoints[-1]["budget"]["decisions"] == 71
    assert len(checkpoints[-1]["turns"]) == 71
    assert checkpoints[-1]["publication_stop_reason"] == "native_partial_published"


def test_zero_evidence_discloses_precise_gap_without_uncited_legal_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    invocation = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            return response("Her bedelsiz makine vergiden muaftır.")
        assert (
            user_payload(arguments["prompt"][-1])["publication_gap"]["missing"]
            == "original legal evidence"
        )
        return response(
            calls=[
                (
                    "submit_partial_answer",
                    {
                        "answer": "Bedelsiz yeni makinenin vergilendirilmesine ilişkin özgün hüküm henüz incelenmedi.",
                        "_language": "tr",
                    },
                )
            ]
        )

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 2
    assert "vergiden muaftır" not in kwargs["state_container"].answer_tokens
    assert checkpoints[-1]["publication_status"] == "partial"
    assert not any(isinstance(packet.obj, CitationInfo) for packet in packets(queue))


def test_clarification_can_end_turn_without_research_or_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    llm.invoke.side_effect = [
        response(
            calls=[
                (
                    "ask_user",
                    {
                        "question": "Makine garanti kapsamında mı gönderiliyor?",
                        "_language": "tr",
                    },
                )
            ]
        )
    ]
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 1
    assert (
        kwargs["state_container"].answer_tokens
        == "Makine garanti kapsamında mı gönderiliyor?"
    )
    assert checkpoints[-1]["publication_stop_reason"] == "clarification_requested"


def test_cancel_discards_late_parallel_results_and_durable_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    broker.block = True
    errors: list[Exception] = []

    def run() -> None:
        try:
            runtime.run_asv3_loop(**kwargs)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert broker.entered.wait(3)
        writes_before_cancel = len(checkpoints)
        broker.cancelled.set()
        thread.join(3)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], RunStopped)
        assert len(checkpoints) == writes_before_cancel
        assert llm.invoke.call_count == 1
        assert kwargs["state_container"].answer_tokens is None
        output = packets(queue)
        assert not any(
            isinstance(packet.obj, (AgentResponseStart, AgentResponseDelta))
            for packet in output
        )
        assert [
            packet.obj.phase
            for packet in output
            if isinstance(packet.obj, ASv3Progress)
        ][-1] == "cancelled"
    finally:
        broker.release.set()
        thread.join(3)


def test_canonical_revalidation_rejects_acl_revocation_before_any_answer_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    writes_before_cancel = 0

    def revoke(_items: list[EvidenceItem], context: RunContext) -> None:
        nonlocal writes_before_cancel
        writes_before_cancel = len(checkpoints)
        broker.cancelled.set()
        context.check_active()

    monkeypatch.setattr(broker, "revalidate_evidence", revoke)
    with pytest.raises(RunStopped):
        runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 2
    assert len(checkpoints) == writes_before_cancel
    assert kwargs["state_container"].answer_tokens is None
    assert not any(
        isinstance(packet.obj, (AgentResponseStart, AgentResponseDelta, CitationInfo))
        for packet in packets(queue)
    )


def test_final_evidence_keeps_uncited_same_source_exception_without_unrelated_sources() -> (
    None
):
    from onyx.context.search.models import SearchDoc

    ledger, context = EvidenceLedger(), RunContext()
    for identity, chunk_id, text in [
        ("law", "rule", "Operative result."),
        ("law", "exception", "Only after the approved document is supplied."),
        ("unrelated", "other", "An unrelated rule."),
    ]:
        ledger.add(
            [
                EvidenceItem(
                    source_id=identity,
                    chunk_id=chunk_id,
                    text=text,
                    search_doc=SearchDoc(
                        document_id=identity,
                        chunk_ind=0,
                        semantic_identifier=identity,
                        link="https://example.test/law",
                        blurb=text,
                        source_type=DocumentSource.USER_FILE,
                        boost=0,
                        hidden=False,
                        score=1,
                        metadata={},
                        match_highlights=[],
                    ),
                )
            ],
            context,
        )
    records = json.loads(runtime._evidence_record(ledger, "Result [1]."))
    assert [row["citation"] for row in records] == [1, 2]
    assert records[1]["text"] == "Only after the approved document is supplied."
