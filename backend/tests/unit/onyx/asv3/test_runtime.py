"""Exercise the native harness and publication path with external boundaries faked."""

import copy
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

from onyx.asv3 import llm_adapter, runtime
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
from onyx.asv3.source_metadata_transport import expand_source_metadata
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
from onyx.prompts.asv3.research import COORDINATOR_PROMPT, RESEARCHER_PROMPT
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
    CitationInfo,
    Packet,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool

pytestmark = pytest.mark.usefixtures("empty_source_inventory")

ASSEMBLY_TOOLS = {
    "assemble_answers",
    "repair_question_answer",
    "read_evidence",
    "resolve_source",
    "read_provision",
    "read_chunk",
    "read_chunk_context",
    "read_source_range",
    "follow_reference",
    "search_source_text",
    "search_corpus",
    "query_corpus",
    "diagnose_source",
}


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
        return cast(dict[str, Any], json.loads(content))
    assert isinstance(content, str)
    payload, offset = json.JSONDecoder().raw_decode(content)
    assert content[offset:] in ("", "\n\n" + llm_adapter.DEFAULT_RESPONSE_PREFERENCES)
    return cast(dict[str, Any], payload)


def delivered_originals(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    catalogue = {
        row["citation"]: row
        for row in expand_source_metadata(user_payload(arguments["prompt"][-1]))
    }
    for message in arguments["prompt"]:
        if isinstance(message, ToolMessage):
            result.extend(json.loads(message.content).get("original_evidence", []))
        elif isinstance(message, UserMessage):
            result.extend(user_payload(message).get("original_evidence", []))
    return [
        {**catalogue[row["identity_ref"]], **row} if "identity_ref" in row else row
        for row in result
    ]


def run_independent(
    *,
    external_requested: bool = False,
    first_decision: ModelResponse | None = None,
    **kwargs: Any,
) -> None:
    """Script the two host decisions while exercising the real child source path."""
    if kwargs.get("research_profile") == "normal":
        runtime.run_asv3_loop(
            **{key: value for key, value in kwargs.items() if key != "test_language"}
        )
        return
    llm = kwargs["llm"]
    child_script = llm.invoke.side_effect

    def invoke(**arguments: Any) -> ModelResponse:
        prompt = arguments["prompt"]
        if prompt[0].content == COORDINATOR_PROMPT:
            names = {tool["function"]["name"] for tool in arguments["tools"]}
            if "research_questions" in names:
                assert {"submit_answer", "ask_user"} <= names
                assert arguments["tool_choice"] == llm_adapter.ToolChoiceOptions.AUTO
                if first_decision is not None:
                    return first_decision
                request = user_payload(prompt[1])["request"]
                return response(
                    calls=[
                        (
                            "research_questions",
                            {
                                "questions": [
                                    {
                                        "question_id": "comparison",
                                        "question": request,
                                        "parent_question_ids": [1],
                                        "public_title": "İstenen karşılaştırma"
                                        if kwargs.get("test_language", "tr") == "tr"
                                        else "Requested comparison",
                                        "public_message": "Bu sonucun koşulları özgün kaynaklardan inceleniyor."
                                        if kwargs.get("test_language", "tr") == "tr"
                                        else "I am checking the conditions in original sources.",
                                    }
                                ],
                                "_language": kwargs.get("test_language", "tr"),
                                "_external_requested": external_requested,
                            },
                        )
                    ]
                )
            assert "assemble_answers" in names and names <= ASSEMBLY_TOOLS
            footer = user_payload(prompt[-1])
            originals = {row["citation"]: row for row in delivered_originals(arguments)}
            for answer in footer["independent_answers"]:
                for number in answer["evidence_numbers"]:
                    assert number in originals
                    original = originals[number]
                    assert original["start_char"] == 0
                    assert original["end_char"] == original["total_chars"]
            if kwargs.get("resume_message_id") is not None and callable(child_script):
                child_script(**arguments)
            return response(
                calls=[
                    (
                        "assemble_answers",
                        {
                            "order": [
                                item["question_id"]
                                for item in footer["independent_answers"]
                            ]
                        },
                    )
                ]
            )
        assert prompt[0].content != COORDINATOR_PROMPT
        if callable(child_script):
            return child_script(**arguments)
        return next(child_script)

    llm.invoke.side_effect = invoke
    call_kwargs = {
        key: value for key, value in kwargs.items() if key != "test_language"
    }
    try:
        runtime.run_asv3_loop(**call_kwargs)
    finally:
        llm.invoke.side_effect = child_script


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

    def related_sources_for_evidence(
        self, _item: EvidenceItem, _context: RunContext
    ) -> dict[str, JsonValue] | None:
        return None

    def related_source_navigation(self) -> list[dict[str, JsonValue]]:
        return []

    def shared_read_fence(self, source_id: str, context: RunContext) -> str:
        context.check_active()
        if source_id not in self.chunks:
            raise PermissionError("Source is outside the captured test scope")
        return f"captured-test-source:{source_id}"

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
    monkeypatch.setattr(runtime, "load_asv3_session_checkpoint", lambda **_kwargs: None)
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="scripted",
        temperature=0,
        max_input_tokens=500000,
    )

    def with_seed(seed: int) -> MagicMock:
        selected = MagicMock(spec=LLM)
        selected.config = llm.config.model_copy(update={"seed": seed})
        selected.invoke.side_effect = llm.invoke
        return selected

    llm.with_seed.side_effect = with_seed
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
        test_language=language,
    )
    return kwargs, broker, llm, checkpoints, queue


def test_experimental_runtime_keeps_selected_model_for_search_and_source_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, selected, _, _ = setup_run(monkeypatch)
    secondary = MagicMock(spec=LLM)
    secondary.invoke.side_effect = AssertionError("Secondary model must not run")
    vision_models: list[LLM] = []

    def create_broker(
        _user: User,
        scope: IndexFilters,
        *,
        vision_llm: LLM,
        allow_numbered_title_fallback: bool,
    ) -> CorpusBoundary:
        assert allow_numbered_title_fallback is True
        vision_models.append(vision_llm)
        broker.scope = scope
        return broker

    monkeypatch.setattr(runtime, "CorpusBroker", create_broker)
    search = MagicMock(spec=SearchTool)
    scoped_search = MagicMock(spec=SearchTool)
    search.llm = secondary
    search.fork_for_independent_context.return_value = scoped_search
    kwargs.update(
        research_profile="experimental", research_llm=secondary, tools=[search]
    )
    runtime.run_asv3_loop(
        **{key: value for key, value in kwargs.items() if key != "test_language"}
    )

    assert vision_models == [selected]
    assert scoped_search.llm is selected
    assert search.llm is secondary
    assert selected.invoke.call_count == 2
    secondary.invoke.assert_not_called()


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
    run_independent(**kwargs)
    assert llm.invoke.call_count == 4
    assert broker.peak == 2
    assert [call.kwargs["prompt"][0].content for call in llm.invoke.call_args_list] == [
        COORDINATOR_PROMPT,
        RESEARCHER_PROMPT,
        RESEARCHER_PROMPT,
        COORDINATOR_PROMPT,
    ]
    for call in llm.invoke.call_args_list:
        assert call.kwargs["structured_response_format"] is None
        assert call.kwargs["timeout_override"] is None
        assert (
            user_payload(call.kwargs["prompt"][1])["assistant_instructions"]
            == kwargs["custom_agent_prompt"]
        )
    assert llm.config.model_provider == provider and llm.config.model_name == model_name
    saved = checkpoints[-1]
    assert saved["execution_mode"] == "native"
    assert saved["prompt_version"] == runtime.PROMPT_VERSION
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
        "research_questions",
    ]
    child_messages = llm.invoke.call_args_list[2].kwargs["prompt"]
    child_turn = next(
        message for message in child_messages if isinstance(message, AssistantMessage)
    )
    child_results = [
        message for message in child_messages if isinstance(message, ToolMessage)
    ]
    assert [call.function.name for call in child_turn.tool_calls] == [
        "read_source_range",
        "read_source_range",
    ]
    assert [message.tool_call_id for message in child_results] == [
        call.id for call in child_turn.tool_calls
    ]
    assert saved["question_research"]["answers"][0]["answer"] in saved["last_draft"]
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


@pytest.mark.parametrize("research_profile", ["normal", "deep"])
def test_followup_reuses_session_originals_without_repeating_source_research(
    monkeypatch: pytest.MonkeyPatch,
    research_profile: str,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    kwargs["research_profile"] = research_profile
    run_independent(**kwargs)
    prior = copy.deepcopy(checkpoints[-1])
    original_question = kwargs["simple_chat_history"][-1].message
    prior_request = prior["request"]
    seen_loads: list[dict[str, Any]] = []

    def load(**arguments: Any) -> dict[str, Any]:
        seen_loads.append(arguments)
        return prior

    monkeypatch.setattr(runtime, "load_asv3_session_checkpoint", load)

    def no_source_read(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Originals were already retained")

    monkeypatch.setattr(broker, "page", no_source_read)
    kwargs["user_message_id"] += 2
    kwargs["assistant_message_id"] += 2
    kwargs["simple_chat_history"] = [
        *kwargs["simple_chat_history"],
        ChatMessageSimple(
            message="Önceki kaynaklı yanıt",
            token_count=10,
            message_type=MessageType.ASSISTANT,
        ),
        ChatMessageSimple(
            message="Peki bu durumda hangi hüküm uygulanıyor?",
            token_count=12,
            message_type=MessageType.USER,
        ),
    ]
    llm.reset_mock()
    kwargs["state_container"] = ChatStateContainer()

    def answer_from_memory(**arguments: Any) -> ModelResponse:
        assert (arguments["prompt"][0].content == COORDINATOR_PROMPT) == (
            research_profile == "deep"
        )
        payload = user_payload(arguments["prompt"][1])
        assert original_question in payload["conversation"]
        footer = user_payload(arguments["prompt"][-1])
        assert prior_request in footer["session_research"]["requests"]
        originals = delivered_originals(arguments)
        assert {row["text"] for row in originals} == {
            chunk.text for chunk in broker.chunks.values()
        }
        assert arguments["timeout_override"] is None
        return response(
            calls=[
                (
                    "submit_answer",
                    {
                        "answer": "Önceden okunmuş özgün hüküm uygulanır [1].\n\nKoşul ve uygulama ayrıntısı [2].",
                        "basis": "originals",
                    },
                )
            ]
        )

    llm.invoke.side_effect = answer_from_memory
    runtime.run_asv3_loop(
        **{key: value for key, value in kwargs.items() if key != "test_language"}
    )
    assert llm.invoke.call_count == 1
    assert seen_loads == [
        {
            "chat_session_id": kwargs["chat_session_id"],
            "user_message_id": kwargs["user_message_id"],
            "user_id": kwargs["user"].id,
        }
    ]
    final = checkpoints[-1]
    assert final["publication_status"] == "found"
    assert (
        final["session_research"]["requests"][-1]
        == kwargs["simple_chat_history"][-1].message
    )
    assert final["session_research"]["reused_evidence_numbers"] == [1, 2]
    assert final["budget"]["tools"] == 0
    assert final["workers"]["tasks"] == []
    assert final["question_research"]["answers"] == []


def test_related_questions_use_fresh_children_and_keep_full_bodies_and_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    tasks = [
        "Garanti kapsamındaki tamirin sonucu nedir?",
        "Yeni makine gönderilmesinin sonucu nedir?",
    ]
    decisions = {task: 0 for task in tasks}
    bodies: dict[str, str] = {}
    lock = threading.Lock()

    def scripted(**arguments: Any) -> ModelResponse:
        prompt = arguments["prompt"]
        prefix, footer = user_payload(prompt[1]), user_payload(prompt[-1])
        if prompt[0].content == COORDINATOR_PROMPT:
            if not footer["question_research_started"]:
                return response(
                    calls=[
                        (
                            "research_questions",
                            {
                                "questions": [
                                    {
                                        "question_id": str(index),
                                        "question": task,
                                        "parent_question_ids": [1],
                                        "public_title": "İstenen sonuç araştırılıyor",
                                        "public_message": "Bu sonucun koşulları özgün kaynaklardan inceleniyor.",
                                    }
                                    for index, task in enumerate(tasks)
                                ],
                                "_language": "tr",
                            },
                        )
                    ]
                )
            originals = delivered_originals(arguments)
            assert len(originals) == 2
            assert {item["text"] for item in originals} == {
                item.text for item in broker.chunks.values()
            }
            assert all(
                item["start_char"] == 0 and item["end_char"] == item["total_chars"]
                for item in originals
            )
            assert {item["answer"] for item in footer["independent_answers"]} == set(
                bodies.values()
            )
            return response(calls=[("assemble_answers", {"order": ["1", "0"]})])
        assert prompt[0].content == RESEARCHER_PROMPT
        task = prefix["request"]
        assert task in tasks
        assert kwargs["simple_chat_history"][0].message in prefix["conversation"]
        assert footer["request"] == task
        assert "independent_answers" not in footer
        with lock:
            decisions[task] += 1
            ordinal = decisions[task]
        index = tasks.index(task)
        if ordinal == 1:
            assert not any(
                isinstance(message, (AssistantMessage, ToolMessage))
                for message in prompt
            )
            assert not delivered_originals(arguments)
            return response(
                calls=[
                    ("read_source_range", {"source_id": str(broker.sources[index].id)})
                ]
            )
        assert ordinal == 2
        assistant = [
            message for message in prompt if isinstance(message, AssistantMessage)
        ]
        assert len(assistant) == 1
        assert assistant[0].tool_calls[0].function.name == "read_source_range"
        assert json.loads(assistant[0].tool_calls[0].function.arguments)[
            "source_id"
        ] == str(broker.sources[index].id)
        originals = delivered_originals(arguments)
        assert len(originals) == 1
        assert originals[0]["text"] == broker.chunks[str(broker.sources[index].id)].text
        body = (
            f"Sonuç {index}: "
            + "Kaynaklı koşul ve sonraki işlem. " * 450
            + f"[{originals[0]['citation']}]"
        )
        assert len(body) > 12_000
        with lock:
            bodies[task] = body
        return response(body)

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(
        **{key: value for key, value in kwargs.items() if key != "test_language"}
    )
    assert llm.invoke.call_count == 6
    assert broker.peak == 2
    assert decisions == {task: 2 for task in tasks}
    saved = checkpoints[-1]
    assert saved["publication_status"] == "found"
    assert len(saved["workers"]["tasks"]) == 2
    assert all(
        item["local_budget"]["used"] == {"tools": 1, "decisions": 2}
        for item in saved["workers"]["tasks"]
    )
    assert all(body in saved["last_draft"] for body in bodies.values())
    assert saved["last_draft"].index(bodies[tasks[1]]) < saved["last_draft"].index(
        bodies[tasks[0]]
    )
    assert {item.chunk_id for item in broker.revalidated} == {"chunk-0", "chunk-1"}


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
    run_independent(**kwargs)
    assert llm.invoke.call_count == 4 and searches == 1
    assert [receipt["call"]["name"] for receipt in checkpoints[-1]["receipts"]] == [
        "research_questions",
        "assemble_answers",
    ]
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
    run_independent(**kwargs)
    updates = [
        packet.obj for packet in packets(queue) if isinstance(packet.obj, ASv3Progress)
    ]
    assert llm.invoke.call_count == 4
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
    run_independent(**kwargs)
    assert llm.invoke.call_count == 4
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
    run_independent(**kwargs)
    assert llm.invoke.call_count == 5
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
    run_independent(**kwargs)
    assert llm.invoke.call_count == 5
    assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"
    assert checkpoints[-1]["evidence"]["included"] == [direct_number]


def configure_governing_basis_sources(broker: CorpusBoundary) -> tuple[str, str]:
    broker.barrier = threading.Barrier(1)
    lower, governing = (str(source.id) for source in broker.sources)
    for identity, kind, title, text in (
        (lower, "tebliğ", "UYGULAMA TEBLİĞİ", "Başvuruda belge sunulur."),
        (
            governing,
            "kanun",
            "8917 SAYILI FAALİYET KANUNU",
            "Faaliyet için izin alınır.",
        ),
    ):
        chunk = broker.chunks[identity]
        broker.chunks[identity] = replace(
            chunk,
            text=text,
            heading_path=(title, "MADDE 27"),
            metadata={
                **chunk.metadata,
                "document_type": kind,
                "title": title,
                "article_no": "27",
            },
        )
    return lower, governing


@pytest.mark.parametrize(
    "research_profile,closure",
    [("experimental", "original"), ("experimental", "gap"), ("normal", "deleted")],
)
def test_runtime_retains_rejected_governing_basis_only_for_experimental(
    monkeypatch: pytest.MonkeyPatch, research_profile: str, closure: str
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs["research_profile"] = research_profile
    lower, governing = configure_governing_basis_sources(broker)
    page = MagicMock(wraps=broker.page)
    monkeypatch.setattr(broker, "page", page)
    named = "8917 sayılı Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir [1]."
    anonymous = "İzin gerekir [1]."
    partial = (
        "Başvuruda belge sunulur [1].\n\n"
        "8917 sayılı Faaliyet Kanunu'nun 27. maddesinin özgün metni incelenemedi."
    )
    invocation = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            return response(
                calls=[("read_source_range", {"source_id": lower, "_language": "tr"})]
            )
        if invocation == 2:
            assert [row["citation"] for row in delivered_originals(arguments)] == [1]
            return response(
                calls=[("submit_answer", {"answer": named, "basis": "originals"})]
            )
        footer = user_payload(arguments["prompt"][-1])
        if invocation == 3:
            assert footer["draft_to_repair"] == named
            assert footer["publication_gap"]["named_authority_gaps"]
            if research_profile == "experimental":
                return response(anonymous)
            return response(
                calls=[("submit_answer", {"answer": anonymous, "basis": "originals"})]
            )
        assert research_profile == "experimental"
        retained = footer["governing_source_requirements"][
            "retained_authority_requirements"
        ]
        assert len(retained) == 1
        assert retained[0]["instrument_number"] == "8917"
        if 4 <= invocation <= 7:
            assert footer["draft_to_repair"] == anonymous
            assert footer["publication_gap"]["retained_authority_requirements"]
            assert "named_authority_gaps" not in footer["publication_gap"]
            if invocation < 7:
                return response(anonymous)
            if closure == "gap":
                return response(calls=[("submit_partial_answer", {"answer": partial})])
            return response(calls=[("read_source_range", {"source_id": governing})])
        assert invocation == 8 and closure == "original"
        original = next(
            row for row in delivered_originals(arguments) if row["citation"] == 2
        )
        assert original["source_id"] == governing
        assert original["text"] == broker.chunks[governing].text
        return response(
            calls=[
                ("submit_answer", {"answer": "İzin gerekir [2].", "basis": "originals"})
            ]
        )

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    saved = checkpoints[-1]
    submit_receipts = [
        row for row in saved["receipts"] if row["call"]["name"] == "submit_answer"
    ]
    assert submit_receipts[0]["outcome"]["status"] == "partial"
    assert saved["final_publication_gap"] is None
    if research_profile == "normal":
        assert invocation == 3 and page.call_count == 1
        assert submit_receipts[1]["outcome"]["status"] == "found"
        assert "authority_requirements" not in saved
    else:
        blocked = [
            row
            for row in saved["receipts"]
            if row["call"]["name"] == "finalization_status"
        ]
        assert len(blocked) == 4
        assert all(row["outcome"]["status"] == "partial" for row in blocked)
        assert all(
            row["outcome"]["data"]["retained_authority_requirements"] for row in blocked
        )
        assert len(saved["authority_requirements"]["records"]) == 1
        assert invocation == (8 if closure == "original" else 7)
        assert page.call_count == (2 if closure == "original" else 1)
        if closure == "original":
            assert any(
                row["citation"] == 2 and row["complete"] is True
                for row in saved["evidence"]["deliveries"][-1]["records"]
            )
    assert saved["publication_status"] == ("partial" if closure == "gap" else "found")
    assert saved["evidence"]["included"] == ([2] if closure == "original" else [1])


def test_experimental_resume_retains_rejected_basis_without_another_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs["research_profile"] = "experimental"
    lower, _governing = configure_governing_basis_sources(broker)
    page = MagicMock(wraps=broker.page)
    monkeypatch.setattr(broker, "page", page)
    named = "8917 sayılı Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir [1]."
    anonymous = "İzin gerekir [1]."
    invocation = 0

    def interrupted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation == 1:
            return response(
                calls=[("read_source_range", {"source_id": lower, "_language": "tr"})]
            )
        if invocation == 2:
            return response(
                calls=[("submit_answer", {"answer": named, "basis": "originals"})]
            )
        assert invocation == 3
        footer = user_payload(arguments["prompt"][-1])
        assert footer["publication_gap"]["retained_authority_requirements"]
        broker.cancelled.set()
        raise RunStopped("cancelled")

    llm.invoke.side_effect = interrupted
    with pytest.raises(RunStopped):
        runtime.run_asv3_loop(**kwargs)
    previous = copy.deepcopy(checkpoints[-1])
    assert len(previous["authority_requirements"]["records"]) == 1
    broker.cancelled.clear()
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: previous)
    kwargs["resume_message_id"] = 2
    kwargs["research_profile"] = "normal"
    llm.reset_mock()
    invocation = 0

    def resumed(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        footer = user_payload(arguments["prompt"][-1])
        assert (
            len(
                footer["governing_source_requirements"][
                    "retained_authority_requirements"
                ]
            )
            == 1
        )
        assert any(row["citation"] == 1 for row in delivered_originals(arguments))
        if invocation == 1:
            return response(
                calls=[("submit_answer", {"answer": anonymous, "basis": "originals"})]
            )
        assert invocation == 2
        assert footer["draft_to_repair"] == anonymous
        assert footer["publication_gap"]["retained_authority_requirements"]
        return response(
            calls=[
                (
                    "submit_partial_answer",
                    {
                        "answer": "Başvuruda belge sunulur [1].\n\n8917 sayılı Faaliyet Kanunu'nun 27. maddesinin özgün metni incelenemedi."
                    },
                )
            ]
        )

    llm.invoke.side_effect = resumed
    runtime.run_asv3_loop(**kwargs)
    saved = checkpoints[-1]
    assert invocation == 2 and page.call_count == 1
    assert saved["research_profile"] == "experimental"
    assert saved["authority_requirements"] == previous["authority_requirements"]
    assert saved["publication_status"] == "partial"
    assert saved["final_publication_gap"] is None


@pytest.mark.parametrize("rejected_citation", ["parentheses", "unknown"])
def test_child_partial_rejection_reaches_next_native_decision_and_repairs_without_read(
    monkeypatch: pytest.MonkeyPatch, rejected_citation: str
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    source_id = str(broker.sources[1].id)
    chunk = broker.chunks[source_id]
    broker.chunks[source_id] = replace(
        chunk,
        text="Başvuru ve inceleme (2) aşamadır.",
        heading_path=("8917 SAYILI FAALİYET KANUNU", "MADDE 27"),
        metadata={
            **chunk.metadata,
            "document_type": "kanun",
            "title": "8917 SAYILI FAALİYET KANUNU",
            "article_no": "27",
        },
    )
    page = MagicMock(wraps=broker.page)
    monkeypatch.setattr(broker, "page", page)
    baseline = llm.invoke.side_effect
    invocation = 0
    direct_number = 0
    rejected = ""
    repaired = ""

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation, direct_number, rejected, repaired
        invocation += 1
        if invocation == 1:
            return baseline(**arguments)
        if invocation == 2:
            originals = delivered_originals(arguments)
            number = next(
                row["citation"] for row in originals if row["chunk_id"] == "chunk-1"
            )
            direct_number = number
            claim = "8917 sayılı Faaliyet Kanunu m. 27 uyarınca başvuru ve inceleme (2) aşamadır"
            marker = f"({number})" if rejected_citation == "parentheses" else "[999]"
            rejected = f"{claim} {marker}."
            repaired = f"{claim} [{number}]."
            return response(calls=[("submit_partial_answer", {"answer": rejected})])
        assert invocation == 3
        footer = user_payload(arguments["prompt"][-1])
        assert footer["draft_to_repair"] == rejected
        gap = footer["publication_gap"]
        if rejected_citation == "parentheses":
            assert gap["citation_format"] == "[n]"
            entry = gap["named_authority_gaps"][0]
            assert entry["inline_evidence"] == []
            assert entry["matching_original_evidence"] == [direct_number]
        else:
            assert gap["unknown_citations"] == [999]
        assert any(
            row["citation"] == direct_number for row in footer["original_evidence"]
        )
        return response(calls=[("submit_partial_answer", {"answer": repaired})])

    llm.invoke.side_effect = scripted
    run_independent(**kwargs)
    assert invocation == 3
    assert llm.invoke.call_count == 5
    assert checkpoints[-1]["publication_status"] == "partial"
    assert checkpoints[-1]["final_publication_gap"] is None
    assert checkpoints[-1]["evidence"]["included"] == [direct_number]
    assert page.call_count == 2
    assert checkpoints[-1]["question_research"]["answers"][0]["answer"] == repaired
    assert rejected not in kwargs["state_container"].answer_tokens
    assert "inceleme (2) aşamadır " in kwargs["state_container"].answer_tokens


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
    run_independent(**kwargs)
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
    run_independent(**kwargs)
    assert llm.invoke.call_count == 6


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
    run_independent(**kwargs)
    assert llm.invoke.call_count == 5
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
            assert ("external_read_public_source" in names) is (consent and intent)
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
    run_independent(**kwargs, allow_external=consent, external_requested=intent)
    assert llm.invoke.call_count == 4


def test_resume_retains_native_history_and_revalidates_originals_without_profile_or_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    run_independent(**kwargs)
    previous = checkpoints[-1]
    previous["public_profile"]["requires_sources"] = False
    packets(queue)
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: previous)
    llm.reset_mock()

    def resume(**arguments: Any) -> ModelResponse:
        assert {row["text"] for row in delivered_originals(arguments)} == {
            chunk.text for chunk in broker.chunks.values()
        }
        return response("Tamir [1], değiştirme [2].")

    llm.invoke.side_effect = resume
    run_independent(**kwargs, resume_message_id=2)
    assert checkpoints[-1]["public_profile"]["requires_sources"] is True
    assert llm.invoke.call_count == 1
    assert checkpoints[-1]["evidence"]["records"] == previous["evidence"]["records"]
    assert {item.chunk_id for item in broker.revalidated} == {"chunk-0", "chunk-1"}
    assert all(
        packet.obj.language == "tr"
        for packet in packets(queue)
        if isinstance(packet.obj, ASv3Progress)
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_native_runtime_seed_profile_survives_checkpoint_and_legacy_resume(
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    llm.config = llm.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )
    run_independent(**kwargs)
    assert [call.args[0] for call in llm.with_seed.call_args_list] == [31, 1424088823]
    previous = dict(checkpoints[-1])
    saved = previous["native_coordinator_sampling"]
    assert saved["first_decision_started"] is True
    assert saved["first_decision_completed"] is True
    assert len(previous["evidence"]["records"]) == 2
    assert broker.search_adapter is not None
    assert llm.config.seed is None
    if legacy:
        previous.pop("native_coordinator_sampling")
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: previous)
    llm.reset_mock()
    llm.invoke.side_effect = lambda **_kwargs: response("Tamir [1], değiştirme [2].")
    run_independent(**kwargs, resume_message_id=2)
    llm.with_seed.assert_called_once_with(1424088823)
    assert llm.invoke.call_count == 1
    assert checkpoints[-1]["evidence"]["records"] == previous["evidence"]["records"]


def test_native_runtime_explicit_seed_zero_is_uniform_and_profile_change_blocks_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    llm.config = llm.config.model_copy(
        update={
            "model_provider": "vertex_ai",
            "model_name": "gemini-3.8-flash",
            "seed": 0,
        }
    )
    run_independent(**kwargs)
    assert llm.invoke.call_count == 4
    llm.with_seed.assert_not_called()
    previous = checkpoints[-1]
    assert previous["native_coordinator_sampling"]["settings"] == {
        "mode": "explicit_seed",
        "seed": 0,
    }
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: previous)
    llm.reset_mock()
    llm.config = llm.config.model_copy(update={"seed": None})
    with pytest.raises(ValueError, match="same coordinator sampling profile"):
        run_independent(**kwargs, resume_message_id=2)
    llm.invoke.assert_not_called()
    llm.with_seed.assert_not_called()


def test_independent_local_budget_retains_more_than_six_native_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    invocation = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        invocation += 1
        if invocation <= 8:
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
    run_independent(**kwargs)
    assert llm.invoke.call_count == 11
    assert checkpoints[-1]["budget"]["decisions"] == 11
    assert len(checkpoints[-1]["turns"]) == 2
    worker = checkpoints[-1]["workers"]["tasks"][0]
    assert worker["local_budget"]["used"]["decisions"] == 9
    assert checkpoints[-1]["publication_status"] == "partial"


@pytest.mark.parametrize("delegate", [True, False])
def test_zero_evidence_discloses_precise_gap_without_uncited_legal_memory(
    monkeypatch: pytest.MonkeyPatch,
    delegate: bool,
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
    if delegate:
        run_independent(**kwargs)
    else:
        runtime.run_asv3_loop(
            **{key: value for key, value in kwargs.items() if key != "test_language"}
        )
    assert llm.invoke.call_count == (4 if delegate else 2)
    assert "vergiden muaftır" not in kwargs["state_container"].answer_tokens
    assert checkpoints[-1]["publication_status"] == "partial"
    assert not any(isinstance(packet.obj, CitationInfo) for packet in packets(queue))


def test_clarification_can_end_turn_without_research_or_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    clarification = response(
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
    run_independent(**kwargs, first_decision=clarification)
    assert llm.invoke.call_count == 1
    assert (
        kwargs["state_container"].answer_tokens
        == "Makine garanti kapsamında mı gönderiliyor?"
    )
    assert checkpoints[-1]["publication_stop_reason"] == "clarification_requested"


@pytest.mark.parametrize(
    "question,answer,basis",
    [
        (
            "nasılsın",
            "İyiyim, teşekkürler. Sana nasıl yardımcı olabilirim?",
            "conversation",
        ),
        ("Good morning", "Good morning! How can I help?", "conversation"),
        ("4000 avronun iki katı kaçtır?", "8.000 avro.", "scenario"),
    ],
)
def test_first_decision_publishes_social_or_fact_answer_without_research(
    monkeypatch: pytest.MonkeyPatch, question: str, answer: str, basis: str
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs["simple_chat_history"][0].message = question
    llm.invoke.side_effect = [
        response(calls=[("submit_answer", {"answer": answer, "basis": basis})])
    ]
    runtime.run_asv3_loop(
        **{key: value for key, value in kwargs.items() if key != "test_language"}
    )
    assert llm.invoke.call_count == 1
    assert kwargs["state_container"].answer_tokens == answer
    assert checkpoints[-1]["publication_status"] == "found"
    assert checkpoints[-1]["public_profile"]["requires_sources"] is True
    assert checkpoints[-1]["workers"]["tasks"] == []


@pytest.mark.parametrize(
    "candidate,basis,gap_key",
    [
        ("Önceki sonuç [999].", "conversation", "unknown_citations"),
        ("Önceki sonuç [999].", "scenario", "unknown_citations"),
        ("Bu işlemde vergi ödenmez.", "originals", "missing"),
        (
            "4458 sayılı Gümrük Kanunu'nun 142 nci maddesi uyarınca vergi ödenmez.",
            "conversation",
            "named_authority_gaps",
        ),
    ],
)
def test_rejected_terminal_candidate_does_not_change_later_source_policy(
    monkeypatch: pytest.MonkeyPatch, candidate: str, basis: str, gap_key: str
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    responses = iter(
        [
            response(calls=[("submit_answer", {"answer": candidate, "basis": basis})]),
            response("Bu işlemde vergi ödenmez."),
            response(
                calls=[("ask_user", {"question": "Makinenin kullanım amacı nedir?"})]
            ),
        ]
    )

    def invoke(**arguments: Any) -> ModelResponse:
        if llm.invoke.call_count == 2:
            footer = user_payload(arguments["prompt"][-1])
            assert gap_key in footer["publication_gap"]
        if llm.invoke.call_count == 3:
            footer = user_payload(arguments["prompt"][-1])
            assert footer["publication_gap"]["missing"] == "original legal evidence"
        return next(responses)

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(
        **{key: value for key, value in kwargs.items() if key != "test_language"}
    )
    assert llm.invoke.call_count == 3
    assert kwargs["state_container"].answer_tokens == "Makinenin kullanım amacı nedir?"
    assert checkpoints[-1]["public_profile"]["requires_sources"] is True


def test_terminal_answer_cannot_publish_beside_another_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    llm.invoke.side_effect = [
        response(
            calls=[
                ("submit_answer", {"answer": "Erken cevap.", "basis": "conversation"}),
                ("ask_user", {"question": "Makinenin kullanım amacı nedir?"}),
            ]
        )
    ]
    runtime.run_asv3_loop(
        **{key: value for key, value in kwargs.items() if key != "test_language"}
    )
    assert kwargs["state_container"].answer_tokens == "Makinenin kullanım amacı nedir?"
    assert checkpoints[-1]["receipts"][0]["outcome"]["status"] == "denied"


def test_cancel_discards_late_parallel_results_and_durable_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    broker.block = True
    errors: list[Exception] = []

    def run() -> None:
        try:
            run_independent(**kwargs)
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
        assert llm.invoke.call_count == 2
        assert kwargs["state_container"].answer_tokens is None
        output = packets(queue)
        assert not any(
            isinstance(packet.obj, (AgentResponseStart, AgentResponseDelta))
            for packet in output
        )
        assert [
            packet.obj.phase
            for packet in output
            if isinstance(packet.obj, ASv3Progress) and packet.obj.task_id is None
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
        run_independent(**kwargs)
    assert llm.invoke.call_count == 4
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


def test_followup_rebinds_retained_native_citation_before_root_and_child_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hashlib import sha256

    from onyx.asv3.corpus_tools import CorpusBroker
    from onyx.server.asv3_citations import saved_item_from_checkpoint

    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    kwargs["user_message_id"], kwargs["assistant_message_id"] = 11, 12
    source = broker.sources[0]
    original_text = broker.chunks[str(source.id)].text
    native = EvidenceItem(
        source_id=str(source.id),
        text=original_text,
        metadata={
            "derived": True,
            "source_sha256": sha256(original_text.encode()).hexdigest(),
            "locator": {"page": 1},
        },
    )
    monkeypatch.setattr(broker, "source", lambda *_args: source, raising=False)
    native = CorpusBroker.attach_native_citation(
        cast(CorpusBroker, broker), native, 5, 2, RunContext()
    )
    assert native.search_doc is not None
    assert native.search_doc.chunk_ind == -5
    earlier = [
        EvidenceItem(
            source_id=f"external-{index}", text="External", metadata={"external": True}
        )
        if index % 2
        else EvidenceItem(
            source_id=f"unavailable-{index}", chunk_id=f"chunk-{index}", text="Old text"
        )
        for index in range(1, 5)
    ]
    prior: dict[str, Any] = {
        "request": "Önceki sorunun tam senaryosu",
        "evidence": {
            "version": 1,
            "records": [
                {"citation": number, "item": item.model_dump(mode="json")}
                for number, item in enumerate([*earlier, native], 1)
            ],
        },
    }
    unchanged_records = copy.deepcopy(prior["evidence"])

    def load_prior(**_arguments: Any) -> dict[str, Any]:
        assert broker.scope is not None
        prior["scope"] = broker.scope.model_dump(mode="json")
        return prior

    monkeypatch.setattr(runtime, "load_asv3_session_checkpoint", load_prior)
    original_revalidate = broker.revalidate_evidence

    def revalidate(items: list[EvidenceItem], context: RunContext) -> None:
        if any(item.source_id != str(source.id) for item in items):
            raise PermissionError("Prior source is no longer authorized")
        original_revalidate(items, context)

    monkeypatch.setattr(broker, "revalidate_evidence", revalidate)
    rebound: list[EvidenceItem] = []

    def attach(
        item: EvidenceItem, number: int, message_id: int, context: RunContext
    ) -> EvidenceItem:
        attached = CorpusBroker.attach_native_citation(
            cast(CorpusBroker, broker), item, number, message_id, context
        )
        rebound.append(attached)
        return attached

    monkeypatch.setattr(broker, "attach_native_citation", attach, raising=False)
    decision_roles: list[str] = []

    def invoke(**arguments: Any) -> ModelResponse:
        assert len(rebound) == 1
        current = rebound[0]
        assert current.text == original_text and current.text_hash == native.text_hash
        assert current.search_doc is not None
        assert current.search_doc.chunk_ind == -1
        assert current.search_doc.metadata["asv3_citation_preview_url"] == (
            "/api/asv3/citation/12/1"
        )
        originals = delivered_originals(arguments)
        assert [(row["citation"], row["text"]) for row in originals] == [
            (1, original_text)
        ]
        assert originals[0]["start_char"] == 0
        assert originals[0]["end_char"] == len(original_text)
        if arguments["prompt"][0].content == RESEARCHER_PROMPT:
            decision_roles.append("child")
            return response("Önceki özgün hükmün koşulları bu soruya uygulanır [1].")
        decision_roles.append("root")
        names = {tool["function"]["name"] for tool in arguments["tools"]}
        if "research_questions" in names:
            return response(
                calls=[
                    (
                        "research_questions",
                        {
                            "questions": [
                                {
                                    "question_id": "followup",
                                    "question": "Önceki hükmün bu senaryoya uygulanması",
                                    "parent_question_ids": [1],
                                    "public_title": "Hükmün uygulanması",
                                    "public_message": "Önceki özgün hükmün koşulları inceleniyor.",
                                }
                            ],
                            "_language": "tr",
                        },
                    )
                ]
            )
        assert "assemble_answers" in names and names <= ASSEMBLY_TOOLS
        return response(calls=[("assemble_answers", {"order": ["followup"]})])

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(
        **{key: value for key, value in kwargs.items() if key != "test_language"}
    )
    assert decision_roles == ["root", "child", "root"]
    assert prior["evidence"] == unchanged_records
    assert checkpoints[-1]["session_research"]["reused_evidence_numbers"] == [1]
    saved, _scope = saved_item_from_checkpoint(checkpoints[-1], 1)
    assert saved.text == original_text and saved.text_hash == native.text_hash
    assert saved.search_doc is not None and saved.search_doc.chunk_ind == -1
    emitted = [
        packet.obj for packet in packets(queue) if isinstance(packet.obj, CitationInfo)
    ]
    assert len(emitted) == 1
    assert emitted[0].citation_number == 1 and emitted[0].chunk_ind == -1
    assert emitted[0].preview_url == "/api/asv3/citation/12/1"


@pytest.mark.parametrize("research_profile", ["normal", "deep"])
def test_first_decision_profiles_can_answer_without_research(
    monkeypatch: pytest.MonkeyPatch, research_profile: str
) -> None:
    from onyx.prompts.asv3.coordinator_reference import COORDINATOR_REFERENCE_PROMPT

    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    kwargs["research_profile"] = research_profile
    kwargs.pop("test_language")
    kwargs["simple_chat_history"] = [
        ChatMessageSimple(
            message="Merhaba", message_type=MessageType.USER, token_count=1
        )
    ]

    def greet(**arguments: Any) -> ModelResponse:
        instruction = arguments["prompt"][0].content
        tools = {item["function"]["name"] for item in arguments["tools"]}
        if research_profile == "normal":
            assert instruction.startswith(COORDINATOR_REFERENCE_PROMPT)
            assert "research_questions" not in tools
            assert "assemble_answers" not in tools
        else:
            assert instruction == COORDINATOR_PROMPT
            assert "research_questions" in tools
        return response(
            calls=[
                (
                    "submit_answer",
                    {
                        "answer": "Merhaba, nasıl yardımcı olabilirim?",
                        "basis": "conversation",
                        "_language": "tr",
                    },
                )
            ]
        )

    llm.invoke.side_effect = greet
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 1
    assert checkpoints[-1]["research_profile"] == research_profile
    assert any(isinstance(packet.obj, AgentResponseDelta) for packet in packets(queue))


@pytest.mark.parametrize("original_profile", ["normal", "deep"])
def test_checkpoint_resume_pins_original_research_profile(
    monkeypatch: pytest.MonkeyPatch, original_profile: str
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs["research_profile"] = original_profile
    kwargs["simple_chat_history"] = [
        ChatMessageSimple(
            message="Merhaba", token_count=1, message_type=MessageType.USER
        )
    ]
    llm.invoke.side_effect = lambda **_args: response(
        calls=[
            (
                "submit_answer",
                {"answer": "Merhaba!", "basis": "conversation", "_language": "tr"},
            )
        ]
    )
    runtime.run_asv3_loop(**kwargs)
    previous = copy.deepcopy(checkpoints[-1])
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: previous)
    kwargs["research_profile"] = "normal" if original_profile == "deep" else "deep"
    kwargs["resume_message_id"] = 2
    llm.reset_mock()

    def finish(**arguments: Any) -> ModelResponse:
        tools = {item["function"]["name"] for item in arguments["tools"]}
        assert ("research_questions" in tools) == (original_profile == "deep")
        assert arguments["timeout_override"] is None
        return response(
            calls=[("submit_answer", {"answer": "Merhaba!", "basis": "conversation"})]
        )

    llm.invoke.side_effect = finish
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 1
    assert checkpoints[-1]["research_profile"] == original_profile
