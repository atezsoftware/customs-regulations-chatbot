"""Exercise the production harness and citation path with only external boundaries faked."""

import json
import threading
from datetime import date
from queue import Queue
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.asv3 import runtime
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
from onyx.configs.constants import MessageType
from onyx.context.search.models import BaseFilters
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
from onyx.llm.models import ReasoningEffort
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
    CitationInfo,
    Packet,
)


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
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.block = False
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.revalidated: list[EvidenceItem] = []
        self.search_adapter: Any = None

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
    monkeypatch: pytest.MonkeyPatch,
    final: str = "Tamir sonucu [1]; değiştirme sonucu [2].",
) -> tuple[dict[str, Any], CorpusBoundary, MagicMock, list[dict[str, Any]], Queue[Any]]:
    broker = CorpusBoundary()
    monkeypatch.setattr(runtime, "CorpusBroker", lambda *_args, **_kwargs: broker)
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
        max_input_tokens=100000,
    )
    notifications = {
        phase: [
            "Tamir koşullarını inceliyorum",
            "Tamir ve değişim için uygulanacak koşulları karşılaştırıyorum.",
        ]
        for phase in (
            "started",
            "tools",
            "worker",
            "final",
            "completed",
            "failed",
            "cancelled",
            "native_citation",
            "resume",
            "interrupted",
        )
    }
    llm.invoke.side_effect = [
        response(json.dumps({"language": "tr", "notifications": notifications})),
        response(
            calls=[
                ("read_source_range", {"source_id": str(source.id)})
                for source in broker.sources
            ]
            + [
                (
                    "report_progress",
                    {
                        "title": "Tamir ve değişim",
                        "message": "Aynı makinenin tamiri ile yeni makine gönderilmesine ilişkin hükümleri karşılaştırıyorum.",
                    },
                )
            ]
        ),
        response("Tamir [1], değiştirme [2]."),
        response(
            '{"status":"supported","explanation":"Koşullar sağlandı.","required_conditions":[],"missing_conditions":[],"evidence_numbers":[1,2]}'
        ),
        response(final),
    ]
    cache = MagicMock(spec=CacheBackend)
    cache.exists.side_effect = lambda _key: broker.cancelled.is_set()
    user = MagicMock(spec=User)
    user.id = uuid4()
    queue: Queue[Any] = Queue()
    state = ChatStateContainer()
    kwargs = dict(
        emitter=Emitter(queue),
        state_container=state,
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
        filters=BaseFilters(regulatory_chunks_only=True),
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


def packets(queue: Queue[Any]) -> list[Packet]:
    result = []
    while not queue.empty():
        _, packet = queue.get_nowait()
        result.append(packet)
    return result


@pytest.mark.parametrize(
    "consent,intent", [(False, False), (False, True), (True, False), (True, True)]
)
def test_runtime_external_capabilities_require_both_permission_and_user_intent(
    monkeypatch: pytest.MonkeyPatch, consent: bool, intent: bool
) -> None:
    kwargs, _broker, llm, _checkpoints, _queue = setup_run(monkeypatch)
    script = list(llm.invoke.side_effect)
    profile = json.loads(script[0].choice.message.content)
    profile["external_requested"] = intent
    script[0] = response(json.dumps(profile))
    llm.invoke.side_effect = script
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
    runtime.run_asv3_loop(**kwargs, allow_external=consent)
    names = {
        tool["function"]["name"]
        for tool in llm.invoke.call_args_list[1].kwargs["tools"]
    }
    assert ("external_read_public_source" in names) is (consent and intent)


def test_resume_reuses_saved_question_language_without_reclassifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    runtime.run_asv3_loop(**kwargs)
    previous = checkpoints[-1]
    packets(queue)
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: previous)
    llm.reset_mock()
    llm.invoke.side_effect = [
        response("Tamir [1], değiştirme [2]."),
        response(
            '{"status":"supported","explanation":"Koşullar sağlandı.","required_conditions":[],"missing_conditions":[],"evidence_numbers":[1,2]}'
        ),
        response("Tamir sonucu [1]; değiştirme sonucu [2]."),
    ]
    runtime.run_asv3_loop(**kwargs, resume_message_id=2)
    assert llm.invoke.call_count == 3
    assert all(
        packet.obj.language == "tr"
        for packet in packets(queue)
        if isinstance(packet.obj, ASv3Progress)
    )


@pytest.mark.parametrize(
    "provider,model_name",
    [
        ("vertex_ai", "gemini-3.8-flash"),
        ("openai", "scripted"),
        ("anthropic", "scripted"),
    ],
)
def test_runtime_parallel_sources_full_original_review_and_final_citations(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model_name: str,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    llm.config = LLMConfig(
        model_provider=provider,
        model_name=model_name,
        temperature=0,
        max_input_tokens=100000,
    )
    runtime.run_asv3_loop(**kwargs)
    assert broker.peak == 2
    assert {item.chunk_id for item in broker.revalidated} == {"chunk-0", "chunk-1"}
    for call in llm.invoke.call_args_list[-2:]:
        data = json.loads(call.kwargs["prompt"][1].content)
        evidence = json.loads(data["evidence"])
        assert {item["text"] for item in evidence} == {
            chunk.text for chunk in broker.chunks.values()
        }
        assert not any(item["truncated"] for item in evidence)
    output = packets(queue)
    narration = [
        packet.obj for packet in output if isinstance(packet.obj, ASv3Progress)
    ]
    assert all(event.language == "tr" for event in narration)
    assert any("Aynı makinenin" in (event.message or "") for event in narration)
    assert not any(
        "read_source" in (event.message or "")
        or "report_progress" in (event.message or "")
        for event in narration
    )
    completed = [
        i
        for i, packet in enumerate(output)
        if isinstance(packet.obj, ASv3Progress) and packet.obj.status == "completed"
    ]
    assert len(completed) == 1
    assert completed[0] > max(
        i
        for i, packet in enumerate(output)
        if isinstance(packet.obj, (AgentResponseDelta, CitationInfo))
    )
    state = kwargs["state_container"]
    assert "[[1]](https://example.test/law-0)" in state.answer_tokens
    assert "[[2]](https://example.test/law-1)" in state.answer_tokens
    assert len(state.citation_to_doc) == 2
    assert checkpoints[-1]["evidence"]["included"] == [1, 2]
    assert llm.config.model_provider == provider and llm.config.model_name == model_name


def test_runtime_rejects_unrecorded_citation_before_any_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, _llm, _checkpoints, queue = setup_run(monkeypatch, "Kaynak [999].")
    with pytest.raises(ValueError, match="unrecorded citation"):
        runtime.run_asv3_loop(**kwargs)
    output = packets(queue)
    assert not any(
        isinstance(packet.obj, (AgentResponseStart, AgentResponseDelta))
        for packet in output
    )
    assert [
        packet.obj.status for packet in output if isinstance(packet.obj, ASv3Progress)
    ][-1] == "failed"
    assert kwargs["state_container"].answer_tokens is None


def test_runtime_cancel_discards_late_source_results_and_durable_writes(
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
        broker.release.set()
        assert len(checkpoints) == writes_before_cancel
        assert llm.invoke.call_count == 2
        assert kwargs["state_container"].answer_tokens is None
        output = packets(queue)
        assert not any(
            isinstance(packet.obj, (AgentResponseStart, AgentResponseDelta))
            for packet in output
        )
        assert [
            packet.obj.status
            for packet in output
            if isinstance(packet.obj, ASv3Progress)
        ][-1] == "cancelled"
    finally:
        broker.release.set()
        thread.join(3)


def test_runtime_researchers_keep_scenario_facts_isolated_and_selected_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.prompts.asv3.research import (
        COORDINATOR_PROMPT,
        FINAL_PROMPT,
        RESEARCHER_PROMPT,
        VERIFICATION_PROMPT,
    )

    kwargs, _broker, llm, _checkpoints, _queue = setup_run(
        monkeypatch, "Araştırma tamamlandı."
    )
    lock = threading.Lock()
    coordinator_started = False
    child_steps: dict[str, int] = {}
    child_facts: dict[str, list[str]] = {}
    final_scenario: dict[str, Any] = {}
    notifications = {
        phase: ["Araştırma", "Tamir ve değişim koşullarını inceliyorum."]
        for phase in (
            "started",
            "tools",
            "worker",
            "final",
            "completed",
            "failed",
            "cancelled",
            "native_citation",
            "resume",
            "interrupted",
        )
    }

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal coordinator_started
        prompt = arguments["prompt"]
        instruction, content = prompt[0].content, prompt[1].content
        if isinstance(content, list):
            content = content[0].text
        data = json.loads(content) if content.startswith("{") else {}
        with lock:
            if instruction == COORDINATOR_PROMPT:
                if not coordinator_started:
                    coordinator_started = True
                    return response(
                        calls=[
                            (
                                "record_scenario",
                                {"questions": ["parent"], "facts": ["parent fact"]},
                            ),
                            ("spawn_researcher", {"task": "branch-A"}),
                            ("spawn_researcher", {"task": "branch-B"}),
                        ]
                    )
                return response("Bağımsız çalışmalar değerlendirildi.")
            if instruction == RESEARCHER_PROMPT:
                task = data["request"]
                step = child_steps.get(task, 0)
                child_steps[task] = step + 1
                if step == 0:
                    assert data["facts"] == []
                    return response(
                        calls=[
                            (
                                "record_scenario",
                                {"questions": [task], "facts": [task + " fact"]},
                            )
                        ]
                    )
                child_facts[task] = data["facts"]
                return response(task + " tamamlandı.")
            if instruction.startswith(VERIFICATION_PROMPT):
                return response(
                    '{"status":"incomplete","explanation":"Kaynak araştırması tamamlanmadı.","required_conditions":[],"missing_conditions":[],"evidence_numbers":[]}'
                )
            if instruction == FINAL_PROMPT:
                final_scenario.update(data["scenario"])
                return response("Araştırma tamamlandı.")
            return response(
                json.dumps({"language": "tr", "notifications": notifications})
            )

    llm.invoke.side_effect = invoke
    selected_name = llm.config.model_name
    runtime.run_asv3_loop(**kwargs)
    assert child_facts == {"branch-A": ["branch-A fact"], "branch-B": ["branch-B fact"]}
    assert final_scenario["facts"] == ["parent fact"]
    assert kwargs["state_container"].answer_tokens == "Araştırma tamamlandı."
    assert llm.config.model_name == selected_name
