"""Exercise parallel publication with real child harnesses and fake I/O boundaries."""

import copy
import hashlib
import threading
from queue import Queue
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest

from onyx.asv3 import runtime
from onyx.asv3.models import RunContext
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.llm.interfaces import LLM
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import UserMessage
from onyx.server.query_and_chat.streaming_models import AgentResponseDelta, ASv3Progress
from tests.unit.onyx.asv3.test_runtime import (
    CorpusBoundary,
    delivered_originals,
    packets,
    response,
    setup_run,
    user_payload,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")

SCENARIO = (
    "İşlem tarihi 6 Ekim; aktör, eşya ve kısmi miktar her iki soruda aynıdır. "
    "Kullanıcının değişmeyen olguları: on bir parça, iki alternatif ve eksik izin tarihi.\n"
    "1. İlk işlemin kaynak koşulları nelerdir?\n"
    "2. Sonraki işlemin kaynak koşulları nelerdir?"
)
TASKS = ("İlk işlemin koşullarını incele.", "Sonraki işlemin koşullarını incele.")


def parallel_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    CorpusBoundary,
    MagicMock,
    list[dict[str, Any]],
    Queue[Any],
    MagicMock,
]:
    kwargs, broker, selected, checkpoints, queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="experimental", parallel_research=True)
    selected.config = selected.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )

    def fence(source_id: str, context: RunContext) -> str:
        assert broker.scope is not None
        assert context.scope == broker.scope.model_dump(mode="json")
        assert source_id in broker.chunks
        return "captured-index-and-source-revision"

    read_fence = MagicMock(side_effect=fence)
    monkeypatch.setattr(broker, "shared_read_fence", read_fence, raising=False)
    return kwargs, broker, selected, checkpoints, queue, read_fence


@pytest.mark.parametrize("action", ["greeting", "clarification"])
def test_parallel_first_decision_can_finish_without_research(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    kwargs, _, selected, checkpoints, queue, fence = parallel_setup(monkeypatch)
    text = (
        "Merhaba! Nasıl yardımcı olabilirim?"
        if action == "greeting"
        else ("İşlemin gerçekleştiği tarih nedir?")
    )
    tool = "submit_answer" if action == "greeting" else "ask_user"
    arguments = (
        {"answer": text, "basis": "conversation", "_language": "tr"}
        if action == "greeting"
        else {"question": text, "_language": "tr"}
    )

    def invoke(**call: Any) -> ModelResponse:
        names = {item["function"]["name"] for item in call["tools"]}
        assert {"submit_answer", "ask_user", "research_questions"} <= names
        assert call["tool_choice"] == "auto"
        return response(calls=[(tool, arguments)])

    selected.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)

    assert selected.invoke.call_count == 1
    assert fence.call_count == 0
    final = checkpoints[-1]
    assert final["parallel_research"] is True
    assert final["research_profile"] == "experimental"
    assert final["workers"]["tasks"] == []
    assert final["parallel_answers"]["receipts"] == []
    assert final["publication_stop_reason"] == (
        "native_answer_published" if action == "greeting" else "clarification_requested"
    )
    assert (
        "".join(
            packet.obj.content
            for packet in packets(queue)
            if isinstance(packet.obj, AgentResponseDelta)
        )
        == text
    )


def script_two_children(
    monkeypatch: pytest.MonkeyPatch,
    *,
    second_result: Literal["full", "partial", "error"] = "full",
) -> tuple[
    dict[str, Any],
    MagicMock,
    MagicMock,
    list[dict[str, Any]],
    Queue[Any],
    MagicMock,
    MagicMock,
    dict[str, str],
    list[str],
]:
    kwargs, broker, selected, checkpoints, queue, fence = parallel_setup(monkeypatch)
    kwargs["simple_chat_history"] = [
        ChatMessageSimple(
            message=SCENARIO, token_count=len(SCENARIO), message_type=MessageType.USER
        )
    ]
    secondary = MagicMock(spec=LLM)
    secondary.invoke.side_effect = AssertionError(
        "Experimental must keep the selected model"
    )
    kwargs["research_llm"] = secondary
    source = broker.sources[0]
    source_id = str(source.id)

    def page(
        requested: str, context: RunContext, **_args: Any
    ) -> tuple[Any, list[Any], bool]:
        assert requested == source_id
        assert broker.scope is not None
        assert context.scope == broker.scope.model_dump(mode="json")
        return source, [broker.chunks[source_id]], False

    physical_read = MagicMock(side_effect=page)
    monkeypatch.setattr(broker, "page", physical_read)
    arrivals = threading.Barrier(2)
    lock = threading.Lock()
    invocations: dict[str, int] = {}
    bodies: dict[str, str] = {}
    requests: list[str] = []

    def invoke(**call: Any) -> ModelResponse:
        prompt = call["prompt"]
        assert isinstance(prompt[1], UserMessage)
        payload = user_payload(prompt[1])
        request = payload["request"]
        assert payload["scenario_request"] == SCENARIO
        with lock:
            requests.append(request)
            invocations[request] = invocations.get(request, 0) + 1
            count = invocations[request]
        if request == SCENARIO:
            assert count == 1, "Host assembly must not invoke another model"
            return response(
                calls=[
                    (
                        "research_questions",
                        {
                            "questions": [
                                {
                                    "question_id": f"branch-{index}",
                                    "question": task,
                                    "parent_question_ids": [index],
                                    "answer_title": f"İşlem {index}",
                                    "public_title": f"İşlem {index} koşulları",
                                    "public_message": f"İşlem {index} için özgün koşullar inceleniyor.",
                                }
                                for index, task in enumerate(TASKS, 1)
                            ],
                            "_language": "tr",
                        },
                    )
                ]
            )
        assert request in TASKS
        assert SCENARIO in str(payload["conversation"])
        assert count <= 2
        if count == 1:
            arrivals.wait(timeout=5)
            return response(
                calls=[
                    (
                        "read_source_range",
                        {
                            "source_id": source_id,
                            "start": 0,
                            "_public_update": [
                                request,
                                "İlgili özgün hüküm ve koşulları okunuyor.",
                            ],
                        },
                    )
                ]
            )
        originals = delivered_originals(call)
        assert len(originals) == 1
        original = originals[0]
        assert original["text"] == broker.chunks[source_id].text
        assert original.get("start_char", 0) == 0
        assert original.get("end_char", len(original["text"])) == len(original["text"])
        citation = original["citation"]
        if request == TASKS[1] and second_result == "error":
            raise RuntimeError("Synthetic provider transport failure")
        body = (
            f"Bu işlemin kaynakta belirtilen koşulu [{citation}].\n\n"
            + ("Özgün koşul, istisna, belge ve sonraki aşama; İĞŞçöü — 東京. " * 650)
            + f"\n\nTAM_SON_{request}\n  "
        )
        if request == TASKS[1] and second_result == "partial":
            body += "\nEksik olan sonraki aşamanın özgün hükmü henüz doğrulanamadı.\n "
        with lock:
            bodies[request] = body
        return response(
            calls=[
                ("submit_partial_answer", {"answer": body})
                if request == TASKS[1] and second_result == "partial"
                else ("submit_answer", {"answer": body, "basis": "originals"})
            ]
        )

    selected.invoke.side_effect = invoke
    return (
        kwargs,
        selected,
        secondary,
        checkpoints,
        queue,
        fence,
        physical_read,
        bodies,
        requests,
    )


def test_parallel_children_share_reads_and_publish_exact_complete_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, selected, secondary, checkpoints, queue, fence, read, bodies, requests = (
        script_two_children(monkeypatch)
    )
    runtime.run_asv3_loop(**kwargs)

    assert selected.invoke.call_count == 5
    assert requests.count(SCENARIO) == 1
    assert all(requests.count(task) == 2 for task in TASKS)
    secondary.invoke.assert_not_called()
    assert selected.config.model_name == "gemini-3.8-flash"
    assert read.call_count == 1
    assert fence.call_count == 2
    final = checkpoints[-1]
    assert final["parallel_research"] is True
    assert final["publication_status"] == "found"
    answers = final["question_research"]["answers"]
    assert [item["answer"] for item in answers] == [bodies[task] for task in TASKS]
    assert all(len(item["answer"]) > 30000 for item in answers)
    assert final["last_draft"] == "\n\n".join(
        f"## {index}. İşlem {index}\n\n{bodies[task]}"
        for index, task in enumerate(TASKS, 1)
    )
    receipts = final["parallel_answers"]["receipts"]
    assert len(receipts) == 2
    assert {item["answer"] for item in receipts} == set(bodies.values())
    assert len({item["model_call_id"] for item in receipts}) == 2
    assert len({item["originals"][0]["citation"] for item in receipts}) == 1
    for receipt in receipts:
        assert (
            receipt["answer_hash"]
            == hashlib.sha256(receipt["answer"].encode()).hexdigest()
        )
        binding = receipt["originals"][0]
        original = next(
            row["item"]
            for row in final["evidence"]["records"]
            if row["citation"] == binding["citation"]
        )
        assert binding["complete"] is True
        assert binding["start_char"] == 0
        assert binding["end_char"] == len(original["text"])
        assert binding["text_hash"] == original["text_hash"]
    tasks = final["workers"]["tasks"]
    assert len(tasks) == 2
    assert {item["assignment_id"] for item in tasks} == {"branch-1", "branch-2"}
    for item in tasks:
        assert item["local_budget"]["limits"] == {"tools": None, "decisions": None}
        child = item["child_checkpoint"]
        assert child["run_id"] == final["run_id"]
        assert child["task_id"] == item["task_id"]
        assert child["assignment_id"] == item["assignment_id"]
        assert child["snapshot"]["request"] == item["task"]
        assert child["snapshot"]["last_draft"] == bodies[item["task"]]
        source_receipts = [
            r
            for r in child["snapshot"]["receipts"]
            if r["call"]["name"] == "read_source_range"
        ]
        assert len(source_receipts) == 1
        assert source_receipts[0]["evidence_ids"] == answers[0]["evidence_numbers"]
    emitted = packets(queue)
    assert set(TASKS) <= {
        packet.obj.title for packet in emitted if isinstance(packet.obj, ASv3Progress)
    }
    rendered = "".join(
        packet.obj.content
        for packet in emitted
        if isinstance(packet.obj, AgentResponseDelta)
    )
    for body in bodies.values():
        assert body.split("\n\n", 1)[1] in rendered


@pytest.mark.parametrize("second_result", ["partial", "error"])
def test_parallel_incomplete_child_preserves_complete_sibling(
    monkeypatch: pytest.MonkeyPatch, second_result: Literal["partial", "error"]
) -> None:
    kwargs, selected, _, checkpoints, queue, _, read, bodies, requests = (
        script_two_children(monkeypatch, second_result=second_result)
    )
    runtime.run_asv3_loop(**kwargs)

    assert selected.invoke.call_count == 5
    assert requests.count(SCENARIO) == 1
    assert read.call_count == 1
    final = checkpoints[-1]
    assert final["publication_status"] == "partial"
    answers = final["question_research"]["answers"]
    assert answers[0]["answer"] == bodies[TASKS[0]]
    assert answers[0]["status"] == "found"
    assert answers[1]["status"] == "partial"
    if second_result == "partial":
        assert answers[1]["answer"] == bodies[TASKS[1]]
        assert answers[1]["host_gap"] is False
        assert len(final["parallel_answers"]["receipts"]) == 2
    else:
        assert answers[1]["host_gap"] is True
        assert "kaynak araştırması tamamlanamadı" in answers[1]["answer"]
        assert len(final["parallel_answers"]["receipts"]) == 1
        assert "Synthetic provider" not in final["last_draft"]
    assert bodies[TASKS[0]] in final["last_draft"]
    rendered = "".join(
        packet.obj.content
        for packet in packets(queue)
        if isinstance(packet.obj, AgentResponseDelta)
    )
    assert bodies[TASKS[0]].split("\n\n", 1)[1] in rendered


def test_parallel_resume_restores_saved_mode_and_bound_bodies_without_new_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, selected, _, checkpoints, queue, _, read, _, _ = script_two_children(
        monkeypatch
    )
    runtime.run_asv3_loop(**kwargs)
    previous = copy.deepcopy(checkpoints[-1])
    packets(queue)
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: previous)
    selected.reset_mock()

    selected.invoke.side_effect = AssertionError(
        "Saved complete branch receipts need no new model call"
    )
    # Resume transport need not carry a new selection: the saved checkpoint owns its mode.
    kwargs.update(
        resume_message_id=2, research_profile="normal", parallel_research=False
    )
    runtime.run_asv3_loop(**kwargs)

    selected.invoke.assert_not_called()
    assert read.call_count == 1
    restored = checkpoints[-1]
    assert restored["research_profile"] == "experimental"
    assert restored["parallel_research"] is True
    assert restored["publication_status"] == "found"
    assert restored["last_draft"] == previous["last_draft"]
    assert restored["parallel_answers"] == previous["parallel_answers"]
    assert (
        restored["question_research"]["answers"]
        == previous["question_research"]["answers"]
    )
    assert {item["task_id"] for item in restored["workers"]["tasks"]} == {
        item["task_id"] for item in previous["workers"]["tasks"]
    }
