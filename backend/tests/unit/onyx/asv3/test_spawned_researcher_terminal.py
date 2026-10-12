"""Ordinary delegated researchers return owned results without publishing them."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from onyx.asv3 import runtime
from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.harness import Harness
from onyx.asv3.models import CapabilityCall, OutcomeStatus
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import ModelResponse
from tests.unit.onyx.asv3.test_runtime import (
    configure_governing_basis_sources,
    delivered_originals,
    response,
    setup_run,
    user_payload,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def capture_harnesses(monkeypatch: pytest.MonkeyPatch) -> list[Harness]:
    harnesses: list[Harness] = []

    def create_harness(**kwargs: Any) -> Harness:
        harness = Harness(**kwargs)
        harnesses.append(harness)
        return harness

    monkeypatch.setattr(runtime, "Harness", create_harness)
    return harnesses


def scripted_coordinator(
    tasks: list[str], child: Callable[..., ModelResponse]
) -> Callable[..., ModelResponse]:
    started = False

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal started
        payload = user_payload(arguments["prompt"][-1])
        if payload["request"] in tasks:
            return child(**arguments)
        if not started:
            started = True
            return response(
                calls=[
                    ("spawn_researcher", {"task": task, "_language": "tr"})
                    for task in tasks
                ]
            )
        research_tasks = payload["research_tasks"]
        assert len(research_tasks) == len(tasks)
        if any(task["status"] in {"queued", "running"} for task in research_tasks):
            return response(calls=[("wait_researcher", {})])
        assert all(task["status"] == "completed" for task in research_tasks)
        outcomes = [task["outcome"] for task in research_tasks]
        assert all(outcome["status"] in {"found", "partial"} for outcome in outcomes)
        delivered = {row["citation"] for row in delivered_originals(arguments)}
        required = {
            number
            for outcome in outcomes
            for number in outcome["data"]["evidence_numbers"]
        }
        if missing := required - delivered:
            return response(
                calls=[
                    ("read_evidence", {"citation": number})
                    for number in sorted(missing)
                ]
            )
        answer = "\n\n".join(outcome["summary"] for outcome in outcomes)
        if any(outcome["status"] == "partial" for outcome in outcomes):
            return response(calls=[("submit_partial_answer", {"answer": answer})])
        return response(
            calls=[("submit_answer", {"answer": answer, "basis": "originals"})]
        )

    return invoke


@pytest.mark.parametrize("research_profile", ["normal", "deep"])
@pytest.mark.parametrize("terminal", ["submit_answer", "submit_partial_answer"])
def test_spawned_researchers_finish_with_one_owned_terminal_call(
    monkeypatch: pytest.MonkeyPatch, research_profile: str, terminal: str
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs["research_profile"] = research_profile
    tasks = ["Tamir koşulunu araştır.", "Değişim koşulunu araştır."]
    source_by_task = {
        task: str(source.id) for task, source in zip(tasks, broker.sources)
    }
    harnesses = capture_harnesses(monkeypatch)
    steps: dict[str, int] = {}
    results: dict[str, str] = {}

    def child(**arguments: Any) -> ModelResponse:
        payload = user_payload(arguments["prompt"][-1])
        task = payload["request"]
        steps[task] = steps.get(task, 0) + 1
        names = {tool["function"]["name"] for tool in arguments["tools"]}
        assert {
            "submit_answer",
            "submit_partial_answer",
            "read_research_state",
        } <= names
        assert not names & {"ask_user", "assemble_answers", "research_questions"}
        if steps[task] == 1:
            return response(
                calls=[("read_source_range", {"source_id": source_by_task[task]})]
            )
        assert steps[task] == 2
        own = next(
            row
            for row in delivered_originals(arguments)
            if row["source_id"] == source_by_task[task]
        )
        result = f"{task} Özgün hükmün koşulları geçerlidir [{own['citation']}]."
        if terminal == "submit_partial_answer":
            result += " Diğer hükmün özgün metni henüz incelenmedi."
        results[task] = result
        return response(
            calls=[
                (
                    terminal,
                    {
                        "answer": result,
                        **(
                            {"basis": "originals"}
                            if terminal == "submit_answer"
                            else {}
                        ),
                    },
                )
            ]
        )

    llm.invoke.side_effect = scripted_coordinator(tasks, child)
    runtime.run_asv3_loop(**kwargs)
    root = next(harness for harness in harnesses if harness.context.depth == 0)
    children = [harness for harness in harnesses if harness.context.depth > 0]
    assert len(children) == 2
    assert len({id(harness.registry) for harness in harnesses}) == 3
    assert steps == {task: 2 for task in tasks}
    assert root.complete_submission is None
    root_submit = root.registry.get("submit_answer")
    assert root_submit is not None and root_submit.description.startswith("Publish")
    for harness in children:
        assert harness.context.services.get("independent_question") is not True
        assert harness.context.services.get("submitted_answer") is None
        assert harness.complete_submission is not None
        assert [receipt.call.name for receipt in harness.receipts] == [
            "read_source_range",
            terminal,
        ]
        assert harness.last_draft == results[harness.request]
        assert harness.publication_gap is None
        if terminal == "submit_answer":
            assert harness.complete_submission() == results[harness.request]
            assert harness.stop_reason == "model_submitted_answer"
        else:
            assert harness.partial_submission is not None
            assert harness.partial_submission() == results[harness.request]
            assert harness.stop_reason == "model_requested_partial_publication"
        # Possessing the parent's catalogue never grants publication authority.
        rejected = root.registry.dispatch(
            CapabilityCall(
                name=terminal,
                arguments={
                    "answer": results[harness.request],
                    **({"basis": "originals"} if terminal == "submit_answer" else {}),
                },
            ),
            harness.context,
        )
        assert rejected.status == OutcomeStatus.DENIED
    tasks_saved = checkpoints[-1]["workers"]["tasks"]
    assert {task["outcome"]["summary"] for task in tasks_saved} == set(results.values())
    assert {task["outcome"]["status"] for task in tasks_saved} == {
        "found" if terminal == "submit_answer" else "partial"
    }
    assert checkpoints[-1]["final_publication_gap"] is None


@pytest.mark.parametrize(
    "invalid", ["unknown_citation", "undelivered_citation", "named_authority"]
)
@pytest.mark.parametrize("terminal", ["submit_answer", "submit_partial_answer"])
def test_spawned_terminal_keeps_source_and_authority_guards(
    monkeypatch: pytest.MonkeyPatch, invalid: str, terminal: str
) -> None:
    kwargs, broker, llm, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs["research_profile"] = "normal"
    lower, governing = configure_governing_basis_sources(broker)
    harnesses = capture_harnesses(monkeypatch)
    task = "Başvurunun özgün koşulunu araştır."
    rejected = (
        "Başvuruda belge sunulur [999]."
        if invalid == "unknown_citation"
        else "Faaliyet için izin alınır [2]."
        if invalid == "undelivered_citation"
        else "8917 sayılı Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir [1]."
    )
    accepted = "Başvuruda belge sunulur [1]."
    if terminal == "submit_partial_answer":
        accepted += "\n\n8917 sayılı Faaliyet Kanunu'nun 27. maddesinin özgün metni incelenemedi."
    step = 0

    def child(**arguments: Any) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return response(calls=[("read_source_range", {"source_id": lower})])
        if step == 3:
            payload = user_payload(arguments["prompt"][-1])
            assert payload["draft_to_repair"] == rejected
            key = (
                "unknown_citations"
                if invalid == "unknown_citation"
                else "undelivered_citations"
                if invalid == "undelivered_citation"
                else "named_authority_gaps"
            )
            assert payload["publication_gap"][key]
        else:
            assert step == 2
            if invalid == "undelivered_citation":
                root = next(
                    harness for harness in harnesses if harness.context.depth == 0
                )
                source = next(
                    source for source in broker.sources if str(source.id) == governing
                )
                assert root.evidence.add(
                    [evidence_for_chunk(source, broker.chunks[governing])], root.context
                ) == [2]
        return response(
            calls=[
                (
                    terminal,
                    {
                        "answer": rejected if step == 2 else accepted,
                        **(
                            {"basis": "originals"}
                            if terminal == "submit_answer"
                            else {}
                        ),
                    },
                )
            ]
        )

    llm.invoke.side_effect = scripted_coordinator([task], child)
    runtime.run_asv3_loop(**kwargs)
    harness = next(harness for harness in harnesses if harness.context.depth)
    assert step == 3
    assert harness.receipts[1].outcome.status == OutcomeStatus.PARTIAL
    assert harness.receipts[2].outcome.status == (
        OutcomeStatus.FOUND if terminal == "submit_answer" else OutcomeStatus.PARTIAL
    )
    assert harness.last_draft == accepted and harness.publication_gap is None

    root = next(harness for harness in harnesses if not harness.context.depth)
    # Recording an authorized original does not prove that this coordinator read it.
    source = next(source for source in broker.sources if str(source.id) == governing)
    extra = evidence_for_chunk(
        source,
        replace(
            broker.chunks[governing],
            id="extra-unread-original",
            text="Ek hükmün özgün metni.",
        ),
    )
    unread = root.evidence.add([extra], root.context)[0]
    submitted_before = root.context.services.get("submitted_answer")
    result = root.registry.dispatch(
        CapabilityCall(
            name=terminal,
            arguments={
                "answer": f"Ek hükmün koşulu uygulanır [{unread}].",
                **({"basis": "originals"} if terminal == "submit_answer" else {}),
            },
        ),
        root.context,
    )
    assert result.status == OutcomeStatus.PARTIAL
    assert result.data["undelivered_citations"] == [unread]
    assert root.context.services.get("submitted_answer") == submitted_before


def test_tuned_spawned_partial_retains_accepted_body_without_failed_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    lower, _governing = configure_governing_basis_sources(broker)
    harnesses = capture_harnesses(monkeypatch)
    body = "Başvuruda belge sunulur [1]. Diğer hükmün özgün metni henüz incelenmedi."
    step = 0

    def child(**_arguments: Any) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return response(calls=[("read_source_range", {"source_id": lower})])
        assert step == 2
        return response(calls=[("submit_partial_answer", {"answer": body})])

    llm.invoke.side_effect = scripted_coordinator(["Başvuru koşulunu araştır."], child)
    runtime.run_asv3_loop(**kwargs)
    harness = next(harness for harness in harnesses if harness.context.depth)
    assert step == 2
    assert harness.partial_submission is not None
    assert harness.partial_submission() == body and harness.last_draft == body
    assert harness.publication_gap is None
    assert harness.view().draft_to_repair is None


@pytest.mark.parametrize("retained", [False, True])
def test_uncited_partial_cannot_assert_an_unsupported_named_rule(
    monkeypatch: pytest.MonkeyPatch, retained: bool
) -> None:
    kwargs, _broker, llm, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs["research_profile"] = "normal"
    if retained:
        kwargs["workflow_variant"] = ASV3_TUNED_VARIANT
    harnesses = capture_harnesses(monkeypatch)
    rejected = "8917 sayılı Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir."
    accepted = (
        "İstenen izin koşulunun özgün metni henüz incelenmedi."
        if retained
        else "8917 sayılı Faaliyet Kanunu'nun 27. maddesinin özgün metni incelenemedi."
    )
    step = 0

    def child(**arguments: Any) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return response(calls=[("submit_partial_answer", {"answer": rejected})])
        assert step == 2
        payload = user_payload(arguments["prompt"][-1])
        assert payload["publication_gap"]["named_authority_gaps"]
        if retained:
            units = payload["draft_to_repair"]["units"]
            assert len(units) == 1
            assert units[0]["text"] == rejected
            return response(
                calls=[
                    (
                        "submit_retained_partial_answer",
                        {
                            "retained_answer_edits": [
                                {
                                    "unit_id": units[0]["unit_id"],
                                    "replacement": accepted,
                                }
                            ]
                        },
                    )
                ]
            )
        assert payload["draft_to_repair"] == rejected
        return response(calls=[("submit_partial_answer", {"answer": accepted})])

    llm.invoke.side_effect = scripted_coordinator(["İzin koşulunu araştır."], child)
    runtime.run_asv3_loop(**kwargs)
    harness = next(harness for harness in harnesses if harness.context.depth)
    assert step == 2 and harness.partial_submission is not None
    assert harness.partial_submission() == accepted
    assert [receipt.outcome.status for receipt in harness.receipts] == [
        OutcomeStatus.PARTIAL,
        OutcomeStatus.PARTIAL,
    ]
    assert harness.last_draft == accepted and harness.publication_gap is None
