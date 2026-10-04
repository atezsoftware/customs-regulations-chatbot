import copy
import threading
import time
from collections.abc import Callable
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    TaskSnapshot,
    TaskStatus,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.question_research import QuestionResearch
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workers import WorkerPool


class ControlledWorkers:
    def __init__(self, outcomes: list[ToolOutcome | None]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, RunContext | None, bool]] = []
        self.waited: list[str] = []
        self.stored_results: list[TaskSnapshot] | None = None

    def spawn(
        self,
        task: str,
        *,
        request_context: RunContext | None = None,
        public_title: str | None = None,
        public_message: str | None = None,
        independent_question: bool = False,
    ) -> str:
        assert public_title and public_message
        self.calls.append((task, request_context, independent_question))
        return f"task-{len(self.calls)}"

    def results(self, *, full: bool = True) -> list[TaskSnapshot]:
        assert full
        if self.stored_results is not None:
            return copy.deepcopy(self.stored_results)
        return [
            TaskSnapshot(
                task_id=f"task-{index + 1}",
                task=self.calls[index][0],
                status=TaskStatus.COMPLETED if outcome else TaskStatus.CANCELLED,
                outcome=outcome,
                independent_question=True,
            )
            for index, outcome in enumerate(self.outcomes[: len(self.calls)])
        ]

    def wait_until_all(self, task_ids: list[str]) -> list[TaskSnapshot]:
        self.waited = list(task_ids)
        results = [item for item in self.results() if item.task_id in task_ids]
        return list(reversed(results))


def question(identifier: str, text: str, parent: int) -> dict[str, JsonValue]:
    return {
        "question_id": identifier,
        "question": text,
        "parent_question_ids": [parent],
        "public_title": "İlgili sonuç araştırılıyor",
        "public_message": "Bu sonucun koşulları özgün kaynaklardan inceleniyor.",
    }


def controlled_research(
    outcomes: list[ToolOutcome | None], original_questions: list[str]
) -> tuple[QuestionResearch, ControlledWorkers, RunContext]:
    context = RunContext(
        services={"full_scenario": "Aktör A, rejim B; iki alternatif de isteniyor."},
        scope={"document_sets": [15], "external": False},
    )
    workers = ControlledWorkers(outcomes)
    return (
        QuestionResearch(context, cast(WorkerPool, workers), original_questions),
        workers,
        context,
    )


def found(body: str) -> ToolOutcome:
    return ToolOutcome(status=OutcomeStatus.FOUND, summary=body)


def test_related_semantic_subquestions_get_distinct_tasks_with_same_full_scenario() -> (
    None
):
    research, workers, context = controlled_research(
        [found("Uygunluk sonucu [1]"), found("Sonraki işlem sonucu [2]")],
        ["Uygun muyum ve uygun bulunursam hangi işlemi yapmalıyım?"],
    )
    scope, services = copy.deepcopy(context.scope), copy.deepcopy(context.services)
    entries = [
        question("eligibility", "Bu işlem için uygunluk koşulları nelerdir?", 1),
        question("procedure", "Uygunluk sonrası işlemler nelerdir?", 1),
    ]

    research.research_questions({"questions": entries}, context)

    assert [task for task, _, _ in workers.calls] == [
        entry["question"] for entry in entries
    ]
    assert all(
        parent is context and independent for _, parent, independent in workers.calls
    )
    assert workers.waited == ["task-1", "task-2"]
    assert context.scope == scope
    assert context.services["full_scenario"] == services["full_scenario"]
    assert [item["task_id"] for item in research.answers] == ["task-1", "task-2"]
    assert context.services["independent_evidence_numbers"] == [1, 2]


def test_full_answer_bodies_are_preserved_verbatim_in_requested_order() -> None:
    first = "Birinci özgün yanıt\n" + "Uzun koşul, istisna ve işlem. " * 600 + "\n[7]"
    second = "İkinci özgün yanıt [11]\n\nSonraki aşama değişmez."
    assert len(first) > 12_000
    research, _, context = controlled_research(
        [found(first), found(second)], ["İlk soru", "İkinci soru"]
    )
    research.research_questions(
        {
            "questions": [
                question("first", "İlk soru", 1),
                question("second", "İkinci soru", 2),
            ]
        },
        context,
    )

    research.assemble_answers({"order": ["second", "first"]}, context)

    answer = context.services["assembled_answer"]
    assert isinstance(answer, str)
    assert answer == f"## 1. İkinci soru\n\n{second}\n\n## 2. İlk soru\n\n{first}"
    assert research.preservation_gap(answer) is None
    shortened = answer.replace(first, first[:12_000])
    gap = research.preservation_gap(shortened)
    assert gap is not None
    assert gap.data["missing_question_answers"] == ["first"]
    assert context.services["independent_partial"] is False


@pytest.mark.parametrize(
    "order",
    [["first"], ["first", "first"], ["first", "other"], ["first", "second", "first"]],
)
def test_assembly_rejects_omitted_duplicate_and_unknown_question_ids(
    order: list[str],
) -> None:
    research, _, context = controlled_research(
        [found("Premier [1]"), found("Second [2]")], ["First", "Second"]
    )
    research.research_questions(
        {"questions": [question("first", "First", 1), question("second", "Second", 2)]},
        context,
    )
    with pytest.raises(ValueError, match="exactly once"):
        research.assemble_answers({"order": order}, context)
    assert "assembled_answer" not in context.services


@pytest.mark.parametrize(
    "entries",
    [
        [question("first", "First", 1)],
        [question("duplicate", "First", 1), question("duplicate", "Second", 2)],
        [question("first", "First", 1), question("second", "Second", 3)],
    ],
)
def test_invalid_original_question_coverage_starts_no_workers(
    entries: list[dict[str, JsonValue]],
) -> None:
    research, workers, context = controlled_research([], ["First", "Second"])
    with pytest.raises(ValueError):
        research.research_questions({"questions": entries}, context)
    assert workers.calls == []
    assert "question_research_started" not in context.services


def test_partial_failed_and_unfinished_tasks_remain_explicit_without_losing_completed_body() -> (
    None
):
    bodies = [
        found("Tam sonuç [1]"),
        ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Desteklenen bölüm [2]. Kalan etki henüz doğrulanamadı.",
        ),
        ToolOutcome(status=OutcomeStatus.ERROR, summary="Araştırma tamamlanamadı."),
        None,
    ]
    research, _, context = controlled_research(bodies, ["Dört ayrı sonuç"])
    research.research_questions(
        {
            "questions": [
                question(str(index), f"Sonuç {index}", 1) for index in range(1, 5)
            ]
        },
        context,
    )
    assert [item["status"] for item in research.answers] == [
        "found",
        "partial",
        "error",
        "cancelled",
    ]
    assert research.answers[0]["answer"] == "Tam sonuç [1]"
    assert "tamamlanamadı" in str(research.answers[-1]["answer"])
    research.assemble_answers({"order": ["1", "2", "3", "4"]}, context)
    assert context.services["independent_partial"] is True
    assert "Tam sonuç [1]" in str(context.services["assembled_answer"])


def test_publication_guard_checks_connections_and_retains_full_answers_on_rejection() -> (
    None
):
    research, _, context = controlled_research([found("Sonuç [4]")], ["Soru"])
    research.research_questions({"questions": [question("only", "Soru", 1)]}, context)
    inspected: list[str] = []
    rejection = ToolOutcome(
        status=OutcomeStatus.PARTIAL, summary="Bağlayıcı cümlenin özgün dayanağı eksik."
    )

    def guard(answer: str) -> ToolOutcome:
        inspected.append(answer)
        return rejection

    research.publish_guard = guard
    outcome = research.assemble_answers(
        {
            "order": ["only"],
            "connections": "Bu sonuç diğer yükümlülüğü de otomatik kaldırır.",
        },
        context,
    )
    assert outcome is rejection
    assert inspected == [
        "## 1. Soru\n\nSonuç [4]\n\nBu sonuç diğer yükümlülüğü de otomatik kaldırır."
    ]
    assert "assembled_answer" not in context.services
    assert research.answers[0]["answer"] == "Sonuç [4]"


def test_no_root_final_can_publish_before_independent_research() -> None:
    research, _, context = controlled_research([], ["Soru"])
    registry = CapabilityRegistry(research.tool_specs())
    result = Harness(
        request="Soru",
        context=context,
        registry=registry,
        decide=lambda _view: Decision(answer="Araştırmasız nihai yanıt"),
        draft_guard=research.preservation_gap,
    ).run()
    assert result.answer is None
    assert result.status == OutcomeStatus.PARTIAL
    assert result.stop_reason == "repeated_publication_gap"
    assert len(result.receipts) == 3


def test_registry_dispatch_accepts_language_metadata_and_waits_without_holding_tool_slot() -> (
    None
):
    context = RunContext(
        services={"lean_native_mode": True},
        budget=SharedBudget(max_inflight_tools=1),
    )
    source = CapabilityRegistry(
        [
            ToolSpec(
                name="read_original",
                description="Read original",
                parameters={"type": "object"},
                handler=lambda _args, _context: found("Özgün metin [1]"),
            )
        ]
    )

    def runner(
        _task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        return source.dispatch(CapabilityCall(name="read_original"), child)

    pool = WorkerPool(context, runner)
    research = QuestionResearch(context, pool, ["Soru"])
    registry = CapabilityRegistry(research.tool_specs())
    results: list[ToolOutcome] = []
    dispatcher = threading.Thread(
        target=lambda: results.append(
            registry.dispatch(
                CapabilityCall(
                    name="research_questions",
                    arguments={
                        "questions": [question("one", "Soru", 1)],
                        "_language": "tr",
                    },
                ),
                context,
            )
        )
    )
    try:
        dispatcher.start()
        dispatcher.join(0.5)
        if dispatcher.is_alive():
            context.cancel()
            dispatcher.join(2)
            pytest.fail(
                "Question orchestration held the only source tool slot while waiting"
            )
        assert results[0].status == OutcomeStatus.FOUND
        assert research.answers[0]["answer"] == "Özgün metin [1]"
    finally:
        context.cancel()
        dispatcher.join(2)
        pool.close()


def test_independent_questions_keep_running_past_a_previous_research_deadline() -> None:
    entered, release = threading.Event(), threading.Event()
    context = RunContext(research_deadline=time.monotonic() + 0.01, timeout_seconds=3)
    body = "Tamamlanan yanıt [8]\n" + "Koşul ayrıntısı " * 1_000

    def runner(
        task: str, _child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        if task == "Yavaş soru":
            entered.set()
            assert release.wait(3)
        return found(body if task == "Hızlı soru" else "Geç kalan yanıt [9]")

    pool = WorkerPool(context, runner, max_workers=2)
    research = QuestionResearch(context, pool, ["İki sonuç"])
    finish = threading.Timer(0.05, release.set)
    finish.start()
    try:
        research.research_questions(
            {
                "questions": [
                    question("fast", "Hızlı soru", 1),
                    question("slow", "Yavaş soru", 1),
                ]
            },
            context,
        )
        assert entered.is_set()
        assert research.answers[0]["answer"] == body
        assert research.answers[1]["status"] == "found"
        assert research.answers[1]["answer"] == "Geç kalan yanıt [9]"
        research.assemble_answers({"order": ["fast", "slow"]}, context)
        assert body in str(context.services["assembled_answer"])
        assert "Geç kalan yanıt [9]" in str(context.services["assembled_answer"])
        assert context.services["independent_partial"] is False
    finally:
        release.set()
        finish.join()
        pool.close()


def test_brief_neutral_headings_do_not_replace_or_shorten_question_bodies() -> None:
    body = (
        "**Hızlı cevap:** Kaynakla desteklenen sonuç [3].\n\n"
        + "Ayrıntı ve koşul. " * 1000
    )
    research, _, context = controlled_research([found(body)], ["Uzun asıl soru"])
    entry = question("only", "Tam olgular ve ayrıntılı alt soru " * 150, 1)
    entry["answer_title"] = "Belge koşulları"
    research.research_questions({"questions": [entry]}, context)
    research.assemble_answers({"order": ["only"]}, context)
    assert context.services["assembled_answer"] == f"## 1. Belge koşulları\n\n{body}"
    assert research.answers[0]["question"] == entry["question"]


def test_repeated_research_call_reuses_completed_answers_without_new_tasks() -> None:
    research, workers, context = controlled_research([found("Sonuç [3]")], ["Soru"])
    args: dict[str, JsonValue] = {"questions": [question("one", "Soru", 1)]}
    research.research_questions(args, context)
    original = copy.deepcopy(research.answers)
    research.research_questions(args, context)
    assert len(workers.calls) == 1
    assert research.answers == original


@pytest.mark.parametrize("partial", [False, True])
def test_host_assembly_finishes_in_two_decisions_without_root_rewriting(
    partial: bool,
) -> None:
    body = "Kaynakla desteklenen tam yanıt [5].\n" + "Koşul ve işlem ayrıntısı. " * 600
    worker_outcome = ToolOutcome(
        status=OutcomeStatus.PARTIAL if partial else OutcomeStatus.FOUND,
        summary=body,
    )
    research, _, context = controlled_research([worker_outcome], ["Soru"])
    registry = CapabilityRegistry(research.tool_specs())
    research.publish_guard = research.preservation_gap
    decisions = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal decisions
        decisions += 1
        if decisions == 1:
            return Decision(
                calls=[
                    CapabilityCall(
                        name="research_questions",
                        arguments={"questions": [question("only", "Soru", 1)]},
                    )
                ]
            )
        assert decisions == 2
        context.research_deadline = time.monotonic() - 1
        return Decision(
            calls=[
                CapabilityCall(name="assemble_answers", arguments={"order": ["only"]})
            ]
        )

    result = Harness(
        request="Soru",
        context=context,
        registry=registry,
        decide=decide,
        draft_guard=research.preservation_gap,
    ).run()
    assert decisions == 2
    assert result.answer == f"## 1. Soru\n\n{body}"
    assert result.status == (OutcomeStatus.PARTIAL if partial else OutcomeStatus.FOUND)
    assert result.stop_reason == "independent_answers_assembled"


def test_completed_checkpoint_resumes_full_bodies_without_dispatching_again() -> None:
    body = "Özgün yanıt [4].\n" + "Korunacak ayrıntı " * 1000
    research, _, context = controlled_research([found(body)], ["Soru"])
    research.research_questions({"questions": [question("only", "Soru", 1)]}, context)
    resumed, workers, restored_context = controlled_research([], ["Soru"])
    resumed.restore(copy.deepcopy(research.export()))
    resumed.research_questions(
        {"questions": [question("only", "Soru", 1)]}, restored_context
    )
    resumed.assemble_answers({"order": ["only"]}, restored_context)
    assert workers.calls == []
    assert restored_context.services["assembled_answer"] == f"## 1. Soru\n\n{body}"
    assert restored_context.services["independent_evidence_numbers"] == [4]


def test_inflight_checkpoint_retains_completed_body_and_marks_interruption_without_respawn() -> (
    None
):
    body = "Tam sonuç [7].\n" + "Ayrıntı korunur. " * 1000
    original, _, _ = controlled_research([], ["First", "Second"])
    original.assignments = [
        {**question("first", "First", 1), "task_id": "old-1"},
        {**question("second", "Second", 2), "task_id": "old-2"},
    ]
    snapshot = original.export()
    resumed, workers, context = controlled_research([], ["First", "Second"])
    workers.stored_results = [
        TaskSnapshot(
            task_id="old-1",
            task="First",
            status=TaskStatus.COMPLETED,
            outcome=found(body),
            independent_question=True,
        ),
        TaskSnapshot(
            task_id="old-2",
            task="Second",
            status=TaskStatus.INTERRUPTED,
            independent_question=True,
        ),
    ]
    resumed.restore(snapshot)
    resumed.research_questions(
        {"questions": [question("new", "Rewritten task", 1)]}, context
    )
    resumed.assemble_answers({"order": ["second", "first"]}, context)
    assert workers.calls == []
    assert workers.waited == ["old-1", "old-2"]
    assert resumed.answers[0]["answer"] == body
    assert resumed.answers[1]["status"] == "interrupted"
    assert body in str(context.services["assembled_answer"])
    assert "tamamlanamadı" in str(context.services["assembled_answer"])
    assert context.services["independent_partial"] is True


def test_checkpoint_missing_task_is_an_explicit_gap_and_never_recreated() -> None:
    resumed, workers, context = controlled_research([], ["First", "Second"])
    resumed.restore(
        {
            "answers": [],
            "assignments": [
                {**question("first", "First", 1), "task_id": "known"},
                {**question("second", "Second", 2), "task_id": None},
            ],
        }
    )
    workers.stored_results = [
        TaskSnapshot(
            task_id="known",
            task="First",
            status=TaskStatus.COMPLETED,
            outcome=found("Known full result [2]"),
            independent_question=True,
        )
    ]
    resumed.research_questions({}, context)
    assert workers.calls == []
    assert resumed.answers[0]["answer"] == "Known full result [2]"
    assert resumed.answers[1]["status"] == "interrupted"
    assert resumed.answers[1]["task_id"] is None
    assert "tamamlanamadı" in str(resumed.answers[1]["answer"])


def test_checkpoint_export_cannot_mutate_retained_full_answer() -> None:
    research, _, context = controlled_research([found("Full answer [6]")], ["Question"])
    research.research_questions(
        {"questions": [question("only", "Question", 1)]}, context
    )
    snapshot = research.export()
    answers = snapshot["answers"]
    assert isinstance(answers, list) and isinstance(answers[0], dict)
    answers[0]["answer"] = "Shortened answer"
    assert research.answers[0]["answer"] == "Full answer [6]"


def test_assignments_are_checkpointed_before_waiting_for_worker_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research, workers, context = controlled_research(
        [found("Completed full body [2]"), None], ["First", "Second"]
    )
    saved: dict[str, JsonValue] = {}

    def interrupted_wait(_task_ids: list[str]) -> list[TaskSnapshot]:
        saved.update(research.export())
        raise RunStopped("Collection interrupted")

    monkeypatch.setattr(workers, "wait_until_all", interrupted_wait)
    with pytest.raises(RunStopped, match="Collection interrupted"):
        research.research_questions(
            {
                "questions": [
                    question("first", "First", 1),
                    question("second", "Second", 2),
                ]
            },
            context,
        )
    assert saved["answers"] == []
    assignments = saved["assignments"]
    assert isinstance(assignments, list)
    assert [item["task_id"] for item in assignments if isinstance(item, dict)] == [
        "task-1",
        "task-2",
    ]
    resumed, resumed_workers, restored_context = controlled_research(
        [], ["First", "Second"]
    )
    resumed_workers.stored_results = workers.results()
    resumed.restore(saved)
    resumed.research_questions({}, restored_context)
    assert resumed_workers.calls == []
    assert resumed.answers[0]["answer"] == "Completed full body [2]"
    assert resumed.answers[1]["status"] != "found"
