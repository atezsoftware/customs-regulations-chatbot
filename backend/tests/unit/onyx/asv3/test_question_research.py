import copy
import threading
import time
from collections.abc import Callable
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
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
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.question_research import QuestionResearch
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workers import WorkerPool


class ControlledWorkers:
    def __init__(self, outcomes: list[ToolOutcome | None]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, RunContext | None, bool]] = []
        self.waited: list[str] = []
        self.stored_results: list[TaskSnapshot] | None = None
        self.outcome_bindings: list[list[str] | None] = []

    def spawn(
        self,
        task: str,
        *,
        request_context: RunContext | None = None,
        public_title: str | None = None,
        public_message: str | None = None,
        independent_question: bool = False,
        outcome_ids: list[str] | None = None,
    ) -> str:
        assert public_title and public_message
        self.calls.append((task, request_context, independent_question))
        self.outcome_bindings.append(copy.deepcopy(outcome_ids))
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
        "answer_title": text,
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
    result = research.research_questions({"questions": entries}, context)
    assert result.status is OutcomeStatus.INVALID
    assert result.data["validation_error"]
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


@pytest.mark.parametrize("repeated_title", [False, True])
def test_missing_or_long_title_never_repeats_the_long_question(
    repeated_title: bool,
) -> None:
    full_question = "Uzun kullanıcı sorusu ve bütün senaryo ayrıntıları " * 100
    body = "**Hızlı cevap:** Kaynakla desteklenen sonuç [3].\n\nEksiksiz ayrıntı [4]."
    research, _, context = controlled_research([found(body)], [full_question])
    entry = question("only", full_question, 1)
    if not repeated_title:
        entry.pop("answer_title")
    research.research_questions({"questions": [entry]}, context)
    research.assemble_answers({"order": ["only"]}, context)
    assert context.services["assembled_answer"] == body
    assert full_question not in str(context.services["assembled_answer"])


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


def repair_arguments(research: QuestionResearch) -> dict[str, JsonValue]:
    return {
        "question_id": "first",
        "expected_answer_hash": research.answers[0]["answer_hash"],
        "gap": "The conclusion omitted the source's approval condition.",
        "edits": [
            {
                "kind": "replace",
                "target_text": "The entitlement is automatic. [1]",
                "text": "The entitlement depends on approval [2].",
            },
            {
                "kind": "insert_after",
                "target_text": "The later stage remains available [3].",
                "text": "The applicant supplies the specified proof before approval [2].",
            },
        ],
    }


def completed_answers_for_repair() -> tuple[
    QuestionResearch, ControlledWorkers, RunContext
]:
    research, workers, context = controlled_research(
        [
            found(
                "The entitlement is automatic. [1]\n\nThe later stage remains available [3]."
            ),
            found("The separate outcome and its conditions remain unchanged [4]."),
        ],
        ["First", "Second"],
    )
    research.research_questions(
        {"questions": [question("first", "First", 1), question("second", "Second", 2)]},
        context,
    )
    research.repair_guard = lambda _candidate: None
    return research, workers, context


def test_targeted_repair_preserves_other_bodies_and_unedited_text_without_new_workers() -> (
    None
):
    research, workers, context = completed_answers_for_repair()
    previous = copy.deepcopy(research.answers)
    arguments = repair_arguments(research)
    result = research.repair_question_answer(arguments, context)
    assert result.status == OutcomeStatus.FOUND
    assert len(workers.calls) == 2
    assert research.answers[1] == previous[1]
    assert (
        research.answers[0]["answer"]
        == "The entitlement depends on approval [2].\n\nThe later stage remains available [3].\n\nThe applicant supplies the specified proof before approval [2]."
    )
    assert research.answer_revisions[0]["previous_answer"] == previous[0]["answer"]
    assert research.answer_revisions[0]["before_hash"] == previous[0]["answer_hash"]
    assert (
        research.answer_revisions[0]["after_hash"] == research.answers[0]["answer_hash"]
    )
    research.assemble_answers({"order": ["first", "second"]}, context)
    assert str(research.answers[0]["answer"]) in str(
        context.services["assembled_answer"]
    )
    assert str(previous[1]["answer"]) in str(context.services["assembled_answer"])
    assert research.preservation_gap(str(context.services["assembled_answer"])) is None


def test_targeted_repair_runs_before_assembly_in_one_native_decision() -> None:
    research, workers, context = completed_answers_for_repair()
    registry = CapabilityRegistry(research.tool_specs())
    other_answer = copy.deepcopy(research.answers[1])
    repair_call = CapabilityCall(
        name="repair_question_answer", arguments=repair_arguments(research)
    )
    assembly_call = CapabilityCall(
        name="assemble_answers", arguments={"order": ["first", "second"]}
    )
    runner = Harness(
        request="First and second",
        context=context,
        registry=registry,
        decide=lambda _view: Decision(answer="Complete"),
    )

    receipts = runner._dispatch([assembly_call, repair_call])

    assert [receipt.call for receipt in receipts] == [assembly_call, repair_call]
    assert all(receipt.outcome.status == OutcomeStatus.FOUND for receipt in receipts)
    assert "The entitlement depends on approval [2]." in str(
        context.services["assembled_answer"]
    )
    assert "The entitlement is automatic." not in str(
        context.services["assembled_answer"]
    )
    assert research.answers[1] == other_answer
    assert str(other_answer["answer"]) in str(context.services["assembled_answer"])
    assert len(workers.calls) == 2
    assert len(research.answer_revisions) == 1


def test_failed_source_guard_leaves_target_and_revision_journal_unchanged() -> None:
    research, _, context = completed_answers_for_repair()
    previous = copy.deepcopy(research.answers)
    rejection = ToolOutcome(
        status=OutcomeStatus.PARTIAL, summary="The cited original was not delivered."
    )
    research.repair_guard = lambda _candidate: rejection
    assert (
        research.repair_question_answer(repair_arguments(research), context)
        is rejection
    )
    assert research.answers == previous
    assert research.answer_revisions == []


def test_rejected_repair_blocks_same_decision_assembly_but_preserves_focused_reads() -> (
    None
):
    research, _, context = completed_answers_for_repair()
    before = copy.deepcopy(research.answers)
    research.repair_guard = lambda _candidate: ToolOutcome(
        status=OutcomeStatus.PARTIAL, summary="The new citation was not delivered."
    )
    reads: list[str] = []

    def read_original(
        _arguments: dict[str, JsonValue], _context: RunContext
    ) -> ToolOutcome:
        reads.append("focused-original")
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Missing original read")

    registry = CapabilityRegistry(
        research.tool_specs()
        + [
            ToolSpec(
                name="read_original",
                description="Read one original",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=read_original,
            )
        ]
    )
    repair_call = CapabilityCall(
        name="repair_question_answer", arguments=repair_arguments(research)
    )
    assembly_call = CapabilityCall(
        name="assemble_answers", arguments={"order": ["first", "second"]}
    )
    read_call = CapabilityCall(name="read_original", arguments={})
    runner = Harness(
        request="First and second",
        context=context,
        registry=registry,
        decide=lambda _view: Decision(answer="Complete"),
    )

    receipts = runner._dispatch([assembly_call, repair_call, read_call])

    assert [receipt.call for receipt in receipts] == [
        assembly_call,
        repair_call,
        read_call,
    ]
    assert receipts[0].outcome.status == OutcomeStatus.PARTIAL
    assert receipts[0].outcome.data["rejected_repair_call_ids"] == [repair_call.call_id]
    assert receipts[1].outcome.status == OutcomeStatus.PARTIAL
    assert receipts[2].outcome.status == OutcomeStatus.FOUND
    assert reads == ["focused-original"]
    assert "assembled_answer" not in context.services
    assert research.answers == before
    assert research.answer_revisions == []


@pytest.mark.parametrize(
    "bad_edit",
    [
        {"kind": "replace", "target_text": "Unknown target", "text": "Corrected [2]."},
        {
            "kind": "replace",
            "target_text": "The entitlement is automatic.",
            "text": "Uncited correction.",
        },
        {
            "kind": "replace",
            "target_text": "The entitlement is automatic. [1]\n\nThe later stage remains available [3].",
            "text": "Shortened answer [2].",
        },
        {
            "kind": "insert_after",
            "target_text": "The entitlement is automatic.",
            "text": "",
        },
    ],
)
def test_invalid_targeted_edits_never_modify_completed_answers(
    bad_edit: dict[str, JsonValue],
) -> None:
    research, _, context = completed_answers_for_repair()
    previous = copy.deepcopy(research.answers)
    arguments = repair_arguments(research)
    arguments["edits"] = [bad_edit]
    with pytest.raises(ValueError):
        research.repair_question_answer(arguments, context)
    assert research.answers == previous
    assert research.answer_revisions == []


def test_overlapping_edits_and_stale_revision_cannot_overwrite_a_repaired_answer() -> (
    None
):
    research, _, context = completed_answers_for_repair()
    arguments = repair_arguments(research)
    assert isinstance(arguments["edits"], list)
    overlapping = copy.deepcopy(arguments)
    overlapping["edits"] = [
        arguments["edits"][0],
        {"kind": "replace", "target_text": "automatic", "text": "conditional [2]"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        research.repair_question_answer(overlapping, context)
    research.repair_question_answer(arguments, context)
    current = copy.deepcopy(research.answers)
    with pytest.raises(ValueError, match="current answer_hash"):
        research.repair_question_answer(arguments, context)
    assert research.answers == current
    assert len(research.answer_revisions) == 1


def test_repair_requires_guard_and_revisions_restore_with_full_bodies() -> None:
    research, _, context = completed_answers_for_repair()
    research.repair_guard = None
    assert (
        research.repair_question_answer(repair_arguments(research), context).status
        == OutcomeStatus.DENIED
    )
    assert research.answer_revisions == []
    research.repair_guard = lambda _candidate: None
    research.repair_question_answer(repair_arguments(research), context)
    snapshot = research.export()
    resumed, workers, _ = controlled_research([], ["First", "Second"])
    resumed.restore(snapshot)
    assert resumed.answers == research.answers
    assert resumed.answer_revisions == research.answer_revisions
    assert workers.calls == []
    damaged = copy.deepcopy(snapshot)
    assert isinstance(damaged["answers"], list) and isinstance(
        damaged["answers"][0], dict
    )
    damaged["answers"][0]["answer"] = "A changed answer [2]."
    with pytest.raises(ValueError):
        resumed.restore(damaged)


@pytest.mark.parametrize("wrong_binding", [False, True])
def test_assignment_outcomes_are_validated_against_their_original_question(
    wrong_binding: bool,
) -> None:
    research, workers, context = controlled_research(
        [found("First [1]"), found("Second [2]")], ["First", "Second"]
    )
    outcomes = OutcomeMap(["First", "Second"], context)
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [
                    {
                        "outcome_id": "one",
                        "question_ids": ["q0"],
                        "detail": "First outcome",
                    },
                    {
                        "outcome_id": "two",
                        "question_ids": ["q1"],
                        "detail": "Second outcome",
                    },
                ]
            }
        ),
        EvidenceLedger(),
    )
    context.services["outcome_map"] = outcomes
    first, second = question("first", "First", 1), question("second", "Second", 2)
    first["outcome_ids"], second["outcome_ids"] = (
        ["two" if wrong_binding else "one"],
        ["two"],
    )
    if wrong_binding:
        result = research.research_questions({"questions": [first, second]}, context)
        assert result.status is OutcomeStatus.INVALID
        assert "original questions" in str(result.data["validation_error"])
        assert workers.calls == []
    else:
        research.research_questions({"questions": [first, second]}, context)
        assert workers.outcome_bindings == [["one"], ["two"]]
        assert research.answers[0]["outcome_ids"] == ["one"]
        assert research.answers[1]["outcome_ids"] == ["two"]


def test_undeclared_assignment_outcomes_return_invalid_without_starting_research() -> (
    None
):
    research, workers, context = controlled_research([found("Answer [1]")], ["First"])
    outcomes = OutcomeMap(["First"], context)
    context.services["outcome_map"] = outcomes
    entry = question("first", "First", 1)
    entry["outcome_ids"] = ["o1", "o2", "o3", "o4"]
    context.services["evidence"] = EvidenceLedger()
    registry = CapabilityRegistry(research.tool_specs())
    result = registry.dispatch(
        CapabilityCall(name="research_questions", arguments={"questions": [entry]}),
        context,
    )
    assert result.status is OutcomeStatus.INVALID
    assert result.data["validation_error"] == (
        "Undeclared assignment outcomes for first: o1, o2, o3, o4"
    )
    assert "Omit outcome_ids" in str(result.data["instruction"])
    assert workers.calls == []
    assert research.assignments == []
    assert research.answers == []
    assert outcomes.outcome_ids() == []
    assert "question_research_started" not in context.services

    entry.pop("outcome_ids")
    retried = registry.dispatch(
        CapabilityCall(name="research_questions", arguments={"questions": [entry]}),
        context,
    )
    assert retried.status is OutcomeStatus.FOUND
    assert len(workers.calls) == 1
    assert workers.outcome_bindings == [None]
    assert outcomes.outcome_ids() == []


def test_assignment_outcomes_may_be_declared_on_the_same_useful_action() -> None:
    research, workers, context = controlled_research([found("Answer [1]")], ["First"])
    outcomes = OutcomeMap(["First"], context)
    context.services["outcome_map"] = outcomes
    entry = question("first", "First", 1)
    entry["outcome_ids"] = ["first-outcome"]
    context.services["evidence"] = EvidenceLedger()
    result = CapabilityRegistry(research.tool_specs()).dispatch(
        CapabilityCall(
            name="research_questions",
            arguments={
                "questions": [entry],
                "_outcomes": [
                    {
                        "outcome_id": "first-outcome",
                        "question_ids": ["q0"],
                        "detail": "The requested first outcome",
                    }
                ],
            },
        ),
        context,
    )
    assert result.status is OutcomeStatus.FOUND
    assert workers.outcome_bindings == [["first-outcome"]]
    assert outcomes.outcome_ids() == ["first-outcome"]


def test_worker_outcome_binding_is_available_before_the_worker_starts() -> None:
    context = RunContext()
    observed: list[object] = []

    def runner(
        _task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        observed.append(child.services.get("task_outcome_ids"))
        return found("The result [1].")

    pool = WorkerPool(context, runner)
    try:
        task_id = pool.spawn(
            "A specific outcome",
            request_context=context,
            independent_question=True,
            outcome_ids=["one"],
        )
        pool.wait_until_all([task_id])
        assert observed == [["one"]]
        assert "task_outcome_ids" not in context.services
    finally:
        context.cancel()
        pool.close()
