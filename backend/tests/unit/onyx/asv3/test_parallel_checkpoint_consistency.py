"""Check concurrent host snapshots and exact parallel publication boundaries."""

from typing import Literal, cast

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.asv3.question_research import QuestionResearch
from tests.unit.onyx.asv3.test_experimental_parallel_runtime import (
    SCENARIO,
    TASKS,
    script_two_children,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def test_checkpoint_refresh_includes_seals_and_originals_created_after_root_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, selected, _, checkpoints, _, _, _, bodies, _ = script_two_children(
        monkeypatch
    )
    contexts: list[RunContext] = []
    original_init = ParallelAnswerReceipts.__init__
    original_export = ParallelAnswerReceipts.export
    injected = False

    def capture_context(
        self: ParallelAnswerReceipts,
        context: RunContext,
        request: str,
        *,
        user_id: str | None,
    ) -> None:
        contexts.append(context)
        original_init(self, context, request, user_id=user_id)

    def export_after_concurrent_acceptance(
        self: ParallelAnswerReceipts,
    ) -> dict[str, JsonValue]:
        nonlocal injected
        current = original_export(self)
        saved = cast(list[JsonValue], current["receipts"])
        if len(saved) < 2 or injected:
            return current
        injected = True
        context = contexts[0]
        ledger = context.services["evidence"]
        assert isinstance(ledger, EvidenceLedger)
        number = next(iter(ledger.citation_mapping()))
        item = ledger.get(number)
        assert item is not None
        # Simulate a sibling finishing after the root captured Harness.snapshot().
        ledger.add(
            [EvidenceItem(source_id="late-source", text="Late original navigation.")],
            context,
        )
        ledger.record_delivery(
            "concurrent-accepted-call",
            "asv3_researcher",
            [{"citation": number, "text": item.text}],
        )
        child = context.independent_child()
        child.services.update(
            task_id="concurrent-task", last_model_call_id="concurrent-accepted-call"
        )
        self.seal(
            child,
            assignment={"task_id": "concurrent-task", "question_id": "concurrent"},
            answer=f"An accepted source condition [{number}].",
            status=OutcomeStatus.FOUND,
            model_call_id="concurrent-accepted-call",
            ledger=ledger,
            validate_body=lambda: None,
        )
        return original_export(self)

    monkeypatch.setattr(ParallelAnswerReceipts, "__init__", capture_context)
    monkeypatch.setattr(
        ParallelAnswerReceipts, "export", export_after_concurrent_acceptance
    )
    runtime.run_asv3_loop(**kwargs)

    assert injected
    assert selected.invoke.call_count == 5
    verified_snapshots = 0
    for snapshot in checkpoints:
        receipts = snapshot["parallel_answers"]["receipts"]
        if not any(
            row["model_call_id"] == "concurrent-accepted-call" for row in receipts
        ):
            continue
        context = RunContext(run_id=snapshot["run_id"], scope=snapshot["scope"])
        ledger = EvidenceLedger()
        ledger.restore(snapshot["evidence"], context)
        assert ledger.completely_delivered("concurrent-accepted-call")
        assert (
            "concurrent-accepted-call" in snapshot["evidence"]["pinned_delivery_calls"]
        )
        assert any(
            row["item"]["source_id"] == "late-source"
            for row in snapshot["evidence"]["records"]
        )
        store = ParallelAnswerReceipts(
            context, SCENARIO, user_id=str(kwargs["user"].id)
        )
        store.restore(snapshot["parallel_answers"], context, SCENARIO, ledger)
        verified_snapshots += 1
    assert verified_snapshots > 0
    assert bodies[TASKS[0]] in checkpoints[-1]["last_draft"]
    assert bodies[TASKS[1]] in checkpoints[-1]["last_draft"]


@pytest.mark.parametrize("second_result", ["full", "partial", "error"])
@pytest.mark.parametrize("extra", ["uncited", "old-citation"])
def test_direct_candidate_cannot_extend_child_seals_with_unexamined_claims(
    monkeypatch: pytest.MonkeyPatch,
    second_result: Literal["full", "partial", "error"],
    extra: Literal["uncited", "old-citation"],
) -> None:
    kwargs, selected, _, checkpoints, _, _, _, bodies, _ = script_two_children(
        monkeypatch, second_result=second_result
    )
    original_assemble = QuestionResearch.assemble_answers
    rejections: list[ToolOutcome | None] = []

    def probe_publication_guard(
        self: QuestionResearch, arguments: dict[str, JsonValue], context: RunContext
    ) -> ToolOutcome:
        order = cast(list[str], arguments["order"])
        answers = {str(row["question_id"]): row for row in self.answers}
        exact = "\n\n".join(
            f"## {index}. {answers[identifier]['answer_title']}\n\n"
            + str(answers[identifier]["answer"])
            for index, identifier in enumerate(order, 1)
        )
        assert self.publish_guard is not None
        claim = "İşlemin bütün yükümlülükleri ayrıca kendiliğinden sona erer."
        if extra == "old-citation":
            number = cast(list[int], self.answers[0]["evidence_numbers"])[0]
            claim += f" [{number}]"
        rejections.append(self.publish_guard(exact + "\n\n" + claim))
        return original_assemble(self, arguments, context)

    monkeypatch.setattr(QuestionResearch, "assemble_answers", probe_publication_guard)
    runtime.run_asv3_loop(**kwargs)

    assert rejections and all(gap is not None for gap in rejections)
    assert selected.invoke.call_count == 5
    final = checkpoints[-1]
    assert bodies[TASKS[0]] in final["last_draft"]
    assert "bütün yükümlülükleri ayrıca kendiliğinden" not in final["last_draft"]
    assert final["question_research"]["answers"][0]["answer"] == bodies[TASKS[0]]
    assert len(bodies[TASKS[0]]) > 12000
    assert "İĞŞçöü — 東京" in bodies[TASKS[0]]
