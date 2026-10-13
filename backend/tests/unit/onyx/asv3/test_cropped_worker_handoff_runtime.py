"""The actual runtime collects partial worker research before guarded root writing."""

import copy
import json
from typing import Any

import pytest

from onyx.asv3 import runtime
from onyx.asv3.cheap_metadata_transport import expand_cheap_payload
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.cropped_worker_handoff import (
    WORKER_HANDOFF_REASON,
    expand_worker_handoff,
)
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CHEAP_VARIANT,
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_TUNED_VARIANT,
)
from onyx.llm.model_response import ModelResponse
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cropped_focused_repair_runtime import (
    priced_scripted_provider as priced_scripted_provider,
)
from tests.unit.onyx.asv3.test_runtime import (
    delivered_originals,
    response,
    setup_run,
    user_payload,
)
from tests.unit.onyx.asv3.test_spawned_researcher_terminal import (
    capture_harnesses,
    scripted_coordinator,
)

pytestmark = pytest.mark.usefixtures(
    "empty_source_inventory", "priced_scripted_provider"
)


def test_long_tail_from_actual_child_read_hands_off_before_another_worker_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _ = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_EXP_CROPPED_VARIANT)
    harnesses = capture_harnesses(monkeypatch)
    source = str(broker.sources[0].id)
    task = "Tamir koşulunun özgün metnini araştır."
    observation = "Kullanıcının olay tarihi verilmedi."
    reasoning = (
        "Özgün metni okuyup istisna kapsamını kök araştırmaya bırakıyorum. " * 500
    )
    calls: list[tuple[int, LLMFlow]] = []
    invoke_once = ResearchModel._invoke_once

    def measured(model: ResearchModel, *args: Any, **options: Any) -> ModelResponse:
        flow = options["flow"] if "flow" in options else args[2]
        calls.append((model.context.depth, flow))
        return invoke_once(model, *args, **options)

    monkeypatch.setattr(ResearchModel, "_invoke_once", measured)

    def page(
        source_id: str, context: RunContext, **_options: Any
    ) -> tuple[Any, list[Any], bool]:
        context.check_active()
        assert context.depth == 1 and source_id == source
        controller = context.services["cropped_research_controller"]
        assert isinstance(controller, CroppedResearchLimit)
        controller.force_finish("actual source-read boundary enters long tail")
        return broker.sources[0], [broker.chunks[source]], False

    monkeypatch.setattr(broker, "page", page)
    started = False
    child_calls = 0
    root_final_packets: list[dict[str, Any]] = []

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal started, child_calls
        payload = expand_cheap_payload(user_payload(arguments["prompt"][-1]))
        if payload["request"] == task:
            child_calls += 1
            assert child_calls == 1, "A long-tail worker must not write or decide again"
            return response(
                text=reasoning,
                calls=[
                    ("record_scenario", {"facts": [observation]}),
                    (
                        "read_source_range",
                        {"source_id": source, "_need_id": "fixture_source_need_0"},
                    ),
                ],
            )
        if not started:
            started = True
            return response(
                calls=[
                    (
                        "update_research",
                        {
                            "needs": [
                                {
                                    "need_id": "fixture_source_need_0",
                                    "question_ids": ["q0"],
                                    "purpose": task,
                                    "completion_test": "Full repair original and applicable exceptions, with unresolved dates retained.",
                                    "material": True,
                                }
                            ]
                        },
                    ),
                    (
                        "spawn_researcher",
                        {
                            "task": task,
                            "_language": "tr",
                            "need_ids": ["fixture_source_need_0"],
                        },
                    ),
                ]
            )
        tasks = payload["research_tasks"]
        assert len(tasks) == 1
        if tasks[0]["status"] != "completed":
            return response(
                calls=[
                    (
                        "wait_researcher",
                        {"task_id": tasks[0]["task_id"], "timeout_seconds": 5},
                    )
                ]
            )
        outcome = tasks[0]["outcome"]
        assert outcome["status"] == "partial"
        handoff = outcome["data"]["cropped_worker_handoff"]
        assert len(json.dumps(handoff, ensure_ascii=False)) > 12_000
        root = next(row for row in harnesses if row.context.depth == 0)
        actual = expand_worker_handoff(
            handoff, ledger=root.evidence, context=root.context
        )
        assert actual["turns"][0]["assistant"]["content"] == reasoning
        assert actual["receipts"][0]["outcome"]["data"]["facts"] == [observation]
        assert (
            handoff["absence_proven"] is False
            and handoff["publication_approved"] is False
        )
        decoded_arguments = {
            **arguments,
            "prompt": [
                *arguments["prompt"][:-1],
                arguments["prompt"][-1].model_copy(
                    update={"content": json.dumps(payload, ensure_ascii=False)}
                ),
            ],
        }
        originals = delivered_originals(decoded_arguments)
        assert (
            len(originals) == 1 and originals[0]["text"] == broker.chunks[source].text
        )
        root_final_packets.append(copy.deepcopy(payload))
        return response(
            calls=[
                (
                    "submit_answer",
                    {
                        "answer": f"Tamir şartları özgün metinde yer alır [{originals[0]['citation']}].",
                        "basis": "originals",
                    },
                )
            ]
        )

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    child = next(row for row in harnesses if row.context.depth)
    root = next(row for row in harnesses if not row.context.depth)
    assert child_calls == 1 and child.stop_reason == WORKER_HANDOFF_REASON
    assert child.complete_submission is not None and child.complete_submission() is None
    assert child.partial_submission is not None and child.partial_submission() is None
    assert len(root_final_packets) == 1
    waits = [
        receipt for receipt in root.receipts if receipt.call.name == "wait_researcher"
    ]
    assert waits
    wait_handoff = waits[-1].outcome.data["tasks"][0]["outcome"]["data"][
        "cropped_worker_handoff"
    ]
    assert (
        expand_worker_handoff(wait_handoff, ledger=root.evidence, context=root.context)[
            "turns"
        ][0]["assistant"]["content"]
        == reasoning
    )
    assert any(depth == 0 and flow == LLMFlow.ASV3_FINAL for depth, flow in calls)
    root.max_context_chars = 300_000
    wait_view = next(
        receipt
        for receipt in root.view().receipts
        if receipt.call.call_id == waits[-1].call.call_id
    )
    assert wait_view.outcome.data == waits[-1].outcome.data
    wait_snapshot = next(
        receipt
        for receipt in root.snapshot()["receipts"]
        if receipt["call"]["call_id"] == waits[-1].call.call_id
    )
    assert wait_snapshot["outcome"]["data"] == waits[-1].outcome.data
    assert not any(depth > 0 and flow == LLMFlow.ASV3_FINAL for depth, flow in calls)
    assert len([row for row in calls if row[0] > 0]) == 1
    tasks = checkpoints[-1]["workers"]["tasks"]
    assert (
        tasks[0]["status"] == "completed" and tasks[0]["outcome"]["status"] == "partial"
    )
    assert tasks[0]["outcome"]["data"]["cropped_worker_handoff"][
        "evidence_numbers"
    ] == [1]
    assert (
        expand_worker_handoff(
            tasks[0]["outcome"]["data"]["cropped_worker_handoff"],
            ledger=root.evidence,
            context=root.context,
        )["turns"][0]["assistant"]["content"]
        == reasoning
    )
    assert checkpoints[-1]["publication_status"] == OutcomeStatus.FOUND.value
    assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"
    assert broker.revalidated


@pytest.mark.parametrize("variant", [ASV3_TUNED_VARIANT, ASV3_EXP_CHEAP_VARIANT])
def test_protected_worker_keeps_ordinary_terminal_lifecycle(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    kwargs, broker, llm, checkpoints, _ = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
    harnesses = capture_harnesses(monkeypatch)
    source = str(broker.sources[0].id)
    task = "Tamir koşulunu araştır."

    def page(
        source_id: str, context: RunContext, **_options: Any
    ) -> tuple[Any, list[Any], bool]:
        context.check_active()
        assert (
            source_id == source
            and "cropped_research_controller" not in context.services
        )
        return broker.sources[0], [broker.chunks[source]], False

    monkeypatch.setattr(broker, "page", page)
    child_calls = 0

    def child(**arguments: Any) -> ModelResponse:
        nonlocal child_calls
        child_calls += 1
        if child_calls == 1:
            return response(calls=[("read_source_range", {"source_id": source})])
        assert child_calls == 2
        original = delivered_originals(arguments)[0]
        return response(
            calls=[
                (
                    "submit_answer",
                    {
                        "answer": f"Tamir şartları geçerlidir [{original['citation']}].",
                        "basis": "originals",
                    },
                )
            ]
        )

    llm.invoke.side_effect = scripted_coordinator([task], child)
    runtime.run_asv3_loop(**kwargs)
    child_harness = next(row for row in harnesses if row.context.depth)
    assert child_calls == 2 and child_harness.stop_reason == "model_submitted_answer"
    assert (
        child_harness.complete_submission is not None
        and child_harness.complete_submission()
    )
    assert (
        "cropped_worker_handoff"
        not in checkpoints[-1]["workers"]["tasks"][0]["outcome"]["data"]
    )
    assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"
