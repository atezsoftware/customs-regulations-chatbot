"""A bounded read planner cannot publish or replace final source checks."""

from unittest.mock import MagicMock, Mock

import pytest

from onyx.asv3 import cropped_read_planning
from onyx.asv3.cropped_research_limit import CroppedResearchLimit, CroppedResearchPolicy
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import EvidenceItem, HarnessView, RunStopped
from onyx.llm.model_response import Choice, Message, ModelResponse
from tests.unit.onyx.asv3.test_cropped_repair_model import current, fixture
from tests.unit.onyx.asv3.test_native_model_adapter import native_response


def planning_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ResearchModel, EvidenceLedger, MagicMock, HarnessView, CroppedResearchLimit, Mock
]:
    adapter, ledger, selected = fixture()
    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    view.tools.append(
        {
            "type": "function",
            "function": {
                "name": "read_provision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "article": {"type": "string"},
                    },
                    "required": ["source_id", "article"],
                    "additionalProperties": False,
                },
            },
        }
    )
    controller = CroppedResearchLimit(
        adapter.context,
        view.request,
        policy=CroppedResearchPolicy(enforce_observed_limits=True),
    )
    controller.force_finish("actual tail")
    adapter.context.services["cropped_research_controller"] = controller
    checkpoint = Mock()
    adapter.context.services["cropped_repair_checkpoint"] = checkpoint
    monkeypatch.setattr(
        cropped_read_planning,
        "build_cropped_read_planning_packet",
        lambda *_args, **_kwargs: {
            "request": view.request,
            "facts": list(view.facts),
            "planning_targets_available": True,
            "operations": [],
            "legal_approval": False,
            "navigation_only": True,
        },
    )
    return adapter, ledger, selected, view, controller, checkpoint


def test_planner_selects_reads_once_then_actual_writer_gets_all_old_and_new_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, selected, view, controller, checkpoint = planning_fixture(
        monkeypatch
    )
    read = native_response("read_provision", '{"source_id":"law","article":"8"}')
    read.choice.message.content = "This condition needs its governing original."
    terminal = native_response("submit_partial_answer", "{}")
    terminal.choice.message.content = "The certificate and consent conditions apply [1][2]. The later step remains [3]."
    selected.invoke.side_effect = [read, terminal]
    decision = adapter._decide_cropped_source_answer(view)
    assert decision.answer is None and decision.calls[0].name == "read_provision"
    assert ledger.completely_delivered(adapter.last_call_id or "") == set()
    assert checkpoint.call_args.kwargs == {"durable": True}
    assert controller.read_plan_reserved(adapter.context, "root")
    assert selected.invoke.call_args.kwargs["max_tokens"] == 2048
    assert not any(
        tool["function"]["name"].startswith("submit_")
        for tool in selected.invoke.call_args.kwargs["tools"]
    )
    [new_number] = ledger.add(
        [
            EvidenceItem(
                source_id="law",
                chunk_id="article8",
                text="The new original retains the full exception.",
            )
        ],
        adapter.context,
    )
    final = adapter._decide_cropped_source_answer(view)
    assert final.calls[0].name == "submit_partial_answer"
    assert selected.invoke.call_count == 2
    assert ledger.completely_delivered(adapter.last_call_id or "") == {
        1,
        2,
        3,
        new_number,
    }
    new_original = ledger.get(new_number)
    assert new_original is not None
    assert new_original.text in str(selected.invoke.call_args.kwargs["prompt"])
    assert view.facts == ["A certificate has not yet been signed."]


@pytest.mark.parametrize(
    "result,truncated",
    [("An unsupported proposed answer.", False), ("READ_PLANNING_DONE", True)],
)
def test_planning_envelope_failure_never_publishes_or_retries(
    monkeypatch: pytest.MonkeyPatch, result: str, truncated: bool
) -> None:
    adapter, _ledger, selected, view, controller, _checkpoint = planning_fixture(
        monkeypatch
    )
    response = ModelResponse(
        id="planning", created="0", choice=Choice(message=Message(content=result))
    )
    response.choice.finish_reason = "length" if truncated else "stop"
    selected.invoke.return_value = response
    with pytest.raises(RunStopped, match="truncated|unsupported completion"):
        adapter._decide_cropped_source_answer(view)
    assert selected.invoke.call_count == 1
    assert controller.read_plan_reserved(adapter.context, "root")


def test_no_concrete_lead_and_existing_repair_skip_new_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ledger, selected, view, _controller, _checkpoint = planning_fixture(
        monkeypatch
    )
    monkeypatch.setattr(
        cropped_read_planning,
        "build_cropped_read_planning_packet",
        lambda *_args, **_kwargs: {"planning_targets_available": False},
    )
    assert adapter._decide_cropped_read_plan(view) is None
    assert (
        adapter._decide_cropped_read_plan(
            view.model_copy(update={"draft_to_repair": "Retained answer."})
        )
        is None
    )
    selected.invoke.assert_not_called()
