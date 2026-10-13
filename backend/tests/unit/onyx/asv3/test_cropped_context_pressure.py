"""Large-packet focus advice leaves source state and ordinary paths intact."""

import copy
import json

import pytest

from onyx.asv3.cropped_context_pressure import cropped_context_pressure
from onyx.asv3.cropped_task_packets import (
    CroppedTaskPacketPolicy,
    CroppedTaskPacketStore,
)
from tests.unit.onyx.asv3.test_cropped_repair_model import current, fixture


def test_large_research_advice_does_not_change_state_or_originals() -> None:
    adapter, ledger, _ = fixture()
    view = current(ledger).model_copy(
        update={
            "draft_to_repair": None,
            "publication_gap": None,
            "research_state": {
                "active_need_ids": ["rule", "procedure"],
                "needs": [{"need_id": "rule", "status": "open", "next_position": 40}],
            },
        }
    )
    store = CroppedTaskPacketStore(
        adapter.context,
        view.request,
        policy=CroppedTaskPacketPolicy(original_records=3),
    )
    adapter.context.services["cropped_task_packet_store"] = store
    before = (view.model_dump(mode="json"), ledger.export(), store.export())
    prompt, _, _ = adapter._fit_native_decision(view)
    assert isinstance(prompt[-1].content, str)
    data = json.loads(prompt[-1].content)
    hint = data["cropped_context_pressure"]
    assert isinstance(hint, dict)
    assert hint["packet_original_count"] == hint["topic_packet_threshold"] == 3
    assert (view.model_dump(mode="json"), ledger.export(), store.export()) == before
    # Fitting again neither accumulates advice nor mutates host research status.
    second, _, _ = adapter._fit_native_decision(view)
    assert isinstance(second[-1].content, str)
    assert json.loads(second[-1].content)["cropped_context_pressure"] == hint
    assert view.research_state["needs"] == before[0]["research_state"]["needs"]


@pytest.mark.parametrize(
    "exception",
    [
        "ordinary",
        "tuned",
        "cheap",
        "child",
        "writer",
        "draft",
        "repair",
        "one_topic",
        "archived",
    ],
)
def test_advice_is_absent_from_protected_and_ordinary_paths(exception: str) -> None:
    adapter, ledger, _ = fixture()
    view = current(ledger).model_copy(
        update={
            "draft_to_repair": None,
            "publication_gap": None,
            "research_state": {"active_need_ids": ["a", "b"]},
        }
    )
    store = CroppedTaskPacketStore(
        adapter.context,
        view.request,
        policy=CroppedTaskPacketPolicy(
            original_records=4 if exception == "ordinary" else 3
        ),
    )
    adapter.context.services["cropped_task_packet_store"] = store
    if exception == "tuned":
        adapter.context.services["asv3_exp_cropped"] = False
    elif exception == "cheap":
        adapter.context.services["asv3_exp_cheap"] = True
    elif exception == "child":
        adapter.context.depth = 1
    elif exception == "draft":
        view.draft_to_repair = "Existing owned draft"
    elif exception == "repair":
        view.publication_gap = {"missing_source": True}
    elif exception == "one_topic":
        view.research_state["active_need_ids"] = ["a"]
    elif exception == "archived":
        view.original_evidence = view.original_evidence[:1]
        view.required_evidence_numbers = []
    before = copy.deepcopy(view.model_dump(mode="json"))
    assert (
        cropped_context_pressure(
            adapter.context, view, writing_answer=exception == "writer"
        )
        is None
    )
    assert view.model_dump(mode="json") == before
