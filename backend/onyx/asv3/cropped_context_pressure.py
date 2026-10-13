"""Advisory topic focus for unusually large Cropped research packets."""

from pydantic import JsonValue

from onyx.asv3.cropped_task_packets import CroppedTaskPacketStore
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import HarnessView, RunContext


def cropped_context_pressure(
    context: RunContext, view: HarnessView, *, writing_answer: bool
) -> dict[str, JsonValue] | None:
    if (
        context.services.get("asv3_exp_cropped") is not True
        or context.services.get("asv3_exp_cheap") is True
        or context.depth
        or writing_answer
        or view.draft_to_repair is not None
        or view.publication_gap is not None
    ):
        return None
    store = context.services.get("cropped_task_packet_store")
    ledger = context.services.get("evidence")
    active = view.research_state.get("active_need_ids")
    if (
        not isinstance(store, CroppedTaskPacketStore)
        or not isinstance(ledger, EvidenceLedger)
        or not isinstance(active, list)
        or len(active) < 2
    ):
        return None
    visible = {
        number
        for row in view.original_evidence
        if type(number := row.get("citation")) is int and ledger.get(number) is not None
    }
    visible.update(view.required_evidence_numbers)
    count = len(visible)
    if count < store.policy.original_records:
        return None
    return {
        "packet_original_count": count,
        "topic_packet_threshold": store.policy.original_records,
        "instruction": (
            "This research packet has grown beyond the ordinary topic-packet threshold. "
            "In the next update_research, select active_need_ids for the actual next "
            "batch and its material dependencies instead of carrying every topic forward. "
            "Keep parked needs and pending cursors open. Record candidate findings with "
            "literal witnesses when the stated completion test is actually supported; "
            "volume alone never establishes completion, irrelevance or absence. "
            "Preserve cross-topic dependencies and reactivate a need if a new source or "
            "fact changes it. All retained originals return for final writing. This "
            "advice does not close a need, discard a source or impose a research cap."
        ),
    }
