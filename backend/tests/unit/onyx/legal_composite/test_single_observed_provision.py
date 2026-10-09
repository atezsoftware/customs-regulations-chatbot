"""Fast terminal reads accept one complete selector without editing navigation."""

from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext, ToolOutcome
from onyx.legal_composite.acquisition import InvalidSourceAction
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    plan,
    registry,
    search_action,
)
from tests.unit.onyx.legal_composite.test_safe_observed_reads import fixture, read


@pytest.mark.parametrize(
    "selector",
    [
        "17",
        "17/2",
        "17/A",
        "17/2/a",
        " 17 / 2 / a ",
        "Madde 17",
        "MADDE17",
        "m. 17",
        "md.17",
        "Ek Madde 17",
        "EK MADDE17",
        "Geçici 17",
        "GEÇİCİ MADDE 17/2",
        "gecici madde 17",
        "Mükerrer Madde 17",
        "mukerrer 17/2/a",
        "Article 17",
        "art. 17",
    ],
)
def test_single_complete_selector_preserves_observed_navigation_identity(
    selector: str,
) -> None:
    acquirer, ledger, _built, invoked = fixture()
    action = read("read_provision", selector)
    original_action = action.model_dump(mode="json")
    evidence = ledger.serialize_records(ledger.citation_numbers())
    budget = acquirer.context.budget.snapshot()

    selected, remaining = acquirer.safe_observed_read_actions([action], plan(), {1})

    assert selected == [action] and selected[0] is action
    assert remaining == []
    assert action.model_dump(mode="json") == original_action
    assert ledger.serialize_records(ledger.citation_numbers()) == evidence
    assert acquirer.context.budget.snapshot() == budget
    assert not invoked and not acquirer._completed and not acquirer.last_receipts


@pytest.mark.parametrize(
    "selector",
    [
        "17-19",
        "17–19",
        "17—19",
        "Madde 17 - 19",
        "17 ve 18",
        "17 veya 18",
        "17,18",
        "17;18",
        "17/2,17/3",
        "17/2-a",
        "Madde 17 ve Madde 18",
        "17 and 18",
        "17 or 18",
        "17, 17",
        "17-17",
        "17/2/a ve 17/2/b",
        "Fictional Regulation Madde 17",
        "9999 sayılı Kanun Madde 17",
        "17 sayılı Karar",
        "Madde 17 tamamı ve istisnalar",
        "17/2/a/b",
        "",
    ],
)
def test_range_list_and_whole_instrument_references_remain_exact_unexecuted_proposals(
    selector: str,
) -> None:
    acquirer, ledger, _built, invoked = fixture()
    action = read("read_provision", selector)
    before = action.model_dump(mode="json")
    selected, remaining = acquirer.safe_observed_read_actions([action], plan(), {1})

    assert selected == [] and remaining == [action] and remaining[0] is action
    assert action.model_dump(mode="json") == before
    assert acquirer.pending_call_counts([action], plan()) == {"read_provision": 1}
    assert not invoked and ledger.citation_numbers() == (1,)


@pytest.mark.parametrize("fault", ["not_delivered", "unknown_source", "hash"])
def test_single_selector_never_bypasses_observed_canonical_identity(fault: str) -> None:
    acquirer, ledger, _built, invoked = fixture()
    action = read("read_provision", "17/2/a")
    delivered = {1}
    if fault == "not_delivered":
        delivered = set()
    elif fault == "unknown_source":
        action.arguments["source_id"] = str(uuid4())
        with pytest.raises(InvalidSourceAction):
            acquirer.safe_observed_read_actions([action], plan(), delivered)
        assert not invoked
        return
    else:
        ledger._items[1].text_hash = "0" * 64
    assert acquirer.safe_observed_read_actions([action], plan(), delivered) == (
        [],
        [action],
    )
    assert not invoked


def test_other_exact_read_tools_and_all_search_lanes_keep_original_behavior() -> None:
    acquirer, _ledger, built, invoked = fixture()
    actions = [
        read("read_provision", "17–19"),
        read("read_chunk"),
        search_action(),
        read("read_chunk_context"),
    ]
    selected, remaining = acquirer.safe_observed_read_actions(actions, plan(), {1})
    assert selected == [actions[1], actions[3]]
    assert remaining == [actions[0], actions[2]]
    assert len(built) == 12 and not invoked
    assert acquirer.pending_call_counts(remaining, plan()) == {
        "read_provision": 1,
        "search_corpus": 12,
    }


@pytest.mark.parametrize("selector", ["17–19", "17 ve 18", "17/2/a"])
def test_ordinary_capability_dispatch_keeps_the_proposed_selector_unchanged(
    selector: str,
) -> None:
    seen: list[dict[str, JsonValue]] = []
    expected = ToolOutcome(
        status=OutcomeStatus.NOT_FOUND, summary="Synthetic response."
    )

    def handler(arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        seen.append(dict(arguments))
        return expected

    action = read("read_provision", selector)
    call = CapabilityCall(name=action.tool, arguments=dict(action.arguments))
    before = call.model_dump(mode="json")
    assert registry(handler, handler).dispatch(call, RunContext()) is expected
    assert seen == [action.arguments]
    assert call.model_dump(mode="json") == before
