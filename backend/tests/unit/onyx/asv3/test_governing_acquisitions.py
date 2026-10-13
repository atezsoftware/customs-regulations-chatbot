"""A failed source episode cannot become a new lookup by changing answer prose."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.governing_acquisitions import (
    GoverningOriginalAcquisitions,
    governing_original_targets,
)
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.model_response import ModelResponse
from tests.unit.onyx.asv3.test_confirmed_authority import captured
from tests.unit.onyx.asv3.test_native_authority import original
from tests.unit.onyx.asv3.test_runtime import response, setup_run
from tests.unit.onyx.asv3.test_spawned_researcher_terminal import capture_harnesses

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def gap(unit: str = "changed-unit") -> dict[str, JsonValue]:
    return {
        "named_authority_gaps": [
            {
                "unit_id": unit,
                "instrument_number": "8917",
                "article": "27",
                "matching_original_evidence": [],
            }
        ]
    }


def receipt(
    call: CapabilityCall,
    status: OutcomeStatus,
    data: dict[str, JsonValue],
    numbers: list[int] | None = None,
) -> ToolReceipt:
    return ToolReceipt(
        call=call,
        outcome=ToolOutcome(status=status, summary="Actual corpus result", data=data),
        elapsed_seconds=0,
        evidence_ids=numbers or [],
    )


def test_historical_54_submission_gaps_do_not_create_52_research_episodes() -> None:
    data = json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("governing_acquisition_5094_gaps.json")
        .read_text()
    )
    assert len(data["attempts"]) == 54
    targets = [
        target
        for row in data["attempts"]
        for target in governing_original_targets(
            {"named_authority_gaps": row["named_authority_gaps"]}
        )
        if target.instrument_number == "6455" and target.article == "11"
    ]
    assert len(targets) >= 52 and len({target.key for target in targets}) == 1
    context, ledger = RunContext(scope={"document_set_id": 15}), EvidenceLedger()
    state = GoverningOriginalAcquisitions(context, "same historical question")
    first = state.next_call(targets[0], context, ledger)
    assert first is not None and state.reserve(targets[0], first, context)
    assert state.record_receipt(
        receipt(
            first,
            OutcomeStatus.NOT_FOUND,
            {"sources": [], "has_more": False, "absence_proven": False},
        ),
        context,
        ledger,
    )
    for target in targets:
        assert state.next_call(target, context, ledger) is None
    episodes = state.export()["episodes"]
    assert isinstance(episodes, dict)
    assert len(episodes) == 1


def test_global_reservation_is_atomic_for_root_and_worker() -> None:
    context = RunContext(scope={"document_set_id": 15})
    child = context.child()
    child.services["task_id"] = "actual-task"
    ledger = EvidenceLedger()
    state = GoverningOriginalAcquisitions(context, "same question")
    target = governing_original_targets(gap())[0]
    calls = [state.next_call(target, current, ledger) for current in (context, child)]
    assert all(call is not None for call in calls)
    with ThreadPoolExecutor(max_workers=2) as executor:
        answers = list(
            executor.map(
                lambda pair: state.reserve(target, pair[0], pair[1]),
                zip(calls, (context, child)),
            )
        )
    assert sorted(answers) == [False, True]


def test_empty_advancing_partial_scan_continues_same_episode() -> None:
    context, ledger = RunContext(), EvidenceLedger()
    state = GoverningOriginalAcquisitions(context, "request")
    target = governing_original_targets(gap())[0]
    call = state.next_call(target, context, ledger)
    assert call is not None and state.reserve(target, call, context)
    state.record_receipt(
        receipt(
            call,
            OutcomeStatus.PARTIAL,
            {
                "source_id": "actual-source",
                "scan_truncated": True,
                "evidence_truncated": False,
                "evidence_next_position": None,
                "next_position": 2000,
            },
        ),
        context,
        ledger,
    )
    continuation = state.next_call(target, context, ledger)
    assert continuation is not None
    assert continuation.name == "read_provision"
    assert continuation.arguments == {
        "source_id": "actual-source",
        "article": "27",
        "start": 2000,
    }
    assert state.reserve(target, continuation, context)
    state.record_receipt(
        receipt(
            continuation,
            OutcomeStatus.PARTIAL,
            {
                "scan_truncated": False,
                "evidence_truncated": False,
                "subunit_verified": False,
                "next_position": 2500,
            },
        ),
        context,
        ledger,
    )
    assert state.next_call(target, context, ledger) is None
    assert state.unavailable_keys(gap(), ledger) == {target.key}


def test_clipped_literal_cursor_does_not_skip_to_scan_cursor() -> None:
    context, ledger = RunContext(), EvidenceLedger()
    state = GoverningOriginalAcquisitions(context, "request")
    target = governing_original_targets(gap())[0]
    call = state.next_call(target, context, ledger)
    assert call is not None and state.reserve(target, call, context)
    state.record_receipt(
        receipt(
            call,
            OutcomeStatus.PARTIAL,
            {
                "source_id": "actual-source",
                "scan_truncated": True,
                "evidence_truncated": True,
                "evidence_next_position": 12,
                "next_position": 2000,
            },
        ),
        context,
        ledger,
    )
    continuation = state.next_call(target, context, ledger)
    assert continuation is not None and continuation.arguments["start"] == 12


def test_missing_numeric_paragraph_and_qualifier_keep_exact_identity() -> None:
    ledger, context = EvidenceLedger(), RunContext()
    ledger.add([captured(291)], context)
    bare = {
        "instrument_number": "5326",
        "article": "15",
        "paragraph": "3",
        "matching_original_evidence": [],
    }
    target = governing_original_targets({"named_authority_gaps": [bare]})[0]
    same = governing_original_targets(
        {
            "unread_cited_statute_references": [
                {**bare, "article": "15/3", "unit_id": "other-prose"}
            ]
        }
    )[0]
    assert target.key == same.key
    state = GoverningOriginalAcquisitions(context, "request")
    assert state.matching_originals(target, ledger) == []
    call = state.next_call(target, context, ledger)
    assert call is not None and call.arguments["paragraph"] == "3"
    temporary = governing_original_targets(
        {"named_authority_gaps": [{**bare, "article_qualifier": "geçici"}]}
    )[0]
    assert temporary.key != target.key


def test_later_actual_original_fulfils_without_reopening_lookup_or_forging_delivery() -> (
    None
):
    ledger, context = EvidenceLedger(), RunContext()
    state = GoverningOriginalAcquisitions(context, "request")
    target = governing_original_targets(gap())[0]
    call = state.next_call(target, context, ledger)
    assert call is not None and state.reserve(target, call, context)
    state.record_receipt(
        receipt(call, OutcomeStatus.NOT_FOUND, {"sources": [], "has_more": False}),
        context,
        ledger,
    )
    assert state.unavailable_keys(gap(), ledger) == {target.key}
    item = original("8917 sayılı Faaliyet Kanunu", "27")
    ledger.add([item], context)
    assert state.next_call(target, context, ledger) is None
    assert state.unavailable_keys(gap(), ledger) == set()
    assert ledger.completely_delivered("future-provider") == set()
    assert state.records_for([target])[0]["legal_approval"] is False


def test_interrupted_reservation_cannot_retry_and_checkpoint_is_typed() -> None:
    context, ledger = RunContext(), EvidenceLedger()
    state = GoverningOriginalAcquisitions(context, "request")
    target = governing_original_targets(gap())[0]
    call = state.next_call(target, context, ledger)
    assert call is not None and state.reserve(target, call, context)
    snapshot = state.export()
    restored = GoverningOriginalAcquisitions(context, "request")
    restored.restore(snapshot)
    assert restored.next_call(target, context, ledger) is None
    assert restored.unavailable_keys(gap(), ledger) == {target.key}
    malformed = copy.deepcopy(snapshot)
    malformed["version"] = True
    with pytest.raises(ValueError):
        GoverningOriginalAcquisitions(context, "request").restore(malformed)
    foreign = RunContext(scope={"document_set_id": 99})
    with pytest.raises(ValueError):
        GoverningOriginalAcquisitions(foreign, "request").restore(snapshot)


def test_actual_tuned_runtime_54_changing_native_submissions_stop_after_unavailable_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    harnesses = capture_harnesses(monkeypatch)
    lookups = 0

    def missing(*_args: Any, **_kwargs: Any) -> tuple[list[Any], bool]:
        nonlocal lookups
        lookups += 1
        return [], False

    monkeypatch.setattr(broker, "sources", missing)
    calls = 0

    def submit(**_arguments: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert calls <= 54
        return response(
            calls=[
                (
                    "submit_partial_answer",
                    {
                        "answer": f"8917 sayılı Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir. Açıklama {calls} değişti."
                    },
                )
            ]
        )

    selected.invoke.side_effect = submit
    with pytest.raises(OnyxError):
        runtime.run_asv3_loop(**kwargs)
    assert lookups == 1 and calls == 4
    root = harnesses[0]
    assert root.context.services.get("submitted_answer") is None
    assert root.partial_submission is not None and root.partial_submission() is None
    assert root.last_draft and root.publication_gap is not None
    assert root.stop_reason == "repeated_unavailable_governing_original"
    assert checkpoints[-1]["governing_original_acquisitions"]["episodes"]


def test_actual_episode_is_durable_before_tool_and_legacy_checkpoint_remains_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    seen_reserved = False

    def missing(*_args: Any, **_kwargs: Any) -> tuple[list[Any], bool]:
        nonlocal seen_reserved
        episodes = checkpoints[-1]["governing_original_acquisitions"]["episodes"]
        seen_reserved = any(row["active_call"] is not None for row in episodes.values())
        return [], False

    monkeypatch.setattr(broker, "sources", missing)
    selected.invoke.side_effect = [
        response(
            calls=[
                (
                    "submit_partial_answer",
                    {
                        "answer": "8917 sayılı Faaliyet Kanunu m.27 uyarınca izin gerekir."
                    },
                )
            ]
        ),
        response(
            calls=[
                (
                    "submit_partial_answer",
                    {
                        "answer": "8917 sayılı Kanun 27. maddesinin özgün metni elde edilemedi."
                    },
                )
            ]
        ),
    ]
    runtime.run_asv3_loop(**kwargs)
    assert seen_reserved and selected.invoke.call_count == 2
    legacy = copy.deepcopy(checkpoints[-1])
    legacy.pop("governing_original_acquisitions")
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: legacy)
    kwargs["resume_message_id"] = kwargs["assistant_message_id"]
    selected.invoke.side_effect = [
        response(
            calls=[
                (
                    "submit_partial_answer",
                    {
                        "answer": "8917 sayılı Kanun 27. maddesinin özgün metni elde edilemedi."
                    },
                )
            ]
        )
    ]
    runtime.run_asv3_loop(**kwargs)
