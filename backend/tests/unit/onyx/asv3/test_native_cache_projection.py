import json
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext, RunStopped
from onyx.asv3.native_cache_projection import project_native_originals
from onyx.llm.models import ChatCompletionMessage, TextContentPart, ToolMessage
from tests.unit.onyx.asv3.test_native_metadata_projection import setup_original
from tests.unit.onyx.asv3.test_native_model_adapter import (
    last_payload,
    model,
    original,
    turn,
    view,
)
from tests.unit.onyx.asv3.test_parallel_authority_navigation import (
    complete,
)
from tests.unit.onyx.asv3.test_parallel_authority_navigation import (
    setup as authority_setup,
)


def parallel(context: RunContext) -> None:
    context.services.update(
        research_profile="experimental",
        experimental_parallel=True,
        scenario_request="Explain every outcome and condition from the fixed scenario.",
    )


def payloads(prompt: list[ChatCompletionMessage]) -> list[dict[str, Any]]:
    result = []
    for message in prompt:
        content = message.content
        texts = (
            [content]
            if isinstance(content, str)
            else [
                part.text for part in content or [] if isinstance(part, TextContentPart)
            ]
        )
        for text in texts:
            try:
                payload = json.loads(text)
            except ValueError:
                continue
            if isinstance(payload, dict):
                result.append(payload)
    return result


def actual_originals(prompt: list[ChatCompletionMessage]) -> list[dict[str, Any]]:
    return [
        row
        for payload in payloads(prompt)
        for row in payload.get("original_evidence", [])
        if isinstance(row, dict) and isinstance(row.get("text"), str)
    ]


def test_native_first_text_and_tail_fallback_are_actually_delivered_once() -> None:
    ledger, context, first = setup_original()
    parallel(context)
    second = original(ledger, context, "A later exception also changes the outcome.")
    native = [turn("first-read", [first]), turn("repeat-read", [first])]
    saved_turns = [item.model_dump(mode="json") for item in native]
    saved_records = ledger.export()["records"]
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(view(turns=native, original_evidence=[first, second]))
    prompt = llm.invoke.call_args.kwargs["prompt"]
    tools = [message for message in prompt if isinstance(message, ToolMessage)]
    acquired = json.loads(tools[0].content)
    repeated = json.loads(tools[1].content)
    assert acquired["original_evidence"][0]["text"] == first["text"]
    assert "original_evidence" not in repeated
    assert repeated["original_evidence_refs"][0]["text_hash"] == first["text_hash"]
    current = last_payload(llm)
    assert current["original_evidence"][0]["text"] == second["text"]
    assert [row["metadata"] for row in current["original_metadata_catalogue"]] == [
        first["metadata"],
        second["metadata"],
    ]
    delivered = actual_originals(prompt)
    assert [
        (
            row["citation"],
            row["source_id"],
            row["chunk_id"],
            row["text_hash"],
            row["text"],
        )
        for row in delivered
    ] == [
        tuple(
            row[key]
            for key in ("citation", "source_id", "chunk_id", "text_hash", "text")
        )
        for row in (first, second)
    ]
    assert all("metadata" not in row for row in delivered)
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert llm.invoke.call_count == 1
    assert [item.model_dump(mode="json") for item in native] == saved_turns
    assert ledger.export()["records"] == saved_records


def test_appending_source_and_refreshing_current_metadata_keeps_native_prefix() -> None:
    ledger, context, first = setup_original()
    parallel(context)
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    native = [turn("first-read", [first])]
    before, _, _ = adapter._fit_native_decision(view(turns=native))
    snapshot = ledger.export()
    records = cast(list[dict[str, Any]], snapshot["records"])
    records[0]["item"]["metadata"]["version_unknown"] = True
    ledger.restore(snapshot, context)
    second = original(ledger, context, "Independent later effect with its own proof.")
    after, _, _ = adapter._fit_native_decision(
        view(turns=[*native, turn("later-read", [second])])
    )
    assert [message.model_dump_json() for message in before[:-1]] == [
        message.model_dump_json() for message in after[: len(before) - 1]
    ]
    metadata = payloads(after)[-1]["original_metadata_catalogue"]
    assert metadata[0]["metadata"]["version_unknown"] is True
    assert "original_evidence" not in payloads(after)[-1]
    assert (
        native[0].results[0].content == turn("first-read", [first]).results[0].content
    )


def test_group_eviction_redelivers_required_original_in_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, context, first = setup_original()
    parallel(context)
    second = original(ledger, context, "The final control must also be performed.")
    llm = model(limit=1000000)
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    monkeypatch.setattr(
        adapter, "_research_instruction", lambda: "Use the exact originals."
    )
    retained_view = view(
        turns=[turn("latest", [second])],
        original_evidence=[first, second],
        required_evidence_numbers=[1, 2],
    )
    fitting_prompt, tools, _ = adapter._fit_native_decision(retained_view)
    ceiling = adapter._input_cost(fitting_prompt, tools) + 5
    monkeypatch.setattr(adapter, "_limits", lambda _: (ceiling, 100))
    adapter.decide(
        view(
            turns=[
                turn("old", [first], extra="OLD_NAVIGATION" * 1000),
                *retained_view.turns,
            ],
            original_evidence=[first, second],
            required_evidence_numbers=[1, 2],
        )
    )
    prompt = llm.invoke.call_args.kwargs["prompt"]
    assert [
        message.tool_call_id for message in prompt if isinstance(message, ToolMessage)
    ] == ["latest"]
    assert [(row["citation"], row["text"]) for row in actual_originals(prompt)] == [
        (2, second["text"]),
        (1, first["text"]),
    ]
    assert last_payload(llm)["original_evidence"][0]["citation"] == 1
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert "OLD_NAVIGATION" not in "".join(
        message.model_dump_json() for message in prompt
    )
    assert adapter._input_cost(prompt, tools) <= ceiling


@pytest.mark.parametrize("wider", [False, True])
def test_partial_and_wider_range_selection_never_invents_complete_delivery(
    wider: bool,
) -> None:
    ledger, context, full = setup_original()
    parallel(context)
    text = cast(str, full["text"])
    parts = [
        {
            **full,
            "text": text[start:end],
            "start_char": start,
            "end_char": end,
            "total_chars": len(text),
            "truncated": True,
        }
        for start, end in [(0, 24), (12, 40), (14, 18)]
    ]
    records = [*parts, full] if wider else parts
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(view(turns=[turn("read-parts", records)], original_evidence=records))
    prompt = llm.invoke.call_args.kwargs["prompt"]
    submitted = actual_originals(prompt)
    expected = (
        [(0, len(text), text)] if wider else [(0, 24, text[:24]), (12, 40, text[12:40])]
    )
    assert [
        (row["start_char"], row["end_char"], row["text"]) for row in submitted
    ] == expected
    assert ledger.completely_delivered(adapter.last_call_id or "") == (
        {1} if wider else set()
    )
    assert llm.invoke.call_count == 1


@pytest.mark.parametrize(
    "change", ["source_id", "chunk_id", "text_hash", "text", "start_char", "end_char"]
)
def test_foreign_original_is_rejected_before_any_provider_delivery(change: str) -> None:
    ledger, context, full = setup_original()
    parallel(context)
    invalid = {
        **full,
        change: 1
        if change == "start_char"
        else 999
        if change == "end_char"
        else "foreign",
    }
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    with pytest.raises(ValueError, match="canonical identity"):
        adapter.decide(
            view(turns=[turn("foreign-read", [cast(dict[str, JsonValue], invalid)])])
        )
    assert llm.invoke.call_count == 0
    assert ledger.export()["deliveries"] == []


def test_evicted_ranges_and_foreign_refs_cannot_address_current_text() -> None:
    ledger, _, full = setup_original()
    native = [turn("old", [full])]
    projected = project_native_originals(native, [], ledger)
    payload = json.loads(projected.turns[0].results[0].content)
    assert "original_evidence" not in payload
    assert "text" not in payload["original_evidence_refs"][0]
    assert "metadata_ref" not in payload["original_evidence_refs"][0]
    assert projected.fallback_originals == projected.metadata_catalogue == []
    foreign = {**full, "source_id": "other", "text_hash": "foreign"}
    foreign.pop("text")
    current = turn("ref-only", [])
    current.results[0].content = json.dumps(
        {
            "original_evidence_refs": [foreign],
            "outcome": {"data": {"text": "keep non-original data"}},
        }
    )
    projected = project_native_originals([current], [full], ledger)
    assert json.loads(projected.turns[0].results[0].content)[
        "original_evidence_refs"
    ] == [foreign]
    assert projected.fallback_originals[0]["text"] == full["text"]
    assert (
        json.loads(projected.turns[0].results[0].content)["outcome"]["data"]["text"]
        == "keep non-original data"
    )


def test_reference_and_catalogue_without_full_text_cannot_create_delivery() -> None:
    ledger, context, full = setup_original()
    parallel(context)
    reference = {key: value for key, value in full.items() if key != "text"}
    ref_turn = turn("ref-only", [])
    ref_turn.results[0].content = json.dumps({"original_evidence_refs": [reference]})
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(view(turns=[ref_turn]))
    assert actual_originals(llm.invoke.call_args.kwargs["prompt"]) == []
    assert ledger.completely_delivered(adapter.last_call_id or "") == set()


def test_physical_source_eviction_keeps_pin_and_actual_required_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, context, required = setup_original()
    parallel(context)
    optional = original(ledger, context, "Older supplementary source. " * 400)
    ledger.record_delivery("accepted-child", "asv3_researcher", [required])
    ledger.pin_delivery("accepted-child")
    llm = model(limit=1000000)
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    monkeypatch.setattr(
        adapter, "_research_instruction", lambda: "Use exact originals."
    )
    minimal, tools, _ = adapter._fit_native_decision(
        view(original_evidence=[required], required_evidence_numbers=[1])
    )
    ceiling = adapter._input_cost(minimal, tools) + 5
    monkeypatch.setattr(adapter, "_limits", lambda _: (ceiling, 100))
    adapter.decide(
        view(
            turns=[turn("old-optional", [optional])],
            original_evidence=[required, optional],
            required_evidence_numbers=[1],
        )
    )
    prompt = llm.invoke.call_args.kwargs["prompt"]
    assert [row["citation"] for row in actual_originals(prompt)] == [1]
    assert last_payload(llm)["original_evidence_omitted"][-1]["citation"] == 2
    assert (
        last_payload(llm)["original_evidence_omitted"][-1]["reason"]
        == "physical_model_context"
    )
    assert [
        row["citation"] for row in last_payload(llm)["original_metadata_catalogue"]
    ] == [1]
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert ledger.completely_delivered("accepted-child") == {1}
    assert ledger.export()["pinned_delivery_calls"] == ["accepted-child"]
    item = ledger.get(2)
    assert item is not None and item.text == optional["text"]
    assert adapter._input_cost(prompt, tools) <= ceiling


def test_required_native_original_cannot_be_clipped_for_physical_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, context, _ = setup_original()
    parallel(context)
    required = original(ledger, context, "Required cumulative condition. " * 200)
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    monkeypatch.setattr(adapter, "_limits", lambda _: (1000, 100))
    with pytest.raises(RunStopped, match="required originals"):
        adapter.decide(
            view(turns=[turn("required", [required])], required_evidence_numbers=[2])
        )
    assert llm.invoke.call_count == 0
    assert ledger.export()["deliveries"] == []


def test_wrong_metadata_pointer_cannot_be_rebound_to_a_current_original() -> None:
    ledger, _, record = setup_original()
    invalid = {key: value for key, value in record.items() if key != "text"}
    invalid["metadata_ref"] = {"citation": 99, "text_hash": record["text_hash"]}
    current = turn("wrong-ref", [])
    current.results[0].content = json.dumps({"original_evidence_refs": [invalid]})
    with pytest.raises(ValueError, match="another original"):
        project_native_originals([current], [record], ledger)


def test_native_navigation_does_not_bind_a_later_narrative_article_to_a_statute() -> (
    None
):
    context, ledger = authority_setup(
        "8917 sayılı Faaliyet Kanunu kapsamındaki yetki kullanılarak hazırlanan "
        "düzenlemede 53 üncü madde usulünü açıklar."
    )
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(view(original_evidence=complete(ledger, 1, 2)))
    current = last_payload(llm)
    assert "research_navigation" not in current
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert llm.invoke.call_count == 1


@pytest.mark.parametrize(
    "profile,parallel_flag",
    [
        ("normal", False),
        ("deep", False),
        ("experimental", False),
        ("normal", True),
        ("deep", True),
    ],
)
def test_other_profiles_keep_existing_reference_and_tail_bytes(
    profile: str, parallel_flag: bool
) -> None:
    ledger, context, full = setup_original()
    context.services.update(
        research_profile=profile,
        experimental_parallel=parallel_flag,
        scenario_request="Fixed full scenario.",
    )
    native = [turn("one", [full]), turn("two", [full])]
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(view(turns=native, original_evidence=[full]))
    prompt = llm.invoke.call_args.kwargs["prompt"]
    actual = [message.content for message in prompt if isinstance(message, ToolMessage)]
    expected = [
        turn.results[0].content
        for turn in ResearchModel._native_turns_with_original_references(native)
    ]
    assert actual == expected
    current = last_payload(llm)
    assert current["original_evidence"] == [full]
    assert "original_metadata_catalogue" not in current
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
