"""Optional source-reference cues reflect the final physical original payload."""

import json
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.llm.models import ChatCompletionMessage
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals, payloads
from tests.unit.onyx.asv3.test_native_model_adapter import last_payload, model, view
from tests.unit.onyx.asv3.test_shared_originals import full_record

REFERRAL = "Bu işlem 8917 sayılı Faaliyet Kanunu esaslarına göre ele alınır."


def source_context(profile: str) -> tuple[RunContext, EvidenceLedger]:
    ledger = ledger_with(
        original("7284 sayılı Veri Kanunu", "8", text=REFERRAL),
        original("8917 sayılı Faaliyet Kanunu", "27"),
    )
    context = RunContext(
        services={
            "evidence": ledger,
            "research_profile": profile,
            "lean_native_mode": True,
            "scenario_request": "Which procedure applies to the fixed scenario?",
        }
    )
    if profile == "parallel":
        context.services.update(
            research_profile="experimental", experimental_parallel=True
        )
    elif profile == "hosted":
        context.services.update(
            research_profile="experimental",
            experimental_parallel=False,
            serial_session_diagnostics=True,
            task_id="owned-session",
        )
    return context, ledger


@pytest.mark.parametrize(
    "profile", ["parallel", "hosted", "experimental", "normal", "deep"]
)
def test_only_owned_parallel_decision_receives_reference_ranges(profile: str) -> None:
    context, ledger = source_context(profile)
    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    adapter.decide(view(original_evidence=[full_record(ledger, 1)]))
    current = last_payload(selected)
    assert selected.invoke.call_count == 1
    if profile not in {"parallel", "hosted"}:
        assert "source_contained_references" not in current
        return
    catalogue = current["source_contained_references"]
    assert all(row["citation"] == 1 for row in catalogue["references"])
    assert {row["instrument_number"] for row in catalogue["references"]} == {"8917"}
    for row in catalogue["references"]:
        assert "text" not in row and "source_quote" not in row
        assert REFERRAL[row["start_char"] : row["end_char"]]
    originals = actual_originals(selected.invoke.call_args.kwargs["prompt"])
    assert [row["text"] for row in originals] == [REFERRAL]
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}


@pytest.mark.parametrize("defect", ["absent", "partial", "source", "hash"])
def test_current_fit_cannot_reuse_prior_source_reference_catalogue(defect: str) -> None:
    context, ledger = source_context("parallel")
    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    adapter.decide(view(original_evidence=[full_record(ledger, 1)]))
    assert "source_contained_references" in last_payload(selected)
    record = full_record(ledger, 1)
    if defect == "partial":
        record.update(text=REFERRAL[:10], start_char=0, end_char=10, truncated=True)
    elif defect == "source":
        record["source_id"] = "another-source"
    elif defect == "hash":
        record["text_hash"] = "0" * 64
    if defect in {"source", "hash"}:
        with pytest.raises(ValueError, match="canonical identity"):
            adapter.decide(view(original_evidence=[record]))
        assert selected.invoke.call_count == 1
        return
    adapter.decide(view(original_evidence=[] if defect == "absent" else [record]))
    assert "source_contained_references" not in last_payload(selected)
    assert selected.invoke.call_count == 2


def test_optional_reference_catalogue_yields_before_its_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, ledger = source_context("parallel")
    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    seen: list[dict[str, Any]] = []

    def cost(
        prompt: list[ChatCompletionMessage], _tools: list[dict[str, JsonValue]]
    ) -> int:
        current = payloads(prompt)[-1]
        seen.append(current)
        return 10**9 if "source_contained_references" in current else 0

    monkeypatch.setattr(adapter, "_input_cost", cost)
    adapter.decide(view(original_evidence=[full_record(ledger, 1)]))
    assert any("source_contained_references" in row for row in seen)
    assert "source_contained_references" not in last_payload(selected)
    prompt = selected.invoke.call_args.kwargs["prompt"]
    assert [row["text"] for row in actual_originals(prompt)] == [REFERRAL]
    assert "original_evidence_omitted" not in last_payload(selected)
    assert selected.invoke.call_count == 1
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert REFERRAL in json.dumps(payloads(prompt), ensure_ascii=False)
