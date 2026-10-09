"""Lossless reviewer context sharing keeps complete originals and readable witnesses."""

import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite import reviewer as reviewer_module
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.gateway import (
    BudgetedGateway,
    MissingRequiredOriginal,
    ModelContextLimit,
    ResearchContextBudgetLimit,
)
from onyx.legal_composite.models import WorkflowPolicy
from onyx.legal_composite.reviewer import (
    GatewayAnswerReviewer,
    ReviewQuestion,
    _decode_review_context,
    _encode_review_context,
    _review_generation_policy,
    _review_protocol_tokens,
    _select_review_context,
)
from onyx.llm.interfaces import LLMConfig

QUOTE = "İşlem ancak başvuru kabul edilirse yapılır; aksi hâlde yapılmaz."


def context() -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "canonical": {"revision": "revision-1", "hash": "sha-1"},
        "nested": [None, False, 0, 0.0, "İstanbul", {"$lc_metadata": 0}],
        "scope": "Exact canonical publication, version and authority metadata. " * 50,
    }
    partial_metadata: dict[str, JsonValue] = {
        "source_id": "source-2",
        "validity": ["2026-01-01", None],
        "$lc_original_metadata": 1,
    }
    text = QUOTE + "\nAra hüküm değişmez.\n" + QUOTE
    check: dict[str, JsonValue] = {
        "check_id": "original:1",
        "question": "Are all decisive qualifications preserved?",
        "need_ids": ["effect"],
        "section_ids": ["answer"],
        "citations": [1],
        "allow_not_applicable": False,
    }
    support: dict[str, JsonValue] = {
        "citation": 1,
        "span_id": "canonical-span-2",
        "quotation": QUOTE,
    }
    partial: dict[str, JsonValue] = {
        "citation": 2,
        "span_id": None,
        "quotation": "Eksik belgenin yalnız doğrulanmış alıntısı.",
    }
    witnesses: list[JsonValue] = [
        {
            "citation": 1,
            "source_id": "source-1",
            "chunk_id": "chunk-1",
            "text_hash": "sha-1",
            "metadata": metadata,
            "quotation": QUOTE,
            "start_char": start,
            "end_char": start + len(QUOTE),
        }
        for start in (0, text.rindex(QUOTE))
    ]
    witnesses.append(
        {
            "citation": 2,
            "source_id": "source-2",
            "chunk_id": "chunk-2",
            "text_hash": "sha-2",
            "metadata": partial_metadata,
            "quotation": partial["quotation"],
            "start_char": 17,
            "end_char": 60,
        }
    )
    return {
        "review_context": {
            "request": "Koşullu işlem yapılabilir mi?",
            "needs": [{"need_id": "effect", "evidence_gaps": []}],
            "sections": [{"section_id": "answer", "need_ids": ["effect"]}],
            "claims": [{"claim_id": "claim-1", "supports": [support, partial]}],
            "requirements": [{"requirement_id": "rule-1", "supports": [support]}],
            "dependencies": [{"edge_id": "edge-1", "need_ids": ["effect"]}],
            "originals": [
                {
                    "citation": 1,
                    "source_id": "source-1",
                    "chunk_id": "chunk-1",
                    "text_hash": "sha-1",
                    "metadata": metadata,
                    "context_field": "original_evidence",
                }
            ],
            "canonical_witnesses": witnesses,
            "original_context_scope": {
                "full_original_citations": [1],
                "quotation_only_citations": [2],
            },
            "checks": {"original:1": check},
        },
        "expected_checks": [{"index": 0, **check}],
        "status_criteria": {"addressed": "Preserves all applicable qualifications."},
        "original_evidence": [
            {
                "citation": 1,
                "source_id": "source-1",
                "chunk_id": "chunk-1",
                "text_hash": "sha-1",
                "metadata": metadata,
                "text": text,
            }
        ],
        "required_evidence_numbers": [1],
    }


def test_roundtrip_retains_whole_originals_partial_witnesses_and_all_ids() -> None:
    baseline = context()
    before = json.dumps(baseline, ensure_ascii=False)
    wire = _encode_review_context(baseline)
    assert _decode_review_context(wire) == baseline
    assert json.dumps(_decode_review_context(wire), ensure_ascii=False) == before
    assert json.dumps(baseline, ensure_ascii=False) == before
    assert wire["original_evidence"] == baseline["original_evidence"]
    assert wire["expected_checks"] == baseline["expected_checks"]
    assert wire["required_evidence_numbers"] == [1]
    assert "sliced" not in _review_generation_policy(wire)
    state = cast(dict[str, JsonValue], wire["review_context"])
    assert state["checks"] == {"$lc_expected_checks": True}
    pools = cast(dict[str, JsonValue], wire["review_context_pools"])
    assert pools["quotations"] == [QUOTE, "Eksik belgenin yalnız doğrulanmış alıntısı."]


def test_repeated_unicode_quote_keeps_both_canonical_offsets_readable() -> None:
    baseline = context()
    wire = _encode_review_context(baseline)
    state = cast(dict[str, JsonValue], wire["review_context"])
    witnesses = cast(list[dict[str, JsonValue]], state["canonical_witnesses"])
    assert witnesses[0]["quotation"] == witnesses[1]["quotation"]
    assert witnesses[0]["start_char"] != witnesses[1]["start_char"]
    assert len(witnesses) == 3
    original = cast(list[dict[str, JsonValue]], wire["original_evidence"])[0]
    assert cast(str, original["text"]).count(QUOTE) == 2
    assert _decode_review_context(wire) == baseline


def test_partial_original_metadata_marker_is_literal_data_and_not_a_reference() -> None:
    wire = _encode_review_context(context())
    decoded = _decode_review_context(wire)
    state = cast(dict[str, JsonValue], decoded["review_context"])
    witnesses = cast(list[dict[str, JsonValue]], state["canonical_witnesses"])
    metadata = cast(dict[str, JsonValue], witnesses[-1]["metadata"])
    assert metadata["$lc_original_metadata"] == 1
    assert state["original_context_scope"] == {
        "full_original_citations": [1],
        "quotation_only_citations": [2],
    }


@pytest.mark.parametrize(
    "reference",
    [
        {"$lc_quotation": True},
        {"$lc_quotation": -1},
        {"$lc_quotation": 100},
        {"$lc_original_metadata": 999},
        {"unknown": 0},
    ],
)
def test_invalid_shared_references_fail_closed(reference: dict[str, JsonValue]) -> None:
    wire = _encode_review_context(context())
    state = cast(dict[str, JsonValue], wire["review_context"])
    witnesses = cast(list[dict[str, JsonValue]], state["canonical_witnesses"])
    witnesses[0]["quotation"] = reference
    with pytest.raises(ValueError):
        _decode_review_context(wire)


def test_configured_token_measurement_selects_only_a_smaller_context() -> None:
    baseline = context()
    selected, before, after = _select_review_context(baseline, "{}", {}, None)
    assert "review_context_codec" in selected
    assert after < before * 0.65
    assert _decode_review_context(selected) == baseline
    assert after == _review_protocol_tokens(selected, "{}", {}, None)


def test_provider_specific_counter_can_reject_the_compact_representation() -> None:
    baseline = context()

    def counter(protocol: str) -> int:
        return 1_000_000 if "lc_review_context_v1" in protocol else 1

    selected, before, after = _select_review_context(baseline, "{}", {}, counter)
    assert selected is baseline
    assert after == before
    assert "review_context_codec" not in selected


def test_unrelated_check_dictionary_is_never_removed() -> None:
    baseline = context()
    state = cast(dict[str, JsonValue], baseline["review_context"])
    state["checks"] = {"additional": {"question": "Distinct question retained."}}
    wire = _encode_review_context(baseline)
    assert (
        cast(dict[str, JsonValue], wire["review_context"])["checks"] == state["checks"]
    )
    assert _decode_review_context(wire) == baseline


def test_codec_rescues_context_cap_without_removing_required_originals() -> None:
    baseline = context()
    selected, before, after = _select_review_context(baseline, "{}", {}, None)
    gateway = object.__new__(BudgetedGateway)
    gateway.share_draft_context = False
    gateway.token_counter = None
    gateway.budget = WorkflowBudget(WorkflowPolicy(max_context_tokens=after))
    llm = Mock(config=Mock(max_input_tokens=before))
    with pytest.raises(RunStopped, match="model context budget"):
        gateway._fit_messages(
            _review_generation_policy(baseline),
            baseline,
            "{}",
            llm,
            {},
            finalizing=True,
        )
    _, measured, records = gateway._fit_messages(
        _review_generation_policy(selected), selected, "{}", llm, {}, finalizing=True
    )
    assert measured == after
    assert records == baseline["original_evidence"]
    assert _decode_review_context(selected) == baseline


def test_wire_mutation_cannot_mutate_source_metadata_or_quotes() -> None:
    baseline = context()
    saved = copy.deepcopy(baseline)
    wire = _encode_review_context(baseline)
    pools = cast(dict[str, JsonValue], wire["review_context_pools"])
    cast(list[JsonValue], pools["quotations"])[0] = "changed"
    assert baseline == saved


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (ValueError("private-source-text"), "invalid_or_unavailable_context"),
        (TypeError("private-source-text"), "invalid_or_unavailable_context"),
        (RunStopped("private-source-text"), "context_or_budget_limit"),
        (MissingRequiredOriginal("private-source-text"), "context_or_budget_limit"),
        (ModelContextLimit("private-source-text"), "context_or_budget_limit"),
        (ResearchContextBudgetLimit("private-source-text"), "context_or_budget_limit"),
    ],
)
def test_singleton_fit_diagnostic_is_safe_and_never_certifies_the_check(
    monkeypatch: pytest.MonkeyPatch, error: Exception, category: str
) -> None:
    config = LLMConfig(
        model_provider="openrouter",
        model_name="openai/gpt-5.6-luna",
        temperature=1,
        max_input_tokens=128_000,
    )
    gateway = Mock(spec=BudgetedGateway)
    gateway.selected_llm = Mock(config=config)
    gateway.token_counter = None
    gateway._fit_messages.side_effect = error
    judge = GatewayAnswerReviewer(
        config=config,
        gateway_factory=lambda: gateway,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=EvidenceLedger(),
    )
    check = ReviewQuestion(
        check_id="original:1",
        question="Verify the original.",
        need_ids=["effect"],
        section_ids=["answer"],
        citations=[1],
    )
    monkeypatch.setattr(
        judge, "expected_checks", lambda *_args: {check.check_id: check}
    )
    monkeypatch.setattr(judge, "_generation_payload", lambda *_args: context())
    observations: list[tuple[str, SimpleNamespace]] = []

    @contextmanager
    def capture(operation: str, *_args: Any, summary: str | None = None):
        step = SimpleNamespace(summary=summary, output_value=None)
        yield step
        observations.append((operation, step))

    monkeypatch.setattr(reviewer_module, "graph_step", capture)
    result = judge.review("Question", Mock(), Mock(), [], [], {1})
    assert result.failure is not None
    assert len(result.checks) == 1
    assert result.checks[0].status == "uncertain"
    assert result.checks[0].confidence == 0
    gateway.complete.assert_not_called()
    fit = next(
        step
        for operation, step in observations
        if operation.endswith("review_context_fit")
    )
    assert fit.output_value["expected_checks"] == 1
    assert fit.output_value["provider_checks"] == 0
    assert fit.output_value["failure_categories"][category] == 1
    assert "saved_tokens=0" in fit.summary and len(fit.summary) <= 160
    assert "private-source-text" not in json.dumps(fit.output_value)
    assert "private-source-text" not in fit.summary
    details = next(
        step
        for operation, step in observations
        if operation.endswith("review_context_failures")
    )
    expected = (
        "required_missing"
        if isinstance(error, MissingRequiredOriginal)
        else "model_capacity"
        if isinstance(error, ModelContextLimit)
        else "priced_allocation"
        if isinstance(error, ResearchContextBudgetLimit)
        else "other_stop"
        if isinstance(error, RunStopped)
        else "invalid_context"
    )
    assert details.output_value[expected] == 1
    assert sum(details.output_value.values()) == 1
    assert "private-source-text" not in json.dumps(details.output_value)
    assert "private-source-text" not in details.summary
