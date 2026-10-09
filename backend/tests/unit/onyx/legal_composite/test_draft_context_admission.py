"""Draft sharing changes admission size without changing delivered evidence."""

import json
from copy import deepcopy
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.draft_context import decode_draft_context
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import StructuredDraftAnswer, WorkflowPolicy
from onyx.llm.models import ChatCompletionMessage


def payload() -> dict[str, JsonValue]:
    prose = "İlgili şart ve istisna birlikte uygulanır. " * 100 + "[1]"
    draft = StructuredDraftAnswer.model_validate(
        {
            "sections": [
                {
                    "section_id": "section",
                    "need_ids": ["issue"],
                    "text": "Başlık",
                    "claim_ids": ["claim"],
                }
            ],
            "claims": [
                {
                    "claim_id": "claim",
                    "section_id": "section",
                    "need_ids": ["issue"],
                    "answer_excerpt": prose,
                    "supports": [{"citation": 1, "quotation": "Tam özgün hüküm."}],
                    "requirement_ids": [],
                }
            ],
            "unresolved_need_ids": [],
        }
    )
    return {
        "draft": draft.model_dump(mode="json"),
        "original_evidence": [
            {
                "citation": 1,
                "source_id": "source",
                "chunk_id": "chunk",
                "text": "Tam özgün hüküm.",
                "text_hash": "exact-binding",
                "metadata": {"literal": "_lc_draft_context"},
            }
        ],
        "required_evidence_numbers": [1],
        "omitted_original_ids": [],
    }


def gateway(share: bool, cap: int = 128_000) -> BudgetedGateway:
    value = object.__new__(BudgetedGateway)
    value.share_draft_context = share
    value.token_counter = None
    value.budget = WorkflowBudget(WorkflowPolicy(max_context_tokens=cap))
    return value


def fit(
    value: BudgetedGateway, body: dict[str, JsonValue]
) -> tuple[list[ChatCompletionMessage], int, list[dict[str, JsonValue]]]:
    return value._fit_messages(
        "Read exact evidence.",
        body,
        "{}",
        Mock(config=Mock(max_input_tokens=128_000)),
        {},
        finalizing=True,
    )


def test_opt_in_saves_tokens_and_keeps_every_original() -> None:
    body = payload()
    before = deepcopy(body)
    _, baseline_tokens, baseline_records = fit(gateway(False), body)
    messages, tokens, records = fit(gateway(True), body)
    wire = json.loads(cast(str, messages[1].content))
    assert tokens < baseline_tokens
    assert decode_draft_context(wire) == body
    assert records == baseline_records == body["original_evidence"]
    assert body == before
    assert "lossless_draft_v1" in cast(str, messages[0].content)


def test_default_view_keeps_literal_host_draft() -> None:
    body = payload()
    messages, _, _ = fit(gateway(False), body)
    assert json.loads(cast(str, messages[1].content)) == body
    assert "lossless_draft_v1" not in cast(str, messages[0].content)


def test_provider_counter_can_reject_compact_view() -> None:
    body = payload()
    value = gateway(True)
    value.token_counter = lambda text: 1_000_000 if "lossless_draft_v1" in text else 1
    messages, _, _ = fit(value, body)
    assert json.loads(cast(str, messages[1].content)) == body


def test_exact_compact_view_fits_without_removing_required_evidence() -> None:
    body = payload()
    _, compact_tokens, _ = fit(gateway(True), body)
    _, baseline_tokens, _ = fit(gateway(False), body)
    assert compact_tokens < baseline_tokens
    with pytest.raises(RunStopped, match="model context budget"):
        fit(gateway(False, compact_tokens), body)
    _, admitted_tokens, records = fit(gateway(True, compact_tokens), body)
    assert admitted_tokens == compact_tokens
    assert records == body["original_evidence"]


def test_codec_never_masks_a_missing_required_original() -> None:
    body = payload()
    body["required_evidence_numbers"] = [1, 2]
    with pytest.raises(RunStopped, match="required original is absent"):
        fit(gateway(True), body)
