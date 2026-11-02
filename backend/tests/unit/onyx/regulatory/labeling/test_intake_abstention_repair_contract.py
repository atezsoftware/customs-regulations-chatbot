"""Source-bound single-attempt semantic repair contracts."""

import json
from copy import deepcopy
from typing import Any

import pytest

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.labeling import intake_abstention_repair as module
from onyx.regulatory.labeling.intake_abstention_repair import (
    validate_abstention_repair,
)
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import (
    REJECTED,
    TEXT,
    audit,
    build,
    response,
    taxonomy,
    validate_audit,
)


@pytest.mark.parametrize("text", ["|---|---|---|", "| | | |\n|---|---|---|\n| | | |"])
def test_model_selected_no_match_is_distinct_from_continued_abstention(
    text: str,
) -> None:
    repair = build(text=text)
    no_match = response(empty=True)
    receipt = audit(repair, no_match, text=text)
    assert not receipt.outcome.abstained and receipt.outcome.labels == []
    assert (
        validate_audit(
            receipt,
            repair,
            text=text,
            source_hash=context_hash(text),
            labels=[],
            assignments=[],
        )
        == receipt
    )
    continued = audit(repair, REJECTED, text=text)
    assert continued.outcome.abstained
    with pytest.raises(ValueError, match="accepted assignments"):
        validate_audit(
            continued,
            repair,
            text=text,
            source_hash=context_hash(text),
            labels=[],
            assignments=[],
        )


@pytest.mark.parametrize(
    "rejected",
    [
        '{"labels":[],"abstained":false}',
        response(),
        response(abstained=True),
        '{"labels":[],"abstained":"true"}',
        '{"labels":[],"abstained":true,"extra":1}',
        "invalid JSON",
    ],
)
def test_only_a_valid_empty_label_abstention_is_eligible(rejected: str) -> None:
    with pytest.raises(ValueError):
        build(rejected_response=rejected)


@pytest.mark.parametrize(
    "change",
    ["target", "source", "taxonomy", "context", "system", "schema", "recursion"],
)
def test_frozen_drift_or_another_repair_cannot_start_reconsideration(
    change: str,
) -> None:
    repair = build()
    values: dict[str, Any] = {}
    if change == "target":
        values["chunk_id"] = "other-target"
    elif change == "source":
        values["text"] = TEXT + "!"
        values["source_hash"] = context_hash(TEXT)
        values["base_request"] = repair.base_request
    elif change == "taxonomy":
        values["taxonomy"] = taxonomy().model_copy(update={"name": "changed"})
    elif change == "context":
        prompt = json.loads(repair.base_request.prompt)
        prompt["interpretation_context"] += " changed"
        values["base_request"] = repair.base_request.model_copy(
            update={"prompt": json.dumps(prompt)}
        )
        # A modified context is a new base request, never this already bound request.
        stored = audit(repair, response())
        with pytest.raises(ValueError):
            validate_audit(stored, repair, **values)
        return
    elif change == "system":
        values["base_request"] = repair.base_request.model_copy(
            update={"system_instruction": "changed"}
        )
    elif change == "schema":
        values["base_request"] = repair.base_request.model_copy(
            update={"generation_config": {}}
        )
    else:
        values["base_request"] = repair.request
    with pytest.raises(ValueError):
        build(**values)


@pytest.mark.parametrize(
    "binding",
    [
        {},
        {"model": "model", "transport": "vertex_gcs_v1"},
        {"transport": "vertex_direct_v1", "model": ""},
    ],
)
def test_non_direct_or_unbound_model_is_ineligible(binding: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build(provider_binding=binding)


@pytest.mark.parametrize(
    "kind",
    ["quote", "unknown", "duplicate", "abstained-label", "extra", "context-only"],
)
def test_normal_taxonomy_schema_and_exact_target_quote_checks_remain_authoritative(
    kind: str,
) -> None:
    repair = build()
    payload = json.loads(response())
    if kind == "quote":
        payload["labels"][0]["evidence_quote"] = TEXT.replace("ortadan", "ORTADAN")
    elif kind == "unknown":
        payload["labels"][0]["label_id"] = "unknown"
    elif kind == "duplicate":
        payload["labels"].append(deepcopy(payload["labels"][0]))
    elif kind == "abstained-label":
        payload["abstained"] = True
    elif kind == "extra":
        payload["extra"] = True
    else:
        payload["labels"][0]["evidence_quote"] = (
            "işlemleri menşe kazanımı açısından dikkate alınmaz."
        )
    with pytest.raises(ValueError):
        validate_abstention_repair(
            json.dumps(payload), repair=repair, text=TEXT, taxonomy=taxonomy()
        )


def test_budgets_fail_explicitly_without_truncating_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 1)
    with pytest.raises(ValueError, match="not truncated"):
        build()
