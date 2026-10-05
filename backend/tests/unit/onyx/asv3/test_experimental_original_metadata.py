"""Verified ledger metadata survives native original-range deduplication."""

import json
from typing import cast

import pytest

from onyx.asv3.llm_adapter import ResearchModel
from tests.unit.onyx.asv3.test_experimental_workflow import experimental_context
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    model,
    turn,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record


@pytest.mark.parametrize("partial", [False, True])
def test_verified_native_original_retains_ledger_metadata_and_exact_range(
    partial: bool,
) -> None:
    context, ledger, _ = experimental_context()
    complete = full_record(ledger, 1)
    original = dict(complete)
    original.pop("metadata")
    if partial:
        text = cast(str, original["text"])
        original.update(
            text=text[5:15],
            start_char=5,
            end_char=15,
            total_chars=len(text),
            truncated=True,
        )
    current = adaptive_tool_view(
        original_evidence=[original],
        turns=[turn("reopened-native-original", [original])],
    )
    before = current.model_dump(mode="json")
    llm = model()
    prompt, _, _ = ResearchModel(
        llm, context, lean_native_mode=True
    )._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    row = payload["original_evidence"][0]
    assert row == {**original, "metadata": complete["metadata"]}
    assert payload["original_evidence_ranges"] == [
        {
            "citation": 1,
            "start_char": 5 if partial else 0,
            "end_char": 15 if partial else len(cast(str, original["text"])),
        }
    ]
    assert current.model_dump(mode="json") == before
    llm.invoke.assert_not_called()


def test_complete_native_copy_cannot_shadow_canonical_metadata() -> None:
    context, ledger, _ = experimental_context()
    complete = full_record(ledger, 1)
    native = {**complete, "metadata": {"title": "Invented Law", "document_type": "law"}}
    current = adaptive_tool_view(
        original_evidence=[complete], turns=[turn("native-copy-first", [native])]
    )
    prompt, _, _ = ResearchModel(
        model(), context, lean_native_mode=True
    )._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    assert payload["original_evidence"] == [complete]


@pytest.mark.parametrize("field", ["source_id", "chunk_id", "text_hash", "text"])
def test_mismatched_original_never_receives_canonical_metadata(field: str) -> None:
    context, ledger, _ = experimental_context()
    original = full_record(ledger, 1)
    original.pop("metadata")
    original[field] = "mismatched"
    current = adaptive_tool_view(original_evidence=[original])
    prompt, _, _ = ResearchModel(
        model(), context, lean_native_mode=True
    )._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    assert payload["original_evidence"] == [original]


@pytest.mark.parametrize("profile", ["normal", "deep"])
def test_existing_workflows_keep_their_native_record_projection(profile: str) -> None:
    context, ledger, _ = experimental_context()
    context.services["research_profile"] = profile
    context.services.pop("legal_source_reviews")
    context.services.pop("legal_source_navigation")
    complete = full_record(ledger, 1)
    native = {key: value for key, value in complete.items() if key != "metadata"}
    current = adaptive_tool_view(
        original_evidence=[complete], turns=[turn("baseline-native-copy", [native])]
    )
    prompt, _, _ = ResearchModel(
        model(), context, lean_native_mode=True
    )._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    assert payload["original_evidence"] == [native]
