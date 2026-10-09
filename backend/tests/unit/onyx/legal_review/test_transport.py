"""Canonical texts and all legal metadata survive model-only normalization."""

import copy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.legal_review.transport import model_state
from onyx.tracing.flows import LLMFlow


def originals() -> list[JsonValue]:
    common: dict[str, JsonValue] = {
        "title": "Türkçe kaynak / tam metin.md",
        "legal_dates": ["2026-06-02", "2026-10-09"],
        "document_date": None,
        "read_as_of_date": "2026-10-09",
        "version_unknown": True,
        "article_closure_complete": False,
        "status": {"amended": True, "annulled": None},
    }
    return [
        {
            "citation": index,
            "source_id": "document-a",
            "chunk_id": f"chunk-{index}",
            "text_hash": f"host-digest-{index}",
            "truncated": False,
            "citable": True,
            "metadata": {
                **common,
                "article_no": str(index),
                "paragraph_no": None if index == 1 else "3",
                "clause_label": "ancak" if index == 1 else None,
                "heading_path": ["KAYNAK", "İstisnalar", f"Madde {index}"],
            },
            "passages": [
                {"span_number": 1, "text": f"İstisna {index}: ancak, şartıyla.\n"},
                {"span_number": 2, "text": "text_hash içerikteyse silinmez.\n\n"},
            ],
        }
        for index in (1, 2)
    ]


def restore_evidence(view: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    registry = cast(dict[str, dict[str, JsonValue]], view["source_registry"])
    restored: list[dict[str, JsonValue]] = []
    for item in cast(list[dict[str, JsonValue]], view["original_evidence"]):
        original = copy.deepcopy(item)
        source = registry[cast(str, original.pop("source_ref"))]
        metadata = {
            **cast(dict[str, JsonValue], source["metadata"]),
            **cast(dict[str, JsonValue], original["metadata"]),
        }
        if "heading_prefix" in source:
            metadata["heading_path"] = [
                *cast(list[JsonValue], source["heading_prefix"]),
                *cast(list[JsonValue], metadata.pop("heading_suffix")),
            ]
        original["source_id"] = source["source_id"]
        original["metadata"] = metadata
        restored.append(original)
    return restored


def test_complete_multispan_unicode_and_metadata_roundtrip_without_host_mutation() -> (
    None
):
    state: dict[str, JsonValue] = {"original_evidence": originals()}
    before = copy.deepcopy(state)
    view = model_state(state, LLMFlow.LEGAL_REVIEW_READING)
    assert state == before
    expected = cast(
        list[dict[str, JsonValue]], copy.deepcopy(state["original_evidence"])
    )
    for original in expected:
        original.pop("text_hash")
    assert restore_evidence(view) == expected
    assert len(cast(dict, view["source_registry"])) == 1
    assert "text_hash" not in cast(list[dict], view["original_evidence"])[0]
    assert (
        cast(list[dict], state["original_evidence"])[0]["text_hash"] == "host-digest-1"
    )


def test_distinct_documents_and_conflicting_status_values_remain_distinct() -> None:
    evidence = cast(list[dict[str, JsonValue]], originals())
    first_metadata = cast(dict[str, JsonValue], evidence[0]["metadata"])
    second_metadata = cast(dict[str, JsonValue], evidence[1]["metadata"])
    first_metadata["typed_flag"] = True
    second_metadata["typed_flag"] = 1
    third = copy.deepcopy(evidence[0])
    third.update(citation=3, source_id="document-b", chunk_id="chunk-3")
    evidence.append(third)
    view = model_state(
        {"original_evidence": cast(list[JsonValue], evidence)},
        LLMFlow.LEGAL_REVIEW_READING,
    )
    restored = restore_evidence(view)
    assert len(cast(dict, view["source_registry"])) == 2
    assert cast(dict, restored[0]["metadata"])["typed_flag"] is True
    assert type(cast(dict, restored[1]["metadata"])["typed_flag"]) is int
    for actual, expected in zip(restored, evidence):
        assert actual == {
            key: value for key, value in expected.items() if key != "text_hash"
        }


def test_heading_metadata_name_collision_is_preserved() -> None:
    evidence = cast(list[dict[str, JsonValue]], originals())
    cast(dict[str, JsonValue], evidence[0]["metadata"])["heading_suffix"] = "legal-data"
    view = model_state(
        {"original_evidence": cast(list[JsonValue], evidence)},
        LLMFlow.LEGAL_REVIEW_READING,
    )
    assert (
        cast(dict, restore_evidence(view)[0]["metadata"])["heading_suffix"]
        == "legal-data"
    )


@pytest.mark.parametrize(
    "flow", [LLMFlow.LEGAL_REVIEW_DRAFT, LLMFlow.LEGAL_REVIEW_REPAIR]
)
def test_writer_has_diagnostics_and_flags_without_tool_definitions_or_raw_receipts(
    flow: LLMFlow,
) -> None:
    state: dict[str, JsonValue] = {
        "tools": [{"name": "navigation"}],
        "source_operations": [
            {
                "action": {"query": "İptal etkisi", "issue_ids": ["i1"]},
                "status": "completed",
                "data": {
                    "unmapped_result_count": 3,
                    "has_more": True,
                    "next_cursor": "c1",
                    "full_tool_response": "navigation-only",
                },
            }
        ],
        "early_review": {
            "flags": [{"id": "finding:r1", "instructions": "Scope defect?"}]
        },
        "final_review": {
            "flags": [{"id": "claim:c1", "instructions": "Condition omitted?"}]
        },
    }
    view = model_state(state, flow)
    assert "tools" not in view and "source_operations" not in view
    assert "early_review" not in view
    assert view["final_review"] == state["final_review"]
    assert cast(list[dict], view["research_record"])[0]["result_limitations"] == {
        "unmapped_result_count": 3,
        "has_more": True,
        "next_cursor": "c1",
    }
    assert "tools" in state and "source_operations" in state
    assert "early_review" in state


def test_reader_retains_navigation_and_empty_evidence_is_explicit() -> None:
    state: dict[str, JsonValue] = {
        "tools": [{"name": "navigation"}],
        "original_evidence": [],
    }
    view = model_state(state, LLMFlow.LEGAL_REVIEW_READING)
    assert view["tools"] == state["tools"]
    assert view["original_evidence"] == [] and view["source_registry"] == {}


def test_initial_followup_keeps_early_flags_until_a_draft_review_exists() -> None:
    state: dict[str, JsonValue] = {
        "early_review": {"flags": [{"id": "finding:r1", "instructions": "Scope?"}]},
        "final_review": None,
    }
    assert (
        model_state(state, LLMFlow.LEGAL_REVIEW_READING)["early_review"]
        == state["early_review"]
    )


@pytest.mark.parametrize(
    "evidence", [None, {}, [None], [{}], [{"source_id": "d", "metadata": None}]]
)
def test_invalid_evidence_cannot_be_silently_dropped(evidence: JsonValue) -> None:
    with pytest.raises(ValueError):
        model_state({"original_evidence": evidence}, LLMFlow.LEGAL_REVIEW_READING)
