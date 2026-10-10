"""Accepted gaps retain exact routing and complete discovery candidate inventories."""

import copy
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from onyx.legal_review.evidence_resolution import EvidenceResolutionPlan
from onyx.legal_review.models import ReviewDiagnosisBatch


def plan() -> EvidenceResolutionPlan:
    common = {
        "assertion": "The route is available.",
        "reason": "A limiting effect is unread.",
        "required_change": "Establish its operative effect.",
        "supports": [],
        "dimension": "exceptions_and_exemptions",
    }
    diagnoses = ReviewDiagnosisBatch.model_validate(
        {
            "diagnoses": [
                {
                    **common,
                    "check_ids": ["a", "b"],
                    "kind": "research",
                    "research_task_ids": ["t1"],
                },
                {
                    **common,
                    "check_ids": ["c"],
                    "kind": "correction",
                    "research_task_ids": [],
                },
                {
                    **common,
                    "check_ids": ["d"],
                    "kind": "research",
                    "research_task_ids": ["t1"],
                },
            ],
            "research_tasks": [
                {
                    "task_id": "t1",
                    "subject": "Permit conditions",
                    "question": "What restrictions govern the permit?",
                    "query": None,
                    "existing_need_id": "n1",
                    "dimension": "exceptions_and_exemptions",
                    "supports": [],
                }
            ],
        }
    )
    return EvidenceResolutionPlan(
        diagnoses, {"a": {"i1"}, "b": {"i2"}, "c": {"i2"}, "d": {"i3"}}
    )


def response() -> dict[str, Any]:
    disposition = {
        "disposition": "unresolved",
        "reason": "Operative restriction is unread.",
        "supports": [],
        "missing_user_facts": [],
    }
    return {
        "dimensions": [],
        "research_resolutions": [
            {"slot": slot, **disposition} for slot in ["r0001", "r0002"]
        ],
    }


def test_provider_schema_binds_only_research_questions_and_host_restores_routing() -> (
    None
):
    mapping = plan()
    schema = mapping.response_model()
    payload = response()
    Draft202012Validator(schema.model_json_schema()).validate(payload)
    compiled = mapping.compile(schema.model_validate(payload))
    assert [row.check_ids for row in compiled.research_resolutions] == [
        ["a", "b"],
        ["d"],
    ]
    assert [row.issue_ids for row in compiled.research_resolutions] == [
        ["i1", "i2"],
        ["i3"],
    ]
    assert compiled.dimensions == []


@pytest.mark.parametrize("mutation", ["missing", "extra", "routing"])
def test_provider_schema_rejects_missing_questions_and_model_owned_routing(
    mutation: str,
) -> None:
    schema = plan().response_model()
    payload = response()
    if mutation == "missing":
        del payload["research_resolutions"][0]
    elif mutation == "extra":
        payload["research_resolutions"][0]["slot"] = "correction-only"
    else:
        payload["research_resolutions"][0]["check_ids"] = ["c"]
    assert not Draft202012Validator(schema.model_json_schema()).is_valid(payload)
    with pytest.raises(ValidationError):
        schema.model_validate(payload)


def test_shared_search_index_retains_each_original_without_answer_specific_selection() -> (
    None
):
    mapping = plan()
    state: dict[str, Any] = {
        "repair_contract": {"required_check_ids": ["a", "b", "c", "d"]},
        "source_operations": [
            {"research_need_ids": ["n1"], "evidence_ids": [11, 12]},
            {"research_need_ids": ["n2"], "evidence_ids": [13]},
            {"research_need_ids": ["n1"], "evidence_ids": [12, 14]},
        ],
        "original_evidence": [
            {
                "citation": n,
                "source_id": str(n),
                "passages": [{"text": f"Original {n}"}],
                "metadata": {"title": f"Source {n}"},
            }
            for n in [11, 12, 13, 14]
        ],
    }
    before = copy.deepcopy(state)
    prepared = mapping.state(state)
    assert state == before
    assert prepared["original_evidence"] == before["original_evidence"]
    assert "repair_contract" not in prepared
    contract = prepared["research_resolution_contract"]
    assert isinstance(contract, dict)
    investigations = contract["investigations"]
    assert isinstance(investigations, dict)
    task = investigations["t1"]
    assert isinstance(task, dict)
    assert task["returned_citations"] == [11, 12, 14]
    sources = contract["source_index"]
    assert isinstance(sources, dict)
    assert set(sources) == {"11", "12", "13", "14"}
    assert sources["12"] == {"title": "Source 12", "citations": [12]}


def test_duplicate_slots_cannot_discard_an_unanswered_question() -> None:
    mapping = plan()
    payload = response()
    payload["research_resolutions"][1]["slot"] = "r0001"
    with pytest.raises(ValueError, match="preserve every accepted question"):
        mapping.compile(mapping.response_model().model_validate(payload))
