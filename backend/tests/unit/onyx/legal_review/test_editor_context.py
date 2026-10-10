import copy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.legal_review.editor_context import editorial_context, group_editor_findings


def original(citation: int, source: str, article: str) -> dict[str, JsonValue]:
    return {
        "citation": citation,
        "source_id": source,
        "chunk_id": f"chunk-{citation}",
        "metadata": {"article_no": article, "status": "current"},
        "passages": [{"span_number": 1, "text": f"Complete original {citation}"}],
    }


def test_editor_keeps_material_alternatives_contrary_sources_and_whole_articles() -> (
    None
):
    state: dict[str, JsonValue] = {
        "request": "Explain the alternatives.",
        "repair_base": {"blocks": [{"claims": [{"supports": [{"citation": 1}]}]}]},
        "requirements": [{"supports": [{"citation": 3}]}],
        "material_source_leads": [{"source_id": "contrary"}],
        "original_evidence": [
            original(1, "law", "8"),
            original(2, "law", "8"),
            original(3, "alternative", "4"),
            original(4, "alternative", "4"),
            original(5, "contrary", "2"),
            original(6, "irrelevant", "8"),
        ],
        "tools": [{"name": "do not replay"}],
        "source_operations": [{"data": "large repeated body"}],
        "dimension_assessments": [{"reason": "private judgment"}],
    }
    before = copy.deepcopy(state)
    context = editorial_context(state)
    assert (
        context["original_evidence"]
        == cast(list[JsonValue], before["original_evidence"])[:5]
    )
    assert state == before
    assert not {"tools", "source_operations", "dimension_assessments"} & context.keys()
    assert context["repair_base"] == before["repair_base"]


def finding(identity: str, *, correction: str = "Preserve the condition.") -> JsonValue:
    return {
        "check_id": identity,
        "target": "assertion",
        "required_change": correction,
        "repair_kind": "correction",
        "supports": [{"citation": 1, "span_number": 1}],
        "answer_quotes": ["One literal paragraph."],
        "reason": "The predicate varies but its concrete defect is identical.",
    }


def test_duplicate_checks_share_one_editor_task_without_losing_any_identity() -> None:
    rows, slots, passages = group_editor_findings(
        [
            finding("claim:a"),
            finding("dimension:b"),
            finding("claim:c", correction="Other defect."),
        ]
    )
    assert slots == {"q0001": ["claim:a", "dimension:b"], "q0002": ["claim:c"]}
    assert passages == {"a0001": "One literal paragraph."}
    assert all("answer_quotes" not in row for row in rows)
    assert rows[0]["answer_passage_ids"] == ["a0001"]


def test_reusing_a_check_identity_is_rejected_instead_of_hiding_a_defect() -> None:
    with pytest.raises(ValueError, match="unique"):
        group_editor_findings([finding("a"), finding("a")])


def test_editor_cannot_select_an_original_that_was_not_acquired() -> None:
    with pytest.raises(ValueError, match="unavailable canonical"):
        editorial_context(
            {"repair_base": {"supports": [{"citation": 99}]}, "original_evidence": []}
        )


def test_same_topic_does_not_merge_different_canonical_defects() -> None:
    first = finding("a")
    second = cast(dict[str, JsonValue], finding("b"))
    second["supports"] = [{"citation": 2, "span_number": 1}]
    rows, slots, _ = group_editor_findings([first, second])
    assert len(rows) == len(slots) == 2
