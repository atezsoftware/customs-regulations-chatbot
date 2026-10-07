"""Connect literal draft citations to physically delivered provision families."""

from copy import deepcopy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.asv3.shared_originals import delivered_provision_navigation
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_native_model_adapter import adaptive_tool_view, model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original, recorded


def test_uncited_sibling_and_every_citing_owned_unit_are_visible_without_mutation() -> (
    None
):
    ledger = recorded(
        [
            original("rule", "An application needs approval."),
            original("relief", "Prior voluntary notification allows a reduction."),
            original("later", "After approval, submit a completion declaration."),
            original("other", "A different rule.", source="another-statute"),
        ]
    )
    records = [full_record(ledger, number) for number in range(1, 5)]
    units: list[dict[str, JsonValue]] = [
        {"unit_id": "summary", "text": "Approval is required [1]."},
        {"unit_id": "detail", "text": "The procedure uses approval [1]."},
        {"unit_id": "heading", "text": "## Background [2]"},
        {"unit_id": "table", "text": "| Step | Result |\n| A | Approval [1] |"},
    ]
    reviews: list[dict[str, JsonValue]] = [
        {
            "lead_id": "related-lead",
            "anchor_source_id": "authorized-statute",
            "article_no": "17",
            "qualifier": None,
            "anchor_evidence_numbers": [1],
        },
        {
            "lead_id": "other-anchor",
            "anchor_source_id": "authorized-statute",
            "article_no": "17",
            "qualifier": None,
            "anchor_evidence_numbers": [4],
        },
    ]
    before = deepcopy((records, units, reviews, ledger.export()))
    navigation = delivered_provision_navigation(
        records, draft_units=units, related_reviews=reviews
    )
    assert navigation[0]["draft_application"] == {
        "cited_original_citations": [1],
        "uncited_full_original_citations": [2, 3],
        "answer_unit_ids": ["summary", "detail", "table"],
        "related_lead_ids": ["related-lead"],
    }
    assert "draft_application" not in navigation[1]
    assert before == (records, units, reviews, ledger.export())
    assert all(
        "draft_application" not in row
        for row in delivered_provision_navigation(records)
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"source_id": "other-statute"},
        {"metadata": {"heading_path": ["Statute", "GEÇİCİ MADDE 17"]}},
        {"metadata": {"heading_path": ["Statute", "EK 2", "MADDE 17"]}},
        {"metadata": {"version": "another-revision"}},
        {"truncated": True},
        {"citable": False},
        {"metadata": {"derived": True}},
    ],
)
def test_navigation_never_implies_a_foreign_or_incomplete_sibling(
    changed: dict[str, JsonValue],
) -> None:
    ledger = recorded([original("rule", "Rule."), original("exception", "Exception.")])
    records = [full_record(ledger, 1), full_record(ledger, 2)]
    for key, value in changed.items():
        if key == "metadata":
            metadata = records[1][key]
            assert isinstance(metadata, dict) and isinstance(value, dict)
            records[1][key] = {**metadata, **value}
        else:
            records[1][key] = value
    rows = delivered_provision_navigation(
        records, draft_units=[{"unit_id": "owned", "text": "Rule [1]."}]
    )
    assert rows[0]["draft_application"] == {
        "cited_original_citations": [1],
        "uncited_full_original_citations": [],
        "answer_unit_ids": ["owned"],
    }


@pytest.mark.parametrize("tuned", [False, True])
def test_native_context_uses_actual_owned_edit_ids_without_an_extra_generation(
    tuned: bool,
) -> None:
    import json

    ledger = recorded([original("rule", "Rule."), original("exception", "Exception.")])
    context = RunContext(
        services={
            "research_profile": "normal",
            "asv3_workflow_variant": ASV3_TUNED_VARIANT if tuned else "standard",
            "lean_native_mode": True,
            "evidence": ledger,
        }
    )
    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    prompt, _, _ = adapter._fit_native_decision(
        adaptive_tool_view(
            original_evidence=[full_record(ledger, 1), full_record(ledger, 2)],
            draft_to_repair="Short outcome [1].\n\nThe procedure also uses the rule [1].",
            publication_gap={"kind": "source_application_test"},
        )
    )
    payload = json.loads(cast(str, prompt[-1].content))
    selected.invoke.assert_not_called()
    if tuned:
        application = payload["delivered_provisions"][0]["draft_application"]
        assert application["uncited_full_original_citations"] == [2]
        owned_ids = [row["unit_id"] for row in payload["draft_to_repair"]["units"]]
        assert application["answer_unit_ids"] == owned_ids
        assert "draft_application" in payload["evidence_note"]
        originals = payload["original_evidence"]
        assert [row["text"] for row in originals] == ["Rule.", "Exception."]
    else:
        assert "draft_application" not in payload["evidence_note"]
        assert "delivered_provisions" not in payload
