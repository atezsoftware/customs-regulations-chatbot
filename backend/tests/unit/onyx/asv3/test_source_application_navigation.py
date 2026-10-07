"""Connect literal draft citations to physically delivered provision families."""

from copy import deepcopy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import uncited_named_authority_bindings
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import CapabilityCall, Decision, EvidenceItem, RunContext
from onyx.asv3.retained_answer import resolve_retained_answer
from onyx.asv3.shared_originals import delivered_provision_navigation
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_native_model_adapter import adaptive_tool_view, model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original, recorded


def canonical_original(
    chunk: str,
    text: str,
    *,
    article: str = "27",
    title: str = "8917 sayılı Faaliyet Kanunu",
    metadata: dict[str, JsonValue] | None = None,
) -> EvidenceItem:
    item = original(
        chunk,
        text,
        source=title,
        headings=[title, f"MADDE {article}"],
        metadata={"title": title, "document_type": "kanun", **(metadata or {})},
    )
    assert item.search_doc is not None
    item.search_doc.metadata["regulatory_chunk_id"] = chunk
    return item


def test_new_uncited_named_family_links_summary_table_and_related_anchor() -> None:
    ledger = recorded(
        [
            original("lower", "An implementing procedure.", source="lower"),
            canonical_original("rule", "Approval is possible."),
            canonical_original("relief", "Prior notification permits a reduction."),
            canonical_original("other", "A foreign provision.", title="Başka Kanunu"),
        ]
    )
    records = [full_record(ledger, number) for number in range(1, 5)]
    units: list[dict[str, JsonValue]] = [
        {"unit_id": "summary", "text": "Faaliyet Kanunu m. 27 uygulanır [1]."},
        {"unit_id": "detail", "text": "8917 sayılı Faaliyet Kanunu m. 27 uygulanır."},
        {
            "unit_id": "table",
            "text": "| Konu | Etki |\n| Faaliyet Kanunu m. 27 | İzin [1] |",
        },
        {"unit_id": "heading", "text": "## Faaliyet Kanunu m. 27 [2]"},
    ]
    reviews: list[dict[str, JsonValue]] = [
        {
            "lead_id": "operative-related-source",
            "anchor_source_id": "8917 sayılı Faaliyet Kanunu",
            "article_no": "27",
            "qualifier": None,
            "anchor_evidence_numbers": [2],
        }
    ]
    before = deepcopy((units, records, reviews, ledger.export()))
    bindings = uncited_named_authority_bindings(units, ledger)
    rows = delivered_provision_navigation(
        records,
        draft_units=units,
        related_reviews=reviews,
        named_original_bindings=bindings,
    )
    assert rows[1]["draft_application"] == {
        "cited_original_citations": [],
        "uncited_full_original_citations": [2, 3],
        "named_original_candidates": [2, 3],
        "answer_unit_ids": ["summary", "detail", "table"],
        "related_lead_ids": ["operative-related-source"],
    }
    assert "draft_application" not in rows[2]
    assert before == (units, records, reviews, ledger.export())


@pytest.mark.parametrize(
    "text",
    [
        "Madde 27 uygulanır [1].",
        "FK m. 27 uygulanır [1].",
        "Başka Kanunu m. 27 uygulanır [1].",
        "Faaliyet Kanunu geçici m. 27 uygulanır [1].",
        "Faaliyet Kanunu m. 27/A uygulanır [1].",
        "## Faaliyet Kanunu m. 27 [1]",
    ],
)
def test_named_navigation_requires_the_actual_bound_identity(text: str) -> None:
    ledger = recorded(
        [
            original("lower", "A lower instrument's text."),
            canonical_original("rule", "An ordinary provision."),
        ]
    )
    units: list[dict[str, JsonValue]] = [{"unit_id": "owned", "text": text}]
    rows = delivered_provision_navigation(
        [full_record(ledger, 2)],
        draft_units=units,
        named_original_bindings=uncited_named_authority_bindings(units, ledger),
    )
    assert "draft_application" not in rows[0]


def test_named_bindings_keep_draft_definitions_and_only_physical_full_ranges() -> None:
    ledger = recorded(
        [
            canonical_original("rule", "A current provision."),
            canonical_original("partial", "An incompletely supplied condition."),
            canonical_original("absent", "An original outside this decision."),
            canonical_original(
                "old", "An older provision.", metadata={"version": "old"}
            ),
        ]
    )
    units: list[dict[str, JsonValue]] = [
        {"unit_id": "definition", "text": "8917 sayılı Faaliyet Kanunu (FK)."},
        {"unit_id": "owned", "text": "FK m. 27 uygulanır."},
    ]
    records = [full_record(ledger, number) for number in [1, 2, 4]]
    records[1]["truncated"] = True
    rows = delivered_provision_navigation(
        records,
        draft_units=units,
        named_original_bindings=uncited_named_authority_bindings(units, ledger),
    )
    assert len(rows) == 2
    current, historical = rows[0]["draft_application"], rows[1]["draft_application"]
    assert isinstance(current, dict) and isinstance(historical, dict)
    assert current["named_original_candidates"] == [1]
    assert historical["named_original_candidates"] == [4]
    answer_units = current["answer_unit_ids"]
    assert isinstance(answer_units, list)
    assert "owned" in answer_units
    assert "definition" not in answer_units


def test_named_binding_never_selects_a_descendant_cross_reference_as_own_article() -> (
    None
):
    item = canonical_original("rule", "A different operative provision.")
    item.metadata["heading_path"] = [
        "8917 sayılı Faaliyet Kanunu",
        "MADDE 27",
        "MADDE 29 uyarınca yapılan işlemler",
    ]
    ledger = recorded([item])
    units: list[dict[str, JsonValue]] = [
        {"unit_id": "owned", "text": "Faaliyet Kanunu m. 29 uygulanır."}
    ]
    rows = delivered_provision_navigation(
        [full_record(ledger, 1)],
        draft_units=units,
        named_original_bindings=uncited_named_authority_bindings(units, ledger),
    )
    assert "draft_application" not in rows[0]


@pytest.mark.parametrize("tuned", [False, True])
def test_actual_context_binds_late_original_to_owned_unit_only_when_tuned(
    tuned: bool,
) -> None:
    import json

    ledger = recorded(
        [
            original("lower", "An implementing rule."),
            canonical_original("rule", "The statute's operative rule."),
            canonical_original("relief", "A conditional relief."),
        ]
    )
    selected = model()
    context = RunContext(
        services={
            "research_profile": "normal",
            "asv3_workflow_variant": ASV3_TUNED_VARIANT if tuned else "standard",
            "lean_native_mode": True,
            "evidence": ledger,
        }
    )
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    prompt, _, _ = adapter._fit_native_decision(
        adaptive_tool_view(
            original_evidence=[full_record(ledger, number) for number in [1, 2, 3]],
            draft_to_repair="Faaliyet Kanunu m. 27 uyarınca izin gerekir [1].",
            publication_gap={"kind": "late_original_test"},
        )
    )
    payload = json.loads(cast(str, prompt[-1].content))
    selected.invoke.assert_not_called()
    if tuned:
        row = next(
            entry
            for entry in payload["delivered_provisions"]
            if entry["source_id"] == "8917 sayılı Faaliyet Kanunu"
        )
        assert row["draft_application"]["named_original_candidates"] == [2, 3]
        assert row["draft_application"]["answer_unit_ids"] == [
            payload["draft_to_repair"]["units"][0]["unit_id"]
        ]
    else:
        assert "delivered_provisions" not in payload


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


def test_native_semantic_edit_outside_host_targets_retains_other_units_exactly() -> (
    None
):
    import json

    ledger = recorded(
        [
            original("rule", "An application requires approval."),
            original("relief", "Prior notification permits a reduction."),
        ]
    )
    context = RunContext(
        services={
            "research_profile": "normal",
            "asv3_workflow_variant": ASV3_TUNED_VARIANT,
            "lean_native_mode": True,
            "evidence": ledger,
        }
    )
    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    draft = (
        "The rule needs approval [1].\n\nThe charge always applies [1].\n\n## Closing"
    )
    view = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)],
        draft_to_repair=draft,
        publication_gap={"target_unit_ids": ["a_mechanical_target"]},
    )
    prompt, _, _ = adapter._fit_native_decision(view)
    payload = json.loads(cast(str, prompt[-1].content))
    instruction = payload["retained_answer"]["instruction"]
    assert "unapproved text" in instruction
    assert "not a complete semantic audit" in instruction
    assert "even when the host did not name it" in instruction
    assert "when the actual gap requires it" not in instruction
    second_unit = payload["draft_to_repair"]["units"][1]
    replacement = "The charge can be reduced if prior notification qualifies [2]."
    decision = Decision(
        calls=[
            CapabilityCall(
                name="submit_retained_answer",
                arguments={
                    "retained_answer_edits": [
                        {"unit_id": second_unit["unit_id"], "replacement": replacement}
                    ]
                },
            )
        ]
    )
    resolved = resolve_retained_answer(decision, context, draft, request=view.request)
    assert resolved.calls[0].arguments["answer"] == (
        "The rule needs approval [1].\n\n" + replacement + "\n\n## Closing"
    )
    selected.invoke.assert_not_called()
