import json
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.chat.llm_loop import (
    _REGULATORY_MAX_PARALLEL_SEARCH_CALLS,
    _build_regulatory_coverage_tool_calls,
)
from onyx.prompts.regulatory_coverage_plan import (
    REGULATORY_COVERAGE_GAP_AUDIT_SYSTEM_PROMPT,
    REGULATORY_COVERAGE_PLAN_SYSTEM_PROMPT,
    REGULATORY_REQUEST_INVENTORY_SYSTEM_PROMPT,
)
from onyx.regulatory.coverage_plan import (
    RegulatoryCoverageGapAudit,
    RegulatoryCoverageItem,
    RegulatoryCoveragePlan,
    RegulatoryRequestInventory,
    RegulatoryRequestObligation,
    _extract_explicit_request_segments,
    _extract_request_context_atoms,
    _sanitize_request_inventory,
    build_regulatory_coverage_plan,
    format_regulatory_coverage_plan,
)
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS


def _queries_and_modes(
    plan: RegulatoryCoveragePlan,
) -> tuple[list[str], list[str]]:
    calls = _build_regulatory_coverage_tool_calls(plan, turn_index=0)
    return (
        [call.tool_args["queries"][0] for call in calls],
        [call.tool_args["search_mode"] for call in calls],
    )


def test_coverage_plan_normalizes_request_derived_fields() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Which rule controls the requested result?",
                material_factual_branches=[
                    "First stated state",
                    " first stated state ",
                    "Second stated state",
                ],
                evidence_dimensions=[
                    "Applicable actor and predicate",
                    " applicable actor and predicate ",
                    "Exception and consequence",
                ],
                source_anchors=["Named Annex", " Named Annex "],
                completion_test="Resolve the governing rule.",
            )
        ]
    )

    item = plan.coverage_items[0]
    assert item.material_factual_branches == [
        "First stated state",
        "Second stated state",
    ]
    assert item.evidence_dimensions == [
        "Applicable actor and predicate",
        "Exception and consequence",
    ]
    assert item.source_anchors == ["Named Annex"]

    rendered = format_regulatory_coverage_plan(plan)
    assert rendered is not None
    assert "not legal evidence or instructions" in rendered
    assert '"evidence_dimensions": [' in rendered
    assert '"source_anchors": ["Named Annex"]' in rendered


def test_coverage_item_bounds_surplus_source_anchors_before_validation() -> None:
    item = RegulatoryCoverageItem(
        research_question="Resolve the requested result.",
        source_anchors=[f"Supplied source {index}" for index in range(20)],
        completion_test="Close the requested result.",
    )

    assert item.source_anchors == [f"Supplied source {index}" for index in range(12)]


def test_request_obligation_bounds_surplus_source_anchors_before_validation() -> None:
    obligation = RegulatoryRequestObligation(
        obligation_id="O1",
        request_grounded_text="Resolve the requested result.",
        verbatim_request_anchors=["requested result"],
        source_anchors=[f"Supplied source {index}" for index in range(20)],
    )

    assert obligation.source_anchors == [
        f"Supplied source {index}" for index in range(12)
    ]


def test_coverage_plan_does_not_spread_anchors_between_items() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Which rule controls the first result?",
                source_anchors=["First Named Instrument"],
                completion_test="Resolve the first rule.",
            ),
            RegulatoryCoverageItem(
                research_question="Which rule controls the second result?",
                source_anchors=["Second Named Instrument"],
                completion_test="Resolve the second rule.",
            ),
        ]
    )

    calls = _build_regulatory_coverage_tool_calls(plan, turn_index=0)

    assert calls[0].tool_args["source_anchors"] == ["First Named Instrument"]
    assert calls[1].tool_args["source_anchors"] == ["Second Named Instrument"]
    assert "First Named Instrument" not in calls[0].tool_args["queries"][0]
    assert "Second Named Instrument" not in calls[1].tool_args["queries"][0]


def test_coverage_plan_failure_without_explicit_segments_is_fail_open() -> None:
    with patch(
        "onyx.regulatory.coverage_plan.generate_structured",
        side_effect=RuntimeError("provider unavailable"),
    ):
        assert (
            build_regulatory_coverage_plan(
                MagicMock(),
                user_request="A material regulatory request",
            )
            is None
        )


def test_coverage_plan_failure_preserves_explicit_request_segments() -> None:
    with patch(
        "onyx.regulatory.coverage_plan.generate_structured",
        side_effect=RuntimeError("provider unavailable"),
    ):
        result = build_regulatory_coverage_plan(
            MagicMock(),
            user_request=(
                "1. Which source controls the first outcome?\n"
                "2. What exact deadline controls the second outcome?"
            ),
        )

    assert result is not None
    assert [item.request_segment_ids for item in result.coverage_items] == [
        ["R1"],
        ["R2"],
    ]
    assert len(_build_regulatory_coverage_tool_calls(result, turn_index=0)) == 4


def test_coverage_plan_retries_truncated_structured_output() -> None:
    expected_plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Which exact rule controls?",
                completion_test="Resolve the controlling rule.",
            )
        ]
    )
    with patch(
        "onyx.regulatory.coverage_plan.generate_structured",
        side_effect=[
            RegulatoryRequestInventory(),
            expected_plan,
            expected_plan,
        ],
    ) as generate:
        result = build_regulatory_coverage_plan(
            MagicMock(),
            user_request="A multi-part regulatory request",
        )

    assert result is expected_plan
    assert generate.call_count == 3
    inventory_call = generate.call_args_list[0]
    initial_call = generate.call_args_list[1]
    audit_call = generate.call_args_list[2]
    assert inventory_call.kwargs["max_attempts"] == 2
    assert inventory_call.kwargs["max_tokens"] == 12_000
    assert initial_call.kwargs["max_attempts"] == 2
    assert initial_call.kwargs["max_tokens"] == 12_000
    assert inventory_call.kwargs["reasoning_effort"].value == "high"
    assert initial_call.kwargs["reasoning_effort"].value == "high"
    assert audit_call.kwargs["max_attempts"] == 1
    assert audit_call.kwargs["max_tokens"] == 12_000
    assert audit_call.kwargs["reasoning_effort"].value == "high"
    assert audit_call.kwargs["response_model"] is RegulatoryCoverageGapAudit
    audit_schema = RegulatoryCoverageGapAudit.model_json_schema()
    assert audit_schema["properties"]["coverage_items"]["maxItems"] == 4


def test_explicit_request_outline_is_syntax_derived_and_deduplicated() -> None:
    segments = _extract_explicit_request_segments(
        "Background only.\n"
        "1. Which source controls the first outcome?\n"
        "2. Explain the second procedure and its deadline?"
    )

    assert [segment.segment_id for segment in segments] == ["R1", "R2"]
    assert [segment.text for segment in segments] == [
        "Which source controls the first outcome?",
        "Explain the second procedure and its deadline?",
    ]


def test_request_context_atoms_preserve_preface_and_colon_list_clauses() -> None:
    atoms = _extract_request_context_atoms(
        "A regulated object crossed two countries and an incident occurred. "
        "The inspection found: the first object was damaged, a fee remained unpaid, "
        "however the second object was not delivered.\n"
        "1. What result follows?\n"
        "2. May the object leave?"
    )

    assert atoms == [
        "A regulated object crossed two countries and an incident occurred",
        "the first object was damaged",
        "a fee remained unpaid",
        "however the second object was not delivered",
    ]


def test_context_atoms_receive_independent_hybrid_retrieval_rows() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve the express question.",
                evidence_dimensions=["Express requested relationship"],
                completion_test="Resolve the express question.",
            )
        ],
        request_context_atoms=[
            "first supplied scenario fact",
            "second supplied scenario fact",
        ],
    )

    calls = _build_regulatory_coverage_tool_calls(plan, turn_index=0)

    atom_calls = [
        call
        for call in calls
        if call.tool_args["evidence_target"].startswith("Request context atom:")
    ]
    assert [call.tool_args["queries"][0] for call in atom_calls] == [
        "first supplied scenario fact",
        "second supplied scenario fact",
    ]
    assert [call.tool_args["search_mode"] for call in atom_calls] == [
        "hybrid",
        "hybrid",
    ]


def test_coverage_plan_appends_only_unmapped_explicit_request_segment() -> None:
    model_plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve the first explicit outcome.",
                evidence_dimensions=["First controlling text"],
                request_segment_ids=["R1", "R999"],
                completion_test="Resolve the first outcome.",
            )
        ]
    )
    with patch(
        "onyx.regulatory.coverage_plan.generate_structured",
        side_effect=[
            RegulatoryRequestInventory(),
            model_plan,
            model_plan,
        ],
    ) as generate:
        result = build_regulatory_coverage_plan(
            MagicMock(),
            user_request=(
                "1. Which source controls the first outcome?\n"
                "2. What exact deadline controls the second outcome?"
            ),
        )

    assert result is not None
    assert [item.request_segment_ids for item in result.coverage_items] == [
        ["R1"],
        ["R2"],
    ]
    assert result.coverage_items[1].research_question == (
        "What exact deadline controls the second outcome?"
    )
    payload = json.loads(generate.call_args_list[0].kwargs["user_prompt"])
    assert payload["request_outline"] == [
        {"segment_id": "R1", "text": "Which source controls the first outcome?"},
        {
            "segment_id": "R2",
            "text": "What exact deadline controls the second outcome?",
        },
    ]


def test_coverage_queries_allocate_atomic_dimensions_and_branches_in_order() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="First independent deliverable",
                material_factual_branches=["First factual state"],
                evidence_dimensions=["First evidence question"],
                source_anchors=["First Source"],
                completion_test="Close first deliverable",
            ),
            RegulatoryCoverageItem(
                research_question="Second independent deliverable",
                material_factual_branches=["Second factual state"],
                evidence_dimensions=["Second evidence question"],
                completion_test="Close second deliverable",
            ),
        ]
    )

    queries, modes = _queries_and_modes(plan)

    assert [query.splitlines()[0] for query in queries[:2]] == [
        "First evidence question",
        "Second evidence question",
    ]
    assert modes == [
        "hybrid",
        "hybrid",
        "hybrid",
        "hybrid",
        "keyword",
        "keyword",
    ]
    assert queries[2:4] == [
        "First factual state",
        "Second factual state",
    ]
    assert "First independent deliverable" not in queries[0]
    assert "Second independent deliverable" not in queries[1]
    assert "First factual state" not in queries[0]
    assert "First Source" not in queries[0]
    calls = _build_regulatory_coverage_tool_calls(plan, turn_index=0)
    assert [call.placement.tab_index for call in calls] == list(range(6))
    assert calls[0].tool_args["coverage_item"] == "First independent deliverable"
    assert calls[0].tool_args["evidence_target"] == "First evidence question"
    assert calls[0].tool_args["source_anchors"] == ["First Source"]


def test_atomic_query_contains_only_request_derived_plan_text() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Alpha beta gamma?",
                material_factual_branches=["Delta epsilon"],
                evidence_dimensions=["Zeta eta"],
                source_anchors=["Theta Iota"],
                completion_test="Kappa lambda",
            )
        ]
    )

    queries, modes = _queries_and_modes(plan)
    assert modes == ["hybrid", "hybrid", "keyword"]
    assert queries[0] == "Zeta eta"
    assert queries[1] == "Delta epsilon"
    assert "Theta Iota" not in queries[0]
    assert "Alpha beta gamma?" not in queries[0]
    assert "Delta epsilon" not in queries[0]
    assert "Kappa lambda" not in queries[0]


def test_terse_retrieval_queries_map_one_to_one_to_atomic_dimensions() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve the requested relationship.",
                evidence_dimensions=[
                    "First long explanatory evidence question",
                    "Second long explanatory evidence question",
                ],
                retrieval_queries=[
                    "abbreviation expanded regulated object consequence",
                    "same-language lexical alternative",
                ],
                source_anchors=["Named Instrument"],
                completion_test="Close the requested relationship.",
            )
        ]
    )

    queries, modes = _queries_and_modes(plan)

    assert queries[:2] == [
        "First long explanatory evidence question",
        "Second long explanatory evidence question",
    ]
    assert queries[2:] == [
        "abbreviation expanded regulated object consequence",
        "same-language lexical alternative",
    ]
    assert modes == ["hybrid", "hybrid", "keyword", "keyword"]
    assert all("Named Instrument" not in query for query in queries)
    calls = _build_regulatory_coverage_tool_calls(plan, turn_index=0)
    assert calls[0].tool_args["evidence_target"] == (
        "First long explanatory evidence question"
    )
    assert calls[1].tool_args["evidence_target"] == (
        "Second long explanatory evidence question"
    )


def test_misaligned_retrieval_queries_cannot_drop_atomic_dimensions() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve every independent result.",
                evidence_dimensions=[
                    "First independent evidence question",
                    "Second independent evidence question",
                    "Third independent evidence question",
                ],
                retrieval_queries=["one broad whole-item alternative"],
                completion_test="Close every independent result.",
            )
        ]
    )

    queries, modes = _queries_and_modes(plan)

    assert queries[:3] == [
        "First independent evidence question",
        "Second independent evidence question",
        "Third independent evidence question",
    ]
    assert modes[:3] == ["hybrid", "hybrid", "hybrid"]
    assert "one broad whole-item alternative" not in queries


def test_full_text_query_uses_only_verified_request_language_and_source() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve the requested outcome.",
                evidence_dimensions=["Translated planner terminology"],
                source_anchors=["Kaynak Adı"],
                request_anchors=["eşyanın teslim edilmemesi"],
                request_anchor_groups=[["eşyanın teslim edilmemesi"]],
                completion_test="Close the request.",
            )
        ]
    )

    queries, modes = _queries_and_modes(plan)

    assert modes == ["hybrid", "hybrid", "keyword"]
    assert queries[0] == "Translated planner terminology"
    assert queries[1] == "eşyanın teslim edilmemesi"


def test_obligation_anchor_groups_remain_separate_search_rows() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve two independent facts.",
                evidence_dimensions=["First planner target", "Second planner target"],
                source_anchors=["Named Source"],
                request_anchor_groups=[
                    ["first exact request phrase"],
                    ["second exact request phrase"],
                ],
                completion_test="Close both facts.",
            )
        ]
    )

    queries, modes = _queries_and_modes(plan)

    assert modes == [
        "hybrid",
        "hybrid",
        "hybrid",
        "hybrid",
        "keyword",
        "keyword",
    ]
    assert queries[2] == "first exact request phrase"
    assert queries[3] == "second exact request phrase"
    assert "second exact request phrase" not in queries[2]


def test_anchors_within_one_obligation_stay_in_one_relationship_query() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve one multi-fact obligation.",
                evidence_dimensions=["Planner target"],
                source_anchors=["Named Source"],
                request_anchor_groups=[
                    ["first exact fact", "second exact fact", "requested consequence"]
                ],
                completion_test="Close the obligation.",
            )
        ]
    )

    queries, modes = _queries_and_modes(plan)

    assert queries == [
        "Planner target",
        "first exact fact; second exact fact; requested consequence",
        "Planner target",
    ]
    assert modes == ["hybrid", "hybrid", "keyword"]
    assert all("Named Source" not in query for query in queries)


def test_request_inventory_rejects_non_verbatim_model_anchors() -> None:
    inventory = RegulatoryRequestInventory(
        obligations=[
            RegulatoryRequestObligation(
                obligation_id="O1",
                request_grounded_text="Teslim edilmeyen eşya",
                verbatim_request_anchors=["eşyanın teslim edilmemesi"],
            ),
            RegulatoryRequestObligation(
                obligation_id="O2",
                request_grounded_text="Translated consequence",
                verbatim_request_anchors=["goods were not delivered"],
            ),
            RegulatoryRequestObligation(
                obligation_id="O3",
                request_grounded_text="Overly broad scenario span",
                verbatim_request_anchors=[
                    "İşlemde eşyanın teslim edilmemesi ayrıca değerlendirilmelidir"
                    " ve farklı sonuçlara bağlanan tüm ayrıntılar korunmalıdır"
                ],
            ),
        ]
    )

    sanitized = _sanitize_request_inventory(
        inventory,
        "İşlemde eşyanın teslim edilmemesi ayrıca değerlendirilmelidir ve farklı "
        "sonuçlara bağlanan tüm ayrıntılar korunmalıdır.",
    )

    assert [item.obligation_id for item in sanitized.obligations] == ["O1"]


def test_initial_searches_use_full_bounded_budget_fairly() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question=f"Independent deliverable {item_index}",
                evidence_dimensions=[
                    f"Dimension {item_index}-{dimension_index}"
                    for dimension_index in range(6)
                ],
                completion_test=f"Close deliverable {item_index}",
            )
            for item_index in range(8)
        ]
    )

    queries, _ = _queries_and_modes(plan)

    assert len(queries) == _REGULATORY_MAX_PARALLEL_SEARCH_CALLS
    assert [query.splitlines()[0] for query in queries[:8]] == [
        f"Dimension {item_index}-0" for item_index in range(8)
    ]
    assert all(
        f"Independent deliverable {item_index}" not in queries[item_index]
        for item_index in range(8)
    )
    assert [query.splitlines()[0] for query in queries[8:16]] == [
        f"Dimension {item_index}-1" for item_index in range(8)
    ]
    assert [query.splitlines()[0] for query in queries[16:20]] == [
        f"Dimension {item_index}-2" for item_index in range(4)
    ]
    assert (
        _queries_and_modes(plan)[1]
        == ["hybrid"] * _REGULATORY_MAX_PARALLEL_SEARCH_CALLS
    )


def test_coverage_queries_are_bounded_to_search_contract() -> None:
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Q " + "long research term " * 28,
                material_factual_branches=["branch distinction " * 12],
                evidence_dimensions=["evidence dimension " * 18],
                completion_test="completion evidence " * 18,
            )
        ]
    )

    calls = _build_regulatory_coverage_tool_calls(plan, turn_index=0)

    assert calls
    for call in calls:
        query = call.tool_args["queries"][0]
        assert len(query) <= REGULATORY_MAX_SEARCH_QUERY_CHARS
        assert not query.endswith((" ", ",", ";", ":", "\n"))


def test_coverage_planner_contract_forbids_source_and_answer_prediction() -> None:
    prompt = REGULATORY_COVERAGE_PLAN_SYSTEM_PROMPT.casefold()

    assert "user's language" in prompt
    assert "source-neutral" in prompt
    assert "never add a predicted source" in prompt
    assert "not encode a possible answer" in prompt

    audit_prompt = REGULATORY_COVERAGE_GAP_AUDIT_SYSTEM_PROMPT.casefold()
    assert "return only genuinely missing" in audit_prompt
    assert "do not answer" in audit_prompt
    assert "structural closure" in audit_prompt
    assert "conventional subject-matter checklist" in audit_prompt

    inventory_prompt = REGULATORY_REQUEST_INVENTORY_SYSTEM_PROMPT.casefold()
    assert "do not answer" in inventory_prompt
    assert "source-neutral" in inventory_prompt
    assert "commonly associated topic" in inventory_prompt


def test_coverage_prompts_atomize_coordinated_requested_outputs() -> None:
    inventory_prompt = " ".join(
        REGULATORY_REQUEST_INVENTORY_SYSTEM_PROMPT.casefold().split()
    )
    plan_prompt = " ".join(REGULATORY_COVERAGE_PLAN_SYSTEM_PROMPT.casefold().split())
    audit_prompt = " ".join(
        REGULATORY_COVERAGE_GAP_AUDIT_SYSTEM_PROMPT.casefold().split()
    )

    assert "container" in inventory_prompt
    assert "coordinates multiple requested outputs" in inventory_prompt
    assert "independently answerable" in inventory_prompt
    assert "coordinated" in plan_prompt
    assert "one independently answerable requested output" in plan_prompt
    assert "coordinated requested outputs" in audit_prompt


def test_coverage_gap_audit_appends_missing_item_without_repeating_draft() -> None:
    initial_item = RegulatoryCoverageItem(
        research_question="Resolve the expressly requested procedure.",
        evidence_dimensions=["Procedure entry condition"],
        completion_test="Close the requested procedure.",
    )
    missing_item = RegulatoryCoverageItem(
        research_question="Resolve the separately stated factual branch.",
        evidence_dimensions=["Effect of the separately stated branch"],
        completion_test="Close the separate factual branch.",
    )
    initial_plan = RegulatoryCoveragePlan(coverage_items=[initial_item])
    audit_plan = RegulatoryCoveragePlan(coverage_items=[initial_item, missing_item])
    with patch(
        "onyx.regulatory.coverage_plan.generate_structured",
        side_effect=[
            RegulatoryRequestInventory(),
            initial_plan,
            audit_plan,
        ],
    ):
        result = build_regulatory_coverage_plan(
            MagicMock(),
            user_request="A request with a procedure and a separate factual branch",
        )

    assert result is not None
    assert result.coverage_items == [
        initial_item.model_copy(
            update={"retrieval_queries": ["Procedure entry condition"]}
        ),
        missing_item,
    ]


def test_coverage_audit_merges_new_dimension_into_matching_item() -> None:
    initial_item = RegulatoryCoverageItem(
        research_question="Resolve the requested consequence.",
        evidence_dimensions=["Operative trigger and scope"],
        retrieval_queries=["stated conduct trigger scope"],
        request_obligation_ids=["O1"],
        completion_test="Close the requested consequence.",
    )
    expanded_item = RegulatoryCoverageItem(
        research_question="Resolve the requested consequence.",
        evidence_dimensions=["Independent status consequence"],
        retrieval_queries=["stated conduct status consequence"],
        request_segment_ids=["R1"],
        completion_test="Close the requested consequence.",
    )
    initial_plan = RegulatoryCoveragePlan(coverage_items=[initial_item])
    with patch(
        "onyx.regulatory.coverage_plan.generate_structured",
        side_effect=[
            RegulatoryRequestInventory(),
            initial_plan,
            RegulatoryCoveragePlan(coverage_items=[expanded_item]),
        ],
    ):
        result = build_regulatory_coverage_plan(
            MagicMock(),
            user_request="Resolve the requested consequence.",
        )

    assert result is not None
    assert len(result.coverage_items) == 1
    merged = result.coverage_items[0]
    assert merged.evidence_dimensions == [
        "Operative trigger and scope",
        "Independent status consequence",
    ]
    assert merged.retrieval_queries == [
        "stated conduct trigger scope",
        "stated conduct status consequence",
    ]
    assert merged.request_segment_ids == ["R1"]
    assert merged.request_obligation_ids == ["O1"]


def test_coverage_plan_rejects_extra_independent_sampling_budget() -> None:
    with pytest.raises(TypeError):
        cast(Any, build_regulatory_coverage_plan)(
            MagicMock(),
            user_request="What is the requested consequence?",
            independent_samples=2,
        )


def test_unmapped_request_inventory_obligation_gets_bounded_fallback() -> None:
    mapped_obligation = RegulatoryRequestObligation(
        obligation_id="O1",
        request_grounded_text="The first requested consequence",
        verbatim_request_anchors=["first requested consequence"],
    )
    omitted_obligation = RegulatoryRequestObligation(
        obligation_id="O2",
        request_grounded_text="The separately stated material event",
        verbatim_request_anchors=["separately stated material event"],
        source_anchors=["Named Instrument"],
    )
    inventory = RegulatoryRequestInventory(
        obligations=[mapped_obligation, omitted_obligation]
    )
    mapped_item = RegulatoryCoverageItem(
        research_question="Resolve the first requested consequence.",
        evidence_dimensions=["First consequence"],
        request_obligation_ids=["O1"],
        completion_test="Close the first consequence.",
    )
    plan = RegulatoryCoveragePlan(coverage_items=[mapped_item])

    with patch(
        "onyx.regulatory.coverage_plan.generate_structured",
        side_effect=[
            inventory,
            plan,
            RegulatoryCoveragePlan(),
        ],
    ):
        result = build_regulatory_coverage_plan(
            MagicMock(),
            user_request=(
                "Resolve the first requested consequence and the separately "
                "stated material event under the Named Instrument."
            ),
        )

    assert result is not None
    assert result.coverage_items[0].research_question == mapped_item.research_question
    assert result.coverage_items[0].request_anchors == ["first requested consequence"]
    assert result.coverage_items[0].request_anchor_groups == [
        ["first requested consequence"]
    ]
    fallback = result.coverage_items[1]
    assert fallback.request_obligation_ids == ["O2"]
    assert fallback.research_question == "The separately stated material event"
    assert fallback.request_anchors == ["separately stated material event"]
    assert fallback.request_anchor_groups == [["separately stated material event"]]
    assert fallback.source_anchors == ["Named Instrument"]
