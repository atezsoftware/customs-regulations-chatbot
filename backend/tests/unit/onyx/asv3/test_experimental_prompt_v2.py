"""Protect the experimental prompt's source and workflow boundary contracts."""

import pytest

from onyx.prompts.asv3.coordinator_reference import (
    COORDINATOR_REFERENCE_PROMPT,
    RESEARCHER_REFERENCE_PROMPT,
)
from onyx.prompts.asv3.experimental import (
    EXPERIMENTAL_COORDINATOR_PROMPT,
    EXPERIMENTAL_PROMPT_VERSION,
    EXPERIMENTAL_RESEARCHER_PROMPT,
    RESEARCH_INSTRUCTIONS,
    SOURCE_NAVIGATION,
    _source_navigation,
)


@pytest.mark.parametrize(
    "prompt",
    [EXPERIMENTAL_COORDINATOR_PROMPT, EXPERIMENTAL_RESEARCHER_PROMPT],
    ids=["coordinator", "researcher"],
)
def test_independent_prompts_reuse_the_complete_baseline_map_once(prompt: str) -> None:
    heading = "TOPIC-TO-SOURCE NAVIGATION\n"
    coordinator_map = COORDINATOR_REFERENCE_PROMPT.split(heading, 1)[1].split(
        "\nCONSTRUCT SOURCE-BOUND ANSWERS\n", 1
    )[0]
    researcher_map = RESEARCHER_REFERENCE_PROMPT.split(heading, 1)[1].split(
        "\nRETURN OPERATIVE FINDINGS\n", 1
    )[0]
    assert coordinator_map == researcher_map
    assert SOURCE_NAVIGATION == (heading + coordinator_map).strip()
    assert prompt.count(SOURCE_NAVIGATION) == 1
    assert prompt.count(RESEARCH_INSTRUCTIONS) == 1
    assert COORDINATOR_REFERENCE_PROMPT not in prompt
    assert RESEARCHER_REFERENCE_PROMPT not in prompt
    assert EXPERIMENTAL_PROMPT_VERSION == "asv3-experimental-2026-10-06.7"


@pytest.mark.parametrize(
    "reference",
    [
        "No navigation section",
        "\nTOPIC-TO-SOURCE NAVIGATION\n| Incomplete |",
        "\nTOPIC-TO-SOURCE NAVIGATION\nNot a complete table\n"
        "CONSTRUCT SOURCE-BOUND ANSWERS\n",
        COORDINATOR_REFERENCE_PROMPT + "\nTOPIC-TO-SOURCE NAVIGATION\n",
        COORDINATOR_REFERENCE_PROMPT + "\nCONSTRUCT SOURCE-BOUND ANSWERS\n",
    ],
)
def test_source_map_changes_fail_loudly_instead_of_silently_omitting_rows(
    reference: str,
) -> None:
    with pytest.raises(ValueError):
        _source_navigation(reference)


def test_related_source_review_requires_read_effect_and_scope_witnesses() -> None:
    for requirement in (
        "final disposition from its reasoning, the referring court's request",
        "party submissions and appended materials",
        "compare every material changed or preserved part",
        "known source_id and supplied continuation cursor or focused local lookup",
        "has_more means more text exists, not that every remaining passage is required",
        "valid witness ranges prove delivery, not legal entailment",
        "status=examined requires original witnesses establishing the effect AND its material limits",
        "status=not_material needs original-bound factual/scope exclusion",
        "argument_only or unknown cannot prove a holding",
        "status=unresolved with its precise gap",
        "actual delivered citation numbers and supplied start_char/end_char",
        "No separate review,\nplanning call or model stage is required",
    ):
        assert requirement in RESEARCH_INSTRUCTIONS


def test_arguments_and_scenario_branches_are_source_bound_not_categorical() -> None:
    for requirement in (
        "underlying obligation, the legal basis",
        "surviving obligation does not prove a surviving sanction",
        "administrative instruction does not establish legal authority",
        "strongest material original-supported favorable ground and its counterargument",
        "ordinary administrative practice, an arguable challenge and an established exception",
        "neither categorical liability nor categorical relief follows",
        "Use already delivered conditions and favorable clauses",
        "Give each supported branch beside its outcome and name the",
        "conditional branch beside its outcome and name the fact changing it",
        "same conditions, controversy and uncertainty in quick answers",
        "Complete material exception/objection analysis now",
        "optional follow-up questions grounded in this case's supported findings",
    ):
        assert requirement in RESEARCH_INSTRUCTIONS


def test_primary_tax_details_and_citation_contracts_survive_the_rebuild() -> None:
    for requirement in (
        "its own governing Kanun or higher binding original",
        "KDV conclusions need applicable KDV Kanunu originals",
        "ÖTV conclusions need",
        "applicable ÖTV Kanunu originals and relevant lists",
        "Customs-duty relief does not establish",
        "Preserve AND/OR",
        "proof issuer, form/authentication, triggering event, deadline/start, calculation",
        "security release or settlement",
        "precise adjacent recorded global\n[n] citations",
        "verified official instrument name",
        "jointly support ALL material clauses and qualifications",
        "short contiguous literal operative quotation",
        "Never\nsummarize, truncate or simplify away",
    ):
        assert requirement in RESEARCH_INSTRUCTIONS


def test_public_updates_follow_current_native_contract_without_imported_artifacts() -> (
    None
):
    for requirement in (
        "Each material call exposing _public_update, including a batch",
        "Name the verified source/article when known",
        "do not announce a finding before reading its original",
        "applicable nested step arguments, not unsupported wrapper fields",
        "BCP-47 _language on the first useful call",
        "phases required by the actual tool schema",
        "Follow the supplied application communication preferences",
        "Never repeat the full question/scenario as a bold heading",
        "No quota of time, calls, sources or words determines research completion",
    ):
        assert requirement in RESEARCH_INSTRUCTIONS
    for benchmark_artifact in (
        "May 10, 2026",
        "submit_final_result",
        "courtlistener_search",
        "law_query_rewrite",
        "fact_law_relevance_check",
        "<think>",
        "trailing JSON",
        "max_turns",
        "m. 241",
        "Vaka 4",
    ):
        assert benchmark_artifact not in RESEARCH_INSTRUCTIONS
