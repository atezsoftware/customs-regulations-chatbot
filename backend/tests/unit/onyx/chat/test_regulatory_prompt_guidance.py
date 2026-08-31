from unittest.mock import patch

from onyx.chat.prompt_utils import append_grounding_guidance, build_system_prompt
from onyx.prompts.regulatory_guidance import (
    REGULATORY_ANALYSIS_GUIDANCE,
    REGULATORY_COVERAGE_REMINDER,
    REGULATORY_RESEARCH_EXECUTION_GUIDANCE,
    REGULATORY_SEARCH_GUIDANCE,
    REGULATORY_SYNTHESIS_GUIDANCE,
)
from onyx.prompts.tool_prompts import INTERNAL_SEARCH_GUIDANCE


def test_regulatory_guidance_is_mandatory_and_idempotent() -> None:
    once = append_grounding_guidance("Custom prompt")
    twice = append_grounding_guidance(once)

    assert REGULATORY_ANALYSIS_GUIDANCE.strip() in once
    assert twice == once


def test_regulatory_guidance_is_in_default_system_prompt() -> None:
    with patch("onyx.chat.prompt_utils.get_company_context", return_value=None):
        prompt = build_system_prompt("Base prompt.")

    assert "silent issue ledger from the current request only" in prompt
    assert "exact operative text" in prompt
    assert "Validate every material claim at citation level" in prompt


def test_regulatory_guidance_is_dataset_blind() -> None:
    all_regulatory_guidance = "\n".join(
        (
            REGULATORY_ANALYSIS_GUIDANCE,
            REGULATORY_SEARCH_GUIDANCE,
            REGULATORY_COVERAGE_REMINDER,
            REGULATORY_RESEARCH_EXECUTION_GUIDANCE,
            REGULATORY_SYNTHESIS_GUIDANCE,
        )
    )

    assert "current request only" in all_regulatory_guidance
    assert "subject-matter checklist" in all_regulatory_guidance
    assert "prior examples" in all_regulatory_guidance


def test_grounding_guidance_resolves_source_supported_priority_before_conflict() -> (
    None
):
    prompt = append_grounding_guidance("Custom prompt")

    assert "source roles, scope, specificity, cross-references, or validity" in prompt
    assert "dedicated operative provision" in prompt
    assert "general definition's incidental description" in prompt
    assert "lead with the controlling passage" in prompt
    assert "If the supplied material does not establish priority" in prompt


def test_regulatory_search_remains_dynamic_and_evidence_led() -> None:
    assert "You decide what to search" in REGULATORY_SEARCH_GUIDANCE
    assert "There is no required call count" in REGULATORY_SEARCH_GUIDANCE
    assert "navigation leads, not evidence" in REGULATORY_SEARCH_GUIDANCE
    assert "connected parent, child, or sibling text" in REGULATORY_SEARCH_GUIDANCE
    assert "do not repeat successful searches" in REGULATORY_SEARCH_GUIDANCE
    assert "Retrieval silence is not proof" in REGULATORY_SEARCH_GUIDANCE


def test_regulatory_research_uses_corpus_language_and_recovers_only_open_rows() -> None:
    assert "terminology and drafting style of Turkish legislation" in (
        INTERNAL_SEARCH_GUIDANCE
    )
    assert "never invent a source identifier or a new issue" in INTERNAL_SEARCH_GUIDANCE
    assert (
        "compare the gathered evidence with the open request-derived propositions"
        in (INTERNAL_SEARCH_GUIDANCE)
    )
    assert "search only that missing proposition" in INTERNAL_SEARCH_GUIDANCE


def test_regulatory_final_gate_is_request_and_citation_bounded() -> None:
    assert "every express current-request deliverable" in REGULATORY_COVERAGE_REMINDER
    assert "exact inline citation directly entails" in REGULATORY_COVERAGE_REMINDER
    assert "smallest sufficient set" in REGULATORY_COVERAGE_REMINDER
    assert "add rows from a conventional legal checklist" in (
        REGULATORY_COVERAGE_REMINDER
    )
    assert "Stop once the request-derived rows are supported" in (
        REGULATORY_COVERAGE_REMINDER
    )


def test_regulatory_synthesis_does_not_expand_the_request() -> None:
    assert (
        "mirror the user's express material requests" in REGULATORY_SYNTHESIS_GUIDANCE
    )
    assert "Do not let an umbrella discussion replace" in (
        REGULATORY_SYNTHESIS_GUIDANCE
    )
    assert "do not invent missing detail" in REGULATORY_SYNTHESIS_GUIDANCE
    assert "smallest directly entailing citation set" in (REGULATORY_SYNTHESIS_GUIDANCE)


def test_internal_search_does_not_force_a_server_owned_plan() -> None:
    assert "coverage_item" not in INTERNAL_SEARCH_GUIDANCE
    assert "evidence_target" not in INTERNAL_SEARCH_GUIDANCE
    assert "Decide yourself" in INTERNAL_SEARCH_GUIDANCE
    assert "one focused query and an explicit `search_mode` per call" in (
        INTERNAL_SEARCH_GUIDANCE
    )
