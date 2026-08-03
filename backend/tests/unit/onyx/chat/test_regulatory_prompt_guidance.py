from unittest.mock import patch

from onyx.chat.prompt_utils import append_grounding_guidance, build_system_prompt
from onyx.prompts.chat_prompts import CITATION_REMINDER
from onyx.prompts.regulatory_guidance import (
    REGULATORY_ANALYSIS_GUIDANCE,
    REGULATORY_COVERAGE_REMINDER,
    REGULATORY_RESEARCH_EXECUTION_GUIDANCE,
    REGULATORY_SEARCH_GUIDANCE,
    REGULATORY_SYNTHESIS_GUIDANCE,
)
from onyx.prompts.search_prompts import KEYWORD_REPHRASE_USER_PROMPT
from onyx.prompts.tool_prompts import INTERNAL_SEARCH_GUIDANCE


def test_regulatory_guidance_is_mandatory_and_idempotent() -> None:
    once = append_grounding_guidance("Custom prompt")
    twice = append_grounding_guidance(once)

    assert REGULATORY_ANALYSIS_GUIDANCE.strip() in once
    assert twice == once


def test_regulatory_guidance_is_in_default_system_prompt() -> None:
    with patch("onyx.chat.prompt_utils.get_company_context", return_value=None):
        prompt = build_system_prompt("Base prompt.")

    assert "You control the research path and legal analysis" in prompt
    assert "governing source and version" in prompt
    assert "Keep established facts" in prompt
    assert "Prefer controlling legal text" in prompt
    assert "Validate citations at claim level" in prompt


def test_regulatory_guidance_does_not_embed_benchmark_scenario() -> None:
    scenario_specific_terms = (
        "TIR",
        "UND",
        "DAC",
        "ADR",
        "Basel",
        "2207.10",
        "50.000",
        "Milano",
        "Bakü",
        "Kapıkule",
        "C2 yetki",
    )

    all_regulatory_guidance = "\n".join(
        (
            REGULATORY_ANALYSIS_GUIDANCE,
            REGULATORY_SEARCH_GUIDANCE,
            REGULATORY_COVERAGE_REMINDER,
            REGULATORY_SYNTHESIS_GUIDANCE,
            INTERNAL_SEARCH_GUIDANCE,
        )
    )

    assert not any(term in all_regulatory_guidance for term in scenario_specific_terms)


def test_regulatory_coverage_audit_is_close_to_the_final_turn() -> None:
    assert "Decide whether the retrieved evidence is sufficient" in CITATION_REMINDER
    assert "Do not spend calls to satisfy a mechanical checklist" in CITATION_REMINDER


def test_regulatory_search_strategy_remains_model_directed() -> None:
    assert "You decide what to search" in REGULATORY_SEARCH_GUIDANCE
    assert "There is no required decomposition or call count" in (
        REGULATORY_SEARCH_GUIDANCE
    )
    assert "split or combine retrieval attempts" in REGULATORY_SEARCH_GUIDANCE
    assert "materially different query or retrieval mode" in (
        REGULATORY_SEARCH_GUIDANCE
    )
    assert "Stop searching when direct controlling text supports" in (
        REGULATORY_SEARCH_GUIDANCE
    )
    assert "Retrieval silence is not proof" in REGULATORY_SEARCH_GUIDANCE
    assert "first tool round" not in REGULATORY_SEARCH_GUIDANCE
    assert "Select and explicitly provide `search_mode` independently" in (
        INTERNAL_SEARCH_GUIDANCE
    )
    assert "high proportion" not in INTERNAL_SEARCH_GUIDANCE
    assert "literal alternatives" not in INTERNAL_SEARCH_GUIDANCE


def test_regulatory_search_preserves_material_user_identifiers() -> None:
    assert "identifier supplied by the user" in REGULATORY_SEARCH_GUIDANCE
    assert "acronym, number, or code verbatim" in REGULATORY_SEARCH_GUIDANCE
    assert "only disambiguating identifier" in REGULATORY_SEARCH_GUIDANCE
    assert "material name, acronym, number, or code" in INTERNAL_SEARCH_GUIDANCE


def test_regulatory_guidance_distinguishes_grouped_labels_without_splitting_aliases() -> (
    None
):
    assert "slash, conjunction, parenthetical" in REGULATORY_ANALYSIS_GUIDANCE
    assert "labels are aliases or one controlling rule" in (
        REGULATORY_ANALYSIS_GUIDANCE
    )
    assert "aliases, variants, or distinct legal objects" in (
        REGULATORY_SEARCH_GUIDANCE
    )
    assert "do not silently substitute an umbrella term" in (REGULATORY_SEARCH_GUIDANCE)
    assert "unless the retrieved text establishes an alias" in (
        REGULATORY_SYNTHESIS_GUIDANCE
    )


def test_regulatory_search_conditionally_resolves_scope_and_linked_text() -> None:
    assert "can materially affect" in REGULATORY_ANALYSIS_GUIDANCE
    assert "Apply only the distinctions" in REGULATORY_ANALYSIS_GUIDANCE
    assert "decide yourself which propositions require retrieval" in (
        REGULATORY_ANALYSIS_GUIDANCE
    )
    assert "navigation lead, not evidence" in REGULATORY_SEARCH_GUIDANCE
    assert "could change the current proposition" in REGULATORY_SEARCH_GUIDANCE
    assert "decide whether a focused follow-up is warranted" in (
        REGULATORY_SEARCH_GUIDANCE
    )
    assert "lead-in that supplies its negation" in REGULATORY_ANALYSIS_GUIDANCE
    assert "enumerated subparagraph" in REGULATORY_SEARCH_GUIDANCE
    assert "Do not infer the logical direction" in REGULATORY_SEARCH_GUIDANCE
    assert "orphan list item" in REGULATORY_SYNTHESIS_GUIDANCE


def test_unknown_provision_discovery_uses_the_legal_relationship() -> None:
    assert "even if the user did not supply a provision number" in (
        REGULATORY_SEARCH_GUIDANCE
    )
    assert "using the discovered source identity" in REGULATORY_SEARCH_GUIDANCE
    assert "without guessing one" in REGULATORY_RESEARCH_EXECUTION_GUIDANCE

    unknown_provision_guidance = (
        REGULATORY_SEARCH_GUIDANCE + REGULATORY_RESEARCH_EXECUTION_GUIDANCE
    )
    assert "outcome-changing question of scope, prohibition, exception" in (
        unknown_provision_guidance
    )
    assert "Madde 4A" not in unknown_provision_guidance
    assert "Article 4A" not in unknown_provision_guidance


def test_regulatory_final_gate_rejects_unsupported_adverse_inferences() -> None:
    assert "materially outcome-changing point" in REGULATORY_COVERAGE_REMINDER
    assert "If the controlling chunks already support" in (REGULATORY_COVERAGE_REMINDER)
    assert "exact citation chunk states the claim" in REGULATORY_COVERAGE_REMINDER
    assert "Distinguish established facts from allegations" in (
        REGULATORY_COVERAGE_REMINDER
    )
    assert "mechanical checklist" in REGULATORY_COVERAGE_REMINDER
    assert "general condition, review power, or discretionary standard" in (
        REGULATORY_COVERAGE_REMINDER
    )
    assert "requires direct support for both the trigger and the stated effect" in (
        REGULATORY_COVERAGE_REMINDER
    )
    assert "direct operative support for its trigger and effect" in (
        REGULATORY_SYNTHESIS_GUIDANCE
    )


def test_regulatory_guidance_separates_classification_regime_and_sequence() -> None:
    assert "Treat classification as a separate supported inference" in (
        REGULATORY_ANALYSIS_GUIDANCE
    )
    assert "does not by itself establish" in REGULATORY_ANALYSIS_GUIDANCE
    assert "legal regime active at the relevant event" in (REGULATORY_ANALYSIS_GUIDANCE)
    assert "authority that issued or administers" in REGULATORY_ANALYSIS_GUIDANCE
    assert "Do not manufacture procedural order" in REGULATORY_ANALYSIS_GUIDANCE
    assert "fact-to-category classification" in REGULATORY_SYNTHESIS_GUIDANCE
    assert "Never infer a mandatory order" in REGULATORY_SYNTHESIS_GUIDANCE


def test_internal_search_does_not_force_a_server_owned_plan() -> None:
    assert "coverage_item" not in INTERNAL_SEARCH_GUIDANCE
    assert "evidence_target" not in INTERNAL_SEARCH_GUIDANCE
    assert "Decide yourself" in INTERNAL_SEARCH_GUIDANCE


def test_keyword_rewrite_covers_regulatory_procedure_dimensions() -> None:
    assert "lexically complementary" in KEYWORD_REPHRASE_USER_PROMPT
    assert "exact source or provision identifier" in KEYWORD_REPHRASE_USER_PROMPT
    assert "do not add dimensions" in KEYWORD_REPHRASE_USER_PROMPT
