from typing import Any, cast

from onyx.deep_research.dr_mock_tools import (
    MAX_RESEARCH_AGENT_TASK_CHARS,
    RESEARCH_AGENT_TASK_KEY,
    RESEARCH_AGENT_TOOL_DESCRIPTION,
)
from onyx.prompts.deep_research.orchestration_layer import (
    FINAL_REPORT_PROMPT,
    INTERNAL_SEARCH_CLARIFICATION_GUIDANCE,
    INTERNAL_SEARCH_RESEARCH_TASK_GUIDANCE,
    ORCHESTRATOR_PROMPT,
    ORCHESTRATOR_PROMPT_REASONING,
    USER_ORCHESTRATOR_PROMPT,
)
from onyx.prompts.deep_research.research_agent import (
    RESEARCH_AGENT_PROMPT,
    RESEARCH_AGENT_PROMPT_REASONING,
    RESEARCH_REPORT_PROMPT,
    USER_REPORT_QUERY,
)
from onyx.prompts.regulatory_guidance import (
    REGULATORY_RESEARCH_REPORT_GUIDANCE,
    REGULATORY_SYNTHESIS_GUIDANCE,
)


def test_orchestrator_scopes_each_research_agent_to_relevant_context() -> None:
    for prompt in (ORCHESTRATOR_PROMPT, ORCHESTRATOR_PROMPT_REASONING):
        assert "one focused research fragment" in prompt
        assert "Include only the context needed" in prompt
        assert "Do not copy unrelated facts" in prompt
        assert "only for materially distinct unresolved topics" in prompt
        assert "provide all of the context" not in prompt
        assert "MANY times" not in prompt

    function_schema = cast(dict[str, Any], RESEARCH_AGENT_TOOL_DESCRIPTION["function"])
    task_schema = cast(
        dict[str, Any],
        function_schema["parameters"]["properties"][RESEARCH_AGENT_TASK_KEY],
    )
    assert task_schema["minLength"] == 1
    assert task_schema["maxLength"] == MAX_RESEARCH_AGENT_TASK_CHARS
    assert MAX_RESEARCH_AGENT_TASK_CHARS == 1200


def test_internal_only_deep_research_guidance_does_not_offer_web_search() -> None:
    assert "web search is not available" in INTERNAL_SEARCH_CLARIFICATION_GUIDANCE
    assert "only the administrator-indexed internal corpus" in (
        INTERNAL_SEARCH_RESEARCH_TASK_GUIDANCE
    )
    assert "combination of both" not in INTERNAL_SEARCH_RESEARCH_TASK_GUIDANCE


def test_unknown_provision_task_keeps_semantic_relationship_without_guessing() -> None:
    guidance = INTERNAL_SEARCH_RESEARCH_TASK_GUIDANCE

    assert "governing provision is unknown" in guidance
    assert "do not guess its number" in guidance
    assert "smallest discriminative description" in guidance
    for relationship_anchor in (
        "actor",
        "status or mechanism",
        "trigger or condition",
        "consequence or exception",
    ):
        assert relationship_anchor in guidance

    assert "TIR" not in guidance
    assert "Basel" not in guidance


def test_orchestrator_does_not_stop_on_plan_or_one_low_novelty_cycle_alone() -> None:
    for prompt in (ORCHESTRATOR_PROMPT, ORCHESTRATOR_PROMPT_REASONING):
        assert "decision signals, not independent shortcuts" in prompt
        assert "Completing the written plan" in prompt
        assert "little novelty in one cycle does not alone justify reporting" in prompt
        assert "current evidence or results expose a material unresolved" in prompt
        assert "focused research direction, query, or retrieval mode" in prompt
        assert (
            "Decide whether that distinct attempt is useful before stopping" in prompt
        )
        assert "no materially different useful attempt remains" in prompt

        assert "Basel" not in prompt
        assert "4A" not in prompt


def test_non_reasoning_agents_do_not_pay_for_mechanical_think_rounds() -> None:
    prompts = ORCHESTRATOR_PROMPT + USER_ORCHESTRATOR_PROMPT + RESEARCH_AGENT_PROMPT

    assert "Do not call it mechanically" in prompts
    assert "Do not call it merely because a search has completed" in prompts
    assert "call the generate_report tool directly" in prompts
    assert "between every call" not in prompts
    assert "after every set of searches" not in prompts


def test_nested_research_agent_does_not_silently_drop_parallel_searches() -> None:
    for prompt in (RESEARCH_AGENT_PROMPT, RESEARCH_AGENT_PROMPT_REASONING):
        assert "at most one retrieval tool call in each decision" in prompt
        assert "parent research layer handles parallel independent fragments" in prompt


def test_intermediate_report_length_is_proportional_but_final_report_stays_detailed() -> (
    None
):
    intermediate_prompt = RESEARCH_REPORT_PROMPT + "\n" + USER_REPORT_QUERY
    assert "every material sourced finding" in intermediate_prompt
    assert "make the length proportional" in intermediate_prompt
    assert "Do not target a page or word count" in intermediate_prompt
    assert "AS MUCH INFORMATION AS POSSIBLE" not in intermediate_prompt
    assert "SHOULD BE SEVERAL PAGES LONG" not in intermediate_prompt

    # Deep Research's user-facing synthesis remains deliberately detailed; only
    # per-fragment intermediate reports lose the arbitrary length target.
    assert "several pages long" in FINAL_REPORT_PROMPT


def test_final_report_preserves_source_language_and_formal_legal_register() -> None:
    assert "same language as the user's query" in FINAL_REPORT_PROMPT
    assert "formal legal register" in FINAL_REPORT_PROMPT
    assert "Do not translate quoted phrases" in FINAL_REPORT_PROMPT


def test_intermediate_report_reminder_does_not_duplicate_focused_task() -> None:
    assert "first user message above" in USER_REPORT_QUERY
    assert "{research_topic}" not in USER_REPORT_QUERY


def test_regulatory_intermediate_report_stays_within_its_focused_proposition() -> None:
    assert REGULATORY_RESEARCH_REPORT_GUIDANCE in RESEARCH_REPORT_PROMPT
    assert REGULATORY_SYNTHESIS_GUIDANCE not in RESEARCH_REPORT_PROMPT

    for required_detail in (
        "condition",
        "exception",
        "contradictory",
        "source gap",
        "proposition-level application",
    ):
        assert required_detail in RESEARCH_REPORT_PROMPT

    assert "Do not add unsupported conclusions or expand into a global analysis" in (
        RESEARCH_REPORT_PROMPT
    )
    assert "focused research topic" in RESEARCH_REPORT_PROMPT
    assert "another agent instead of a user" in RESEARCH_REPORT_PROMPT
    assert "answer the user's material requests directly" not in RESEARCH_REPORT_PROMPT
