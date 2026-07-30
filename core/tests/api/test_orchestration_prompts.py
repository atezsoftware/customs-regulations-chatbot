from fs_explorer_api.orchestration_prompts import (
    APPLICATION_TASK_SYSTEM_PROMPT,
    EVIDENCE_WORKER_SYSTEM_PROMPT,
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    GLOBAL_PLANNER_SYSTEM_PROMPT,
    INTEGRATION_TASK_SYSTEM_PROMPT,
    SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT,
    TASK_COORDINATOR_SYSTEM_PROMPT,
    TASK_REVIEW_SYSTEM_PROMPT,
    build_global_planner_prompt,
    build_task_coordinator_prompt,
)


def test_global_planner_prompt_routes_without_separate_intent_call() -> None:
    prompt = build_global_planner_prompt(max_tasks=4)

    assert "In this one call, understand the request" in prompt
    assert "intent-analysis task/call" in prompt
    assert "problem_type=lookup, mode=direct" in prompt
    assert "execution_strategy=single_pass only" in prompt
    assert "Decomposed plans use adaptive and contain 2 to 4 tasks" in prompt
    assert "Do not split a simple lookup" in prompt
    assert "OUTPUT CONTRACT — GlobalPlan" in prompt
    assert 'version="3"' in prompt


def test_global_planner_prompt_builds_decision_graph_not_topic_outline() -> None:
    prompt = build_global_planner_prompt()

    assert "Facts are\n  inputs, not evidence" in prompt
    assert "MaterialUnknowns" in prompt
    assert "DecisionBranches" in prompt
    assert "Consolidate shared evidence" in prompt
    assert "Application tasks do no search" in prompt
    assert "consumes={task_id, output_id}" in prompt
    assert 'BAD: independent topic tasks named "law"' in prompt
    assert "GOOD: one shared rule/exception evidence task" in prompt


def test_task_coordinator_prompt_enforces_bounded_non_recursive_fanout() -> None:
    prompt = build_task_coordinator_prompt(
        max_assignments_per_wave=2,
        max_worker_rounds=1,
    )

    assert "issue 1 to 2 independent assignments" in prompt
    assert "at most 1 waves" in prompt
    assert "Do not spawn coordinators, create recursive tasks" in prompt
    assert "OUTPUT CONTRACT — SearchAssignmentBatch" in prompt


def test_worker_prompt_distinguishes_no_evidence_from_failure() -> None:
    assert "no_evidence: search executed normally" in EVIDENCE_WORKER_SYSTEM_PROMPT
    assert "failed: a tool, provider, or infrastructure error" in (
        EVIDENCE_WORKER_SYSTEM_PROMPT
    )
    assert "Never invent, repair, or guess a source ID" in (
        EVIDENCE_WORKER_SYSTEM_PROMPT
    )
    assert "claim_id prefixed by task_id and assignment_id" in (
        EVIDENCE_WORKER_SYSTEM_PROMPT
    )
    assert "evidence_requirement_ids" in EVIDENCE_WORKER_SYSTEM_PROMPT
    assert "OUTPUT CONTRACT — WorkerArtifact" in EVIDENCE_WORKER_SYSTEM_PROMPT


def test_task_review_prompt_has_strict_coverage_semantics() -> None:
    assert "Map retained verified claims to requirement_ids" in (
        TASK_REVIEW_SYSTEM_PROMPT
    )
    assert "Set complete only when every assigned requirement ID" in (
        TASK_REVIEW_SYSTEM_PROMPT
    )
    assert "application_findings=[]" in TASK_REVIEW_SYSTEM_PROMPT
    assert "covered_requirement_ids" in TASK_REVIEW_SYSTEM_PROMPT
    assert "OUTPUT CONTRACT — TaskArtifact" in TASK_REVIEW_SYSTEM_PROMPT


def test_application_prompt_enforces_fact_claim_and_dependency_grounding() -> None:
    assert "Do not search" in APPLICATION_TASK_SYSTEM_PROMPT
    assert "User facts are inputs, not" in APPLICATION_TASK_SYSTEM_PROMPT
    assert "Never guess it" in APPLICATION_TASK_SYSTEM_PROMPT
    assert "at least one applied\n  fact_id" in APPLICATION_TASK_SYSTEM_PROMPT
    assert "supporting_claim_ids" in APPLICATION_TASK_SYSTEM_PROMPT
    assert "exact dependency_refs" in APPLICATION_TASK_SYSTEM_PROMPT
    assert "Do not create EvidenceClaims" in APPLICATION_TASK_SYSTEM_PROMPT
    assert "OUTPUT CONTRACT — TaskArtifact" in APPLICATION_TASK_SYSTEM_PROMPT


def test_integration_prompt_uses_upstream_grounding_without_requiring_facts() -> None:
    prompt = INTEGRATION_TASK_SYSTEM_PROMPT

    assert "Do not search" in prompt
    assert "supporting_claim_id already present" in prompt
    assert "dependency application finding" in prompt
    assert "fact_ids and branch_ids may be empty" in prompt
    assert "conditional application finding" in prompt


def test_final_prompt_requires_grounded_citations_and_discloses_gaps() -> None:
    assert "[Readable Document Title, Article/Section]" in (
        FINAL_SYNTHESIS_SYSTEM_PROMPT
    )
    assert "required task is partial or failed" in FINAL_SYNTHESIS_SYSTEM_PROMPT
    assert "clearly labeled missing-information section" in (
        FINAL_SYNTHESIS_SYSTEM_PROMPT
    )
    assert "Never present a partial" in FINAL_SYNTHESIS_SYSTEM_PROMPT
    assert "result as complete" in FINAL_SYNTHESIS_SYSTEM_PROMPT
    assert "Never expose raw" in FINAL_SYNTHESIS_SYSTEM_PROMPT
    assert "## Sources" in FINAL_SYNTHESIS_SYSTEM_PROMPT


def test_scenario_final_prompt_preserves_unknowns_and_decision_branches() -> None:
    prompt = SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT

    assert "ScenarioFacts are inputs and need no source citation" in prompt
    assert "Applied outcomes must come from" in prompt
    assert "MaterialUnknown or\nDecisionBranch" in prompt
    assert "never silently\nchoose a branch" in prompt
    assert "Distinguish source-backed rules, user facts" in prompt
    assert "smallest precise question(s)" in prompt
    assert "facts only the user can provide" in prompt
    assert "Never expose task, requirement, fact, claim" in prompt
    assert "## Sources" in prompt


def test_default_prompts_remain_lean() -> None:
    prompts = [
        TASK_COORDINATOR_SYSTEM_PROMPT,
        EVIDENCE_WORKER_SYSTEM_PROMPT,
        TASK_REVIEW_SYSTEM_PROMPT,
        APPLICATION_TASK_SYSTEM_PROMPT,
        INTEGRATION_TASK_SYSTEM_PROMPT,
        FINAL_SYNTHESIS_SYSTEM_PROMPT,
        SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT,
    ]

    assert all(len(prompt.split()) < 300 for prompt in prompts)
    assert len(GLOBAL_PLANNER_SYSTEM_PROMPT.split()) < 500
