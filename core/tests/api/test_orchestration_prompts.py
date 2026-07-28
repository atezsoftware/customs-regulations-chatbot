from fs_explorer_api.orchestration_prompts import (
    EVIDENCE_WORKER_SYSTEM_PROMPT,
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    GLOBAL_PLANNER_SYSTEM_PROMPT,
    TASK_COORDINATOR_SYSTEM_PROMPT,
    TASK_REVIEW_SYSTEM_PROMPT,
    build_global_planner_prompt,
    build_task_coordinator_prompt,
)


def test_global_planner_prompt_routes_without_separate_intent_call() -> None:
    prompt = build_global_planner_prompt(max_tasks=4)

    assert "mode=direct with exactly one task" in prompt
    assert "mode=decomposed" in prompt
    assert "2 to\n  4 tasks" in prompt
    assert "Do not add a separate intent-analysis task" in prompt
    assert "OUTPUT CONTRACT — GlobalPlan" in prompt


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
    assert "OUTPUT CONTRACT — WorkerArtifact" in EVIDENCE_WORKER_SYSTEM_PROMPT


def test_task_review_prompt_has_strict_coverage_semantics() -> None:
    assert "Set complete only when all" in TASK_REVIEW_SYSTEM_PROMPT
    assert "partial when a useful supported conclusion exists" in (
        TASK_REVIEW_SYSTEM_PROMPT
    )
    assert "OUTPUT CONTRACT — TaskArtifact" in TASK_REVIEW_SYSTEM_PROMPT


def test_final_prompt_requires_grounded_citations_and_discloses_gaps() -> None:
    assert "[Readable Document Title, Article/Section]" in (
        FINAL_SYNTHESIS_SYSTEM_PROMPT
    )
    assert "required task is partial or failed" in FINAL_SYNTHESIS_SYSTEM_PROMPT
    assert "Never expose raw" in FINAL_SYNTHESIS_SYSTEM_PROMPT
    assert "## Sources" in FINAL_SYNTHESIS_SYSTEM_PROMPT


def test_default_prompts_remain_lean() -> None:
    prompts = [
        GLOBAL_PLANNER_SYSTEM_PROMPT,
        TASK_COORDINATOR_SYSTEM_PROMPT,
        EVIDENCE_WORKER_SYSTEM_PROMPT,
        TASK_REVIEW_SYSTEM_PROMPT,
        FINAL_SYNTHESIS_SYSTEM_PROMPT,
    ]

    assert all(len(prompt.split()) < 300 for prompt in prompts)
