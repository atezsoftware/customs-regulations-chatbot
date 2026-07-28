import pytest
from pydantic import ValidationError

from fs_explorer_api.orchestration_models import (
    ExecutionStrategy,
    GlobalPlan,
    PlanMode,
    PlanValidationError,
    TaskSpec,
    make_direct_fallback_plan,
    resolve_global_plan,
    validate_global_plan,
)


def _task(task_id: str, *, depends_on: list[str] | None = None) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        question=f"Research {task_id}",
        purpose=f"Resolve {task_id}",
        expected_output=f"Supported findings for {task_id}",
        success_criteria=[f"{task_id} is supported by evidence"],
        depends_on=depends_on or [],
        required=True,
        as_of_date=None,
        filters=None,
    )


def _plan(
    mode: PlanMode,
    tasks: list[TaskSpec],
    *,
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
) -> GlobalPlan:
    return GlobalPlan(
        version="2",
        mode=mode,
        execution_strategy=execution_strategy,
        normalized_question="What rules apply?",
        answer_requirements=["Explain the applicable rules."],
        tasks=tasks,
        synthesis_requirements=[],
        assumptions=[],
    )


def test_strict_contract_rejects_unknown_fields() -> None:
    payload = _plan(PlanMode.DIRECT, [_task("task_1")]).model_dump()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        GlobalPlan.model_validate(payload)


def test_validate_global_plan_accepts_direct_plan() -> None:
    plan = _plan(PlanMode.DIRECT, [_task("task_1")])

    assert validate_global_plan(plan) is plan


def test_validate_global_plan_accepts_single_pass_only_for_direct_mode() -> None:
    plan = _plan(
        PlanMode.DIRECT,
        [_task("task_1")],
        execution_strategy=ExecutionStrategy.SINGLE_PASS,
    )

    assert validate_global_plan(plan) is plan


def test_validate_global_plan_rejects_single_pass_decomposed_plan() -> None:
    plan = _plan(
        PlanMode.DECOMPOSED,
        [_task("one"), _task("two")],
        execution_strategy=ExecutionStrategy.SINGLE_PASS,
    )

    with pytest.raises(PlanValidationError, match="must use adaptive execution"):
        validate_global_plan(plan)


def test_validate_global_plan_accepts_acyclic_dependency_graph() -> None:
    plan = _plan(
        PlanMode.DECOMPOSED,
        [
            _task("rule"),
            _task("exception", depends_on=["rule"]),
            _task("procedure", depends_on=["rule", "exception"]),
        ],
    )

    assert validate_global_plan(plan) is plan


@pytest.mark.parametrize(
    ("mode", "tasks", "expected_error"),
    [
        (
            PlanMode.DIRECT,
            [_task("one"), _task("two")],
            "direct mode must contain exactly one task",
        ),
        (
            PlanMode.DECOMPOSED,
            [_task("one")],
            "decomposed mode must contain at least two tasks",
        ),
    ],
)
def test_validate_global_plan_enforces_mode_task_count(
    mode: PlanMode,
    tasks: list[TaskSpec],
    expected_error: str,
) -> None:
    with pytest.raises(PlanValidationError, match=expected_error):
        validate_global_plan(_plan(mode, tasks))


def test_validate_global_plan_enforces_maximum_task_count() -> None:
    plan = _plan(PlanMode.DECOMPOSED, [_task(f"task_{i}") for i in range(4)])

    with pytest.raises(PlanValidationError) as exc_info:
        validate_global_plan(plan, max_tasks=3)

    assert "exceeds the maximum of 3 tasks" in str(exc_info.value)


def test_validate_global_plan_rejects_duplicate_ids() -> None:
    plan = _plan(PlanMode.DECOMPOSED, [_task("same"), _task("same")])

    with pytest.raises(PlanValidationError, match="task IDs must be unique"):
        validate_global_plan(plan)


@pytest.mark.parametrize("task_id", ["unsafe id", "../escape", "x" * 65])
def test_validate_global_plan_rejects_unsafe_task_ids(task_id: str) -> None:
    plan = _plan(PlanMode.DIRECT, [_task(task_id)])

    with pytest.raises(PlanValidationError, match="safe identifier"):
        validate_global_plan(plan)


def test_validate_global_plan_rejects_unknown_dependency() -> None:
    plan = _plan(
        PlanMode.DECOMPOSED,
        [_task("one"), _task("two", depends_on=["missing"])],
    )

    with pytest.raises(PlanValidationError, match="depends on unknown task"):
        validate_global_plan(plan)


def test_validate_global_plan_rejects_dependency_cycle() -> None:
    plan = _plan(
        PlanMode.DECOMPOSED,
        [
            _task("one", depends_on=["two"]),
            _task("two", depends_on=["one"]),
        ],
    )

    with pytest.raises(PlanValidationError, match="acyclic graph"):
        validate_global_plan(plan)


def test_make_direct_fallback_plan_preserves_original_question() -> None:
    plan = make_direct_fallback_plan("  What is the deadline?  ")

    assert plan.mode == PlanMode.DIRECT
    assert plan.execution_strategy == ExecutionStrategy.ADAPTIVE
    assert plan.normalized_question == "What is the deadline?"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].question == "What is the deadline?"
    assert validate_global_plan(plan) is plan


def test_resolve_global_plan_falls_back_after_validation_error() -> None:
    invalid = _plan(PlanMode.DIRECT, [_task("one"), _task("two")])

    resolution = resolve_global_plan(
        invalid,
        original_question="Original question",
    )

    assert resolution.used_fallback is True
    assert resolution.plan.mode == PlanMode.DIRECT
    assert resolution.plan.tasks[0].question == "Original question"
    assert resolution.validation_errors


def test_resolve_global_plan_falls_back_after_planner_failure() -> None:
    resolution = resolve_global_plan(
        None,
        original_question="Original question",
        planner_error="provider_timeout",
    )

    assert resolution.used_fallback is True
    assert resolution.validation_errors == ("provider_timeout",)


def test_resolve_global_plan_returns_valid_candidate_unchanged() -> None:
    candidate = _plan(PlanMode.DIRECT, [_task("one")])

    resolution = resolve_global_plan(
        candidate,
        original_question="Original question",
    )

    assert resolution.used_fallback is False
    assert resolution.plan is candidate
    assert resolution.validation_errors == ()
