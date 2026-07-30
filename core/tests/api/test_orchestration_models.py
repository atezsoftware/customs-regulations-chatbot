import pytest
from pydantic import ValidationError

from fs_explorer_api.orchestration_models import (
    AnswerRequirement,
    AnswerRequirementKind,
    ArtifactValidationError,
    DecisionBranch,
    DerivedConclusion,
    EvidenceClaim,
    EvidenceConfidence,
    EvidenceRequirement,
    EvidenceRequirementKind,
    ExecutionStrategy,
    GlobalPlan,
    MaterialUnknown,
    PlanMode,
    PlanValidationError,
    ProblemType,
    ScenarioFact,
    ScenarioSpec,
    TaskArtifact,
    TaskKind,
    TaskOutput,
    TaskOutputRef,
    TaskSpec,
    TaskStatus,
    validate_global_plan,
    validate_task_artifact,
)


def _task(
    task_id: str,
    *,
    kind: TaskKind = TaskKind.EVIDENCE,
    depends_on: list[str] | None = None,
    requirement_ids: list[str] | None = None,
    fact_ids: list[str] | None = None,
    unknown_ids: list[str] | None = None,
    branch_ids: list[str] | None = None,
    output_id: str = "result",
) -> TaskSpec:
    mapped_requirements = requirement_ids or ["req_1"]
    return TaskSpec(
        task_id=task_id,
        kind=kind,
        issue=f"Resolve {task_id}",
        search_question=(f"Research {task_id}" if kind == TaskKind.EVIDENCE else None),
        requirement_ids=mapped_requirements,
        fact_ids=fact_ids or [],
        unknown_ids=unknown_ids or [],
        branch_ids=branch_ids or [],
        evidence_requirements=(
            [
                EvidenceRequirement(
                    evidence_requirement_id=f"evidence_{task_id}",
                    kind=EvidenceRequirementKind.GOVERNING_RULE,
                    description=f"Evidence needed for {task_id}",
                    requirement_ids=mapped_requirements,
                )
            ]
            if kind == TaskKind.EVIDENCE
            else []
        ),
        consumes=[
            TaskOutputRef(task_id=dependency, output_id="result")
            for dependency in depends_on or []
        ],
        produces=[
            TaskOutput(
                output_id=output_id,
                description=f"Grounded result for {task_id}",
            )
        ],
        required=True,
        as_of_date=None,
        filters=None,
    )


def _plan(
    mode: PlanMode,
    tasks: list[TaskSpec],
    *,
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    problem_type: ProblemType | None = None,
    requirements: list[AnswerRequirement] | None = None,
    scenario: ScenarioSpec | None = None,
) -> GlobalPlan:
    return GlobalPlan(
        version="3",
        problem_type=problem_type
        or (ProblemType.LOOKUP if mode == PlanMode.DIRECT else ProblemType.MIXED),
        mode=mode,
        execution_strategy=execution_strategy,
        normalized_question="What rules apply?",
        answer_requirements=requirements
        or [
            AnswerRequirement(
                requirement_id="req_1",
                kind=AnswerRequirementKind.RULE,
                description="Explain the applicable rules.",
                required=True,
            )
        ],
        scenario=scenario,
        tasks=tasks,
        synthesis_requirements=[],
        assumptions=[],
    )


def _scenario() -> ScenarioSpec:
    return ScenarioSpec(
        jurisdiction="Türkiye",
        law_as_of_date="2026-07-29",
        facts=[
            ScenarioFact(
                fact_id="fact_import",
                description="The user imports the goods commercially.",
                requirement_ids=["req_1"],
            )
        ],
        material_unknowns=[
            MaterialUnknown(
                unknown_id="unknown_origin",
                description="The goods' preferential origin is unknown.",
                why_material="Origin may change the applicable duty treatment.",
                requirement_ids=["req_1"],
            )
        ],
        decision_branches=[
            DecisionBranch(
                branch_id="branch_preferential",
                condition="Preferential origin is documented.",
                consequence="Evaluate preferential treatment.",
                requirement_ids=["req_1"],
            )
        ],
    )


def _scenario_plan() -> GlobalPlan:
    evidence = _task(
        "rules",
        fact_ids=["fact_import"],
        unknown_ids=["unknown_origin"],
        branch_ids=["branch_preferential"],
        output_id="rules_output",
    )
    application = _task(
        "apply",
        kind=TaskKind.APPLICATION,
        depends_on=[],
        fact_ids=["fact_import"],
        unknown_ids=["unknown_origin"],
        branch_ids=["branch_preferential"],
        output_id="outcome",
    ).model_copy(
        update={"consumes": [TaskOutputRef(task_id="rules", output_id="rules_output")]}
    )
    return _plan(
        PlanMode.DECOMPOSED,
        [evidence, application],
        problem_type=ProblemType.SCENARIO_APPLICATION,
        scenario=_scenario(),
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


def test_validate_global_plan_rejects_decomposed_lookup() -> None:
    plan = _plan(
        PlanMode.DECOMPOSED,
        [_task("one"), _task("two")],
        problem_type=ProblemType.LOOKUP,
    )

    with pytest.raises(
        PlanValidationError, match="lookup plans must use one direct task"
    ):
        validate_global_plan(plan)


def test_validate_global_plan_rejects_duplicate_evidence_searches() -> None:
    first = _task("one")
    second = _task("two").model_copy(update={"search_question": first.search_question})
    plan = _plan(PlanMode.DECOMPOSED, [first, second])

    with pytest.raises(PlanValidationError, match="consolidate shared evidence"):
        validate_global_plan(plan)


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


def test_validate_global_plan_enforces_structural_list_budget() -> None:
    plan = _plan(PlanMode.DIRECT, [_task("task_1")]).model_copy(
        update={"assumptions": ["first", "second"]}
    )

    with pytest.raises(PlanValidationError, match="assumptions exceeds"):
        validate_global_plan(plan, max_list_items=1)


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

    with pytest.raises(PlanValidationError, match="consumes unknown task"):
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


def test_v3_schema_uses_flat_task_models_and_typed_ids() -> None:
    schema = GlobalPlan.model_json_schema()
    task_properties = schema["$defs"]["TaskSpec"]["properties"]

    assert schema["properties"]["version"]["const"] == "3"
    assert "kind" in task_properties
    assert "issue" in task_properties
    assert "search_question" in task_properties
    assert "requirement_ids" in task_properties
    assert "consumes" in task_properties
    assert "produces" in task_properties
    assert "question" not in task_properties
    assert "depends_on" not in task_properties
    assert "discriminator" not in str(schema)


def test_validate_global_plan_accepts_scenario_decision_graph() -> None:
    plan = _scenario_plan()

    assert validate_global_plan(plan) is plan
    assert plan.tasks[1].question == plan.tasks[1].issue
    assert plan.tasks[1].depends_on == ["rules"]


def test_validate_global_plan_rejects_scenario_single_pass() -> None:
    plan = _scenario_plan().model_copy(
        update={"execution_strategy": ExecutionStrategy.SINGLE_PASS}
    )

    with pytest.raises(PlanValidationError) as exc_info:
        validate_global_plan(plan)

    assert "decomposed mode must use adaptive execution" in str(exc_info.value)
    assert "single_pass is allowed only for one direct lookup task" in str(
        exc_info.value
    )


def test_validate_global_plan_requires_application_for_scenario() -> None:
    scenario_plan = _scenario_plan()
    plan = scenario_plan.model_copy(
        update={
            "tasks": [
                scenario_plan.tasks[0],
                _task(
                    "exceptions",
                    fact_ids=["fact_import"],
                    unknown_ids=["unknown_origin"],
                    branch_ids=["branch_preferential"],
                ),
            ]
        }
    )

    with pytest.raises(
        PlanValidationError,
        match="scenario plans require at least one application task",
    ):
        validate_global_plan(plan)


def test_validate_global_plan_requires_all_scenario_branches_to_be_applied() -> None:
    plan = _scenario_plan()
    application = plan.tasks[1].model_copy(update={"branch_ids": []})
    plan = plan.model_copy(update={"tasks": [plan.tasks[0], application]})

    with pytest.raises(
        PlanValidationError,
        match="decision branch .* is not mapped to an application task",
    ):
        validate_global_plan(plan)


def test_scenario_items_must_carry_requirements_into_their_application_task() -> None:
    plan = _scenario_plan()
    outcome = AnswerRequirement(
        requirement_id="req_outcome",
        kind=AnswerRequirementKind.OUTCOME,
        description="Apply the rule to the scenario.",
        required=True,
    )
    scenario = plan.scenario
    assert scenario is not None
    scenario = scenario.model_copy(
        update={
            "facts": [
                fact.model_copy(update={"requirement_ids": ["req_outcome"]})
                for fact in scenario.facts
            ],
            "material_unknowns": [
                unknown.model_copy(update={"requirement_ids": ["req_outcome"]})
                for unknown in scenario.material_unknowns
            ],
            "decision_branches": [
                branch.model_copy(update={"requirement_ids": ["req_outcome"]})
                for branch in scenario.decision_branches
            ],
        }
    )
    evidence = plan.tasks[0]
    evidence = evidence.model_copy(
        update={
            "requirement_ids": ["req_1", "req_outcome"],
            "evidence_requirements": [
                requirement.model_copy(
                    update={"requirement_ids": ["req_1", "req_outcome"]}
                )
                for requirement in evidence.evidence_requirements
            ],
        }
    )
    miswired = plan.model_copy(
        update={
            "answer_requirements": [*plan.answer_requirements, outcome],
            "scenario": scenario,
            "tasks": [evidence, plan.tasks[1]],
        }
    )

    with pytest.raises(
        PlanValidationError,
        match="requirement_ids are not carried by its application task: req_outcome",
    ):
        validate_global_plan(miswired)


def test_integration_task_cannot_substitute_for_scenario_application_mapping() -> None:
    plan = _scenario_plan()
    application = plan.tasks[1].model_copy(update={"unknown_ids": []})
    integration = _task(
        "integrate_unknown",
        kind=TaskKind.INTEGRATION,
        depends_on=[],
        unknown_ids=["unknown_origin"],
    ).model_copy(
        update={"consumes": [TaskOutputRef(task_id="rules", output_id="rules_output")]}
    )
    invalid = plan.model_copy(
        update={"tasks": [plan.tasks[0], application, integration]}
    )

    with pytest.raises(
        PlanValidationError,
        match="material unknown .* is not mapped to an application task",
    ):
        validate_global_plan(invalid)


@pytest.mark.parametrize(
    ("scenario_field", "value_field"),
    [
        ("facts", "description"),
        ("material_unknowns", "why_material"),
        ("decision_branches", "condition"),
    ],
)
def test_validate_global_plan_rejects_blank_scenario_semantics(
    scenario_field: str,
    value_field: str,
) -> None:
    plan = _scenario_plan()
    scenario = plan.scenario
    assert scenario is not None
    values = list(getattr(scenario, scenario_field))
    values[0] = values[0].model_copy(update={value_field: "   "})
    invalid = plan.model_copy(
        update={"scenario": scenario.model_copy(update={scenario_field: values})}
    )

    with pytest.raises(PlanValidationError, match="must not be blank"):
        validate_global_plan(invalid)


def test_validate_global_plan_rejects_unknown_scenario_reference() -> None:
    plan = _scenario_plan()
    application = plan.tasks[1].model_copy(update={"fact_ids": ["fact_missing"]})
    plan = plan.model_copy(update={"tasks": [plan.tasks[0], application]})

    with pytest.raises(PlanValidationError, match="references unknown IDs"):
        validate_global_plan(plan)


def test_validate_global_plan_requires_required_requirement_coverage() -> None:
    requirements = [
        AnswerRequirement(
            requirement_id="req_1",
            kind=AnswerRequirementKind.RULE,
            description="State the rule.",
            required=True,
        ),
        AnswerRequirement(
            requirement_id="req_2",
            kind=AnswerRequirementKind.OUTCOME,
            description="State the outcome.",
            required=True,
        ),
    ]
    plan = _plan(
        PlanMode.DIRECT,
        [_task("rules")],
        requirements=requirements,
    )

    with pytest.raises(
        PlanValidationError,
        match="required answer requirements are not mapped to tasks: req_2",
    ):
        validate_global_plan(plan)


def test_validate_global_plan_rejects_unknown_consumed_output() -> None:
    plan = _scenario_plan()
    application = plan.tasks[1].model_copy(
        update={"consumes": [TaskOutputRef(task_id="rules", output_id="not_declared")]}
    )
    plan = plan.model_copy(update={"tasks": [plan.tasks[0], application]})

    with pytest.raises(PlanValidationError, match="consumes unknown output"):
        validate_global_plan(plan)


def test_application_must_consume_an_evidence_task_output() -> None:
    plan = _scenario_plan()
    integration = _task(
        "intermediate",
        kind=TaskKind.INTEGRATION,
        fact_ids=["fact_import"],
        unknown_ids=["unknown_origin"],
        branch_ids=["branch_preferential"],
        output_id="integrated",
    ).model_copy(
        update={"consumes": [TaskOutputRef(task_id="rules", output_id="rules_output")]}
    )
    application = plan.tasks[1].model_copy(
        update={
            "consumes": [TaskOutputRef(task_id="intermediate", output_id="integrated")]
        }
    )
    plan = plan.model_copy(update={"tasks": [plan.tasks[0], integration, application]})

    with pytest.raises(
        PlanValidationError,
        match="application task .* must consume evidence task output",
    ):
        validate_global_plan(plan)


def _claim() -> EvidenceClaim:
    return EvidenceClaim(
        claim_id="claim_rules_1",
        claim="The cited provision establishes the applicable rule.",
        document_id="doc-1",
        chunk_id="chunk-1",
        readable_title="Customs Rule",
        locator="Article 1",
        evidence_excerpt="The applicable rule is established.",
        requirement_ids=["req_1"],
        evidence_requirement_ids=["evidence_rules"],
        fact_ids=["fact_import"],
        confidence=EvidenceConfidence.HIGH,
        effective_start_date=None,
        effective_end_date=None,
    )


def _evidence_artifact() -> TaskArtifact:
    return TaskArtifact(
        task_id="rules",
        status=TaskStatus.COMPLETE,
        answer_fragment="The rule is supported.",
        covered_requirement_ids=["req_1"],
        uncovered_requirement_ids=[],
        claims=[_claim()],
        application_findings=[],
        conflicts=[],
        gaps=[],
        contributing_worker_ids=["worker_1"],
    )


def _application_artifact(
    *,
    supporting_claim_ids: list[str] | None = None,
    dependency_refs: list[TaskOutputRef] | None = None,
) -> TaskArtifact:
    return TaskArtifact(
        task_id="apply",
        status=TaskStatus.COMPLETE,
        answer_fragment="The rule applies conditionally.",
        covered_requirement_ids=["req_1"],
        uncovered_requirement_ids=[],
        claims=[],
        application_findings=[
            DerivedConclusion(
                conclusion_id="conclusion_1",
                finding="The rule applies if preferential origin is documented.",
                requirement_ids=["req_1"],
                fact_ids=["fact_import"],
                branch_ids=["branch_preferential"],
                supporting_claim_ids=(
                    ["claim_rules_1"]
                    if supporting_claim_ids is None
                    else supporting_claim_ids
                ),
                dependency_refs=(
                    [TaskOutputRef(task_id="rules", output_id="rules_output")]
                    if dependency_refs is None
                    else dependency_refs
                ),
                confidence=EvidenceConfidence.HIGH,
                limitations=["Preferential origin remains unconfirmed."],
            )
        ],
        conflicts=[],
        gaps=["unknown_origin"],
        contributing_worker_ids=[],
    )


def test_validate_task_artifact_accepts_grounded_evidence_artifact() -> None:
    plan = _scenario_plan()
    artifact = _evidence_artifact()

    assert validate_task_artifact(artifact, plan=plan) is artifact


def test_validate_task_artifact_accepts_grounded_application_finding() -> None:
    plan = _scenario_plan()
    artifact = _application_artifact()

    assert (
        validate_task_artifact(
            artifact,
            plan=plan,
            dependency_artifacts=[_evidence_artifact()],
        )
        is artifact
    )


def test_integration_can_reuse_claim_grounding_from_application_findings() -> None:
    scenario_plan = _scenario_plan()
    integration = _task(
        "integrate",
        kind=TaskKind.INTEGRATION,
        depends_on=[],
        output_id="integrated",
    ).model_copy(
        update={"consumes": [TaskOutputRef(task_id="apply", output_id="outcome")]}
    )
    plan = scenario_plan.model_copy(
        update={"tasks": [*scenario_plan.tasks, integration]}
    )
    validate_global_plan(plan)
    application = _application_artifact()
    artifact = TaskArtifact(
        task_id="integrate",
        status=TaskStatus.COMPLETE,
        answer_fragment="The grounded scenario outcome is integrated.",
        covered_requirement_ids=["req_1"],
        uncovered_requirement_ids=[],
        claims=[],
        application_findings=[
            DerivedConclusion(
                conclusion_id="integrated_conclusion",
                finding="The conditional scenario outcome remains controlling.",
                requirement_ids=["req_1"],
                fact_ids=[],
                branch_ids=[],
                supporting_claim_ids=["claim_rules_1"],
                dependency_refs=[TaskOutputRef(task_id="apply", output_id="outcome")],
                confidence=EvidenceConfidence.HIGH,
                limitations=["Preferential origin remains unconfirmed."],
            )
        ],
        conflicts=[],
        gaps=[],
        contributing_worker_ids=[],
    )

    assert (
        validate_task_artifact(
            artifact,
            plan=plan,
            dependency_artifacts=[application],
        )
        is artifact
    )


def test_validate_task_artifact_rejects_unknown_dependency_claim() -> None:
    plan = _scenario_plan()
    artifact = _application_artifact(supporting_claim_ids=["claim_hallucinated"])

    with pytest.raises(
        ArtifactValidationError,
        match="supporting_claim_ids references unknown IDs",
    ):
        validate_task_artifact(
            artifact,
            plan=plan,
            dependency_artifacts=[_evidence_artifact()],
        )


def test_validate_task_artifact_rejects_unavailable_dependency_output() -> None:
    plan = _scenario_plan()
    artifact = _application_artifact(
        dependency_refs=[TaskOutputRef(task_id="rules", output_id="not_declared")]
    )

    with pytest.raises(
        ArtifactValidationError,
        match="references unavailable dependency output",
    ):
        validate_task_artifact(
            artifact,
            plan=plan,
            dependency_artifacts=[_evidence_artifact()],
        )


def test_application_finding_requires_fact_and_dependency_refs() -> None:
    plan = _scenario_plan()
    no_dependency_ref = _application_artifact(dependency_refs=[])

    with pytest.raises(
        ArtifactValidationError,
        match="must reference a dependency output",
    ):
        validate_task_artifact(
            no_dependency_ref,
            plan=plan,
            dependency_artifacts=[_evidence_artifact()],
        )

    finding = (
        _application_artifact()
        .application_findings[0]
        .model_copy(update={"fact_ids": []})
    )
    no_fact = _application_artifact().model_copy(
        update={"application_findings": [finding]}
    )
    with pytest.raises(
        ArtifactValidationError,
        match="must apply a fact",
    ):
        validate_task_artifact(
            no_fact,
            plan=plan,
            dependency_artifacts=[_evidence_artifact()],
        )


def test_compatibility_properties_do_not_expand_structured_schema() -> None:
    plan = _scenario_plan()
    artifact = _application_artifact()

    assert plan.tasks[0].question == "Research rules"
    assert plan.tasks[1].depends_on == ["rules"]
    assert artifact.covered_success_criteria == ["req_1"]
    assert artifact.uncovered_success_criteria == []
    assert "depends_on" not in plan.tasks[1].model_dump()
    assert "covered_success_criteria" not in artifact.model_dump()
