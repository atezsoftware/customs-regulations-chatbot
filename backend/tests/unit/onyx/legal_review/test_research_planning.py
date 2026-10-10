import pytest
from pydantic import ValidationError

from onyx.legal_review.models import (
    LegalDimension,
    PassageSupport,
    ReviewDiagnosis,
    ReviewDiagnosisBatch,
    ReviewResearchTask,
)
from onyx.legal_review.research_planning import ResearchPlan


def compilation() -> ResearchPlan:
    tasks = [
        ReviewResearchTask(
            task_id=identity,
            subject="Enabling norm",
            question=question,
            dimension=dimension,
            supports=[PassageSupport(citation=index, span_number=1)],
            query=question,
            existing_need_id=None,
        )
        for index, (identity, question, dimension) in enumerate(
            [
                ("authority", "Was the power annulled?", LegalDimension.CASE_LAW),
                ("conditions", "What conditions apply?", LegalDimension.LEGAL_BASIS),
                ("amount", "What is the amount?", LegalDimension.PENALTIES),
            ],
            1,
        )
    ]
    return ResearchPlan(
        ReviewDiagnosisBatch(
            research_tasks=tasks,
            diagnoses=[
                ReviewDiagnosis(
                    kind="research",
                    assertion=task.question,
                    reason="The controlling text is missing.",
                    required_change="Investigate the material gap.",
                    supports=task.supports,
                    dimension=task.dimension,
                    research_task_ids=[task.task_id],
                    check_ids=[f"check:{task.task_id}"],
                )
                for task in tasks
            ],
        )
    )


def test_grouping_shares_operative_search_without_losing_judicial_investigation() -> (
    None
):
    plan = compilation()
    tasks = list(plan.slots.values())
    result = plan.compile(
        plan.response_model().model_validate(
            dict(
                zip(
                    plan.slots,
                    [
                        {
                            "coverage_reason": "A separate authority-change investigation.",
                            "investigations": [
                                tasks[0].model_dump(exclude={"task_id"})
                            ],
                        },
                        {
                            "coverage_reason": "Read the operative conditions and amount.",
                            "investigations": [
                                tasks[1].model_dump(exclude={"task_id"})
                            ],
                        },
                        {
                            "coverage_reason": "The same operative text supplies the amount.",
                            "investigations": [{"reuse_group": "g0002"}],
                        },
                    ],
                )
            )
        )
    )
    assert len(result.research_tasks) == 2
    assert result.diagnoses[0].research_task_ids == ["investigation_1"]
    assert result.diagnoses[1].research_task_ids == ["investigation_2"]
    assert result.diagnoses[2].research_task_ids == ["investigation_2"]
    assert [s.citation for s in result.research_tasks[1].supports] == [2, 3]
    assert {c for d in result.diagnoses for c in d.check_ids} == {
        "check:authority",
        "check:conditions",
        "check:amount",
    }


def test_grouping_cannot_omit_a_gap_or_accept_a_cycle() -> None:
    plan = compilation()
    response = plan.response_model()
    with pytest.raises(ValidationError):
        response.model_validate({})
    rows = [
        {"coverage_reason": "Shared", "investigations": [{"reuse_group": "g0003"}]}
    ] * 3
    with pytest.raises(ValueError, match="acyclic"):
        plan.compile(response.model_validate(dict(zip(plan.slots, rows))))


def test_forward_references_preserve_indirect_bindings_and_all_trigger_supports() -> (
    None
):
    plan = compilation()
    tasks = list(plan.slots.values())
    rows = [
        {
            "coverage_reason": "Shared operative rule",
            "investigations": [{"reuse_group": "g0002"}],
        },
        {
            "coverage_reason": "Shared operative rule",
            "investigations": [{"reuse_group": "g0003"}],
        },
        {
            "coverage_reason": "Complete operative rule",
            "investigations": [tasks[2].model_dump(exclude={"task_id"})],
        },
    ]
    result = plan.compile(
        plan.response_model().model_validate(dict(zip(plan.slots, rows)))
    )
    assert len(result.research_tasks) == 1
    assert all(row.research_task_ids == ["investigation_1"] for row in result.diagnoses)
    assert {support.citation for support in result.research_tasks[0].supports} == {
        1,
        2,
        3,
    }
    assert set(result.research_coverage["investigation_1"]) == {
        LegalDimension.LEGAL_BASIS,
        LegalDimension.PENALTIES,
        LegalDimension.CASE_LAW,
    }


def test_mutual_references_cannot_hide_a_cycle_behind_new_questions() -> None:
    plan = compilation()
    tasks = list(plan.slots.values())
    rows = [
        {
            "coverage_reason": "One",
            "investigations": [
                tasks[0].model_dump(exclude={"task_id"}),
                {"reuse_group": "g0002"},
            ],
        },
        {"coverage_reason": "Two", "investigations": [{"reuse_group": "g0001"}]},
        {
            "coverage_reason": "Three",
            "investigations": [tasks[2].model_dump(exclude={"task_id"})],
        },
    ]
    with pytest.raises(ValueError, match="acyclic"):
        plan.compile(plan.response_model().model_validate(dict(zip(plan.slots, rows))))


def test_composite_gap_can_split_into_independent_investigations() -> None:
    plan = compilation()
    tasks = list(plan.slots.values())
    rows = [
        {
            "coverage_reason": "Independent source targets.",
            "investigations": [
                task.model_dump(exclude={"task_id"}) for task in tasks[:2]
            ],
        },
        {
            "coverage_reason": "Covered by this group",
            "investigations": [{"reuse_group": "g0001"}],
        },
        {
            "coverage_reason": "Its distinct financial effect.",
            "investigations": [tasks[2].model_dump(exclude={"task_id"})],
        },
    ]
    result = plan.compile(
        plan.response_model().model_validate(dict(zip(plan.slots, rows)))
    )
    assert result.diagnoses[0].research_task_ids == [
        "investigation_1",
        "investigation_2",
    ]
    assert result.diagnoses[1].research_task_ids == [
        "investigation_1",
        "investigation_2",
    ]
    assert result.diagnoses[2].research_task_ids == ["investigation_3"]
