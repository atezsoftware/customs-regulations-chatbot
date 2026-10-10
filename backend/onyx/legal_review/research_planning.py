"""Batch overlapping findings into focused investigations without losing any gap."""

from graphlib import CycleError, TopologicalSorter
from types import GenericAlias
from typing import Any, Literal, Union

from pydantic import Field, JsonValue, create_model

from onyx.legal_review.models import (
    AttemptedResearchQuestion,
    FreshResearchQuestion,
    LegalDimension,
    ResearchQuestion,
    ReviewDiagnosisBatch,
    ReviewResearchTask,
    StrictModel,
)


def question_response_type(needs: list[dict[str, JsonValue]]) -> Any:
    variants: list[Any] = [FreshResearchQuestion]
    for dimension in LegalDimension:
        eligible = tuple(
            identity
            for row in needs
            if isinstance(identity := row.get("need_id"), str)
            and row.get("attempted") is True
            and row.get("origin") != "question"
            and (
                row.get("dimension") == dimension
                or isinstance(covered := row.get("covered_dimensions"), list)
                and dimension in covered
            )
        )
        if eligible:
            variants.append(
                create_model(
                    f"AttemptedQuestion_{dimension.value}",
                    __base__=AttemptedResearchQuestion,
                    dimension=(Literal.__getitem__((dimension.value,)), Field()),
                    existing_need_id=(Literal.__getitem__(eligible), Field()),
                )
            )
    return Union[tuple(variants)]


class SharedInvestigation(StrictModel):
    reuse_group: str


class InvestigationGroup(StrictModel):
    coverage_reason: str = Field(min_length=1)
    investigations: list[ResearchQuestion | SharedInvestigation] = Field(min_length=1)


class ResearchPlan:
    def __init__(self, diagnoses: ReviewDiagnosisBatch) -> None:
        self.diagnoses = diagnoses
        ordered = sorted(
            diagnoses.research_tasks,
            key=lambda task: (task.subject.casefold(), task.dimension, task.question),
        )
        self.slots = {f"g{index:04d}": task for index, task in enumerate(ordered, 1)}

    def response_model(
        self, needs: list[dict[str, JsonValue]] | None = None
    ) -> type[StrictModel]:
        reference = create_model(
            "KnownSharedInvestigation",
            __base__=SharedInvestigation,
            reuse_group=(Literal.__getitem__(tuple(self.slots)), Field()),
        )
        group = create_model(
            "BoundInvestigationGroup",
            __base__=InvestigationGroup,
            investigations=(
                GenericAlias(list, question_response_type(needs or []) | reference),
                Field(min_length=1),
            ),
        )
        fields: dict[str, Any] = {slot: (group, Field()) for slot in self.slots}
        return create_model(
            "ResearchInvestigationPlan",
            __base__=StrictModel,
            **fields,
        )

    def state(
        self, request: str, research_needs: list[JsonValue]
    ) -> dict[str, JsonValue]:
        return {
            "request": request,
            "accepted_gaps": [
                {"slot": slot, **task.model_dump(mode="json", exclude={"query"})}
                for slot, task in self.slots.items()
            ],
            "research_needs": research_needs,
        }

    def compile(self, result: StrictModel) -> ReviewDiagnosisBatch:
        rows = result.model_dump()
        if rows.keys() != self.slots.keys():
            raise ValueError("Research planning must preserve every accepted gap")
        dependencies = {
            slot: {
                raw["reuse_group"]
                for raw in row["investigations"]
                if "reuse_group" in raw
            }
            for slot, row in rows.items()
        }
        if any(references - self.slots.keys() for references in dependencies.values()):
            raise ValueError("Shared investigation must refer to a known gap")
        try:
            order = list(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            raise ValueError(
                "Shared investigation references must be acyclic"
            ) from error

        direct: dict[str, list[ReviewResearchTask]] = {}
        questions: list[ReviewResearchTask] = []
        for slot, row in rows.items():
            direct[slot] = []
            for raw in row["investigations"]:
                if "reuse_group" not in raw:
                    question = ReviewResearchTask(
                        task_id=f"investigation_{len(questions) + 1}", **raw
                    )
                    questions.append(question)
                    direct[slot].append(question)

        groups: dict[str, list[ReviewResearchTask]] = {}
        for slot in order:
            investigations = list(direct[slot])
            for raw in rows[slot]["investigations"]:
                if "reuse_group" in raw:
                    for shared in groups[raw["reuse_group"]]:
                        if shared not in investigations:
                            investigations.append(shared)
            groups[slot] = investigations

        mapped: dict[str, list[str]] = {}
        coverage: dict[str, list[LegalDimension]] = {}
        for slot, task in self.slots.items():
            for question in groups[slot]:
                dimensions = coverage.setdefault(question.task_id, [])
                for dimension in [
                    task.dimension,
                    *self.diagnoses.research_coverage.get(task.task_id, []),
                ]:
                    if dimension not in dimensions:
                        dimensions.append(dimension)
                for support in task.supports:
                    if support not in question.supports:
                        question.supports.append(support)
            mapped[task.task_id] = [question.task_id for question in groups[slot]]
        return ReviewDiagnosisBatch(
            research_tasks=questions,
            research_coverage=coverage,
            diagnoses=[
                row.model_copy(
                    update={
                        "research_task_ids": list(
                            dict.fromkeys(
                                question
                                for original in row.research_task_ids
                                for question in mapped[original]
                            )
                        )
                    }
                )
                for row in self.diagnoses.diagnoses
            ],
        )
