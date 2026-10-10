"""Question-derived issues, early evidence review, one answer and one repair."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from datetime import date
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped
from onyx.legal_review.adjudication import cited_source_checks
from onyx.legal_review.audit import evidence_journey
from onyx.legal_review.contracts import ReadingContractError
from onyx.legal_review.dimensions import DIMENSION_GUIDANCE
from onyx.legal_review.drafting import (
    DraftEdits,
    GeneratedDraft,
    apply_draft_edits,
    compile_draft,
)
from onyx.legal_review.evidence_resolution import EvidenceResolutionPlan
from onyx.legal_review.models import (
    DimensionAssessment,
    DraftAnswer,
    EvidenceResolutionDecision,
    InitialDiscoveryPlan,
    InitialReadingDecision,
    IssueClosure,
    IssuePlan,
    LegalDimension,
    PassageSupport,
    PublicationReview,
    ReadingDecision,
    RepairReadingDecision,
    RepairResolution,
    Requirement,
    RequirementRecord,
    ResearchResolution,
    ReviewCheck,
    ReviewDiagnosis,
    ReviewDiagnosisBatch,
    ReviewResearchTask,
    ReviewResult,
    SourceAction,
    WorkflowPolicy,
    WorkflowResult,
)
from onyx.legal_review.passages import (
    CanonicalPassage,
    canonical_evidence_view,
    resolve_passage,
)
from onyx.legal_review.research import ResearchLedger
from onyx.legal_review.review_scope import scope_review_state
from onyx.legal_review.source_accounting import SourceAccountant
from onyx.legal_review.transport import model_state
from onyx.prompts.legal_review.prompts import (
    DRAFT_PROMPT,
    EVIDENCE_RESOLUTION_PROMPT,
    PLAN_PROMPT,
    READING_PROMPT,
    REPAIR_PROMPT,
    REPAIR_READING_PROMPT,
)
from onyx.regulatory.structured_llm import StructuredOutputValidationError
from onyx.tracing.answer_graph import graph_step
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
SourceTriggerBinding = tuple[
    RequirementRecord, tuple[DimensionAssessment, ...], tuple[CanonicalPassage, ...]
]


class ModelGateway(Protocol):
    def complete(
        self,
        prompt: str,
        state: dict[str, JsonValue],
        response_model: type[ResponseModel],
        flow: LLMFlow,
        *,
        finalizing: bool = False,
    ) -> ResponseModel: ...


class Acquirer(Protocol):
    receipts: list[dict[str, JsonValue]]
    searches: int

    def definitions(self) -> list[dict[str, JsonValue]]: ...

    def acquire(
        self, actions: list[SourceAction], plan: IssuePlan, *, finalizing: bool = False
    ) -> None: ...


class Reviewer(Protocol):
    def review(
        self,
        state: dict[str, JsonValue],
        checks: Sequence[ReviewCheck],
        timeout_seconds: float,
    ) -> ReviewResult: ...


class Diagnoser(Protocol):
    def diagnose(
        self,
        state: dict[str, JsonValue],
        checks: Sequence[ReviewCheck],
        timeout_seconds: float,
    ) -> ReviewDiagnosisBatch: ...

    def examine(
        self,
        state: dict[str, JsonValue],
        checks: Sequence[ReviewCheck],
        timeout_seconds: float,
    ) -> PublicationReview: ...


def validate_support(support: PassageSupport, ledger: EvidenceLedger) -> None:
    resolve_passage(support, ledger)


def recorded_validity(
    requirement: Requirement, ledger: EvidenceLedger, event_date: JsonValue = None
) -> str:
    statuses = []
    for support in requirement.supports:
        item = ledger.get(support.citation)
        assert item is not None
        start = item.metadata.get("validity_start")
        end = item.metadata.get("validity_end")
        as_of = item.metadata.get("read_as_of_date")
        if not start or not as_of or not event_date or str(event_date) != str(as_of):
            statuses.append("unknown")
            continue
        try:
            event = date.fromisoformat(str(as_of))
            lower = date.fromisoformat(str(start))
            upper = date.fromisoformat(str(end)) if end else None
        except ValueError:
            statuses.append("unknown")
            continue
        statuses.append(
            "within_recorded_window"
            if lower <= event and (upper is None or event < upper)
            else "outside_recorded_window"
        )
    if "outside_recorded_window" in statuses:
        return "outside_recorded_window"
    return "unknown" if "unknown" in statuses else "within_recorded_window"


def recorded_legal_status(requirement: Requirement, ledger: EvidenceLedger) -> str:
    statuses: set[str] = set()
    for support in requirement.supports:
        item = ledger.get(support.citation)
        assert item is not None
        canonical = item.metadata.get("canonical_metadata")
        layers = [item.metadata, canonical if isinstance(canonical, dict) else {}]
        verified = next(
            (
                str(layer["legal_status"])
                for layer in layers
                if layer.get("legal_status_verified") is True
                and layer.get("legal_status") in {"in_force", "amended", "annulled"}
            ),
            "unknown",
        )
        statuses.add(verified)
    if "annulled" in statuses:
        return "annulled"
    if len(statuses) != 1:
        return "unknown"
    return next(iter(statuses), "unknown")


class LegalReviewEngine:
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        acquirer: Acquirer,
        reviewer: Reviewer,
        ledger: EvidenceLedger,
        context: RunContext,
        policy: WorkflowPolicy,
        diagnoser: Diagnoser | None = None,
        source_accountant: SourceAccountant | None = None,
        report: Callable[[str, str], None] = lambda _phase, _language: None,
        record_review_usage: Callable[[int, int], None] = lambda _input, _output: None,
        complete_articles: Callable[[set[int], bool], bool] | None = None,
    ) -> None:
        self.gateway = gateway
        self.acquirer = acquirer
        self.reviewer = reviewer
        self.ledger = ledger
        self.context = context
        self.policy = policy
        self.diagnoser = diagnoser
        self.source_accountant = source_accountant
        self.complete_articles = complete_articles
        self.source_assessments: list[dict[str, JsonValue]] = []
        self.review_diagnoses: ReviewDiagnosisBatch | None = None
        self.final_adjudication: PublicationReview | None = None
        self.draft_adjudication: PublicationReview | None = None
        self.review_work_history: list[JsonValue] = []
        self.research_resolutions: list[ResearchResolution] = []
        self.research = ResearchLedger()
        self._review_receipts_start = 0
        self.report = report
        self.record_review_usage = record_review_usage
        self.plan: IssuePlan | None = None
        self.requirements: dict[str, RequirementRecord] = {}
        self.requirement_history: dict[str, RequirementRecord] = {}
        self.dimensions: list[DimensionAssessment] = []
        self.requirement_associations: dict[
            tuple[str, str], tuple[DimensionAssessment, ...]
        ] = {}
        self.source_issue_triggers: dict[str, tuple[SourceTriggerBinding, ...]] = {}
        self.gaps: list[str] = []
        self.early_review: ReviewResult | None = None
        self.final_review: ReviewResult | None = None
        self.repair_used = False
        self.reading_correction_used = False
        self.pending_actions: list[SourceAction] = []
        self.last_draft: DraftAnswer | None = None
        self.repair_checks: dict[str, ReviewCheck] = {}
        self.repair_resolutions: list[RepairResolution] = []
        self._repair_requirements: dict[str, RequirementRecord] = {}
        self._repair_dimensions: dict[
            tuple[str, LegalDimension], DimensionAssessment
        ] = {}
        self._repair_check_issues: dict[str, set[str]] = {}

    def state(
        self, request: str, history: str, draft: DraftAnswer | None = None
    ) -> dict[str, JsonValue]:
        originals: list[JsonValue] = list(canonical_evidence_view(self.ledger))
        full_required = (
            self._required_assessment_issue_ids(self.plan) if self.plan else []
        )
        full_required_set = set(full_required)
        return {
            "request": request,
            "history": history,
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "standard_dimensions": [dimension.value for dimension in LegalDimension],
            "corpus_currency": {
                "assume_current_versions": self.policy.assume_current_corpus,
                "basis": "User-supplied current corpus; explicit source effects and event dates still govern",
            },
            "dimension_guidance": {
                dimension.value: guidance
                for dimension, guidance in DIMENSION_GUIDANCE.items()
            },
            "reading_contract": {
                "mode": "assessment_update"
                if self.dimensions
                else "initial_assessment",
                "full_matrix_required_issue_ids": full_required,
                "sparse_update_issue_ids": [
                    issue.issue_id
                    for issue in self.plan.issues
                    if issue.issue_id not in full_required_set
                ]
                if self.plan
                else [],
                "new_source_issues_require_full_matrix": True,
            },
            "dimension_assessments": [
                assessment.model_dump(mode="json") for assessment in self.dimensions
            ],
            "requirements": [
                requirement.model_dump(mode="json")
                for requirement in self.requirements.values()
            ],
            "original_evidence": originals,
            "source_assessments": list(self.source_assessments),
            "material_source_leads": [
                row
                for row in self.source_assessments
                if row["disposition"] in {"material_limitation", "needs_operative_read"}
            ],
            "source_issue_triggers": self._source_trigger_state(),
            "source_operations": list(self.acquirer.receipts),
            "evidence_gaps": list(self.gaps),
            "issue_closures": [
                row.model_dump(mode="json") for row in self.issue_closures()
            ],
            "pending_source_operations": [
                action.model_dump(mode="json") for action in self.pending_actions
            ],
            "limits": {
                "remaining_research_seconds": max(
                    0, self.context.research_deadline - time.monotonic()
                )
                if math.isfinite(self.context.research_deadline)
                else None,
                "searches_performed": self.acquirer.searches,
                "max_searches_per_research_need": 2,
                "remaining_total_seconds": max(
                    0, self.context.deadline - time.monotonic()
                )
                if math.isfinite(self.context.deadline)
                else None,
                "max_postdraft_repairs": 1,
                "remaining_reading_contract_corrections": int(
                    not self.reading_correction_used
                ),
            },
            "tools": self.acquirer.definitions(),
            "draft": draft.model_dump(mode="json") if draft else None,
            "early_review": self.early_review.model_dump(mode="json")
            if self.early_review
            else None,
            "final_review": self.final_review.model_dump(mode="json")
            if self.final_review
            else None,
            "repair_contract": {
                "required_check_ids": list(self.repair_checks),
                "grouped_check_ids_allowed": True,
            },
            "repair_resolutions": [
                row.model_dump(mode="json") for row in self.repair_resolutions
            ],
            "review_diagnoses": self.review_diagnoses.model_dump(mode="json")
            if self.review_diagnoses
            else None,
            "final_adjudication": self.final_adjudication.model_dump(mode="json")
            if self.final_adjudication
            else None,
            "draft_adjudication": self.draft_adjudication.model_dump(mode="json")
            if self.draft_adjudication
            else None,
            "research_resolutions": [
                row.model_dump(mode="json") for row in self.research_resolutions
            ],
            "research_needs": [
                row.model_dump(mode="json") for row in self.research.needs.values()
            ],
            "skipped_searches": list(self.research.skipped),
        }

    def _stage_requirements(
        self, incoming: list[Requirement]
    ) -> tuple[dict[str, RequirementRecord], dict[str, RequirementRecord]]:
        records: dict[str, RequirementRecord] = {}
        for requirement in incoming:
            identity = requirement.requirement_id
            if identity in records:
                raise ValueError("Duplicate requirement identity")
            for support in requirement.supports:
                validate_support(support, self.ledger)
            record = RequirementRecord.model_validate(
                {
                    **requirement.model_dump(mode="python"),
                    "validity": recorded_validity(
                        requirement, self.ledger, self.context.scope.get("as_of_date")
                    ),
                    "legal_status": recorded_legal_status(requirement, self.ledger),
                }
            )
            if record.validity == "outside_recorded_window":
                raise ValueError(
                    "Requirement is outside its recorded event-date window"
                )
            existing = self.requirements.get(identity) or self.requirement_history.get(
                identity
            )
            if existing is not None and existing != record:
                raise ValueError("Requirement identity is immutable")
            records[identity] = record
        pending = {
            **self.requirements,
            **{
                identity: record
                for identity, record in records.items()
                if identity not in self.requirement_history
            },
        }
        replacements: dict[str, str] = {}
        edges: dict[str, list[str]] = {}
        for identity, record in records.items():
            # Unchanged active or archived echoes never repeat a version operation.
            if identity in self.requirements or identity in self.requirement_history:
                continue
            edges[identity] = list(record.supersedes_requirement_ids)
            for previous in record.supersedes_requirement_ids:
                if previous == identity:
                    raise ValueError("Requirement supersession cannot replace itself")
                if previous not in pending:
                    raise ValueError(
                        "Supersession must replace a known active or batch requirement"
                    )
                if previous in replacements and replacements[previous] != identity:
                    raise ValueError(
                        "Requirement supersession has ambiguous replacements"
                    )
                replacements[previous] = identity
        indegrees = dict.fromkeys(pending, 0)
        for previous_ids in edges.values():
            for previous in previous_ids:
                indegrees[previous] += 1
        ready = [identity for identity, degree in indegrees.items() if degree == 0]
        visited = 0
        while ready:
            identity = ready.pop()
            visited += 1
            for previous in edges.get(identity, []):
                indegrees[previous] -= 1
                if indegrees[previous] == 0:
                    ready.append(previous)
        if visited != len(indegrees):
            raise ValueError("Requirement supersession must be acyclic")
        history = dict(self.requirement_history)
        for previous in replacements:
            history[previous] = pending.pop(previous)
        return pending, history

    def _required_assessment_issue_ids(self, proposed: IssuePlan) -> list[str]:
        previous = {(row.issue_id, row.dimension) for row in self.dimensions}
        return [
            issue.issue_id
            for issue in proposed.issues
            if any(
                (issue.issue_id, dimension) not in previous
                for dimension in LegalDimension
            )
        ]

    def _stage_dimensions(
        self,
        proposed: IssuePlan,
        updates: list[DimensionAssessment],
        requirements: dict[str, RequirementRecord],
        history: dict[str, RequirementRecord],
    ) -> list[DimensionAssessment]:
        known = {issue.issue_id for issue in proposed.issues}
        changes: dict[tuple[str, LegalDimension], DimensionAssessment] = {}
        for row in updates:
            key = (row.issue_id, row.dimension)
            if row.issue_id not in known:
                raise ValueError("Dimension assessment refers to an unknown issue")
            if key in changes:
                raise ValueError(
                    "Dimension assessments must be unique per issue and dimension"
                )
            changes[key] = row
        required = {
            (issue_id, dimension)
            for issue_id in self._required_assessment_issue_ids(proposed)
            for dimension in LegalDimension
        }
        missing = required - changes.keys()
        if missing:
            raise ValueError(
                "Initial and new issues require every standard dimension exactly once: "
                + ", ".join(
                    f"{issue_id}:{dimension.value}"
                    for issue_id, dimension in sorted(missing)
                )
            )
        previous = {(row.issue_id, row.dimension): row for row in self.dimensions}
        result: list[DimensionAssessment] = []
        for issue in proposed.issues:
            for dimension in LegalDimension:
                key = (issue.issue_id, dimension)
                row = changes.get(key) or previous.get(key)
                assert row is not None
                row = row.model_copy(deep=True)
                if any(
                    identity not in requirements and identity not in history
                    for identity in row.requirement_ids
                ):
                    raise ValueError(
                        "Dimension assessment refers to an unknown finding"
                    )
                obsolete = [
                    identity
                    for identity in row.requirement_ids
                    if identity not in requirements
                ]
                if obsolete:
                    row.status = "unresolved"
                    row.reason = (
                        "Referenced findings were superseded and their application needs "
                        f"reassessment: {', '.join(obsolete)}. " + row.reason
                    )
                    row.requirement_ids = [
                        identity
                        for identity in row.requirement_ids
                        if identity in requirements
                    ]
                if row.status == "addressed" and not row.requirement_ids:
                    raise ValueError(
                        "Addressed dimensions require original-backed findings"
                    )
                if row.status == "addressed" and any(
                    requirements[identity].legal_status == "annulled"
                    for identity in row.requirement_ids
                ):
                    raise ValueError(
                        "An annulled requirement cannot close an operative dimension"
                    )
                result.append(row)
        return result

    def _accept_reading(self, decision: ReadingDecision) -> None:
        assert self.plan is not None
        if any(issue.origin != "source" for issue in decision.additional_issues):
            raise ValueError(
                "Additional issues must be material source-derived dependencies"
            )
        proposed = IssuePlan.model_validate(
            {
                **self.plan.model_dump(mode="python"),
                "issues": [*self.plan.issues, *decision.additional_issues],
            }
        )
        pending, history = self._stage_requirements(decision.requirements)
        dimensions = self._stage_dimensions(
            proposed, decision.dimensions, pending, history
        )
        triggers = self._validate_source_issues(proposed, pending, history, dimensions)
        if isinstance(decision, RepairReadingDecision):
            self._validate_repair_resolutions(decision, pending, dimensions)
        if isinstance(decision, (EvidenceResolutionDecision, RepairReadingDecision)):
            self._validate_research_resolutions(decision, proposed)
        associations = dict(self.requirement_associations)
        for row in dimensions:
            for identity in row.requirement_ids:
                key = (row.issue_id, identity)
                previous = associations.get(key, ())
                if row not in previous:
                    associations[key] = (*previous, row.model_copy(deep=True))
        actions = list(decision.actions)
        for issue in decision.additional_issues:
            actions.extend(
                SourceAction(
                    issue_ids=[issue.issue_id],
                    tool="search_corpus",
                    arguments={"query": query, "mode": "hybrid"},
                )
                for query in issue.research_queries
            )
        known = {issue.issue_id for issue in proposed.issues}
        combined: dict[tuple[str, str], SourceAction] = {}
        for action in actions:
            if set(action.issue_ids) - known:
                raise ValueError("Source action refers to an unknown issue")
            key = (
                action.tool,
                json.dumps(action.arguments, sort_keys=True, ensure_ascii=False),
            )
            previous_action = combined.get(key)
            if previous_action is None:
                combined[key] = action.model_copy(deep=True)
            else:
                previous_action.issue_ids = list(
                    dict.fromkeys([*previous_action.issue_ids, *action.issue_ids])
                )
        self.requirements = pending
        self.requirement_history = history
        self.dimensions = dimensions
        self.requirement_associations = associations
        self.source_issue_triggers = triggers
        if isinstance(decision, RepairReadingDecision):
            self.repair_resolutions = [
                row.model_copy(deep=True) for row in decision.repair_resolutions
            ]
        if isinstance(decision, (EvidenceResolutionDecision, RepairReadingDecision)):
            self.research_resolutions = [
                row.model_copy(deep=True) for row in decision.research_resolutions
            ]
        self.gaps = list(
            dict.fromkeys(
                [
                    *decision.evidence_gaps,
                    *(
                        row.correction
                        for row in self.repair_resolutions
                        if row.disposition == "unresolved"
                    ),
                ]
            )
        )
        self.plan = proposed
        decision.actions = list(combined.values())
        self.pending_actions = list(decision.actions)

    def _start_repair(
        self,
        draft: DraftAnswer | None,
        review: ReviewResult | None = None,
    ) -> None:
        if self.repair_checks and review is None:
            return
        selected_review = review or self.final_review
        assert self.plan is not None and selected_review is not None
        if self.repair_checks:
            self.review_work_history.append(
                {
                    "checks": [
                        check.model_dump(mode="json")
                        for check in self.repair_checks.values()
                    ],
                    "diagnoses": self.review_diagnoses.model_dump(mode="json")
                    if self.review_diagnoses
                    else None,
                    "resolutions": [
                        row.model_dump(mode="json") for row in self.repair_resolutions
                    ],
                    "research_resolutions": [
                        row.model_dump(mode="json") for row in self.research_resolutions
                    ],
                }
            )
        self.repair_checks = {
            check.id: check for check in self._review_work(selected_review)
        }
        self.repair_resolutions = []
        self.review_diagnoses = None
        self.research_resolutions = []
        self._review_receipts_start = len(self.acquirer.receipts)
        self._repair_check_issues = {}
        self._repair_requirements = {
            identity: record.model_copy(deep=True)
            for identity, record in self.requirements.items()
        }
        self._repair_dimensions = {
            (row.issue_id, row.dimension): row.model_copy(deep=True)
            for row in self.dimensions
        }
        claims = {claim.claim_id: claim for claim in draft.claims} if draft else {}
        for check in self.repair_checks.values():
            if check.issue_id is not None:
                issues = {check.issue_id}
            elif check.requirement_id is not None:
                issues = {
                    row.issue_id
                    for row in self.dimensions
                    if check.requirement_id in row.requirement_ids
                }
            elif check.claim_id is not None and check.claim_id in claims:
                issues = set(claims[check.claim_id].issue_ids)
            else:
                issues = {issue.issue_id for issue in self.plan.issues}
            self._repair_check_issues[check.id] = issues

    def _diagnose_review(
        self,
        request: str,
        history: str,
        draft: DraftAnswer | None,
        review: ReviewResult,
    ) -> None:
        self._start_repair(draft, review)
        if self.diagnoser is None:
            return
        deadline = (
            self.context.deadline - self.policy.publication_reserve_seconds - 60
            if draft
            else self.context.research_deadline
        )
        confirmed = (
            {
                row.check_id: row
                for row in self.draft_adjudication.findings
                if row.disposition == "defect"
            }
            if draft is not None and self.draft_adjudication is not None
            else {}
        )
        checks = self._review_work(review)
        if confirmed and all(
            check.id in confirmed and confirmed[check.id].repair_kind is not None
            for check in checks
        ):
            tasks = {
                check.id: ReviewResearchTask(
                    task_id=f"publication:{check.id}",
                    subject=confirmed[check.id].research_query or check.id,
                    question=confirmed[check.id].required_change
                    or confirmed[check.id].reason,
                    query=confirmed[check.id].research_query,
                    dimension=LegalDimension(check.dimension)
                    if check.dimension is not None
                    else LegalDimension.LEGAL_BASIS,
                    supports=confirmed[check.id].supports,
                    existing_need_id=None,
                )
                for check in checks
                if confirmed[check.id].repair_kind == "research"
            }
            diagnoses = ReviewDiagnosisBatch(
                diagnoses=[
                    ReviewDiagnosis(
                        check_ids=[check.id],
                        kind="research" if check.id in tasks else "correction",
                        dimension=LegalDimension(check.dimension)
                        if check.dimension is not None
                        else None,
                        assertion="\n\n".join(confirmed[check.id].answer_quotes),
                        reason=confirmed[check.id].reason,
                        required_change=confirmed[check.id].required_change
                        or "Correct the identified defect",
                        supports=confirmed[check.id].supports,
                        research_task_ids=[tasks[check.id].task_id]
                        if check.id in tasks
                        else [],
                    )
                    for check in checks
                ],
                research_tasks=list(tasks.values()),
            )
        else:
            diagnoses = self.diagnoser.diagnose(
                self.state(request, history, draft),
                checks,
                max(0, deadline - time.monotonic())
                if math.isfinite(deadline)
                else self.policy.max_call_seconds,
            )
        self._validate_diagnoses(diagnoses, self._review_work(review))
        self.review_diagnoses = diagnoses
        with graph_step("legal_review.review_diagnoses", {}) as step:
            step.output_value = diagnoses.model_dump(mode="json")
        actions: dict[str, SourceAction] = {}
        for task in diagnoses.research_tasks:
            check_ids = {
                check_id
                for row in diagnoses.diagnoses
                if task.task_id in row.research_task_ids
                for check_id in row.check_ids
            }
            issue_ids = sorted(
                {
                    issue_id
                    for check_id in check_ids
                    for issue_id in self._repair_check_issues[check_id]
                }
            )
            need = self.research.bind_task(
                task,
                issue_ids,
                covered_dimensions=diagnoses.research_coverage.get(task.task_id, []),
            )
            if task.query:
                query = task.query
                key = " ".join(query.casefold().split())
                if key in actions:
                    actions[key].issue_ids = sorted(
                        set(actions[key].issue_ids) | set(issue_ids)
                    )
                    actions[key].research_need_ids.append(need.need_id)
                    for field, value in (
                        ("coverage_item", task.subject),
                        ("evidence_target", task.question),
                    ):
                        previous = str(actions[key].arguments[field])
                        if value not in previous.split("\n\n"):
                            actions[key].arguments[field] = previous + "\n\n" + value
                else:
                    actions[key] = SourceAction(
                        tool="search_corpus",
                        arguments={
                            "query": query,
                            "mode": "hybrid",
                            "coverage_item": task.subject,
                            "evidence_target": task.question,
                        },
                        issue_ids=issue_ids,
                        retry_reason=task.question if need.attempted else None,
                        research_need_ids=[need.need_id],
                    )
        if actions:
            assert self.plan is not None
            self.report("tools", self.plan.language)
            self._acquire(list(actions.values()), finalizing=draft is not None)

    def _validate_diagnoses(
        self, diagnoses: ReviewDiagnosisBatch, checks: Sequence[ReviewCheck]
    ) -> None:
        identities = [
            identity for row in diagnoses.diagnoses for identity in row.check_ids
        ]
        if len(identities) != len(set(identities)) or set(identities) != {
            check.id for check in checks
        }:
            raise ValueError(
                "Independent diagnosis must cover the exact review inventory"
            )
        for row in diagnoses.diagnoses:
            for check in checks:
                if (
                    check.id in row.check_ids
                    and check.dimension is not None
                    and row.dimension != check.dimension
                ):
                    raise ValueError("Diagnosis must retain the assessed dimension")
            for support in row.supports:
                validate_support(support, self.ledger)
        for task in diagnoses.research_tasks:
            for support in task.supports:
                validate_support(support, self.ledger)

    def _final_review_is_publishable(
        self, request: str, history: str, draft: DraftAnswer
    ) -> bool:
        self.final_adjudication = None
        review = self.final_review
        if review is None or not review.completed:
            return False
        if not review.flags:
            return True
        if self.diagnoser is None:
            return False
        self.context.check_active()
        adjudication = self.diagnoser.examine(
            self.state(request, history, draft),
            review.flags,
            max(0, self.context.deadline - time.monotonic())
            if math.isfinite(self.context.deadline)
            else self.policy.max_call_seconds,
        )
        self._validate_publication_review(adjudication, review.flags, draft)
        self.final_adjudication = adjudication
        with graph_step("legal_review.final_adjudication", {}) as step:
            step.output_value = adjudication.model_dump(mode="json")
        self.context.check_active()
        return all(row.disposition != "defect" for row in adjudication.findings)

    def _validate_publication_review(
        self,
        adjudication: PublicationReview,
        checks: Sequence[ReviewCheck],
        draft: DraftAnswer,
    ) -> None:
        identities = [row.check_id for row in adjudication.findings]
        if len(identities) != len(set(identities)) or set(identities) != {
            check.id for check in checks
        }:
            raise ValueError(
                "Publication adjudication must cover the exact review inventory"
            )
        for row in adjudication.findings:
            if any(quote not in draft.answer for quote in row.answer_quotes):
                raise ValueError(
                    "Publication assessment quotes text absent from the answer"
                )
            for support in row.supports:
                validate_support(support, self.ledger)

    def _validate_repair_resolutions(
        self,
        decision: RepairReadingDecision,
        requirements: dict[str, RequirementRecord],
        dimensions: list[DimensionAssessment],
    ) -> None:
        identities = [
            identity
            for resolution in decision.repair_resolutions
            for identity in resolution.check_ids
        ]
        if (
            not self.repair_checks
            or len(identities) != len(set(identities))
            or set(identities) != set(self.repair_checks)
        ):
            raise ValueError(
                "Repair resolutions must cover every flagged check exactly once"
            )
        rows = {(row.issue_id, row.dimension): row for row in dimensions}
        changed_rows = {
            key for key, row in rows.items() if self._repair_dimensions.get(key) != row
        }
        research_changed = requirements != self._repair_requirements or bool(
            changed_rows
        )
        for resolution in decision.repair_resolutions:
            for support in resolution.supports:
                validate_support(support, self.ledger)
            if resolution.disposition != "correct":
                continue
            for identity in resolution.check_ids:
                check = self.repair_checks[identity]
                if check.requirement_id is not None:
                    bound_rows = {
                        key
                        for key, row in self._repair_dimensions.items()
                        if check.requirement_id in row.requirement_ids
                    }
                    finding_changed = requirements.get(
                        check.requirement_id
                    ) != self._repair_requirements.get(check.requirement_id)
                    if resolution.scope != "research" or not (
                        finding_changed or changed_rows & bound_rows
                    ):
                        raise ValueError(
                            "A corrected finding flag requires a changed finding or linked assessment"
                        )
                elif resolution.scope == "research":
                    if check.issue_id is not None and check.dimension is not None:
                        changed = (
                            check.issue_id,
                            LegalDimension(check.dimension),
                        ) in changed_rows
                    else:
                        changed = research_changed
                    if not changed:
                        raise ValueError(
                            "A research correction must change its bound research state"
                        )

    def _unresolved_repair_issue_ids(self) -> set[str]:
        return {
            issue_id
            for resolution in self.repair_resolutions
            if resolution.disposition == "unresolved"
            for identity in resolution.check_ids
            for issue_id in self._repair_check_issues.get(identity, ())
        } | {
            issue_id
            for row in self.research_resolutions
            if row.disposition
            in {"request_sources", "needs_user_facts", "exhausted", "unresolved"}
            for issue_id in row.issue_ids
        }

    def _validate_research_resolutions(
        self,
        decision: EvidenceResolutionDecision | RepairReadingDecision,
        proposed: IssuePlan,
    ) -> None:
        questions = (
            [row for row in self.review_diagnoses.diagnoses if row.kind == "research"]
            if self.review_diagnoses
            else []
        )
        expected = {identity for row in questions for identity in row.check_ids}
        supplied = [
            identity
            for row in decision.research_resolutions
            for identity in row.check_ids
        ]
        if len(supplied) != len(set(supplied)) or set(supplied) != expected:
            raise ValueError(
                "Every independent research question requires its own source-work disposition; a corrected assessment or caveat does not resolve it"
            )
        known = {issue.issue_id for issue in proposed.issues}
        actionable = {
            identity for action in decision.actions for identity in action.issue_ids
        } | {
            issue.issue_id
            for issue in decision.additional_issues
            if issue.research_queries
        }
        for row in decision.research_resolutions:
            if set(row.issue_ids) - known:
                raise ValueError("Research question refers to an unknown issue")
            for support in row.supports:
                validate_support(support, self.ledger)
            if (
                row.disposition == "request_sources"
                and not set(row.issue_ids) <= actionable
            ):
                raise ValueError(
                    "A request_sources disposition must include executable source operations for its issues"
                )
            task_ids = {
                task_id
                for question in questions
                if set(question.check_ids) & set(row.check_ids)
                for task_id in question.research_task_ids
            }
            bound_needs = (
                {
                    task.existing_need_id
                    for task in self.review_diagnoses.research_tasks
                    if task.task_id in task_ids and task.existing_need_id is not None
                }
                if self.review_diagnoses
                else set()
            )
            read_citations: set[int] = set()
            for index, receipt in enumerate(self.acquirer.receipts):
                receipt_issues = receipt.get("issue_ids")
                evidence_ids = receipt.get("evidence_ids")
                receipt_needs = receipt.get("research_need_ids")
                bound_investigation = isinstance(receipt_needs, list) and any(
                    identity in bound_needs
                    for identity in receipt_needs
                    if isinstance(identity, str)
                )
                continuation = (
                    index >= self._review_receipts_start
                    and isinstance(receipt.get("tool"), str)
                    and receipt.get("tool") != "search_corpus"
                )
                if (
                    (bound_investigation or continuation)
                    and receipt.get("status") in {"found", "partial"}
                    and isinstance(receipt_issues, list)
                    and any(identity in receipt_issues for identity in row.issue_ids)
                    and isinstance(evidence_ids, list)
                ):
                    read_citations.update(
                        identity
                        for identity in evidence_ids
                        if isinstance(identity, int)
                    )
            if row.disposition in {"resolved", "disputed"} and not any(
                support.citation in read_citations for support in row.supports
            ):
                raise ValueError(
                    f"Research resolution {row.check_ids} needs supporting originals from an executed source operation bound to its research need, or a subsequent canonical continuation; earlier bound search results remain eligible"
                )
            if row.disposition == "needs_user_facts" and not row.missing_user_facts:
                raise ValueError(
                    "A user-fact blocker must identify the actual missing facts"
                )
            if row.disposition == "exhausted" and not (
                math.isfinite(self.context.research_deadline)
                and self.context.research_deadline - time.monotonic()
                <= self.policy.max_call_seconds
            ):
                raise ValueError(
                    "There is no exhausted execution limit; resolve the research question or request sources"
                )

    def _acquire(self, actions: list[SourceAction], *, finalizing: bool = False) -> int:
        assert self.plan is not None
        admitted = self.research.admit(actions, self.plan)
        self.pending_actions = admitted
        if not admitted:
            return 0
        before = set(self.ledger.citation_numbers())
        receipt_start = len(self.acquirer.receipts)
        self.acquirer.acquire(admitted, self.plan, finalizing=finalizing)
        self.research.record_results(
            admitted, self.acquirer.receipts[receipt_start:], before
        )
        self.pending_actions = []
        return len(admitted)

    def _validate_source_issues(
        self,
        proposed: IssuePlan,
        requirements: dict[str, RequirementRecord],
        history: dict[str, RequirementRecord],
        dimensions: list[DimensionAssessment],
    ) -> dict[str, tuple[SourceTriggerBinding, ...]]:
        triggers = dict(self.source_issue_triggers)
        for issue in proposed.issues:
            if issue.origin != "source":
                continue
            existing = triggers.get(issue.issue_id)
            if existing is not None:
                if {row.requirement_id for row, _, _ in existing} != set(
                    issue.supporting_requirement_ids
                ) or any(
                    row.issue_id != issue.parent_issue_id
                    for _, associations, _ in existing
                    for row in associations
                ):
                    raise ValueError(
                        "Adopted source issue trigger identity is immutable"
                    )
                for record, _, _ in existing:
                    for support in record.supports:
                        validate_support(support, self.ledger)
                continue
            bindings: list[SourceTriggerBinding] = []
            for identity in dict.fromkeys(issue.supporting_requirement_ids):
                requirement = requirements.get(identity) or history.get(identity)
                associations = tuple(
                    row.model_copy(deep=True)
                    for row in dimensions
                    if row.issue_id == issue.parent_issue_id
                    and identity in row.requirement_ids
                ) or self.requirement_associations.get(
                    (issue.parent_issue_id or "", identity), ()
                )
                if requirement is None or not associations:
                    raise ValueError(
                        "Source issue must be triggered by an exact parent-backed requirement association"
                    )
                for support in requirement.supports:
                    validate_support(support, self.ledger)
                bindings.append(
                    (
                        requirement.model_copy(deep=True),
                        tuple(row.model_copy(deep=True) for row in associations),
                        tuple(
                            resolve_passage(support, self.ledger)
                            for support in requirement.supports
                        ),
                    )
                )
            triggers[issue.issue_id] = tuple(bindings)
        return triggers

    def _source_trigger_state(self) -> list[JsonValue]:
        if self.plan is None:
            return []
        result: list[JsonValue] = []
        for issue in self.plan.issues:
            if issue.origin != "source":
                continue
            result.append(
                {
                    "issue_id": issue.issue_id,
                    "parent_issue_id": issue.parent_issue_id,
                    "trigger_dimension": issue.trigger_dimension.value
                    if issue.trigger_dimension is not None
                    else None,
                    "findings": [
                        {
                            "finding": record.model_dump(mode="json"),
                            "parent_assessments": [
                                row.model_dump(mode="json") for row in associations
                            ],
                            "canonical_passages": [
                                passage.model_dump(mode="json", exclude={"quotation"})
                                for passage in passages
                            ],
                        }
                        for record, associations, passages in self.source_issue_triggers.get(
                            issue.issue_id, ()
                        )
                    ],
                }
            )
        return result

    def _issue_requirements(self, issue_id: str) -> list[RequirementRecord]:
        identities = dict.fromkeys(
            identity
            for row in self.dimensions
            if row.issue_id == issue_id
            for identity in row.requirement_ids
        )
        return [
            self.requirements[identity]
            for identity in identities
            if identity in self.requirements
        ]

    def _uncertain_issue_ids(self) -> set[str]:
        if self.plan is None:
            return set()
        return {
            issue.issue_id
            for issue in self.plan.issues
            if any(
                row.legal_status == "annulled"
                or (
                    not self.policy.assume_current_corpus
                    and (row.validity == "unknown" or row.legal_status == "unknown")
                )
                for row in self._issue_requirements(issue.issue_id)
            )
        }

    def issue_closures(self) -> list[IssueClosure]:
        if self.plan is None:
            return []
        closures: dict[str, IssueClosure] = {}
        children: dict[str, list[str]] = {}
        unresolved_repairs = self._unresolved_repair_issue_ids()
        for issue in self.plan.issues:
            if issue.parent_issue_id is not None:
                children.setdefault(issue.parent_issue_id, []).append(issue.issue_id)
            rows = [row for row in self.dimensions if row.issue_id == issue.issue_id]
            requirements = self._issue_requirements(issue.issue_id)
            reasons = [row.reason for row in rows if row.status == "unresolved"]
            status = "closed"
            if len(rows) != len(LegalDimension) or reasons:
                status = "open"
                if len(rows) != len(LegalDimension):
                    reasons.append("The complete issue assessment is pending")
            if issue.origin == "source" and not requirements:
                status = "open"
                reasons.append(
                    "The material source-derived question has no extracted supported resolution"
                )
            if any(
                issue.issue_id in action.issue_ids for action in self.pending_actions
            ):
                status = "open"
                reasons.append("Requested source operations remain unexecuted")
            if issue.issue_id in unresolved_repairs:
                status = "open"
                reasons.append(
                    "A material legal limitation identified during repair remains unresolved"
                )
            for assessment in self.source_assessments:
                continuation = assessment.get("requested_read")
                if (
                    assessment.get("content_status") == "operative_effect_missing"
                    and isinstance(continuation, dict)
                    and isinstance(identities := continuation.get("issue_ids"), list)
                    and issue.issue_id in identities
                ):
                    status = "open"
                    reasons.append(str(assessment["missing_effect"]))
            if status == "closed" and any(
                row.legal_status == "annulled"
                or (
                    not self.policy.assume_current_corpus
                    and (row.validity == "unknown" or row.legal_status == "unknown")
                )
                for row in requirements
            ):
                status = "partial"
                reasons.append("The recorded legal validity remains uncertain")
            closures[issue.issue_id] = IssueClosure.model_validate(
                {"issue_id": issue.issue_id, "status": status, "reasons": reasons}
            )
        # IssuePlan rejects cycles, so bottom-up dependency propagation is bounded.
        for _ in self.plan.issues:
            for parent_id, child_ids in children.items():
                blocked = [
                    identity
                    for identity in child_ids
                    if closures[identity].status != "closed"
                ]
                if blocked:
                    parent = closures[parent_id]
                    parent.blocking_child_ids = blocked
                    if parent.status != "open":
                        parent.status = (
                            "open"
                            if any(
                                closures[identity].status == "open"
                                for identity in blocked
                            )
                            else "partial"
                        )
        return [closures[issue.issue_id] for issue in self.plan.issues]

    def _account_sources(
        self,
        request: str,
        history: str,
        *,
        finalizing: bool = False,
        draft: DraftAnswer | None = None,
    ) -> None:
        if self.source_accountant is None:
            return
        assert self.plan is not None
        while True:
            try:
                self.source_assessments = self.source_accountant.scan(
                    self.state(request, history, draft), finalizing=finalizing
                )
            except (RunStopped, TimeoutError):
                self.source_assessments = self.source_accountant.snapshot()
                raise
            executed = {
                json.dumps(
                    {"tool": row["tool"], "arguments": row["arguments"]}, sort_keys=True
                )
                for row in self.acquirer.receipts
            }
            actions: dict[str, SourceAction] = {}
            for assessment in self.source_assessments:
                if assessment.get("requested_read") is None:
                    continue
                action = SourceAction.model_validate(assessment["requested_read"])
                key = json.dumps(
                    {"tool": action.tool, "arguments": action.arguments}, sort_keys=True
                )
                if key in executed:
                    continue
                if key in actions:
                    actions[key].issue_ids = sorted(
                        set(actions[key].issue_ids) | set(action.issue_ids)
                    )
                else:
                    actions[key] = action
            if not actions:
                return
            before = set(self.ledger.citation_numbers())
            self.report("tools", self.plan.language)
            self._acquire(list(actions.values()), finalizing=finalizing)
            if set(self.ledger.citation_numbers()) == before:
                return

    def _reading(
        self,
        request: str,
        history: str,
        *,
        finalizing: bool = False,
        draft: DraftAnswer | None = None,
    ) -> ReadingDecision:
        assert self.plan is not None
        self.report("reading", self.plan.language)
        self._account_sources(request, history, finalizing=finalizing, draft=draft)
        repairing = bool(self.repair_checks) or (
            draft is not None
            and self.final_review is not None
            and bool(self.final_review.flags)
        )
        if repairing and draft is not None:
            self._start_repair(draft)
        independent_plan = repairing and self.review_diagnoses is not None
        resolution_plan = (
            EvidenceResolutionPlan(self.review_diagnoses, self._repair_check_issues)
            if independent_plan and self.review_diagnoses is not None
            else None
        )
        response_model = (
            resolution_plan.response_model()
            if resolution_plan is not None
            else RepairReadingDecision
            if repairing
            else InitialReadingDecision
            if not self.dimensions
            else ReadingDecision
        )
        prompt = (
            EVIDENCE_RESOLUTION_PROMPT
            if independent_plan
            else REPAIR_READING_PROMPT
            if repairing
            else READING_PROMPT
        )
        state = self.state(request, history, draft)
        while True:
            if resolution_plan is not None:
                state = resolution_plan.state(state)
            try:
                decision = self.gateway.complete(
                    prompt,
                    state,
                    response_model,
                    LLMFlow.LEGAL_REVIEW_READING,
                    finalizing=finalizing,
                )
                if resolution_plan is not None:
                    decision = resolution_plan.compile(decision)
                try:
                    self._accept_reading(decision)
                except ValueError as error:
                    raise ReadingContractError(
                        str(error), decision.model_dump(mode="json")
                    ) from error
                return decision
            except ReadingContractError as error:
                if self.reading_correction_used:
                    raise
                self.reading_correction_used = True
                # The rejected proposal was never committed; this is a new admission.
                state = {
                    **self.state(request, history, draft),
                    "reading_contract_correction": {
                        "diagnostics": error.diagnostics,
                        "rejected_proposal": error.candidate,
                    },
                }
                prompt += (
                    "\nThe previous completed proposal violated the reading contract. "
                    "Use reading_contract_correction diagnostics to return one complete "
                    "corrected replacement. Preserve the canonical source identities and "
                    "all unaffected assessments. No part of the rejected proposal was accepted. "
                    "Include requirement_ids in every row; addressed needs real supporting "
                    "findings, never an invented link. If evidence is insufficient, mark "
                    "unresolved and state the gap. This is the sole contract correction."
                )

    def _research(
        self,
        request: str,
        history: str,
        *,
        finalizing: bool = False,
        draft: DraftAnswer | None = None,
    ) -> None:
        seen: set[str] = set()
        while True:
            decision = self._reading(
                request, history, finalizing=finalizing, draft=draft
            )
            expanded = (
                self.complete_articles(
                    {
                        support.citation
                        for record in self.requirements.values()
                        for support in record.supports
                    },
                    finalizing,
                )
                if self.complete_articles is not None
                else False
            )
            if not decision.actions:
                if expanded:
                    continue
                return
            signature = json.dumps(
                {
                    "actions": [
                        action.model_dump(mode="json") for action in decision.actions
                    ],
                    "originals": sorted(self.ledger.citation_numbers()),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if signature in seen:
                raise ValueError(
                    "Research repeated the same source operations without new evidence"
                )
            seen.add(signature)
            assert self.plan is not None
            self.report("tools", self.plan.language)
            if not self._acquire(decision.actions, finalizing=finalizing):
                return

    def _should_review_research(self) -> bool:
        if self.policy.always_review_research:
            return True
        material_gap = any(
            row.status == "unresolved" and row.dimension != LegalDimension.FACTS
            for row in self.dimensions
        ) or any(
            row.get("disposition") in {"material_limitation", "needs_operative_read"}
            for row in self.source_assessments
        )
        return material_gap and (
            self.context.research_deadline - time.monotonic()
            >= 2 * self.policy.max_call_seconds
        )

    def _collect_evidence(
        self, request: str, history: str, initial: list[SourceAction]
    ) -> None:
        try:
            self._acquire(initial)
            if not self.ledger.citation_numbers():
                return
            self._research(request, history)
            if self._should_review_research():
                self.early_review = self._review(request, history, None)
                if not self._review_can_repair(self.early_review):
                    raise ValueError(
                        self.early_review.failure_reason or "Early review failed"
                    )
                if self._review_work(self.early_review):
                    self._diagnose_review(request, history, None, self.early_review)
                    self._research(request, history)
            else:
                self.review_work_history.append(
                    {
                        "stage": "early_review",
                        "status": "skipped",
                        "reason": "No material gap or insufficient research time; full final review retained",
                    }
                )
        except (RunStopped, TimeoutError):
            if (
                self.context.is_cancelled()
                or time.monotonic() < self.context.research_deadline
            ):
                raise
            self.gaps.append(
                "Research time ended; preserve unresolved source questions as specific limitations."
            )
            self.report("finalizing", self.context.language)

    def _checks(
        self, *, draft: bool, answer: DraftAnswer | None = None
    ) -> list[ReviewCheck]:
        assert self.plan is not None
        purpose = (
            "Does the literal answer materially omit or misapply"
            if draft
            else "Does the extracted research materially lack or misinterpret"
        )
        checks = [
            ReviewCheck(
                id=f"{'draft' if draft else 'evidence'}:{issue.issue_id}:{dimension.value}",
                instructions=(
                    f"{purpose} {dimension.value} for the requested outcome of issue "
                    f"{issue.issue_id}? Judge within the request's scope from the complete "
                    f"originals using this criterion: {DIMENSION_GUIDANCE[dimension]} "
                    "originals and actual user facts. An affirmed not_applicable needs a "
                    "reason; absent evidence cannot establish non-applicability. Precisely "
                    "disclosed unresolved law is acceptable only without an unsupported "
                    "positive conclusion. Apply corpus_currency: absent date or status metadata "
                    "alone is not a defect under the supplied current-corpus assumption. "
                    "Explicit amendments, annulments and event-date restrictions still govern."
                    + (
                        " Evaluate the literal draft's treatment of this dimension, including "
                        "material omissions. Internal research status is not an assertion "
                        "in the answer. Do not require a separate paragraph per dimension."
                        if draft
                        else " Assess whether each dimension's reason and linked findings "
                        "establish its relevance and result from the originals; a finding can "
                        "support multiple dimensions, but an ID link alone proves no entailment."
                    )
                ),
                issue_id=issue.issue_id,
                dimension=dimension.value,
            )
            for issue in self.plan.issues
            for dimension in LegalDimension
        ]
        checks.extend(
            ReviewCheck(
                id=f"finding:{finding.requirement_id}",
                requirement_id=finding.requirement_id,
                instructions=(
                    (
                        f"Does the literal answer materially misapply or omit a decisive "
                        f"effect of the sources bound by finding_sources entry {finding.requirement_id}? "
                        "Inspect their use and any contrary effect on a requested outcome. "
                        "An unused or superseded interpretation in private research is not "
                        "an answer defect. Does the answer overgeneralize the "
                        if draft
                        else f"Does active global finding {finding.requirement_id} materially "
                        "misinterpret its selected canonical passages or overgeneralize the "
                    )
                    + "rule beyond the original's operative scope, regime, persons, conditions, "
                    "exceptions, deadline or event-date application? Judge against the complete "
                    "originals. An amendment or annulment must be bound to the exact norm "
                    "and temporal effect; a sector-specific condition cannot establish a "
                    "general rule. A real passage selector alone proves no entailment."
                ),
            )
            for finding in self.requirements.values()
        )
        checks.extend(
            [
                ReviewCheck(
                    id="request_coverage",
                    instructions=(
                        "Does the ENTIRE literal answer materially omit any explicit user subquestion, requested alternative or interaction, including one absent from the issue inventory? Assess answer coverage, not whether the frozen original plan included it. Do not invent unrelated topics."
                        if draft
                        else "Does the issue plan materially omit any explicit user subquestion, requested alternative or interaction from the original request? Do not invent unrelated topics."
                    ),
                ),
                ReviewCheck(
                    id="source_conditions",
                    instructions=(
                        "Does the literal answer"
                        if draft
                        else "Does the extracted research"
                    )
                    + " materially contradict or omit a decisive condition, exception, qualification or contrary effect of the complete original evidence, or conceal a partial/unread provision or temporal-law gap? No search result or review score proves absence of other law.",
                ),
                ReviewCheck(
                    id="issue_dependencies",
                    instructions=(
                        "Does the literal answer present an unsupported complete conclusion "
                        "while a material dependency remains unresolved, or omit that dependency's "
                        "effect on the requested outcome? Internal open/partial issue status alone "
                        "is not a defect: inspect whether the actual answer properly conditions "
                        "the affected conclusion."
                        if draft
                        else "Does any source-derived issue lack a material connection to its "
                        "parent's requested outcome or its exact canonical trigger, or does an "
                        "issue claimed resolved fail its explicit closure criteria?"
                    )
                    + " Review issue_closures and dependencies against originals and supplied "
                    "facts; do not demand new issues for incidental references or every category.",
                ),
            ]
        )
        if draft:
            checks.extend(
                [
                    ReviewCheck(
                        id="all_answer_claims",
                        instructions="Does ANY material legal assertion in the ENTIRE literal answer, including headings, tables, tax bases, calculations, sanctions, security, deadlines, or must/may/cannot language, lack exact original support, misapply that support or fall outside the claim inventory? Examine statements even when no planned issue covers them. Conditional conclusions must preserve qualifying facts.",
                    ),
                    ReviewCheck(
                        id="answer_consistency",
                        instructions="Does the integrated answer contain a material contradiction, repetition with conflicting conditions, or unsupported cross-issue conclusion or summary? Judge the entire answer against actual facts and originals, not only isolated issue paragraphs.",
                    ),
                    ReviewCheck(
                        id="answer_quotations",
                        instructions="Does any purported direct source quotation in the literal answer change the quoted wording or its legally material boundaries? Compare with the identified original passage and its role, including the exact text amended, repealed or annulled. Words appearing elsewhere in a quoted older rule do not establish that they form part of the operative disposition. Clearly marked omissions may shorten a quotation without changing its meaning; a paraphrase must not be presented as verbatim. Exclude quotations of the user's own question and mere labels.",
                    ),
                ]
            )
            if answer is not None:
                checks.extend(cited_source_checks(answer))
                checks.extend(
                    ReviewCheck(
                        id=f"claim:{claim.claim_id}",
                        claim_id=claim.claim_id,
                        issue_id=claim.issue_ids[0]
                        if len(claim.issue_ids) == 1
                        else None,
                        instructions=(
                            f"Does draft claim {claim.claim_id}, associated with issues "
                            f"{', '.join(claim.issue_ids)}, materially lack entailment from "
                            "its selected canonical passages, omit a decisive condition or "
                            "exception, or extend the supported assertion beyond the "
                            "original's operative scope, regime or temporal effect? Locate "
                            "the assertion this claim's selected supports purport to justify "
                            "within its literal answer_excerpt. A rendered block may contain "
                            "other separately inventoried claims; do not demand this claim "
                            "independently support unrelated items in the shared block. "
                            "Judge this bound claim and preserve factual qualifications."
                        ),
                    )
                    for claim in answer.claims
                )
        return checks

    @staticmethod
    def _review_work(review: ReviewResult) -> list[ReviewCheck]:
        return [*review.flags, *review.unassessed_checks]

    @staticmethod
    def _review_can_repair(review: ReviewResult) -> bool:
        return review.completed or (
            review.failure_reason == "openai_decision_refusal"
            and bool(review.unassessed_checks)
        )

    def _review(
        self, request: str, history: str, draft: DraftAnswer | None
    ) -> ReviewResult:
        self.last_draft = draft
        assert self.plan is not None
        self.report("review", self.plan.language)
        self.context.check_active()
        checks = self._checks(draft=draft is not None, answer=draft)
        state = scope_review_state(self.state(request, history, draft))
        # Preserve actual facts and discovery limitations without duplicate source bodies.
        review_state = {
            key: value
            for key, value in state.items()
            if key
            not in {
                "tools",
                "source_operations",
                "early_review",
                "final_review",
                "repair_contract",
                "repair_resolutions",
                "review_diagnoses",
                "research_resolutions",
                "source_assessments",
                "material_source_leads",
            }
        }
        diagnostic_keys = {
            "unmapped_chunk_count",
            "unmapped_count",
            "evidence_truncated",
            "scan_truncated",
            "outline_truncated",
            "has_more",
            "truncated",
            "degraded",
            "error",
            "reason",
            "missing",
            "access_denied",
            "continuation",
            "next_cursor",
            "total_hits",
            "matched_count",
            "unmapped_result_count",
            "incomplete_closure_count",
            "unhydrated_centers",
        }
        research_record: list[JsonValue] = []
        for receipt in self.acquirer.receipts:
            data = receipt.get("data")
            research_record.append(
                {
                    **{key: value for key, value in receipt.items() if key != "data"},
                    "result_limitations": {
                        key: value
                        for key, value in (
                            data.items() if isinstance(data, dict) else []
                        )
                        if key in diagnostic_keys
                    },
                }
            )
        review_state["research_record"] = research_record
        review_state = model_state(review_state, LLMFlow.LEGAL_REVIEW_DECISION)
        deadline = self.context.deadline if draft else self.context.research_deadline
        result = self.reviewer.review(
            review_state,
            checks,
            timeout_seconds=min(
                self.policy.max_call_seconds,
                max(0, deadline - time.monotonic()),
            ),
        )
        self.record_review_usage(result.input_tokens, result.output_tokens)
        self.context.check_active()
        expected_checks = {check.id: check for check in checks}
        expected = set(expected_checks)
        flags = {check.id for check in result.flags}
        unassessed = {check.id for check in result.unassessed_checks}
        if self._review_can_repair(result) and (
            len(result.flags) != len(flags)
            or len(result.unassessed_checks) != len(unassessed)
            or bool(result.completed and unassessed)
            or bool(unassessed & result.scores.keys())
            or any(
                check != expected_checks.get(check.id)
                for check in result.unassessed_checks
            )
            or any(flag != expected_checks.get(flag.id) for flag in result.flags)
            or set(result.scores) | unassessed != expected
            or not flags <= expected
            or flags != {key for key, value in result.scores.items() if value >= 0.5}
            or any(not 0 <= score <= 1 for score in result.scores.values())
        ):
            return ReviewResult(
                completed=False, failure_reason="review_inventory_invalid"
            )
        with graph_step("legal_review.review_result", {}) as step:
            step.output_value = result.model_dump(mode="json")
        return result

    def _validate_draft(self, draft: DraftAnswer) -> None:
        assert self.plan is not None
        known = {issue.issue_id for issue in self.plan.issues}
        if set(draft.unresolved_issue_ids) - known:
            raise ValueError("Draft unresolved issue is not in the plan")
        ids = [claim.claim_id for claim in draft.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("Draft claim identities must be unique")
        cited = set(extract_citation_numbers(draft.answer))
        supported_citations: set[int] = set()
        for claim in draft.claims:
            if set(claim.issue_ids) - known or claim.answer_excerpt not in draft.answer:
                raise ValueError(
                    "Claim is not bound to its literal answer and known issues"
                )
            for support in claim.supports:
                validate_support(support, self.ledger)
                if support.citation not in extract_citation_numbers(
                    claim.answer_excerpt
                ):
                    raise ValueError(
                        "Claim support must appear in its published passage"
                    )
                supported_citations.add(support.citation)
        if not draft.claims and not draft.unresolved_issue_ids:
            raise ValueError("A claimless draft must disclose unresolved issues")
        if (draft.claims and not cited) or cited - supported_citations:
            raise ValueError("Published citations must bind to supported answer claims")
        unresolved = (
            {row.issue_id for row in self.dimensions if row.status == "unresolved"}
            | self._uncertain_issue_ids()
            | {row.issue_id for row in self.issue_closures() if row.status != "closed"}
        )
        if unresolved - set(draft.unresolved_issue_ids):
            raise ValueError(
                "Draft must disclose all unresolved evidence/validity issues"
            )

    def _bind_unresolved_issues(self, draft: DraftAnswer) -> DraftAnswer:
        """Keep closure bookkeeping separate from the writer's reviewed legal prose."""
        unresolved = (
            self._uncertain_issue_ids()
            | {row.issue_id for row in self.dimensions if row.status == "unresolved"}
            | {row.issue_id for row in self.issue_closures() if row.status != "closed"}
        )
        return draft.model_copy(
            update={
                "unresolved_issue_ids": list(
                    dict.fromkeys([*draft.unresolved_issue_ids, *sorted(unresolved)])
                ),
            }
        )

    def _result(
        self,
        status: str,
        answer: str | None = None,
        gap: str | None = None,
        *,
        non_publication_reason: Literal["review_rejected", "time_exhausted"]
        | None = None,
    ) -> WorkflowResult:
        return WorkflowResult.model_validate(
            {
                "status": status,
                "non_publication_reason": non_publication_reason,
                "answer": answer,
                "plan": self.plan,
                "requirements": list(self.requirements.values()),
                "dimensions": self.dimensions,
                "issue_closures": self.issue_closures(),
                "source_issue_triggers": self._source_trigger_state(),
                "gaps": [
                    *self.gaps,
                    *(
                        ["Further requested source operations remain unexecuted"]
                        if self.pending_actions
                        else []
                    ),
                    *([gap] if gap else []),
                ],
                "early_review": self.early_review,
                "final_review": self.final_review,
                "repair_used": self.repair_used,
                "reading_correction_used": self.reading_correction_used,
                "repair_checks": list(self.repair_checks.values()),
                "repair_resolutions": self.repair_resolutions,
                "review_diagnoses": self.review_diagnoses,
                "final_adjudication": self.final_adjudication,
                "draft_adjudication": self.draft_adjudication,
                "review_work_history": self.review_work_history,
                "research_resolutions": self.research_resolutions,
                "research_needs": list(self.research.needs.values()),
                "skipped_searches": self.research.skipped,
                "source_journey": evidence_journey(
                    self.ledger,
                    self.acquirer.receipts,
                    self.source_assessments,
                    self.requirements,
                    self.last_draft,
                ),
            }
        )

    def run(self, request: str, history: str) -> WorkflowResult:
        try:
            return self._run(request, history)
        except (RunStopped, TimeoutError) as error:
            return self._result(
                "cancelled" if self.context.is_cancelled() else "unavailable",
                gap=str(error),
                non_publication_reason="time_exhausted"
                if not self.context.is_cancelled()
                and time.monotonic()
                >= min(
                    self.context.deadline,
                    float(
                        self.context.services.get(
                            "legal_review_phase_deadline", self.context.deadline
                        )
                    ),
                )
                else None,
            )
        except ValueError as error:
            return self._result("unavailable", gap=str(error))

    def _run(self, request: str, history: str) -> WorkflowResult:
        self.report("planning", "tr")
        planning_state = self.state(request, history)
        try:
            initial_plan = self.gateway.complete(
                PLAN_PROMPT,
                planning_state,
                InitialDiscoveryPlan,
                LLMFlow.LEGAL_REVIEW_PLANNER,
            )
        except StructuredOutputValidationError:
            # A separate admission preserves accounting for the sole schema correction.
            initial_plan = self.gateway.complete(
                PLAN_PROMPT
                + "\nThe previous initial plan did not satisfy its schema. Return a complete "
                "replacement including the required nonempty discovery_queries array with "
                "query text and known issue_ids. Preserve every requested outcome. "
                "Do not invent source identities or require a query per issue.",
                {**self.state(request, history), "planning_schema_correction": True},
                InitialDiscoveryPlan,
                LLMFlow.LEGAL_REVIEW_PLANNER,
            )
        self.plan = IssuePlan.model_validate(
            initial_plan.model_dump(mode="python", exclude={"discovery_queries"})
        )
        self.context.language = self.plan.language
        if any(issue.origin != "question" for issue in self.plan.issues):
            return self._result(
                "unavailable",
                gap="The initial plan must come from the question, before source-derived issues exist",
            )
        self.report("tools", self.plan.language)
        shared_queries: dict[str, list[str]] = {}
        for query in initial_plan.discovery_queries:
            identities = shared_queries.setdefault(query.query, [])
            for identity in query.issue_ids:
                if identity not in identities:
                    identities.append(identity)
        initial = [
            SourceAction(
                issue_ids=identities,
                tool="search_corpus",
                arguments={"query": query, "mode": "hybrid"},
            )
            for query, identities in shared_queries.items()
        ]
        if not initial:
            return self._result(
                "unavailable",
                gap="The initial plan supplied no discovery query for authorized original research",
            )
        self._collect_evidence(request, history, initial)
        if not self.ledger.citation_numbers():
            return self._result(
                "unavailable", gap="No authorized original evidence was read"
            )
        self.report("final", self.plan.language)
        generated = self.gateway.complete(
            DRAFT_PROMPT,
            self.state(request, history),
            GeneratedDraft,
            LLMFlow.LEGAL_REVIEW_DRAFT,
            finalizing=True,
        )
        draft = compile_draft(generated, ledger=self.ledger)
        draft = self._bind_unresolved_issues(draft)
        self._validate_draft(draft)
        self.final_review = self._review(request, history, draft)
        if not self._review_can_repair(self.final_review):
            return self._result("unavailable", gap=self.final_review.failure_reason)
        if self._review_work(self.final_review):
            repair_review = self.final_review
            if self.diagnoser is not None and self.final_review.completed:
                if self._final_review_is_publishable(request, history, draft):
                    partial = bool(
                        draft.unresolved_issue_ids or self.gaps or self.pending_actions
                    )
                    return self._result(
                        "partial" if partial else "verified", draft.answer
                    )
                assert self.final_adjudication is not None
                self.draft_adjudication = self.final_adjudication
                self.final_adjudication = None
                defects = {
                    row.check_id
                    for row in self.draft_adjudication.findings
                    if row.disposition == "defect"
                }
                repair_review = self.final_review.model_copy(
                    update={
                        "scores": {
                            identity: score
                            for identity, score in self.final_review.scores.items()
                            if identity in defects
                        },
                        "flags": [
                            check
                            for check in self.final_review.flags
                            if check.id in defects
                        ],
                    }
                )
            if (
                self.context.deadline - time.monotonic()
                <= self.policy.publication_reserve_seconds
            ):
                return self._result(
                    "unavailable",
                    gap="Insufficient time for a checked repair",
                    non_publication_reason="time_exhausted",
                )
            self.repair_used = True
            self.report("repair", self.plan.language)
            evidence_deadline = (
                self.context.deadline - self.policy.publication_reserve_seconds - 60
            )
            self.context.services["legal_review_phase_deadline"] = evidence_deadline
            try:
                self._diagnose_review(request, history, draft, repair_review)
                if (
                    self.review_diagnoses is None
                    or self.review_diagnoses.research_tasks
                ):
                    self._research(request, history, finalizing=True, draft=draft)
            except (RunStopped, TimeoutError):
                if self.context.is_cancelled() or time.monotonic() < evidence_deadline:
                    raise
                self.gaps.append(
                    "Repair research time ended; disclose the specific unresolved effects at their conclusions."
                )
            edits = self.gateway.complete(
                REPAIR_PROMPT,
                {
                    **self.state(request, history, draft),
                    "repair_base": generated.model_dump(mode="json"),
                },
                DraftEdits,
                LLMFlow.LEGAL_REVIEW_REPAIR,
                finalizing=True,
            )
            generated = apply_draft_edits(generated, edits)
            draft = compile_draft(generated, ledger=self.ledger)
            draft = self._bind_unresolved_issues(draft)
            self._validate_draft(draft)
            self.final_review = self._review(request, history, draft)
        if not self._final_review_is_publishable(request, history, draft):
            return self._result(
                "unavailable",
                gap="Final legal review did not pass",
                non_publication_reason="review_rejected"
                if self.final_review.completed
                else None,
            )
        partial = bool(draft.unresolved_issue_ids or self.gaps or self.pending_actions)
        return self._result("partial" if partial else "verified", draft.answer)
