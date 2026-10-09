"""Question-derived issues, early evidence review, one answer and one repair."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import date
from typing import Protocol, TypeVar

from pydantic import BaseModel, JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped
from onyx.legal_review.models import (
    DimensionAssessment,
    DraftAnswer,
    Issue,
    IssueClosure,
    IssuePlan,
    LegalDimension,
    PassageSupport,
    ReadingDecision,
    Requirement,
    RequirementRecord,
    ReviewCheck,
    ReviewResult,
    SourceAction,
    WorkflowPolicy,
    WorkflowResult,
)
from onyx.prompts.legal_review.prompts import (
    DRAFT_PROMPT,
    PLAN_PROMPT,
    READING_PROMPT,
    REPAIR_PROMPT,
)
from onyx.tracing.answer_graph import graph_step
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


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


def validate_support(support: PassageSupport, ledger: EvidenceLedger) -> None:
    item = ledger.get(support.citation)
    if item is None or item.search_doc is None or item.chunk_id is None:
        raise ValueError("Support has no canonical original citation")
    canonical = item.metadata.get("canonical_metadata")
    layers = (
        item.metadata,
        item.search_doc.metadata,
        canonical if isinstance(canonical, dict) else {},
    )
    if (
        not support.quotation.strip()
        or support.quotation not in item.text
        or hashlib.sha256(item.text.encode()).hexdigest() != item.text_hash
        or item.search_doc.document_id != item.source_id
        or item.search_doc.metadata.get("regulatory_chunk_id") != item.chunk_id
        or any(
            layer.get(flag) is True
            for layer in layers
            for flag in ("external", "derived", "untrusted", "truncated")
        )
    ):
        raise ValueError("Support is not an exact authorized original passage")


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
        report: Callable[[str, str], None] = lambda _phase, _language: None,
        record_review_usage: Callable[[int, int], None] = lambda _input, _output: None,
    ) -> None:
        self.gateway = gateway
        self.acquirer = acquirer
        self.reviewer = reviewer
        self.ledger = ledger
        self.context = context
        self.policy = policy
        self.report = report
        self.record_review_usage = record_review_usage
        self.plan: IssuePlan | None = None
        self.requirements: dict[str, RequirementRecord] = {}
        self.requirement_history: dict[str, RequirementRecord] = {}
        self.dimensions: list[DimensionAssessment] = []
        self.gaps: list[str] = []
        self.early_review: ReviewResult | None = None
        self.final_review: ReviewResult | None = None
        self.repair_used = False
        self.pending_actions: list[SourceAction] = []

    def state(
        self, request: str, history: str, draft: DraftAnswer | None = None
    ) -> dict[str, JsonValue]:
        numbers = self.ledger.citation_numbers()
        originals: list[JsonValue] = json.loads(
            self.ledger.serialize_records(numbers, required=numbers, max_chars=None)
        )
        return {
            "request": request,
            "history": history,
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "standard_dimensions": [dimension.value for dimension in LegalDimension],
            "dimension_assessments": [
                assessment.model_dump(mode="json") for assessment in self.dimensions
            ],
            "requirements": [
                requirement.model_dump(mode="json")
                for requirement in self.requirements.values()
            ],
            "original_evidence": originals,
            "source_operations": list(self.acquirer.receipts),
            "evidence_gaps": list(self.gaps),
            "issue_closures": [
                row.model_dump(mode="json") for row in self.issue_closures()
            ],
            "pending_source_operations": [
                action.model_dump(mode="json") for action in self.pending_actions
            ],
            "limits": {
                "total_search_budget": self.policy.max_searches,
                "remaining_search_budget": max(
                    0, self.policy.max_searches - self.acquirer.searches
                ),
                "remaining_research_seconds": max(
                    0, self.context.research_deadline - time.monotonic()
                ),
                "max_research_rounds": self.policy.max_research_rounds,
                "max_postdraft_repairs": 1,
            },
            "tools": self.acquirer.definitions(),
            "draft": draft.model_dump(mode="json") if draft else None,
            "early_review": self.early_review.model_dump(mode="json")
            if self.early_review
            else None,
            "final_review": self.final_review.model_dump(mode="json")
            if self.final_review
            else None,
        }

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
        known = {issue.issue_id for issue in proposed.issues}
        pending = dict(self.requirements)
        history = dict(self.requirement_history)
        seen: set[str] = set()
        for requirement in decision.requirements:
            if requirement.requirement_id in seen:
                raise ValueError("Duplicate requirement identity")
            seen.add(requirement.requirement_id)
            if requirement.issue_id not in known:
                raise ValueError("Requirement refers to an unknown issue")
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
            if (
                record.requirement_id in pending
                and pending[record.requirement_id] != record
            ):
                raise ValueError("Requirement identity is immutable")
            if record.requirement_id in history:
                raise ValueError("Superseded requirement identity cannot be reused")
            for identity in record.supersedes_requirement_ids:
                previous = pending.get(identity)
                if (
                    previous is None
                    or previous.issue_id != record.issue_id
                    or previous.requirement_id == record.requirement_id
                ):
                    raise ValueError(
                        "Supersession must replace an existing same-issue requirement"
                    )
                history[identity] = pending.pop(identity)
            pending[record.requirement_id] = record
        self._validate_source_issues(proposed, pending, history)
        expected = {
            (issue_id, dimension) for issue_id in known for dimension in LegalDimension
        }
        keys = [(row.issue_id, row.dimension) for row in decision.dimensions]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("Every issue needs every standard dimension exactly once")
        for assessment in decision.dimensions:
            if assessment.status == "addressed" and not assessment.requirement_ids:
                raise ValueError(
                    "Addressed dimensions require original-backed requirements"
                )
            for identity in assessment.requirement_ids:
                requirement = pending.get(identity)
                if requirement is None or requirement.issue_id != assessment.issue_id:
                    raise ValueError(
                        "Dimension requirements must belong to the same issue"
                    )
                if requirement.dimension != assessment.dimension:
                    raise ValueError(
                        "Dimension requirement must match the assessed dimension"
                    )
                if (
                    assessment.status == "addressed"
                    and requirement.legal_status == "annulled"
                ):
                    raise ValueError(
                        "An annulled requirement cannot close an operative dimension"
                    )
        self.requirements = pending
        self.requirement_history = history
        self.dimensions = decision.dimensions
        self.gaps = list(decision.evidence_gaps)
        self.plan = proposed
        for issue in decision.additional_issues:
            decision.actions.extend(
                SourceAction(
                    issue_ids=[issue.issue_id],
                    tool="search_corpus",
                    arguments={"query": query, "mode": "hybrid"},
                )
                for query in issue.research_queries
            )
        self.pending_actions = list(decision.actions)

    def _acquire(
        self, actions: list[SourceAction], *, finalizing: bool = False
    ) -> None:
        assert self.plan is not None
        self.pending_actions = list(actions)
        self.acquirer.acquire(actions, self.plan, finalizing=finalizing)
        self.pending_actions = []

    def _source_key(
        self, issue: Issue
    ) -> tuple[str | None, LegalDimension | None, tuple[tuple[str, str], ...]]:
        triggers: set[tuple[str, str]] = set()
        for citation in issue.supporting_citations:
            item = self.ledger.get(citation)
            if item is None or item.chunk_id is None or item.search_doc is None:
                raise ValueError("Source issue trigger is not a canonical original")
            article = item.metadata.get("article_no") or item.search_doc.metadata.get(
                "article_no"
            )
            triggers.add((item.source_id, str(article or item.chunk_id)))
        return issue.parent_issue_id, issue.trigger_dimension, tuple(sorted(triggers))

    def _validate_source_issues(
        self,
        proposed: IssuePlan,
        requirements: dict[str, RequirementRecord],
        history: dict[str, RequirementRecord],
    ) -> None:
        identities: set[
            tuple[str | None, LegalDimension | None, tuple[tuple[str, str], ...]]
        ] = set()
        for issue in proposed.issues:
            if issue.origin != "source":
                continue
            key = self._source_key(issue)
            if key in identities:
                raise ValueError(
                    "Duplicate parent/source/dimension dependency; resolve the existing issue"
                )
            identities.add(key)
            supported: set[int] = set()
            for identity in issue.supporting_requirement_ids:
                requirement = requirements.get(identity) or history.get(identity)
                if requirement is None or requirement.issue_id != issue.parent_issue_id:
                    raise ValueError(
                        "Source issue must be triggered by an exact parent-backed requirement"
                    )
                supported.update(support.citation for support in requirement.supports)
            if set(issue.supporting_citations) - supported:
                raise ValueError(
                    "Source issue citations must bind to its parent requirement supports"
                )

    def issue_closures(self) -> list[IssueClosure]:
        if self.plan is None:
            return []
        closures: dict[str, IssueClosure] = {}
        children: dict[str, list[str]] = {}
        for issue in self.plan.issues:
            if issue.parent_issue_id is not None:
                children.setdefault(issue.parent_issue_id, []).append(issue.issue_id)
            rows = [row for row in self.dimensions if row.issue_id == issue.issue_id]
            requirements = [
                row
                for row in self.requirements.values()
                if row.issue_id == issue.issue_id
            ]
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
            if status == "closed" and any(
                row.validity == "unknown" or row.legal_status in {"unknown", "annulled"}
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
        decision = self.gateway.complete(
            READING_PROMPT,
            self.state(request, history, draft),
            ReadingDecision,
            LLMFlow.LEGAL_REVIEW_READING,
            finalizing=finalizing,
        )
        self._accept_reading(decision)
        return decision

    def _checks(self, *, draft: bool) -> list[ReviewCheck]:
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
                    "originals and actual user facts. An affirmed not_applicable needs a "
                    "reason; absent evidence cannot establish non-applicability. Precisely "
                    "disclosed unresolved law is acceptable only without an unsupported "
                    "positive conclusion. Unknown chunk dates do not prove legal validity."
                ),
                issue_id=issue.issue_id,
                dimension=dimension.value,
            )
            for issue in self.plan.issues
            for dimension in LegalDimension
        ]
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
                    instructions="Does the research or literal answer materially contradict or omit a decisive condition, exception, qualification or contrary effect of the complete original evidence, or conceal a partial/unread provision or temporal-law gap? No search result or JEV score proves absence of other law.",
                ),
                ReviewCheck(
                    id="issue_dependencies",
                    instructions="Does any source-derived issue lack a material connection to its parent's requested outcome or its exact canonical trigger, or does an issue claimed resolved fail its explicit closure criteria? Does the answer treat a parent conclusion as complete while a material dependent child remains open or partial? Review all issue_closures and dependencies against the originals, supplied facts and literal answer; do not demand new issues for incidental references or every category.",
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
                ]
            )
        return checks

    def _review(
        self, request: str, history: str, draft: DraftAnswer | None
    ) -> ReviewResult:
        assert self.plan is not None
        self.report("review", self.plan.language)
        self.context.check_active()
        checks = self._checks(draft=draft is not None)
        state = self.state(request, history, draft)
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
        expected = {check.id for check in checks}
        flags = {check.id for check in result.flags}
        if result.completed and (
            set(result.scores) != expected
            or not flags <= expected
            or flags != {key for key, value in result.scores.items() if value >= 0.5}
            or any(not 0 <= score <= 1 for score in result.scores.values())
        ):
            return ReviewResult(
                completed=False, failure_reason="review_inventory_invalid"
            )
        with graph_step("legal_review.jev_result", {}) as step:
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
        if not cited or cited - supported_citations:
            raise ValueError("Published citations must bind to supported answer claims")
        unresolved = (
            {row.issue_id for row in self.dimensions if row.status == "unresolved"}
            | {
                row.issue_id
                for row in self.requirements.values()
                if row.validity == "unknown"
                or row.legal_status in {"unknown", "annulled"}
            }
            | {row.issue_id for row in self.issue_closures() if row.status != "closed"}
        )
        if unresolved - set(draft.unresolved_issue_ids):
            raise ValueError(
                "Draft must disclose all unresolved evidence/validity issues"
            )

    def _disclose_validity(self, draft: DraftAnswer) -> DraftAnswer:
        assert self.plan is not None
        affected = {
            row.issue_id
            for row in self.requirements.values()
            if row.validity == "unknown" or row.legal_status in {"unknown", "annulled"}
        }
        dependencies = [
            row
            for row in self.issue_closures()
            if row.status == "open" or row.blocking_child_ids
        ]
        unresolved = affected | {row.issue_id for row in dependencies}
        questions = "; ".join(
            issue.question for issue in self.plan.issues if issue.issue_id in affected
        )
        disclosure = (
            (
                "Yürürlük sınırı: Şu meselelerde dayanılan kaynakların olay tarihi itibarıyla "
                f"yürürlük ve iptal durumu kesinleştirilemedi: {questions}. "
                "Bu meselelerin sonuçları yürürlük ve iptal durumu teyidine bağlıdır."
                if self.plan.language.startswith("tr")
                else f"Validity limitation: The operative and annulment status of sources for {questions} "
                "on the event date could not be established. These conclusions remain conditional "
                "on verification of their operative and annulment status."
            )
            if affected
            else ""
        )
        if dependencies:
            dependency_questions = "; ".join(
                issue.question
                for issue in self.plan.issues
                if issue.issue_id in {row.issue_id for row in dependencies}
            )
            dependency_disclosure = (
                f"İnceleme sınırı: Şu sonucu etkileyen meseleler veya bağlı alt meseleleri tamamlanamadı: {dependency_questions}. İlgili sonuçlar bu açık meselelerin çözümüne bağlıdır."
                if self.plan.language.startswith("tr")
                else f"Research limitation: These material issues or their dependent questions remain unresolved: {dependency_questions}. The affected conclusions remain conditional on resolving these questions."
            )
            disclosure = "\n\n".join(
                part for part in (disclosure, dependency_disclosure) if part
            )
        if not disclosure:
            return draft
        return draft.model_copy(
            update={
                "answer": draft.answer
                if disclosure in draft.answer
                else draft.answer + "\n\n" + disclosure,
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
    ) -> WorkflowResult:
        return WorkflowResult.model_validate(
            {
                "status": status,
                "answer": answer,
                "plan": self.plan,
                "requirements": list(self.requirements.values()),
                "dimensions": self.dimensions,
                "issue_closures": self.issue_closures(),
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
            }
        )

    def run(self, request: str, history: str) -> WorkflowResult:
        try:
            return self._run(request, history)
        except (RunStopped, TimeoutError) as error:
            return self._result(
                "cancelled" if self.context.is_cancelled() else "unavailable",
                gap=str(error),
            )
        except ValueError as error:
            return self._result("unavailable", gap=str(error))

    def _run(self, request: str, history: str) -> WorkflowResult:
        self.report("planning", "tr")
        self.plan = self.gateway.complete(
            PLAN_PROMPT,
            self.state(request, history),
            IssuePlan,
            LLMFlow.LEGAL_REVIEW_PLANNER,
        )
        self.context.language = self.plan.language
        if any(issue.origin != "question" for issue in self.plan.issues):
            return self._result(
                "unavailable",
                gap="The initial plan must come from the question, before source-derived issues exist",
            )
        self.report("tools", self.plan.language)
        shared_queries: dict[str, list[str]] = {}
        for issue in self.plan.issues:
            for query in issue.research_queries:
                identities = shared_queries.setdefault(query, [])
                if issue.issue_id not in identities:
                    identities.append(issue.issue_id)
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
        if len(initial) > self.policy.max_searches:
            return self._result(
                "unavailable",
                gap="Planner discovery queries exceed the explicit total search budget",
            )
        self._acquire(initial)
        if not self.ledger.citation_numbers():
            return self._result(
                "unavailable", gap="No authorized original evidence was read"
            )
        decision = ReadingDecision(dimensions=[])
        early_return_used = False
        for round_number in range(self.policy.max_research_rounds):
            self.context.check_research_active()
            decision = self._reading(request, history)
            if (
                not decision.actions
                or round_number + 1 >= self.policy.max_research_rounds
            ):
                break
            self.report("tools", self.plan.language)
            self._acquire(decision.actions)
        if decision.actions and time.monotonic() < self.context.research_deadline:
            self.report("tools", self.plan.language)
            self._acquire(decision.actions)
            early_return_used = True
            decision = self._reading(request, history)
        self.early_review = self._review(request, history, None)
        if not self.early_review.completed:
            return self._result("unavailable", gap=self.early_review.failure_reason)
        if (
            (self.early_review.flags or decision.actions)
            and not early_return_used
            and time.monotonic() < self.context.research_deadline
        ):
            # One evidence-directed response to the early batched review.
            if self.early_review.flags:
                decision = self._reading(request, history)
            if decision.actions:
                self._acquire(decision.actions)
                self._reading(request, history)
        self.report("final", self.plan.language)
        draft = self.gateway.complete(
            DRAFT_PROMPT,
            self.state(request, history),
            DraftAnswer,
            LLMFlow.LEGAL_REVIEW_DRAFT,
            finalizing=True,
        )
        draft = self._disclose_validity(draft)
        self._validate_draft(draft)
        self.final_review = self._review(request, history, draft)
        if not self.final_review.completed:
            return self._result("unavailable", gap=self.final_review.failure_reason)
        if self.final_review.flags:
            self.repair_used = True
            self.report("repair", self.plan.language)
            decision = self._reading(request, history, finalizing=True, draft=draft)
            if decision.actions:
                self._acquire(decision.actions, finalizing=True)
                self._reading(request, history, finalizing=True, draft=draft)
            draft = self.gateway.complete(
                REPAIR_PROMPT,
                self.state(request, history, draft),
                DraftAnswer,
                LLMFlow.LEGAL_REVIEW_REPAIR,
                finalizing=True,
            )
            draft = self._disclose_validity(draft)
            self._validate_draft(draft)
            self.final_review = self._review(request, history, draft)
        if not self.final_review.completed or self.final_review.flags:
            return self._result("unavailable", gap="Final legal review did not pass")
        partial = bool(draft.unresolved_issue_ids or self.gaps or self.pending_actions)
        return self._result("partial" if partial else "verified", draft.answer)
