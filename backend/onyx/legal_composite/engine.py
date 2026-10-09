from __future__ import annotations

import json
import time
from collections.abc import Callable
from math import ceil
from typing import Protocol, TypeVar

from pydantic import BaseModel, JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, RunStopped
from onyx.db.legal_composite_sources import SourceKind
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.budget import ResearchPhaseClosed
from onyx.legal_composite.claim_edits import (
    CLAIM_REPAIR_EDITS_PROMPT,
    ClaimRepairEdits,
    claim_edits_to_delta,
)
from onyx.legal_composite.dependencies import (
    CompositeDependencyExpander,
    DependencyExpander,
    assess_dependencies,
    dependency_required_citations,
    material_dependency_gaps,
)
from onyx.legal_composite.draft_composition import (
    DRAFT_COMPOSITION_PROMPT,
    DraftComposition,
    compose_draft,
)
from onyx.legal_composite.draft_repair import (
    apply_claim_delta,
    canonicalize_delta_supports,
)
from onyx.legal_composite.models import (
    AnswerReview,
    AuthorityDependency,
    CompositeWorkflowResult,
    DraftAnswer,
    GapResolution,
    IssueResearchPlan,
    IssueResearchStep,
    MaterialDependencyRequest,
    PassageSupport,
    ResearchPlan,
    ResearchStep,
    SemanticReview,
    SourceAction,
    SourceRequirement,
    StructuredDraftAnswer,
    WorkflowPolicy,
    WorkflowResult,
)
from onyx.legal_composite.prompts import (
    ANSWER_PROMPT,
    PLAN_PROMPT,
    RESEARCH_PROMPT,
    REVIEW_PROMPT,
)
from onyx.legal_composite.reading_evidence import (
    IssueReadingResponse,
    ReadingWitnessManifest,
    number_reading_witnesses,
    resolve_issue_reading,
)
from onyx.legal_composite.reading_prompt import NUMBERED_READING_PROMPT
from onyx.legal_composite.requirements import (
    RequirementLedger,
    canonicalize_draft_supports,
    draft_binding_gaps,
    support_is_original,
)
from onyx.legal_composite.reviewer import AnswerReviewer
from onyx.legal_composite.selection import (
    SourceSelectionResult,
    SourceSelector,
    selection_request_from_ledger,
)
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS
from onyx.tracing.answer_graph import graph_step
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

_STOP_REASONS = {
    "The model response failed the workflow schema": "output_schema",
    "The typed model response was truncated": "output_truncated",
    "The model returned no structured answer": "output_missing",
    "The typed model returned an undeclared tool call": "output_tool_call",
    "The bounded model invocation timed out": "provider_timeout",
    "Provider call exceeded its deadline; no further spend authorized": "provider_timeout",
    "The bounded model invocation failed": "provider_failure",
    "Provider usage exceeded the estimate; no further spend authorized": "usage_estimate_overrun",
    "Provider actual usage exceeded workflow capacity; no further generations allowed": "usage_capacity",
    "Workflow model budget exhausted; finalization allocation retained": "model_allocation",
    "Workflow deadline reached; finalization time retained": "deadline",
    "Research deadline exceeded; finalization time retained": "research_deadline",
    "Research deadline exceeded": "deadline",
    "Research cancelled": "cancelled",
    "A required original is absent from the generation payload": "required_original",
    "Required originals and protocol exceed the remaining research budget": "model_allocation",
    "Required originals and protocol exceed the model context budget": "model_capacity",
}


def _stop_reason(error: RunStopped | InvalidSourceAction) -> str:
    if isinstance(error, InvalidSourceAction):
        return "source_action"
    return _STOP_REASONS.get(str(error), "other_stop")


def _admission_reason(error: InvalidSourceAction) -> str:
    return {
        "Support is not an exact delivered original: canonical original binding failed": "original_binding",
        "Support span ID is not in this exact original": "span_identity",
        "Reading support does not identify a provided delivered witness": "span_identity",
        "Reading witness manifest does not match the active delivery": "original_binding",
        "Reading witness manifest no longer matches its original": "original_binding",
        "Reading witness catalogue is not an exact original": "original_binding",
        "Support quotation conflicts with its original span": "span_quotation",
        "Support quotation is not an exact delivered original passage": "literal_quotation",
        "Repair must name exactly the affected sections": "patch_sections",
        "Repair cannot add readings or closures for an unaffected issue": "patch_reading_scope",
        "Repair cannot change an unaffected issue's unresolved status": "patch_unresolved_scope",
        "Repair can delete only existing affected claims": "patch_delete_scope",
        "Repair cannot upsert an unaffected claim": "patch_claim_scope",
        "Repair cannot move an existing claim": "patch_claim_move",
        "Repair claim cannot reference another section's issue": "patch_claim_issue",
        "Repair cannot change a section's issue bindings": "patch_section_issue",
        "Repair order must retain every section claim exactly once": "patch_claim_order",
        "Repair heading cannot repeat its legal claim prose": "patch_heading_prose",
        "Repair cannot rewrite an existing requirement": "patch_requirement_rewrite",
        "Repair does not recompose a valid complete structured draft": "patch_recomposition",
        "Existing section is not its exact rendered claim body": "existing_section_render",
        "A literal section needs an explicit repair heading": "existing_section_heading",
        "Claim repair identities must be unique and disjoint": "patch_duplicate_identity",
        "Duplicate source requirement identity": "requirement_duplicate",
        "Source requirement refers to an unknown issue": "requirement_issue",
        "Source requirement identities are immutable": "requirement_immutable",
        "Research gaps refer to an unknown or unaffected issue": "gap_issue_scope",
        "Gap closure lacks a fresh same-issue requirement binding": "gap_requirement_binding",
        "Claim edits failed the repair schema": "patch_schema",
        "Frozen repair identities must be unique": "existing_duplicate_identity",
        "Claim edits target an unknown section": "patch_sections",
        "Heading edits must target affected sections": "patch_heading_scope",
        "Claim edits can delete only existing affected claims": "patch_delete_scope",
        "Claim edits can upsert only affected claims": "patch_claim_scope",
        "Claim edits cannot move an existing claim": "patch_claim_move",
        "Claim edits cannot change existing issue bindings": "patch_claim_issue",
        "New claim edits need an explicit issue subset": "patch_claim_issue",
        "Frozen claim issue bindings exceed the target section": "existing_claim_issue",
        "Frozen section must name every own claim exactly once": "existing_claim_order",
        "Claim edits do not form a valid repair delta": "patch_recomposition",
        "Draft composition failed the transport schema": "draft_composition_schema",
        "Draft composition identities must be unique": "draft_composition_identity",
        "Draft composition section issue bindings must be unique": "draft_composition_issue",
        "Draft composition claim exceeds its section issue scope": "draft_composition_claim",
        "Draft composition cannot form a valid answer": "draft_composition_render",
    }.get(str(error), "source_action")


def _add_span_previews(records: list[JsonValue]) -> None:
    """Expose short navigation labels while preserving the whole original text."""
    for record in records:
        if not isinstance(record, dict):
            continue
        text = record.get("text")
        spans = record.get("witness_spans")
        if not isinstance(text, str) or not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, dict):
                continue
            start, end = span.get("start_char"), span.get("end_char")
            if isinstance(start, int) and isinstance(end, int):
                span["preview"] = text[start : min(start + 140, end)]


def _compact_original_catalogue(
    catalogue: list[dict[str, JsonValue]],
) -> JsonValue:
    """Share exact navigation values without changing row identities or metadata."""
    values: list[JsonValue] = []
    value_indices: dict[str, int] = {}
    metadata_pool: list[JsonValue] = []
    metadata_indices: dict[str, int] = {}
    rows: list[JsonValue] = []

    def intern(value: JsonValue) -> int:
        # Serialized keys distinguish bool/int/float, null/empty, and object order.
        key = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if key not in value_indices:
            value_indices[key] = len(values)
            values.append(json.loads(key))
        return value_indices[key]

    for row in catalogue:
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        pooled_metadata: dict[str, JsonValue] = {
            key: intern(value) for key, value in metadata.items()
        }
        signature = json.dumps(pooled_metadata, separators=(",", ":"))
        if signature not in metadata_indices:
            metadata_indices[signature] = len(metadata_pool)
            metadata_pool.append(pooled_metadata)
        rows.append(
            [
                row["citation"],
                intern(row["source_id"]),
                row["chunk_id"],
                metadata_indices[signature],
            ]
        )
    compact: dict[str, JsonValue] = {
        "codec": "shared_metadata_v1",
        "row_fields": ["citation", "source_id_ref", "chunk_id", "metadata_ref"],
        "values": values,
        "metadata": metadata_pool,
        "rows": rows,
    }
    if len(json.dumps(compact, ensure_ascii=False)) < len(
        json.dumps(catalogue, ensure_ascii=False)
    ):
        return compact
    return catalogue


def source_free_social_request(request: str) -> bool:
    return request.strip().casefold().rstrip(".!?").strip() in {
        "merhaba",
        "selam",
        "günaydın",
        "hello",
        "hi",
        "good morning",
        "thanks",
        "thank you",
        "teşekkürler",
        "tesekkurler",
        "teşekkür ederim",
    }


def initial_discovery_actions(plan: ResearchPlan, request: str) -> list[SourceAction]:
    actions = list(plan.initial_actions)
    if not plan.requires_sources:
        return actions
    searched_needs = {
        need_id
        for action in actions
        if action.tool == "search_corpus"
        for need_id in action.need_ids
    }
    missing = [need for need in plan.needs if need.need_id not in searched_needs]
    if len(missing) == len(plan.needs):
        queries = [
            (plan.discovery_query or request, [need.need_id for need in missing])
        ]
    else:
        queries = [(need.question, [need.need_id]) for need in missing]
    for query, need_ids in queries:
        actions.append(
            SourceAction(
                need_ids=need_ids,
                tool="search_corpus",
                arguments={
                    "query": query.strip()[:REGULATORY_MAX_SEARCH_QUERY_CHARS],
                    "mode": "hybrid",
                    "coverage_item": ", ".join(need_ids),
                    "evidence_target": "Operative governing, implementing, limiting and contrary original passages for the requested outcomes",
                    "expand_query": False,
                },
            )
        )
    return actions


def initial_issue_discovery_actions(
    plan: IssueResearchPlan, request: str
) -> list[SourceAction]:
    """Acquire one cross-issue frontier before further gap-directed discovery."""
    if not plan.requires_sources:
        return list(plan.initial_actions)
    need_ids = [need.need_id for need in plan.needs]
    query = plan.discovery_query.strip() or request.strip()
    return [
        SourceAction(
            need_ids=need_ids,
            tool="search_corpus",
            arguments={
                "query": query[:REGULATORY_MAX_SEARCH_QUERY_CHARS],
                "mode": "hybrid",
                "coverage_item": ", ".join(need_ids),
                "evidence_target": "Operative governing, implementing, limiting and contrary original passages for the requested outcomes",
                "expand_query": False,
            },
        ),
        *(action for action in plan.initial_actions if action.tool != "search_corpus"),
    ]


def _search_action_identity(action: SourceAction) -> str:
    arguments = {
        key: value for key, value in action.arguments.items() if key != "_public_update"
    }
    arguments["expand_query"] = False
    return json.dumps(arguments, sort_keys=True)


class ModelGateway(Protocol):
    last_call_id: str | None

    def complete(
        self,
        system: str,
        payload: dict[str, JsonValue],
        response_type: type[ResponseModel],
        flow: LLMFlow,
        finalizing: bool = False,
    ) -> ResponseModel: ...


class SourceAcquirer(Protocol):
    def definitions(self) -> list[dict[str, JsonValue]]: ...

    def acquire(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]: ...


def review_assessment(
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    ledger: EvidenceLedger,
    delivered: set[int],
    dependencies: list[AuthorityDependency] | None = None,
) -> tuple[bool, bool, list[str]]:
    """A semantic review cannot authorize invented IDs, quotes or omitted needs."""
    planned = {need.need_id: need for need in plan.needs}
    expected = set(planned)
    reviewed = [need.need_id for need in review.needs]
    gaps = list(review.defects)
    structurally_valid = (
        len(reviewed) == len(set(reviewed)) and set(reviewed) == expected
    )
    if not structurally_valid:
        gaps.append(
            "The review does not cover every frozen research need exactly once."
        )
    if set(draft.unresolved_need_ids) - expected:
        structurally_valid = False
        gaps.append("The draft declares an unknown research need.")
    cited = set(extract_citation_numbers(draft.answer))
    if plan.requires_sources and not cited:
        structurally_valid = False
        gaps.append("A legal answer needs original-source citations.")
    for number in cited:
        item = ledger.get(number)
        if item is None or item.search_doc is None or number not in delivered:
            structurally_valid = False
            gaps.append(f"Citation [{number}] was not delivered as a citable original.")
        elif item is not None:
            canonical = item.metadata.get("canonical_metadata")
            layers = [item.metadata, canonical if isinstance(canonical, dict) else {}]
            if any(
                layer.get(key) is True
                for layer in layers
                for key in ("external", "derived", "untrusted", "truncated")
            ):
                structurally_valid = False
                gaps.append(
                    f"Citation [{number}] is not a complete authorized original."
                )
    complete = not draft.unresolved_need_ids
    for need in review.needs:
        planned_need = planned.get(need.need_id)
        if planned_need is None:
            continue
        condition_indices = [row.condition_index for row in need.condition_reviews]
        expected_indices = set(range(len(planned_need.conditions_to_check)))
        if (
            any(isinstance(index, bool) for index in condition_indices)
            or len(condition_indices) != len(set(condition_indices))
            or set(condition_indices) != expected_indices
            or (plan.requires_sources and not expected_indices)
        ):
            structurally_valid = False
            gaps.append(
                f"{need.need_id}: the review does not cover every planned condition exactly once."
            )
        support_numbers = {support.citation for support in need.supports}
        for condition in need.condition_reviews:
            references = condition.support_citations
            if (
                any(isinstance(number, bool) for number in references)
                or len(references) != len(set(references))
                or set(references) - support_numbers
                or set(references) - cited
                or set(references) - delivered
            ):
                structurally_valid = False
                gaps.append(
                    f"{need.need_id}: a condition refers to missing or unrelated original support."
                )
            excerpt = condition.answer_excerpt
            if condition.status == "preserved":
                if (
                    not excerpt.strip()
                    or excerpt not in draft.answer
                    or (plan.requires_sources and not references)
                ):
                    structurally_valid = False
                    gaps.append(
                        f"{need.need_id}: a preserved condition lacks an exact answer witness and original support."
                    )
            elif condition.status == "unresolved":
                complete = False
                if (
                    need.status != "unresolved"
                    or need.need_id not in draft.unresolved_need_ids
                    or not excerpt.strip()
                    or excerpt not in draft.answer
                    or not need.gap_disclosure
                    or excerpt not in need.gap_disclosure
                ):
                    structurally_valid = False
                    gaps.append(
                        f"{need.need_id}: an unresolved condition is not bound to its declared gap."
                    )
            else:
                complete = False
                structurally_valid = False
                gaps.append(
                    f"{need.need_id}: a planned condition is {condition.status}."
                )
            if (
                need.status in {"supported", "conditional"}
                and condition.status != "preserved"
            ):
                structurally_valid = False
                gaps.append(
                    f"{need.need_id}: a supported outcome has an unpreserved condition."
                )
        if need.status in {"unresolved", "incorrect"}:
            complete = False
            gaps.append(f"{need.need_id}: {need.explanation}")
        if need.status == "incorrect":
            structurally_valid = False
        if need.status == "unresolved" and (
            need.need_id not in draft.unresolved_need_ids
            or not need.gap_disclosure
            or need.gap_disclosure not in draft.answer
        ):
            structurally_valid = False
            gaps.append(
                f"{need.need_id}: the unresolved interaction is not explicitly disclosed in the draft."
            )
        if need.status in {"supported", "conditional"}:
            if not need.conditions_preserved or (
                plan.requires_sources and not need.supports
            ):
                structurally_valid = False
                gaps.append(
                    f"{need.need_id}: decisive conditions or original support are missing."
                )
        for support in need.supports:
            item = ledger.get(support.citation)
            if (
                item is None
                or not support.quotation.strip()
                or support.citation not in cited
                or support.citation not in delivered
                or support.quotation not in item.text
            ):
                structurally_valid = False
                gaps.append(
                    f"{need.need_id}: a decisive quotation does not match its cited original."
                )
    dependency_complete, dependency_safe, dependency_gaps = assess_dependencies(
        dependencies or [], plan, draft, review, ledger, delivered
    )
    gaps.extend(dependency_gaps)
    safe = structurally_valid and review.material_claims_supported and dependency_safe
    passed = (
        safe
        and complete
        and review.request_coverage_complete
        and review.counter_authority_checked
        and not review.defects
        and dependency_complete
    )
    if not review.request_coverage_complete:
        gaps.append("The full user request has an uncovered outcome or alternative.")
    if not review.material_claims_supported:
        gaps.append("The draft contains an unsupported material legal claim.")
    if not review.counter_authority_checked:
        gaps.append("Applicable contrary or limiting authority remains unchecked.")
    return passed, safe, list(dict.fromkeys(gaps))


class LegalCompositeEngine:
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        acquirer: SourceAcquirer,
        ledger: EvidenceLedger,
        policy: WorkflowPolicy,
        check_active: Callable[[], None],
        research_available: Callable[[], bool],
        report: Callable[[str, str], None] = lambda _phase, _language: None,
        selector: SourceSelector | None = None,
        dependency_expander: DependencyExpander | None = None,
        source_kinds: dict[str, SourceKind] | None = None,
        reviewer: AnswerReviewer | None = None,
        evidence_context: RunContext | None = None,
        use_numbered_reading_supports: bool = False,
        allow_terminal_observed_reads: bool = False,
    ) -> None:
        if (
            type(use_numbered_reading_supports) is not bool
            or type(allow_terminal_observed_reads) is not bool
        ):
            raise ValueError("Reading behavior must use explicit boolean opt-ins")
        self.gateway = gateway
        self.acquirer = acquirer
        self.ledger = ledger
        self.policy = policy
        self.check_active = check_active
        self.research_available = research_available
        self.report = report
        self.selector = selector
        self.dependency_expander = dependency_expander
        self.source_kinds = dict(source_kinds or {})
        self.reviewer = reviewer
        self.evidence_context = evidence_context
        self.use_numbered_reading_supports = use_numbered_reading_supports
        self.allow_terminal_observed_reads = allow_terminal_observed_reads
        self.requirements = RequirementLedger(ledger)
        self.research_gaps: list[str] = []
        self.protocol_defects: list[str] = []
        self.source_requests: list[JsonValue] = []
        self.deferred_initial_actions: list[tuple[SourceAction, str]] = []
        self.deferred_followup_steps: list[JsonValue] = []
        self.semantic_review: SemanticReview | None = None
        self.dependencies: list[AuthorityDependency] = []
        self.selection: SourceSelectionResult | None = None
        self.pending_reconsidered: set[int] = set()
        self.pending_selection_citations: set[int] = set()
        self.plan: ResearchPlan | None = None
        self.receipts: list[dict[str, JsonValue]] = []
        self.last_review: AnswerReview | None = None
        self._research_seconds: list[float] = []
        self._source_seconds: dict[str, float] = {}
        self._dependency_seconds: list[float] = []

    def _reading_estimate(self) -> float:
        return max(self._research_seconds, default=self.policy.max_call_seconds)

    def _reading_has_runway(self) -> bool:
        return (
            self._repair_runway()
            > self._reading_estimate() + self.policy.selection_reserve_seconds
        )

    def _followup_has_runway(self, step: IssueResearchStep) -> bool:
        remaining = self._repair_runway()
        if remaining <= 0:
            return False
        acquisition = max(
            (
                self._source_seconds.get(action.tool, self.policy.max_call_seconds)
                for action in step.actions
            ),
            default=0.0,
        )
        if isinstance(self.acquirer, CanonicalAcquirer) and self.plan is not None:
            pending = self.acquirer.pending_call_counts(step.actions, self.plan)
            acquisition = max(
                (
                    self._source_seconds.get(tool, self.policy.max_call_seconds)
                    for tool in pending
                ),
                default=0.0,
            ) * ceil(sum(pending.values()) / self.policy.max_parallel_tools)
        dependencies = (
            max(self._dependency_seconds, default=self.policy.max_call_seconds)
            if step.material_dependencies
            else 0.0
        )
        return remaining > (
            acquisition
            + dependencies
            + self._reading_estimate()
            + self.policy.selection_reserve_seconds
        )

    def _defer_followup(self, step: IssueResearchStep, plan: IssueResearchPlan) -> None:
        known = {need.need_id for need in plan.needs}
        if any(set(action.need_ids) - known for action in step.actions):
            raise InvalidSourceAction("Action refers to an unknown research need")
        if step.material_dependencies:
            frontier = {target.origin_citation for target in step.material_dependencies}
            before = set(self.ledger.citation_numbers())
            delivered = getattr(self.gateway, "last_delivered_citations", before)
            if not frontier <= before or not frontier <= delivered:
                raise InvalidSourceAction(
                    "Related-source target was not a delivered original"
                )
            if not isinstance(self.dependency_expander, CompositeDependencyExpander):
                raise InvalidSourceAction(
                    "Material dependency deferral requires an issue-aware collector"
                )
            self.dependencies = self.dependency_expander.register_material(
                plan, frontier=frontier, material_targets=step.material_dependencies
            )
        self.deferred_followup_steps.append(
            {
                "actions": [action.model_dump(mode="json") for action in step.actions],
                "material_dependencies": [
                    target.model_dump(mode="json")
                    for target in step.material_dependencies
                ],
                "status": "unexecuted_navigation",
                "reason": "research_runway",
                "navigation_only": True,
            }
        )
        with graph_step(
            "legal_composite.research_admission",
            {},
            summary="work_kind=sources accepted=0 reason=research_runway",
        ) as admission:
            admission.output_value = {
                "operation": "sources",
                "accepted": False,
                "reason": "research_runway",
            }

    def _terminal_reads_have_runway(self, actions: list[SourceAction]) -> bool:
        if not isinstance(self.acquirer, CanonicalAcquirer) or self.plan is None:
            return False
        remaining = self._repair_runway()
        pending = self.acquirer.pending_call_counts(actions, self.plan)
        acquisition = max(
            (
                self._source_seconds.get(tool, self.policy.max_call_seconds)
                for tool in pending
            ),
            default=0.0,
        ) * ceil(sum(pending.values()) / self.policy.max_parallel_tools)
        return remaining > acquisition + self.policy.selection_reserve_seconds

    def _try_terminal_observed_reads(
        self, step: IssueResearchStep, plan: IssueResearchPlan, request: str
    ) -> bool:
        if not self.allow_terminal_observed_reads or not isinstance(
            self.acquirer, CanonicalAcquirer
        ):
            return False
        self.check_active()
        selected, remaining = self.acquirer.safe_observed_read_actions(
            step.actions,
            plan,
            set(getattr(self.gateway, "last_delivered_citations", set())),
        )
        if not selected or not self._terminal_reads_have_runway(selected):
            return False
        if remaining or step.material_dependencies:
            self._defer_followup(step.model_copy(update={"actions": remaining}), plan)
        with graph_step(
            "legal_composite.research_admission",
            {},
            summary="work_kind=sources accepted=1 reason=terminal_observed_reads",
        ) as admission:
            admission.output_value = {
                "operation": "sources",
                "accepted": True,
                "reason": "terminal_observed_reads",
                "selected_actions": len(selected),
                "deferred_actions": len(remaining),
            }
        self.report("tools", plan.language)
        before = set(self.ledger.citation_numbers())
        self._acquire(selected, plan)
        added = set(self.ledger.citation_numbers()) - before
        if added:
            self._select_sources(request, plan, added)
        return True

    def _complete_source_reading(
        self, payload: dict[str, JsonValue]
    ) -> ResearchStep | IssueResearchStep | IssueReadingResponse:
        if self.reviewer is not None and self.use_numbered_reading_supports:
            return self.gateway.complete(
                NUMBERED_READING_PROMPT,
                payload,
                IssueReadingResponse,
                LLMFlow.LEGAL_COMPOSITE_RESEARCH,
            )
        return self.gateway.complete(
            RESEARCH_PROMPT,
            payload,
            IssueResearchStep if self.reviewer is not None else ResearchStep,
            LLMFlow.LEGAL_COMPOSITE_RESEARCH,
        )

    def _resolve_source_reading(
        self, step: ResearchStep | IssueResearchStep | IssueReadingResponse
    ) -> ResearchStep | IssueResearchStep:
        if not isinstance(step, IssueReadingResponse):
            return step
        manifest = getattr(self.gateway, "last_reading_manifest", None)
        call_id = getattr(self.gateway, "last_call_id", None)
        if not isinstance(manifest, ReadingWitnessManifest) or not isinstance(
            call_id, str
        ):
            raise InvalidSourceAction(
                "Reading witness manifest does not match the active delivery"
            )
        return resolve_issue_reading(
            step,
            self.ledger,
            set(getattr(self.gateway, "last_delivered_citations", set())),
            manifest,
            call_id=call_id,
        )

    def _repair_runway(self) -> float:
        if not self.research_available():
            return 0.0
        if self.evidence_context is None:
            return float("inf")
        return max(0.0, self.evidence_context.research_deadline - time.monotonic())

    def _repair_research_has_runway(self) -> bool:
        planning = max(self._research_seconds, default=self.policy.max_call_seconds)
        acquisition = min(
            self._source_seconds.values(), default=self.policy.max_call_seconds
        )
        return (
            self._repair_runway()
            > planning + acquisition + self.policy.selection_reserve_seconds
        )

    def _repair_actions_have_runway(self, step: IssueResearchStep) -> bool:
        tools = {action.tool for action in step.actions}
        if step.material_dependencies:
            tools.add("search_corpus")
        acquisition = max(
            (
                self._source_seconds.get(tool, self.policy.max_call_seconds)
                for tool in tools
            ),
            default=0.0,
        )
        return (
            self._repair_runway() > acquisition + self.policy.selection_reserve_seconds
        )

    def _payload(
        self,
        request: str,
        history: str,
        *,
        draft: DraftAnswer | None = None,
        gaps: list[str] | None = None,
        instructions: str | None = None,
        source_phase: bool = True,
    ) -> dict[str, JsonValue]:
        # Whole originals are selected atomically; omitted IDs remain explicit navigation.
        recent_numbers = [
            number
            for receipt in reversed(self.receipts)
            if isinstance(numbers := receipt.get("citations"), list)
            for number in numbers
            if isinstance(number, int)
        ]
        required = list(extract_citation_numbers(draft.answer)) if draft else []
        if self.reviewer is not None:
            self.pending_reconsidered -= set(
                getattr(self.gateway, "last_delivered_citations", set())
            )
            required.extend(self.requirements.citations())
            required.extend(sorted(self.pending_reconsidered))
            required.extend(sorted(self.pending_selection_citations))
        required = list(
            dict.fromkeys(
                [
                    *required,
                    *dependency_required_citations(self.dependencies, self.ledger),
                ]
            )
        )
        if self.selection is not None:
            classified = {row.citation for row in self.selection.identities}
            required = list(
                dict.fromkeys(
                    [
                        *self.selection.protected_citations,
                        *(
                            number
                            for number in self.ledger.citation_numbers()
                            if number not in classified
                        ),
                        *required,
                    ]
                )
            )
            recent_numbers = [
                number
                for number in recent_numbers
                if number not in self.selection.rejected_citations
            ]
        available = [
            number
            for number in reversed(self.ledger.citation_numbers())
            if self.selection is None or number not in self.selection.rejected_citations
        ]
        records: JsonValue = json.loads(
            self.ledger.serialize_records(
                [*required, *recent_numbers, *available],
                required=required,
                max_chars=None if required else 50_000,
                include_witness_spans=self.reviewer is not None,
            )
        )
        assert isinstance(records, list)
        _add_span_previews(records)
        if self.reviewer is not None and self.use_numbered_reading_supports:
            records = number_reading_witnesses(records)
        for record in records:
            if not isinstance(record, dict):
                continue
            source_id = record.get("source_id")
            if isinstance(source_id, str):
                record["source_kind"] = self.source_kinds.get(
                    source_id, SourceKind.UNKNOWN
                ).value
        delivered = {
            int(record["citation"])
            for record in records
            if isinstance(record, dict) and isinstance(record.get("citation"), int)
        }
        payload: dict[str, JsonValue] = {
            "request": request,
            "conversation": history,
            "assistant_instructions": instructions,
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "original_evidence": records,
            "required_evidence_numbers": required,
            "omitted_original_ids": [
                number
                for number in self.ledger.citation_numbers()
                if number not in delivered
            ],
            "receipts": self.receipts[-8:],
            "draft": draft.model_dump(mode="json") if draft else None,
            "defects": gaps or [],
            "limits": self.policy.model_dump(mode="json"),
            "source_lane_inventory": getattr(self.acquirer, "lane_inventory", {}),
            "authority_dependencies": [
                edge.model_dump(mode="json") for edge in self.dependencies
            ],
            "source_requirements": self.requirements.export(),
            "research_gaps": self.research_gaps,
            "protocol_defects": self.protocol_defects,
        }
        if self.selection is not None:
            payload["source_selection"] = {
                "selection_complete": self.selection.selection_complete,
                "protected": self.selection.protected_citations,
                "background": self.selection.background_citations,
                "rejected": self.selection.rejected_citations,
                "uncertain_count": sum(
                    "uncertain" in row.roles for row in self.selection.receipts
                ),
                "uncertain_by_need": {
                    need.need_id: [
                        row.citation
                        for row in self.selection.receipts
                        if row.need_id == need.need_id and "uncertain" in row.roles
                    ]
                    for need in self.plan.needs
                }
                if self.plan
                else {},
                "instruction": "Inspect retained uncertainty from full originals; relevance alone does not prove applicability.",
            }
        if self.reviewer is not None:
            payload["coordinator_source_requests"] = self.source_requests
            payload["deferred_followup_source_actions"] = self.deferred_followup_steps
            payload["deferred_initial_source_actions"] = [
                {
                    "action": action.model_dump(mode="json"),
                    "status": status,
                    "navigation_only": True,
                }
                for action, status in self.deferred_initial_actions
            ]
            sources: dict[str, dict[str, JsonValue]] = {}
            for row in self.ledger.provision_metadata():
                source_id, metadata = row.get("source_id"), row.get("metadata")
                if not isinstance(source_id, str) or not isinstance(metadata, dict):
                    continue
                source = sources.setdefault(
                    source_id,
                    {"source_id": source_id, "titles": [], "observed_articles": []},
                )
                for field, target in (
                    ("title", "titles"),
                    ("article_no", "observed_articles"),
                ):
                    values = source[target]
                    assert isinstance(values, list)
                    value = metadata.get(field)
                    if value is not None and value not in values:
                        values.append(value)
            payload["observed_source_directory"] = list(sources.values())
        if source_phase:
            catalogue: list[dict[str, JsonValue]] = [
                {
                    "citation": row["citation"],
                    "source_id": row["source_id"],
                    "chunk_id": row["chunk_id"],
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in {
                            "title",
                            "article_no",
                            "paragraph_no",
                            "clause_label",
                            "heading_path",
                        }
                    },
                }
                for row in self.ledger.provision_metadata()
                if isinstance(metadata := row.get("metadata"), dict)
            ]
            payload["original_catalogue"] = _compact_original_catalogue(catalogue)
            payload["tools"] = self.acquirer.definitions()
        else:
            payload["receipts"] = [
                {
                    key: receipt[key]
                    for key in ("need_ids", "status", "citations")
                    if key in receipt
                }
                for receipt in self.receipts[-8:]
            ]
        return payload

    def _mark_deferred_initial_attempts(
        self, actions: list[SourceAction], status: str
    ) -> None:
        self.deferred_initial_actions = [
            (
                deferred,
                status
                if any(
                    action.tool == "search_corpus"
                    and set(deferred.need_ids) <= set(action.need_ids)
                    and _search_action_identity(deferred)
                    == _search_action_identity(action)
                    for action in actions
                )
                else previous_status,
            )
            for deferred, previous_status in self.deferred_initial_actions
        ]

    def _acquire(self, actions: list[SourceAction], plan: ResearchPlan) -> bool:
        started = time.monotonic()
        request: dict[str, JsonValue] | None = None
        if self.reviewer is not None:
            request = {
                "actions": [action.model_dump(mode="json") for action in actions],
                "status": "requested",
                "navigation_only": True,
            }
            self.source_requests.append(request)
            self._mark_deferred_initial_attempts(actions, "requested")
        try:
            self.receipts.extend(self.acquirer.acquire(actions, plan))
            if self.reviewer is not None and actions:
                elapsed = time.monotonic() - started
                for tool in {action.tool for action in actions}:
                    self._source_seconds[tool] = max(
                        self._source_seconds.get(tool, 0.0), elapsed
                    )
            if request is not None:
                request["status"] = "completed"
                self._mark_deferred_initial_attempts(actions, "attempted")
            return True
        except InvalidSourceAction:
            if request is not None:
                request["status"] = "invalid"
                self._mark_deferred_initial_attempts(actions, "invalid_attempt")
            raise
        except RunStopped:
            if request is not None:
                request["status"] = "source_phase_stopped"
                self._mark_deferred_initial_attempts(actions, "partially_attempted")
            # The finite source phase cannot spend the writer/reviewer reserve.
            self.check_active()
            retained = getattr(self.acquirer, "last_receipts", [])
            self.receipts.extend(retained)
            return False

    def run(
        self, request: str, history: str = "", instructions: str | None = None
    ) -> WorkflowResult:
        try:
            result = self._run(request, history, instructions)
            if self.reviewer is not None and result.status == "unavailable":
                self._trace_stop("semantic_rejection")
            return result
        except (RunStopped, InvalidSourceAction) as error:
            if self.reviewer is not None:
                self._trace_stop(_stop_reason(error))
            status = "cancelled" if "cancel" in str(error).lower() else "unavailable"
            if self.reviewer is not None:
                assert self.plan is None or isinstance(self.plan, IssueResearchPlan)
                return CompositeWorkflowResult(
                    answer=None,
                    status=status,
                    gaps=[str(error)],
                    plan=self.plan,
                    semantic_review=self.semantic_review,
                    source_requirements=self.requirements.records(),
                )
            return WorkflowResult(
                answer=None,
                status=status,
                gaps=[str(error)],
                plan=self.plan,
                review=self.last_review,
            )
        except Exception:
            if self.reviewer is not None:
                self._trace_stop("unhandled_exception")
            raise

    def _trace_stop(self, reason: str) -> None:
        with graph_step(
            "legal_composite.run_stop", {}, summary=f"reason={reason}"
        ) as step:
            step.output_value = {"reason": reason}

    def _trace_admission(
        self,
        kind: str,
        stage: str,
        accepted: bool,
        reason: str = "none",
        changed_claims: int = 0,
        changed_sections: int = 0,
    ) -> None:
        with graph_step(
            "legal_composite.draft_admission",
            {},
            summary=(
                f"kind={kind} accepted={int(accepted)} stage={stage} reason={reason} "
                f"changed_claims={changed_claims} changed_sections={changed_sections}"
            ),
        ) as step:
            step.output_value = {
                "kind": kind,
                "stage": stage,
                "accepted": accepted,
                "reason": reason,
                "changed_claims": changed_claims,
                "changed_sections": changed_sections,
            }

    def _select_sources(
        self, request: str, plan: ResearchPlan, numbers: set[int] | None = None
    ) -> None:
        if self.selector is not None and plan.requires_sources:
            budget = getattr(self.gateway, "budget", None)
            if budget is not None:
                budget.begin_selection()
            self.report("selection", plan.language)
            selection_request = selection_request_from_ledger(
                request, plan, self.ledger, source_kinds=self.source_kinds
            )
            if numbers is not None:
                selection_request = selection_request.model_copy(
                    update={
                        "candidates": [
                            row
                            for row in selection_request.candidates
                            if row.citation in numbers
                        ]
                    }
                )
            if not selection_request.candidates:
                return
            try:
                selected = self.selector.select(selection_request)
            except RunStopped as error:
                if isinstance(error, ResearchPhaseClosed):
                    reason = error.reason
                elif (
                    getattr(
                        self.gateway, "preserve_research_finalization_on_timeout", False
                    )
                    is True
                    and budget is not None
                    and str(error)
                    == "Workflow deadline reached; finalization time retained"
                    and budget.remaining_seconds() < 3
                ):
                    self.check_active()
                    budget.close_research("host_research_deadline")
                    reason = "host_research_deadline"
                else:
                    raise
                self.check_active()
                self.pending_selection_citations.update(
                    row.citation for row in selection_request.candidates
                )
                self.receipts.append(
                    {
                        "status": "selection_stopped",
                        "reason": reason,
                        "citations": [
                            row.citation for row in selection_request.candidates
                        ],
                        "navigation_only": True,
                    }
                )
                with graph_step(
                    "legal_composite.research_admission",
                    {},
                    summary=f"work_kind=selection accepted=0 reason={reason}",
                ) as admission:
                    admission.output_value = {
                        "operation": "selection",
                        "accepted": False,
                        "reason": reason,
                    }
                return
            if self.selection is None or numbers is None:
                self.selection = selected
            else:
                previous = self.selection
                self.selection = selected.model_copy(
                    update={
                        **{
                            key: sorted(
                                set(getattr(previous, key))
                                | set(getattr(selected, key))
                            )
                            for key in (
                                "protected_citations",
                                "background_citations",
                                "rejected_citations",
                                "retained_citations",
                            )
                        },
                        "identities": [*previous.identities, *selected.identities],
                        "receipts": [*previous.receipts, *selected.receipts],
                        "selection_complete": previous.selection_complete
                        and selected.selection_complete,
                        "gaps": list(dict.fromkeys(previous.gaps + selected.gaps)),
                    }
                )

    def _reconsider_sources(
        self,
        numbers: list[int],
        plan: ResearchPlan,
        need_ids: set[str] | None = None,
    ) -> None:
        if not numbers:
            return
        scope = {need.need_id for need in plan.needs} if need_ids is None else need_ids
        if scope - {need.need_id for need in plan.needs}:
            raise InvalidSourceAction(
                "Source reconsideration refers to an unknown issue"
            )
        requested = set(numbers)
        for number in requested:
            item = self.ledger.get(number)
            # Recovery changes eligibility, not writer delivery or legal applicability.
            if item is None or not support_is_original(
                PassageSupport(citation=number, quotation=item.text),
                self.ledger,
                requested,
            ):
                raise InvalidSourceAction(
                    "Source reconsideration requires a canonical original"
                )
        if self.selection is None:
            return
        recovered = requested & set(self.selection.rejected_citations)
        if not recovered:
            return
        if self.evidence_context is None:
            raise InvalidSourceAction("Source reconsideration requires its run context")
        rebound = []
        for number in recovered:
            item = self.ledger.get(number)
            assert item is not None
            item.question_ids = list(
                dict.fromkeys([*item.question_ids, *sorted(scope)])
            )
            rebound.append(item)
        self.ledger.add(rebound, self.evidence_context)
        self.selection = self.selection.model_copy(
            update={
                "rejected_citations": sorted(
                    set(self.selection.rejected_citations) - recovered
                ),
                "background_citations": sorted(
                    set(self.selection.background_citations) | recovered
                ),
                "retained_citations": sorted(
                    set(self.selection.retained_citations) | recovered
                ),
            }
        )
        self.pending_reconsidered.update(recovered)
        self.receipts.append(
            {
                "status": "reconsidered",
                "need_ids": sorted(scope),
                "citations": sorted(recovered),
                "summary": "Previously excluded originals retained for source reading; earlier selection receipts remain audit history.",
            }
        )

    def _refresh_dependencies(
        self,
        request: str,
        plan: ResearchPlan,
        targets: list[int] | None = None,
        material_targets: list[MaterialDependencyRequest] | None = None,
    ) -> None:
        if self.dependency_expander is not None and plan.requires_sources:
            if not targets and not material_targets:
                self.dependencies = self.dependency_expander.synchronize()
                return
            before = set(self.ledger.citation_numbers())
            frontier = set(targets or []) | {
                target.origin_citation for target in material_targets or []
            }
            delivered = getattr(self.gateway, "last_delivered_citations", before)
            if not frontier <= before or not frontier <= delivered:
                raise InvalidSourceAction(
                    "Related-source target was not a delivered original"
                )
            need_bindings: dict[int, set[str]] = {}
            if self.selection is not None:
                for row in self.selection.receipts:
                    if row.citation in frontier and set(row.roles) & {
                        "relevant",
                        "direct",
                        "condition",
                        "exception",
                        "contrary",
                    }:
                        need_bindings.setdefault(row.citation, set()).add(row.need_id)
            self.report("tools", plan.language)
            receipt_start = len(self.dependency_expander.receipts)
            dependency_started = time.monotonic()
            self.dependencies = self.dependency_expander.expand(
                plan,
                frontier=frontier,
                need_bindings=need_bindings,
                **(
                    {"material_targets": material_targets}
                    if material_targets is not None
                    else {}
                ),
            )
            self.receipts.extend(self.dependency_expander.receipts[receipt_start:])
            self._dependency_seconds.append(time.monotonic() - dependency_started)
            added = set(self.ledger.citation_numbers()) - before
            if added:
                self._select_sources(request, plan, added)

    def _run(
        self, request: str, history: str, instructions: str | None
    ) -> WorkflowResult:
        self.check_active()
        planning_started = time.monotonic()
        self.plan = self.gateway.complete(
            PLAN_PROMPT,
            self._payload(request, history, instructions=instructions),
            IssueResearchPlan if self.reviewer is not None else ResearchPlan,
            LLMFlow.LEGAL_COMPOSITE_RESEARCH,
        )
        if self.reviewer is not None:
            self._research_seconds.append(time.monotonic() - planning_started)
        if not source_free_social_request(request):
            self.plan = self.plan.model_copy(update={"requires_sources": True})
        plan = self.plan
        self.report("tools", plan.language)
        if self.reviewer is not None:
            assert isinstance(plan, IssueResearchPlan)
            self.deferred_initial_actions = [
                (action, "unexecuted_navigation")
                for action in plan.initial_actions
                if action.tool == "search_corpus" and plan.requires_sources
            ]
            initial_actions = initial_issue_discovery_actions(plan, request)
        else:
            initial_actions = initial_discovery_actions(plan, request)
        source_phase_open = not initial_actions or self._acquire(initial_actions, plan)
        self._select_sources(request, plan)
        for _round in range(self.policy.max_research_rounds):
            if (
                not source_phase_open
                or not plan.requires_sources
                or not self.research_available()
            ):
                break
            try:
                if self.reviewer is not None:
                    if not self._reading_has_runway():
                        with graph_step(
                            "legal_composite.research_admission",
                            {},
                            summary="work_kind=reading accepted=0 reason=research_runway",
                        ) as admission:
                            admission.output_value = {
                                "operation": "reading",
                                "accepted": False,
                                "reason": "research_runway",
                            }
                        break
                    self.report("reading", plan.language)
                reading_started = time.monotonic()
                response = self._complete_source_reading(
                    self._payload(request, history, instructions=instructions),
                )
                if self.reviewer is not None:
                    self._research_seconds.append(time.monotonic() - reading_started)
            except ResearchPhaseClosed as error:
                self.check_active()
                with graph_step(
                    "legal_composite.research_admission",
                    {},
                    summary=f"work_kind=reading accepted=0 reason={error.reason}",
                ) as admission:
                    admission.output_value = {
                        "operation": "reading",
                        "accepted": False,
                        "reason": error.reason,
                    }
                break
            except RunStopped:
                self.check_active()
                break
            try:
                step = self._resolve_source_reading(response)
                new_requirement_ids = (
                    {row.requirement_id for row in step.requirements}
                    - {row.requirement_id for row in self.requirements.records()}
                    if isinstance(step, IssueResearchStep)
                    else set()
                )
                if isinstance(step, IssueResearchStep):
                    assert isinstance(plan, IssueResearchPlan)
                    self._validate_gap_scope(step, plan)
                self.requirements.update(
                    step.requirements if isinstance(step, IssueResearchStep) else [],
                    plan,
                    getattr(
                        self.gateway,
                        "last_delivered_citations",
                        set(self.ledger.citation_numbers()),
                    ),
                )
            except InvalidSourceAction as error:
                if self.reviewer is not None:
                    with graph_step(
                        "legal_composite.reading_admission",
                        {},
                        summary=f"accepted=0 reason={_admission_reason(error)}",
                    ) as reading_step:
                        reading_step.output_value = {
                            "accepted": False,
                            "reason": _admission_reason(error),
                        }
                self.protocol_defects.append(str(error))
                continue
            if self.reviewer is not None:
                with graph_step(
                    "legal_composite.reading_admission",
                    {},
                    summary=f"accepted=1 new_requirements={len(new_requirement_ids)}",
                ) as reading_step:
                    reading_step.output_value = {
                        "accepted": True,
                        "new_requirements": len(new_requirement_ids),
                    }
            if isinstance(step, IssueResearchStep):
                assert isinstance(plan, IssueResearchPlan)
                self._record_research_gaps(
                    step, plan, new_requirement_ids=new_requirement_ids
                )
                self._reconsider_sources(step.reconsider_citations, plan)
            if (
                not step.actions
                and not step.related_citations
                and not (
                    isinstance(step, IssueResearchStep)
                    and (step.material_dependencies or step.reconsider_citations)
                )
            ):
                if not step.ready_to_answer and not self.research_gaps:
                    self.research_gaps.append(
                        "Research did not establish readiness for the requested outcomes."
                    )
                break
            if isinstance(step, IssueResearchStep) and not self._followup_has_runway(
                step
            ):
                assert isinstance(plan, IssueResearchPlan)
                if not self._try_terminal_observed_reads(step, plan, request):
                    self._defer_followup(step, plan)
                break
            self._refresh_dependencies(
                request,
                plan,
                step.related_citations if self.reviewer is None else [],
                step.material_dependencies
                if isinstance(step, IssueResearchStep)
                else None,
            )
            before = set(self.ledger.citation_numbers())
            source_phase_open = not step.actions or self._acquire(step.actions, plan)
            added = set(self.ledger.citation_numbers()) - before
            if added:
                self._select_sources(request, plan, added)
            self._refresh_dependencies(request, plan)
        self._refresh_dependencies(request, plan)
        self.report("final", plan.language)
        if self.reviewer is not None:
            assert isinstance(plan, IssueResearchPlan)
            return self._finalize_semantic(request, history, instructions, plan)
        gaps: list[str] = []
        draft: DraftAnswer | None = None
        safe_partial: WorkflowResult | None = None
        for attempt in range(self.policy.max_reviews):
            self.check_active()
            payload = self._payload(
                request,
                history,
                draft=draft,
                gaps=gaps,
                instructions=instructions,
                source_phase=False,
            )
            try:
                draft = self.gateway.complete(
                    ANSWER_PROMPT,
                    payload,
                    DraftAnswer,
                    LLMFlow.LEGAL_COMPOSITE_ANSWER,
                    True,
                )
            except RunStopped:
                self.check_active()
                if safe_partial is not None:
                    return safe_partial
                raise
            answer_records = payload["original_evidence"]
            assert isinstance(answer_records, list)
            answer_delivered = {
                record["citation"]
                for record in answer_records
                if isinstance(record, dict) and isinstance(record.get("citation"), int)
            }
            actual_delivery = getattr(
                self.gateway, "last_delivered_citations", answer_delivered
            )
            answer_delivered &= actual_delivery
            review_payload = self._payload(
                request,
                history,
                draft=draft,
                instructions=instructions,
                source_phase=False,
            )
            # No extra originals may silently make an unsupported writer draft acceptable.
            review_payload["original_evidence"] = answer_records
            protected_numbers = payload["required_evidence_numbers"]
            assert isinstance(protected_numbers, list)
            review_payload["required_evidence_numbers"] = list(
                dict.fromkeys(
                    [
                        *protected_numbers,
                        *extract_citation_numbers(draft.answer),
                    ]
                )
            )
            try:
                review = self.gateway.complete(
                    REVIEW_PROMPT,
                    review_payload,
                    AnswerReview,
                    LLMFlow.LEGAL_COMPOSITE_REVIEW,
                    True,
                )
            except RunStopped:
                self.check_active()
                if safe_partial is not None:
                    return safe_partial
                raise
            self.last_review = review
            actual_delivery = getattr(
                self.gateway, "last_delivered_citations", answer_delivered
            )
            answer_delivered &= actual_delivery
            passed, safe, gaps = review_assessment(
                plan, draft, review, self.ledger, answer_delivered, self.dependencies
            )
            if (
                self.selection is not None
                and not self.selection.selection_complete
                and not review.selection_uncertainty_resolved
            ):
                passed = safe = False
                gaps.append(
                    "Retained source relevance uncertainty was not independently resolved against the originals."
                )
            if passed:
                return WorkflowResult(
                    answer=draft.answer,
                    status="verified",
                    gaps=[],
                    plan=plan,
                    review=review,
                )
            if safe:
                safe_partial = WorkflowResult(
                    answer=draft.answer,
                    status="partial",
                    gaps=gaps,
                    plan=plan,
                    review=review,
                )
            if attempt + 1 == self.policy.max_reviews:
                return WorkflowResult(
                    answer=draft.answer if safe else None,
                    status="partial" if safe else "unavailable",
                    gaps=gaps,
                    plan=plan,
                    review=review,
                )
            if review.repair_actions and self.research_available():
                before = set(self.ledger.citation_numbers())
                self._acquire(review.repair_actions, plan)
                added = set(self.ledger.citation_numbers()) - before
                if added:
                    safe_partial = None
                    self._select_sources(request, plan, added)
                self._refresh_dependencies(request, plan)
        raise AssertionError("At least one review is required")

    def _finalize_semantic(
        self,
        request: str,
        history: str,
        instructions: str | None,
        plan: IssueResearchPlan,
    ) -> WorkflowResult:
        assert self.reviewer is not None
        payload = self._payload(
            request, history, instructions=instructions, source_phase=False
        )
        composition = self.gateway.complete(
            DRAFT_COMPOSITION_PROMPT,
            payload,
            DraftComposition,
            LLMFlow.LEGAL_COMPOSITE_ANSWER,
            True,
        )
        try:
            draft = compose_draft(composition)
        except InvalidSourceAction as error:
            self._trace_admission("draft", "scope", False, _admission_reason(error))
            raise
        with graph_step(
            "legal_composite.draft_composition",
            {},
            summary=(
                f"sections={len(draft.sections)} claims={len(draft.claims)} "
                f"characters={len(draft.answer)}"
            ),
        ) as composition_step:
            composition_step.output_value = {
                "sections": len(draft.sections),
                "claims": len(draft.claims),
                "characters": len(draft.answer),
            }
        delivered = set(getattr(self.gateway, "last_delivered_citations", set()))
        pending_binding_gaps: list[str] = []
        admission_stage = "supports"
        try:
            draft = canonicalize_draft_supports(draft, self.ledger, delivered)
            admission_stage = "reading"
            self._accept_draft_reading(
                draft.requirements, draft.gap_resolutions, plan, delivered
            )
            self._trace_admission("draft", admission_stage, True)
        except InvalidSourceAction as error:
            self._trace_admission(
                "draft", admission_stage, False, _admission_reason(error)
            )
            pending_binding_gaps.append(str(error))
        gaps: list[str] = []
        previous: SemanticReview | None = None
        affected: set[str] | None = None
        for attempt in range(self.policy.max_reviews):
            self.check_active()
            requirements = self.requirements.records()
            binding_gaps = draft_binding_gaps(
                draft, plan, requirements, self.ledger, delivered
            )
            binding_gaps.extend(pending_binding_gaps)
            pending_binding_gaps = []
            categories = {
                "invalid_answer_binding",
                "invalid_original_binding",
                "invalid_requirement_binding",
                "not_delivered_original",
                "no_stable_sections",
                "unknown_unresolved_issue",
                "unknown_section_issue",
                "issue_without_section",
                "missing_original_claim_bindings",
            }
            binding_counts: dict[str, int] = {}
            for gap in binding_gaps:
                category = gap.rsplit(":", 1)[-1]
                if category not in categories:
                    category = "other"
                binding_counts[category] = binding_counts.get(category, 0) + 1
            summary_parts: list[str] = []
            for key, value in sorted(binding_counts.items()):
                part = f"{key}={value}"
                if len(" ".join([*summary_parts, part])) > 180:
                    break
                summary_parts.append(part)
            with graph_step(
                "legal_composite.binding_diagnostics",
                {},
                summary=" ".join(summary_parts),
            ) as binding_step:
                binding_step.output_value = binding_counts
            questions = self.reviewer.expected_checks(
                request,
                plan,
                draft,
                requirements,
                self.dependencies,
                delivered,
                previous=previous,
                affected_sections=affected,
            )
            self.report("review", plan.language)
            review = self.reviewer.review(
                request,
                plan,
                draft,
                requirements,
                self.dependencies,
                delivered,
                previous=previous,
                affected_sections=affected,
            )
            self.semantic_review = review
            checked = [check.check_id for check in review.checks]
            shape_valid = len(checked) == len(set(checked)) and set(checked) == set(
                questions
            )
            failed = []
            reviewed_by_id = {check.check_id: check for check in review.checks}
            needs_by_id = {need.need_id: need for need in plan.needs}
            permit_disclosed_partial = (
                attempt + 1 >= self.policy.max_reviews or not self.research_available()
            )
            for check in review.checks:
                question = questions.get(check.check_id)
                if question is None:
                    continue
                disclosed_source_gap = (
                    shape_valid
                    and not review.failure
                    and permit_disclosed_partial
                    and set(check.need_ids) == set(question.need_ids)
                    and set(check.section_ids) == set(question.section_ids)
                    and check.check_id.startswith("evidence:")
                    and check.status == "gap"
                    and check.confidence >= 0.80
                    and bool(check.need_ids)
                    and all(
                        need_id in draft.unresolved_need_ids
                        and need_id in needs_by_id
                        and bool(needs_by_id[need_id].evidence_gaps)
                        and (issue := reviewed_by_id.get(f"issue:{need_id}"))
                        is not None
                        and issue.status == "addressed"
                        and issue.confidence >= 0.80
                        for need_id in check.need_ids
                    )
                )
                if disclosed_source_gap:
                    continue
                if (
                    set(check.need_ids) != set(question.need_ids)
                    or set(check.section_ids) != set(question.section_ids)
                    or check.confidence < 0.80
                    or check.status not in {"addressed", "not_applicable"}
                    or (
                        check.status == "not_applicable"
                        and not question.allow_not_applicable
                    )
                ):
                    failed.append(check)
            gaps = [*binding_gaps]
            if not shape_valid:
                gaps.append("review:missing_or_duplicate_check_identity")
            if review.failure:
                gaps.append(f"review:provider_failure:{review.failure}")
            gaps.extend(f"{check.check_id}:{check.status}" for check in failed)
            dependency_gaps = material_dependency_gaps(
                self.dependencies, self.ledger, delivered, requirements
            )
            by_id = {check.check_id: check for check in review.checks}
            for edge_id, reasons in dependency_gaps.items():
                if not reasons:
                    continue
                edge = next(row for row in self.dependencies if row.edge_id == edge_id)
                origin_numbers = {origin.citation for origin in edge.origins}
                integrity_faults = [
                    reason
                    for reason in reasons
                    if reason.startswith(
                        ("origin_binding_mismatch", "citation_noncanonical")
                    )
                    or (
                        reason.startswith(
                            ("citation_missing@", "citation_undelivered@")
                        )
                        and int(reason.rsplit("@", 1)[1]) in origin_numbers
                    )
                ]
                if integrity_faults:
                    gaps.extend(
                        f"dependency:{edge_id}:{reason}" for reason in integrity_faults
                    )
                    continue
                check = by_id.get(f"dependency:{edge_id}")
                if (
                    check is not None
                    and check.status == "not_applicable"
                    and check.confidence >= 0.80
                ):
                    # Nonmateriality is judged against actual delivered original text,
                    # never inferred merely from an empty relationship lookup.
                    continue
                if set(edge.need_ids) <= set(draft.unresolved_need_ids):
                    continue
                gaps.extend(f"dependency:{edge_id}:{reason}" for reason in reasons)
            unsupported = {
                need.need_id
                for need in plan.needs
                if plan.requires_sources
                and not any(
                    requirement.need_id == need.need_id for requirement in requirements
                )
            }
            undisclosed = unsupported - set(draft.unresolved_need_ids)
            undisclosed |= {
                need.need_id
                for need in plan.needs
                if need.evidence_gaps and need.need_id not in draft.unresolved_need_ids
            }
            gaps.extend(
                f"issue:{need}:no_source_backed_requirement"
                for need in sorted(undisclosed)
            )
            if not gaps:
                partial = bool(draft.unresolved_need_ids or self.research_gaps)
                return CompositeWorkflowResult(
                    answer=draft.answer,
                    status="partial" if partial else "verified",
                    gaps=list(self.research_gaps),
                    plan=plan,
                    semantic_review=review,
                    source_requirements=requirements,
                )
            if attempt + 1 >= self.policy.max_reviews:
                break
            if review.failure or not shape_valid:
                break
            affected_needs = {
                need for check in failed for need in check.need_ids
            } | undisclosed
            affected = {
                section.section_id
                for section in draft.sections
                if set(section.need_ids) & affected_needs
            }
            if binding_gaps or not shape_valid or review.failure or not affected:
                affected = {section.section_id for section in draft.sections}
            # Cross-issue conclusions and summaries are rechecked with their dependencies.
            affected |= {
                section.section_id
                for section in draft.sections
                if len(section.need_ids) > 1
            }
            affected_needs |= {
                need
                for section in draft.sections
                if section.section_id in affected
                for need in section.need_ids
            }
            missing_evidence = any(
                check.status in {"gap", "incorrect"}
                and check.check_id.startswith(("evidence:", "gap-resolution:"))
                for check in failed
            ) or any(dependency_gaps.values())
            negative_counts: dict[str, int] = {}
            for check in failed:
                candidate_category = check.check_id.split(":", 1)[0].replace("-", "_")
                category = (
                    candidate_category
                    if candidate_category
                    in {
                        "issue",
                        "evidence",
                        "dimension",
                        "original",
                        "requirement",
                        "claim",
                        "dependency",
                        "summary",
                        "request",
                        "gap_resolution",
                        "request_coverage",
                        "cross_issue_consistency",
                    }
                    else "other"
                )
                negative_counts[category] = negative_counts.get(category, 0) + 1
            repair_research_ready = (
                missing_evidence and self._repair_research_has_runway()
            )
            with graph_step(
                "legal_composite.repair_plan",
                {},
                summary=(
                    f"failed={len(failed)} binding={len(binding_gaps)} "
                    f"law_gap={int(missing_evidence)} runway={int(repair_research_ready)} "
                    + " ".join(
                        f"{key}={value}"
                        for key, value in sorted(negative_counts.items())
                    )
                )[:160],
            ) as repair_step:
                repair_step.output_value = {
                    "failed_checks": len(failed),
                    "binding_defects": len(binding_gaps),
                    "missing_evidence": missing_evidence,
                    "negative_check_categories": negative_counts,
                }
            if repair_research_ready:
                repair_research = self._focused_payload(
                    request, history, draft, gaps, instructions, affected_needs, True
                )
                repair_research["review_findings"] = [
                    {
                        "check": check.model_dump(mode="json"),
                        "question": questions[check.check_id].question,
                    }
                    for check in failed
                    if check.check_id in questions
                ]
                try:
                    step = self._resolve_source_reading(
                        self._complete_source_reading(repair_research)
                    )
                    assert isinstance(step, IssueResearchStep)
                    self._validate_gap_scope(step, plan, affected_needs)
                    new_requirement_ids = {
                        row.requirement_id for row in step.requirements
                    } - {row.requirement_id for row in self.requirements.records()}
                    self.requirements.update(
                        step.requirements,
                        plan,
                        set(getattr(self.gateway, "last_delivered_citations", set())),
                    )
                    self._record_research_gaps(
                        step,
                        plan,
                        affected_needs,
                        new_requirement_ids=new_requirement_ids,
                    )
                    self._reconsider_sources(
                        step.reconsider_citations, plan, affected_needs
                    )
                    if self._repair_actions_have_runway(step):
                        self._refresh_dependencies(
                            request, plan, [], step.material_dependencies
                        )
                        before = set(self.ledger.citation_numbers())
                        self._acquire(step.actions, plan)
                        added = set(self.ledger.citation_numbers()) - before
                        if added:
                            self._select_sources(request, plan, added)
                        self._refresh_dependencies(request, plan)
                except RunStopped:
                    self.check_active()
            patch_payload = self._focused_payload(
                request, history, draft, gaps, instructions, affected_needs, False
            )
            patch_payload["affected_section_ids"] = sorted(affected)
            patch_payload["repair_targets"] = [
                {
                    "section_id": section.section_id,
                    "need_ids": list(section.need_ids),
                    "existing_claim_ids": list(section.claim_ids),
                }
                for section in draft.sections
                if section.section_id in affected
            ]
            patch_payload["active_requirement_ids"] = [
                row.requirement_id for row in self.requirements.records()
            ]
            patch_payload["review_findings"] = [
                {
                    "check": check.model_dump(mode="json"),
                    "question": questions[check.check_id].question,
                }
                for check in failed
                if check.check_id in questions
            ]
            self.report("repair", plan.language)
            edits = self.gateway.complete(
                CLAIM_REPAIR_EDITS_PROMPT,
                patch_payload,
                ClaimRepairEdits,
                LLMFlow.LEGAL_COMPOSITE_ANSWER,
                True,
            )
            admission_stage = "scope"
            try:
                patch = claim_edits_to_delta(draft, edits, affected)
                admission_stage = "supports"
                patch = canonicalize_delta_supports(
                    patch,
                    self.ledger,
                    set(getattr(self.gateway, "last_delivered_citations", set())),
                )
                admission_stage = "scope"
                candidate_draft = apply_claim_delta(draft, patch, affected)
                admission_stage = "reading"
                self._accept_draft_reading(
                    patch.requirements,
                    patch.gap_resolutions,
                    plan,
                    set(getattr(self.gateway, "last_delivered_citations", set())),
                    affected_needs,
                )
                prior_claims = {claim.claim_id: claim for claim in draft.claims}
                prior_sections = {
                    section.section_id: section for section in draft.sections
                }
                self._trace_admission(
                    "patch",
                    admission_stage,
                    True,
                    changed_claims=sum(
                        prior_claims.get(claim.claim_id) != claim
                        for claim in candidate_draft.claims
                    )
                    + len(patch.deleted_claim_ids),
                    changed_sections=sum(
                        prior_sections.get(section.section_id) != section
                        for section in candidate_draft.sections
                    ),
                )
                draft = candidate_draft
            except InvalidSourceAction as error:
                self._trace_admission(
                    "patch", admission_stage, False, _admission_reason(error)
                )
                pending_binding_gaps.append(str(error))
            delivered |= set(getattr(self.gateway, "last_delivered_citations", set()))
            previous = review
        return CompositeWorkflowResult(
            answer=None,
            status="unavailable",
            gaps=list(dict.fromkeys(gaps)),
            plan=plan,
            semantic_review=self.semantic_review,
            source_requirements=self.requirements.records(),
        )

    def _focused_payload(
        self,
        request: str,
        history: str,
        draft: StructuredDraftAnswer,
        gaps: list[str],
        instructions: str | None,
        need_ids: set[str],
        source_phase: bool,
    ) -> dict[str, JsonValue]:
        payload = self._payload(
            request,
            history,
            draft=draft,
            gaps=gaps,
            instructions=instructions,
            source_phase=source_phase,
        )
        required = self.requirements.citations(need_ids)
        required.update(self.pending_reconsidered)
        required.update(
            support.citation
            for claim in draft.claims
            if set(claim.need_ids) & need_ids
            for support in claim.supports
        )
        dependencies = [
            edge for edge in self.dependencies if set(edge.need_ids) & need_ids
        ]
        required.update(dependency_required_citations(dependencies, self.ledger))
        available = [
            number
            for number in self.ledger.citation_numbers()
            if (item := self.ledger.get(number)) is not None
            and (not item.question_ids or set(item.question_ids) & need_ids)
            and (
                self.selection is None
                or number not in self.selection.rejected_citations
            )
        ]
        records: JsonValue = json.loads(
            self.ledger.serialize_records(
                [*sorted(required), *available],
                required=required,
                max_chars=None if required else 50_000,
                include_witness_spans=True,
            )
        )
        assert isinstance(records, list)
        _add_span_previews(records)
        payload["original_evidence"] = records
        payload["required_evidence_numbers"] = sorted(required)
        payload["omitted_original_ids"] = [
            number
            for number in self.ledger.citation_numbers()
            if number
            not in {row["citation"] for row in records if isinstance(row, dict)}
        ]
        payload["affected_need_ids"] = sorted(need_ids)
        return payload

    def _accept_draft_reading(
        self,
        requirements: list[SourceRequirement],
        resolutions: list[GapResolution],
        plan: IssueResearchPlan,
        delivered: set[int],
        need_ids: set[str] | None = None,
    ) -> None:
        if not resolutions:
            self.requirements.update(requirements, plan, delivered)
            return
        scoped = {row.need_id for row in resolutions}
        if need_ids is not None and scoped - need_ids:
            raise InvalidSourceAction("Draft closure refers to an unaffected issue")
        by_need = {need.need_id: need for need in plan.needs}
        if scoped - set(by_need):
            raise InvalidSourceAction("Draft closure refers to an unknown issue")
        closed = {(row.need_id, row.gap) for row in resolutions}
        step = IssueResearchStep(
            actions=[],
            ready_to_answer=True,
            remaining_gaps=[],
            requirements=requirements,
            gap_resolutions=resolutions,
            issue_gaps={
                identity: [
                    gap
                    for gap in by_need[identity].evidence_gaps
                    if (identity, gap) not in closed
                ]
                for identity in scoped
            },
        )
        fresh = {row.requirement_id for row in requirements} - {
            row.requirement_id for row in self.requirements.records()
        }
        self._validate_gap_scope(step, plan, scoped, fresh)
        self.requirements.update(requirements, plan, delivered)
        self._record_research_gaps(step, plan, scoped, new_requirement_ids=fresh)

    def _validate_gap_scope(
        self,
        step: IssueResearchStep,
        plan: IssueResearchPlan,
        need_ids: set[str] | None = None,
        new_requirement_ids: set[str] | None = None,
    ) -> None:
        needs = {need.need_id: need for need in plan.needs}
        scope = set(needs) if need_ids is None else need_ids
        if scope - set(needs) or set(step.issue_gaps) - scope:
            raise InvalidSourceAction(
                "Research gaps refer to an unknown or unaffected issue"
            )
        candidates = {row.requirement_id: row for row in step.requirements}
        fresh = (
            new_requirement_ids
            if new_requirement_ids is not None
            else (
                set(candidates)
                - {row.requirement_id for row in self.requirements.records()}
            )
        )
        seen: set[tuple[str, str]] = set()
        for resolution in step.gap_resolutions:
            identity = (resolution.need_id, resolution.gap)
            if (
                resolution.need_id not in scope
                or identity in seen
                or resolution.gap
                not in {
                    *needs[resolution.need_id].evidence_gaps,
                    *(
                        row.gap
                        for row in needs[resolution.need_id].evidence_gap_resolutions
                    ),
                }
                or resolution.need_id not in step.issue_gaps
                or resolution.gap in step.issue_gaps[resolution.need_id]
                or len(resolution.requirement_ids)
                != len(set(resolution.requirement_ids))
                or not set(resolution.requirement_ids) <= fresh
                or any(
                    key not in candidates
                    or candidates[key].need_id != resolution.need_id
                    for key in resolution.requirement_ids
                )
            ):
                raise InvalidSourceAction(
                    "Gap closure lacks a fresh same-issue requirement binding"
                )
            seen.add(identity)

    def _record_research_gaps(
        self,
        step: IssueResearchStep,
        plan: IssueResearchPlan,
        need_ids: set[str] | None = None,
        *,
        new_requirement_ids: set[str] | None = None,
    ) -> None:
        self._validate_gap_scope(step, plan, need_ids, new_requirement_ids)
        scope = {need.need_id for need in plan.needs} if need_ids is None else need_ids
        active_ids = {row.requirement_id for row in self.requirements.records()}
        resolutions = {
            (row.need_id, row.gap): row
            for row in step.gap_resolutions
            if set(row.requirement_ids) <= active_ids
        }
        for need in plan.needs:
            if need.need_id not in scope:
                continue
            if need.need_id in step.issue_gaps:
                proposed = list(step.issue_gaps[need.need_id])
                retained = []
                for old_gap in need.evidence_gaps:
                    resolution = resolutions.get((need.need_id, old_gap))
                    if old_gap not in proposed and resolution is None:
                        retained.append(old_gap)
                need.evidence_gaps = list(dict.fromkeys([*proposed, *retained]))
                for resolution in step.gap_resolutions:
                    if resolution.need_id == need.need_id:
                        # The sole semantic reviewer must verify the latest exact interaction.
                        need.evidence_gap_resolutions.append(
                            resolution.model_copy(deep=True)
                        )
            elif step.remaining_gaps and not step.issue_gaps:
                need.evidence_gaps = list(
                    dict.fromkeys([*need.evidence_gaps, *step.remaining_gaps])
                )
        self.research_gaps = list(
            dict.fromkeys(
                [
                    *step.remaining_gaps,
                    *(gap for need in plan.needs for gap in need.evidence_gaps),
                ]
            )
        )
