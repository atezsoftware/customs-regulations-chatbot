from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol, TypeVar

from pydantic import BaseModel, JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import (
    AnswerReview,
    DraftAnswer,
    ResearchPlan,
    ResearchStep,
    SourceAction,
    WorkflowPolicy,
    WorkflowResult,
)
from onyx.legal_composite.prompts import (
    ANSWER_PROMPT,
    PLAN_PROMPT,
    RESEARCH_PROMPT,
    REVIEW_PROMPT,
)
from onyx.legal_composite.selection import (
    SourceSelectionResult,
    SourceSelector,
    selection_request_from_ledger,
)
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


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
    safe = structurally_valid and review.material_claims_supported
    passed = (
        safe
        and complete
        and review.request_coverage_complete
        and review.counter_authority_checked
        and not review.defects
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
    ) -> None:
        self.gateway = gateway
        self.acquirer = acquirer
        self.ledger = ledger
        self.policy = policy
        self.check_active = check_active
        self.research_available = research_available
        self.report = report
        self.selector = selector
        self.selection: SourceSelectionResult | None = None
        self.plan: ResearchPlan | None = None
        self.receipts: list[dict[str, JsonValue]] = []
        self.last_review: AnswerReview | None = None

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
            )
        )
        assert isinstance(records, list)
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

    def _acquire(self, actions: list[SourceAction], plan: ResearchPlan) -> bool:
        try:
            self.receipts.extend(self.acquirer.acquire(actions, plan))
            return True
        except InvalidSourceAction:
            raise
        except RunStopped:
            # The finite source phase cannot spend the writer/reviewer reserve.
            self.check_active()
            retained = getattr(self.acquirer, "last_receipts", [])
            self.receipts.extend(retained)
            return False

    def run(
        self, request: str, history: str = "", instructions: str | None = None
    ) -> WorkflowResult:
        try:
            return self._run(request, history, instructions)
        except RunStopped as error:
            return WorkflowResult(
                answer=None,
                status="cancelled" if "cancel" in str(error).lower() else "unavailable",
                gaps=[str(error)],
                plan=self.plan,
                review=self.last_review,
            )

    def _run(
        self, request: str, history: str, instructions: str | None
    ) -> WorkflowResult:
        self.check_active()
        self.plan = self.gateway.complete(
            PLAN_PROMPT,
            self._payload(request, history, instructions=instructions),
            ResearchPlan,
            LLMFlow.LEGAL_COMPOSITE_RESEARCH,
        )
        if not source_free_social_request(request):
            self.plan = self.plan.model_copy(update={"requires_sources": True})
        plan = self.plan
        self.report("tools", plan.language)
        source_phase_open = not plan.initial_actions or self._acquire(
            plan.initial_actions, plan
        )
        for _round in range(self.policy.max_research_rounds):
            if (
                not source_phase_open
                or not plan.requires_sources
                or not self.research_available()
            ):
                break
            try:
                step = self.gateway.complete(
                    RESEARCH_PROMPT,
                    self._payload(request, history, instructions=instructions),
                    ResearchStep,
                    LLMFlow.LEGAL_COMPOSITE_RESEARCH,
                )
            except RunStopped:
                self.check_active()
                break
            if step.ready_to_answer or not step.actions:
                break
            source_phase_open = self._acquire(step.actions, plan)
        if self.selector is not None and plan.requires_sources:
            budget = getattr(self.gateway, "budget", None)
            if budget is not None:
                budget.begin_selection()
            self.report("selection", plan.language)
            selection_request = selection_request_from_ledger(
                request, plan, self.ledger
            )
            lane_kinds = {
                number: kind
                for receipt in self.receipts
                if isinstance(kind := receipt.get("source_kind"), str)
                if isinstance(numbers := receipt.get("citations"), list)
                for number in numbers
                if isinstance(number, int)
            }
            selection_request = selection_request.model_copy(
                update={
                    "candidates": [
                        candidate.model_copy(
                            update={
                                "source_kind": lane_kinds.get(
                                    candidate.citation, "unknown"
                                )
                            }
                        )
                        for candidate in selection_request.candidates
                    ]
                }
            )
            self.selection = self.selector.select(selection_request)
        self.report("final", plan.language)
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
                plan, draft, review, self.ledger, answer_delivered
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
                self._acquire(review.repair_actions, plan)
        raise AssertionError("At least one review is required")
