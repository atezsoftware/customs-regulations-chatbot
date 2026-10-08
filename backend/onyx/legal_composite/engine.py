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
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


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
    expected = {need.need_id for need in plan.needs}
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
    ) -> None:
        self.gateway = gateway
        self.acquirer = acquirer
        self.ledger = ledger
        self.policy = policy
        self.check_active = check_active
        self.research_available = research_available
        self.report = report
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
        records: JsonValue = json.loads(
            self.ledger.serialize_records(
                [*required, *recent_numbers, *reversed(self.ledger.citation_numbers())],
                required=required,
                max_chars=50_000,
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
            "original_catalogue": [
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
            ],
            "receipts": self.receipts[-8:],
            "draft": draft.model_dump(mode="json") if draft else None,
            "defects": gaps or [],
            "limits": self.policy.model_dump(mode="json"),
        }
        if source_phase:
            payload["tools"] = self.acquirer.definitions()
        else:
            payload.pop("original_catalogue")
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
            review_payload["required_evidence_numbers"] = list(
                extract_citation_numbers(draft.answer)
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
