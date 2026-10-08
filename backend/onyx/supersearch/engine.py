"""One plan/query batch, canonical closure, one writer and witnessed review."""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.dependencies import DependencyExpander
from onyx.legal_composite.engine import (
    ModelGateway,
    SourceAcquirer,
    review_assessment,
    source_free_social_request,
)
from onyx.legal_composite.models import (
    AnswerReview,
    AuthorityDependency,
    DraftAnswer,
    ResearchPlan,
    SourceAction,
    WorkflowResult,
)
from onyx.prompts.supersearch.prompts import (
    ANSWER_PROMPT,
    PLAN_PROMPT,
    REPAIR_PROMPT,
    REVIEW_PROMPT,
)
from onyx.supersearch.models import AnswerRepair, WriterDecision
from onyx.supersearch.witnesses import bind_review_witnesses
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS
from onyx.tracing.flows import LLMFlow


def initial_source_actions(plan: ResearchPlan, request: str) -> list[SourceAction]:
    actions = list(plan.initial_actions)
    if not plan.requires_sources:
        return actions
    covered = {need_id for action in actions for need_id in action.need_ids}
    missing = [need for need in plan.needs if need.need_id not in covered]
    for need in missing:
        actions.append(
            SourceAction(
                need_ids=[need.need_id],
                tool="search_corpus",
                arguments={
                    "query": (
                        plan.discovery_query
                        if len(missing) == len(plan.needs) == 1
                        else need.question
                    ).strip()[:REGULATORY_MAX_SEARCH_QUERY_CHARS]
                    or request[:REGULATORY_MAX_SEARCH_QUERY_CHARS],
                    "mode": "hybrid",
                    "coverage_item": need.need_id,
                    "evidence_target": "Own governing original, material conditions, procedure and applicable limiting authority",
                    "expand_query": False,
                },
            )
        )
    return actions


def apply_passage_patches(answer: str, repair: AnswerRepair) -> str:
    ranges: list[tuple[int, int, str]] = []
    for patch in repair.patches:
        if answer.count(patch.old_text) != 1:
            raise RunStopped(
                "A targeted correction did not identify one exact answer passage"
            )
        start = answer.index(patch.old_text)
        end = start + len(patch.old_text)
        if any(start < old_end and old_start < end for old_start, old_end, _ in ranges):
            raise RunStopped("Targeted answer correction passages overlap")
        ranges.append((start, end, patch.new_text))
    for start, end, new_text in sorted(ranges, reverse=True):
        answer = answer[:start] + new_text + answer[end:]
    if not answer.strip():
        raise RunStopped("Targeted correction removed the complete answer")
    return answer


class SupersearchEngine:
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        acquirer: SourceAcquirer,
        ledger: EvidenceLedger,
        check_active: Callable[[], None],
        dependency_expander: DependencyExpander | None = None,
        report: Callable[[str, str], None] = lambda _phase, _language: None,
    ) -> None:
        self.gateway = gateway
        self.acquirer = acquirer
        self.ledger = ledger
        self.check_active = check_active
        self.dependency_expander = dependency_expander
        self.report = report
        self.plan: ResearchPlan | None = None
        self.dependencies: list[AuthorityDependency] = []
        self.receipts: list[dict[str, JsonValue]] = []
        self.last_review: AnswerReview | None = None
        self._source_frontier: set[int] = set()

    def _payload(
        self,
        request: str,
        history: str,
        instructions: str | None,
        *,
        draft: DraftAnswer | None = None,
        defects: list[str] | None = None,
        tools: bool = False,
    ) -> dict[str, JsonValue]:
        # Every retained complete original reaches the deciding writer and reviewer.
        # A context failure is explicit, never a top-k truncation of legal coverage.
        numbers = list(self.ledger.citation_numbers())
        payload: dict[str, JsonValue] = {
            "request": request,
            "conversation": history,
            "assistant_instructions": instructions,
            "scope": "PC Külliyatı document set only; external research unavailable",
            "plan": self.plan.model_dump(mode="json") if self.plan else None,
            "original_evidence": json.loads(
                self.ledger.serialize_records(numbers, required=numbers, max_chars=None)
            ),
            "required_evidence_numbers": numbers,
            "receipts": self.receipts,
            "authority_dependencies": [
                edge.model_dump(mode="json") for edge in self.dependencies
            ],
            "draft": draft.model_dump(mode="json") if draft else None,
            "defects": defects or [],
        }
        if tools:
            payload["tools"] = self.acquirer.definitions()
        return payload

    def _acquire(self, actions: list[SourceAction]) -> bool:
        assert self.plan is not None
        if not actions:
            return False
        self.report("tools", self.plan.language)
        receipts = self.acquirer.acquire(actions, self.plan)
        self.receipts.extend(receipts)
        self._source_frontier.update(
            number
            for receipt in receipts
            if isinstance(numbers := receipt.get("citations"), list)
            for number in numbers
            if isinstance(number, int)
        )
        # A cached receipt can bind a new need but cannot restart the same failed frontier.
        return any(row.get("reused") is not True for row in receipts)

    def _close_dependencies(self) -> None:
        if (
            self.dependency_expander is None
            or self.plan is None
            or not self.plan.requires_sources
        ):
            return
        self.check_active()
        receipt_start = len(self.dependency_expander.receipts)
        # Only explicit planner/writer/reviewer acquisitions enter the material
        # frontier. Identity openings and incidental candidate references do not
        # recursively turn a focused question into a crawl of the whole corpus.
        self.dependencies = self.dependency_expander.expand(
            self.plan, frontier=set(self._source_frontier)
        )
        self.receipts.extend(self.dependency_expander.receipts[receipt_start:])

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
            self._payload(request, history, instructions, tools=True),
            ResearchPlan,
            LLMFlow.SUPERSEARCH_PLAN,
        )
        if not source_free_social_request(request):
            self.plan = self.plan.model_copy(update={"requires_sources": True})
        plan = self.plan
        if plan.requires_sources and any(
            not need.conditions_to_check for need in plan.needs
        ):
            raise RunStopped(
                "A material legal need omitted its decisive conditions from the frozen plan"
            )
        self._acquire(initial_source_actions(plan, request))
        self._close_dependencies()
        if plan.requires_sources and not self.ledger.citation_numbers():
            return WorkflowResult(
                answer="PC Külliyatı kapsamındaki bu aramada, soruyu yanıtlamaya yeterli erişilebilir özgün dayanak elde edilemedi. Bu sonuç, ilgili hukuki hükmün bulunmadığını göstermez."
                if plan.language.startswith("tr")
                else "This search of the PC Külliyatı document set did not obtain sufficient accessible original evidence to answer the question. This does not establish that the legal rule does not exist.",
                status="partial",
                gaps=[
                    "No accessible canonical originals were delivered by the scoped source actions."
                ],
                plan=plan,
            )
        seen_actions: set[str] = set()
        while True:
            self.check_active()
            self.report("final", plan.language)
            decision = self.gateway.complete(
                ANSWER_PROMPT,
                self._payload(request, history, instructions, tools=True),
                WriterDecision,
                LLMFlow.SUPERSEARCH_ANSWER,
                True,
            )
            if decision.answer:
                draft = DraftAnswer(
                    answer=decision.answer,
                    unresolved_need_ids=decision.unresolved_need_ids,
                )
                answer_delivered = set(
                    getattr(
                        self.gateway,
                        "last_delivered_citations",
                        set(self.ledger.citation_numbers()),
                    )
                )
                break
            signature = json.dumps(
                [action.model_dump(mode="json") for action in decision.actions],
                sort_keys=True,
            )
            if signature in seen_actions or not self._acquire(decision.actions):
                raise RunStopped(
                    "The writer requested an unchanged source frontier; unresolved law cannot be guessed"
                )
            seen_actions.add(signature)
            self._close_dependencies()
        seen_repairs: set[str] = set()
        while True:
            self.check_active()
            self.report("review", plan.language)
            review = self.gateway.complete(
                REVIEW_PROMPT,
                self._payload(request, history, instructions, draft=draft),
                AnswerReview,
                LLMFlow.SUPERSEARCH_REVIEW,
                True,
            )
            review = bind_review_witnesses(review, draft, self.ledger)
            self.last_review = review
            delivered = answer_delivered & set(
                getattr(self.gateway, "last_delivered_citations", answer_delivered)
            )
            passed, safe, defects = review_assessment(
                plan, draft, review, self.ledger, delivered, self.dependencies
            )
            if passed:
                return WorkflowResult(
                    answer=draft.answer,
                    status="verified",
                    gaps=[],
                    plan=plan,
                    review=review,
                )
            if safe and not review.repair_actions and not review.defects:
                return WorkflowResult(
                    answer=draft.answer,
                    status="partial",
                    gaps=defects,
                    plan=plan,
                    review=review,
                )
            signature = json.dumps(
                [
                    list(self.ledger.citation_numbers()),
                    defects,
                    [action.model_dump() for action in review.repair_actions],
                ],
                sort_keys=True,
            )
            if signature in seen_repairs:
                return WorkflowResult(
                    answer=draft.answer if safe else None,
                    status="partial" if safe else "unavailable",
                    gaps=defects,
                    plan=plan,
                    review=review,
                )
            seen_repairs.add(signature)
            new_sources = self._acquire(review.repair_actions)
            if new_sources:
                self._close_dependencies()
            repair = self.gateway.complete(
                REPAIR_PROMPT,
                self._payload(
                    request, history, instructions, draft=draft, defects=defects
                ),
                AnswerRepair,
                LLMFlow.SUPERSEARCH_REPAIR,
                True,
            )
            patched = apply_passage_patches(draft.answer, repair)
            updated = DraftAnswer(
                answer=patched, unresolved_need_ids=repair.unresolved_need_ids
            )
            if updated == draft and not new_sources:
                return WorkflowResult(
                    answer=draft.answer if safe else None,
                    status="partial" if safe else "unavailable",
                    gaps=defects,
                    plan=plan,
                    review=review,
                )
            draft = updated
            answer_delivered |= set(
                getattr(
                    self.gateway,
                    "last_delivered_citations",
                    set(self.ledger.citation_numbers()),
                )
            )
