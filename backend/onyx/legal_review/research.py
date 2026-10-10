"""One discovery attempt per material question, retained across review stages."""

from collections.abc import Sequence

from pydantic import JsonValue

from onyx.legal_review.models import (
    IssuePlan,
    LegalDimension,
    ResearchNeed,
    ReviewResearchTask,
    SourceAction,
)


class ResearchLedger:
    def __init__(self) -> None:
        self.needs: dict[str, ResearchNeed] = {}
        self.skipped: list[dict[str, JsonValue]] = []

    def bind_task(
        self,
        row: ReviewResearchTask,
        issue_ids: list[str],
        *,
        covered_dimensions: Sequence[LegalDimension] = (),
    ) -> ResearchNeed:
        existing = (
            self.needs.get(row.existing_need_id)
            if row.existing_need_id
            else next(
                (
                    need
                    for need in self.needs.values()
                    if need.origin != "question"
                    and row.dimension in {need.dimension, *need.covered_dimensions}
                    and (need.subject or "").casefold().strip()
                    == row.subject.casefold().strip()
                    and " ".join(need.question.casefold().split())
                    == " ".join(row.question.casefold().split())
                ),
                None,
            )
        )
        if row.existing_need_id and existing is None:
            raise ValueError("Diagnosis refers to an unknown research need")
        if existing is not None:
            if existing.origin == "question":
                raise ValueError(
                    "Initial issue discovery cannot consume a specific review gap's search"
                )
            if row.dimension not in {existing.dimension, *existing.covered_dimensions}:
                raise ValueError(
                    "A search in another dimension cannot consume this material gap"
                )
            if not existing.attempted and not row.query:
                raise ValueError("An unsearched material gap needs its discovery query")
            row.existing_need_id = existing.need_id
            return existing
        need = ResearchNeed(
            need_id=f"review_need_{len(self.needs) + 1}",
            issue_ids=issue_ids,
            question=row.question,
            subject=row.subject,
            dimension=row.dimension,
            covered_dimensions=list(
                dict.fromkeys([row.dimension, *covered_dimensions])
            ),
            origin="review",
            trigger_supports=row.supports,
        )
        self.needs[need.need_id] = need
        row.existing_need_id = need.need_id
        return need

    def admit(
        self, actions: Sequence[SourceAction], plan: IssuePlan
    ) -> list[SourceAction]:
        admitted: list[SourceAction] = []
        issues = {issue.issue_id: issue for issue in plan.issues}
        for action in actions:
            if action.tool != "search_corpus":
                admitted.append(action)
                continue
            need_ids = list(action.research_need_ids)
            if not need_ids:
                for identity in action.issue_ids:
                    issue = issues[identity]
                    need_id = f"issue:{identity}"
                    if need_id not in self.needs:
                        parent = next(
                            (
                                need
                                for need in self.needs.values()
                                if issue.parent_issue_id in need.issue_ids
                                and need.attempted
                            ),
                            None,
                        )
                        self.needs[need_id] = ResearchNeed(
                            need_id=need_id,
                            issue_ids=[identity],
                            question=issue.question,
                            origin=issue.origin,
                            dimension=issue.trigger_dimension,
                            parent_need_id=parent.need_id if parent else None,
                        )
                    need_ids.append(need_id)
            if any(identity not in self.needs for identity in need_ids):
                raise ValueError("Search refers to an unknown research need")
            available = list(
                dict.fromkeys(
                    identity
                    for identity in need_ids
                    if not self.needs[identity].attempted
                )
            )
            if not available:
                self.skipped.append(
                    {
                        "research_need_ids": list(need_ids),
                        "reason": "This gap already received its one discovery search",
                        "query": action.arguments.get("query"),
                    }
                )
                continue
            # Consume before dispatch: an empty result or provider failure is still an attempt.
            for identity in available:
                need = self.needs[identity]
                need.attempted = True
                need.query = str(action.arguments["query"])
            admitted.append(action.model_copy(update={"research_need_ids": available}))
        return admitted

    def record_results(
        self,
        actions: Sequence[SourceAction],
        receipts: Sequence[dict[str, JsonValue]],
        prior_evidence: set[int],
    ) -> None:
        for action in actions:
            if action.tool != "search_corpus":
                continue
            for receipt in receipts:
                arguments = receipt.get("arguments")
                if (
                    receipt.get("tool") != action.tool
                    or not isinstance(arguments, dict)
                    or arguments.get("query") != action.arguments.get("query")
                ):
                    continue
                for identity in action.research_need_ids:
                    need = self.needs[identity]
                    call_id = receipt.get("call_id")
                    if isinstance(call_id, str):
                        need.receipt_ids.append(call_id)
                    evidence = receipt.get("evidence_ids")
                    if isinstance(evidence, list):
                        need.new_evidence_ids = list(
                            dict.fromkeys(
                                [
                                    *need.new_evidence_ids,
                                    *(
                                        number
                                        for number in evidence
                                        if type(number) is int
                                        and number not in prior_evidence
                                    ),
                                ]
                            )
                        )
