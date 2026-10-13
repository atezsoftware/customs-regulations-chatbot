"""Section scoping preserves aggregate operative and unresolved-source retention."""

import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_sections import CroppedSectionPlan, plan_cropped_sections
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.models import HarnessView, RunContext, RunStopped
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    navigation,
    review,
    setup_reviews,
)


def cropped_reviews() -> tuple[RunContext, EvidenceLedger, LegalSourceReviews]:
    context, ledger, reviews = setup_reviews()
    context.services.update(
        asv3_exp_cropped=True, asv3_workflow_variant=ASV3_TUNED_VARIANT
    )
    leads = navigation()
    candidates = leads[0]["candidates"]
    assert isinstance(candidates, list) and isinstance(candidates[0], dict)
    candidates[0]["candidate_role"] = "executive_candidate"
    reviews.record_delivery("law-call", context, leads, ledger)
    return context, ledger, reviews


def test_unrelated_section_can_pass_without_closing_pending_owner_dependency() -> None:
    context, ledger, reviews = cropped_reviews()
    deliver(ledger, "unrelated-writer", [3])
    before = copy.deepcopy(reviews.export())
    assert (
        reviews.publication_gap_for_cropped_section(
            "Independent supported result [3].",
            "unrelated-writer",
            context,
            ledger,
            {3},
        )
        is None
    )
    assert reviews.export() == before
    gap = reviews.publication_gap_for_cropped_assembly(
        "Independent supported result [3].", context, ledger, {1, 3}
    )
    assert gap is not None
    assert gap.data["unread_related_sources"]
    assert (
        reviews.publication_gap(
            "Independent supported result [3].", "unrelated-writer", context, ledger
        )
        == gap
    )


def test_aggregate_retains_exact_effect_limitations_and_actual_union_delivery() -> None:
    context, ledger, reviews = cropped_reviews()
    deliver(ledger, "operative-writer", [1, 2])
    assessment = review()
    reviews.apply([assessment], "operative-writer", context, ledger)
    deliver(ledger, "unrelated-writer", [3])
    body = f"{assessment['effect']} {assessment['limitations']} [2]."
    assert (
        reviews.publication_gap_for_cropped_section(
            body, "operative-writer", context, ledger, {1, 2}
        )
        is None
    )
    assert (
        reviews.publication_gap_for_cropped_section(
            "Independent supported result [3].",
            "unrelated-writer",
            context,
            ledger,
            {3},
        )
        is None
    )
    complete = body + "\n\nIndependent supported result [3]."
    assert (
        reviews.publication_gap_for_cropped_assembly(
            complete, context, ledger, {1, 2, 3}
        )
        is None
    )
    assert ledger.completely_delivered("assembly") == set()
    without_limitations = f"{assessment['effect']} [2].\n\nIndependent result [3]."
    gap = reviews.publication_gap_for_cropped_assembly(
        without_limitations, context, ledger, {1, 2, 3}
    )
    assert gap is not None
    assert gap.data["unretained_examined_source_effects"]
    # An intact body does not compensate for a witness omitted from the receipt union.
    gap = reviews.publication_gap_for_cropped_assembly(
        body.replace("[2]", "[1]"), context, ledger, {1, 3}
    )
    assert gap is not None
    assert ledger.completely_delivered("assembly") == set()


def test_unresolved_interaction_requires_its_exact_uncited_paragraph_in_assembly() -> (
    None
):
    context, ledger, reviews = cropped_reviews()
    declared_gap = "The candidate's original and its effect remain unexamined."
    reviews.apply(
        [
            review(
                status="unresolved",
                source_role="unknown",
                gap=declared_gap,
                witnesses=[],
            )
        ],
        "law-call",
        context,
        ledger,
    )
    deliver(ledger, "unrelated-writer", [3])
    body = "Supported independent result [3]."
    gap = reviews.publication_gap_for_cropped_assembly(body, context, ledger, {1, 3})
    assert gap is not None
    assert gap.data["undisclosed_related_source_gaps"]
    assert (
        reviews.publication_gap_for_cropped_assembly(
            body + "\n\n" + declared_gap, context, ledger, {1, 3}
        )
        is None
    )
    assert (
        reviews.publication_gap_for_cropped_assembly(
            body + "\n\n" + declared_gap + " [1]", context, ledger, {1, 3}
        )
        is not None
    )


@pytest.mark.parametrize(
    "failure", ["outside_assignment", "undelivered", "cheap", "protected"]
)
def test_section_scope_cannot_replace_real_delivery_or_change_other_modes(
    failure: str,
) -> None:
    context, ledger, reviews = cropped_reviews()
    deliver(ledger, "writer", [3])
    numbers, answer = {3}, "Supported result [3]."
    if failure == "outside_assignment":
        answer = "Supported result [1]."
    elif failure == "undelivered":
        numbers = {1}
        answer = "Supported result [1]."
    elif failure == "cheap":
        context.services["asv3_exp_cheap"] = True
    else:
        context.services["asv3_exp_cropped"] = False
    with pytest.raises(ValueError):
        reviews.publication_gap_for_cropped_section(
            answer, "writer", context, ledger, numbers
        )


def test_full_saved_requirements_have_owned_source_bound_dependencies() -> None:
    context, ledger, reviews = cropped_reviews()
    deliver(ledger, "writer", [1, 2])
    reviews.apply([review()], "writer", context, ledger)
    requirements = reviews.cropped_requirements(context, ledger)
    rows = requirements["reviews"]
    assert isinstance(rows, list) and len(rows) == 1
    assert isinstance(rows[0], dict)
    assert rows[0]["required_evidence_numbers"] == [1, 2]
    assert rows[0]["review"] == review()
    other = RunContext(
        run_id=context.run_id,
        scope=context.scope,
        services={**context.services, "task_id": "other-owner"},
    )
    assert reviews.cropped_requirements(other, ledger)["reviews"] == []
    typed: dict[str, JsonValue] = rows[0]
    assert typed["first_call_id"] == "law-call"


def test_planner_keeps_related_anchor_and_own_operative_original_in_one_section() -> (
    None
):
    context, ledger, reviews = cropped_reviews()
    context.services["legal_source_reviews"] = reviews
    deliver(ledger, "writer", [1, 2])
    reviews.apply([review()], "writer", context, ledger)
    view = HarnessView(
        request="Can the rule be applied?",
        questions=["Can the rule be applied?"],
        facts=[],
        receipts=[],
        evidence=[],
        tools=[],
        original_evidence=cast(
            list[dict[str, JsonValue]],
            json.loads(ledger.serialize_records([1, 2, 3], max_chars=None)),
        ),
    )

    def count(prepared: HarnessView, _plan: CroppedSectionPlan) -> int:
        return len(prepared.original_evidence) * 90_000

    plans = plan_cropped_sections(view, context, ledger, count)
    assert [plan.evidence_numbers for plan, _ in plans] == [[1, 2], [3]]

    def oversized(prepared: HarnessView, _plan: CroppedSectionPlan) -> int:
        return len(prepared.original_evidence) * 130_000

    with pytest.raises(RunStopped, match="connected source obligation"):
        plan_cropped_sections(view, context, ledger, oversized)
