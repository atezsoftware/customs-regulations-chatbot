"""Accepted owner reviews may close root navigation without fabricated delivery."""

import copy

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.models import RunContext
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    navigation,
    review,
    seen,
    setup_reviews,
)


def assess(
    root: RunContext,
    ledger: EvidenceLedger,
    reviews: LegalSourceReviews,
    owner: str,
    assessment: dict[str, JsonValue],
) -> RunContext:
    child = root.child()
    child.services["task_id"] = owner
    call = owner + "-call"
    deliver(ledger, call, [1, 2])
    reviews.record_delivery(call, child, navigation(), ledger)
    reviews.apply([assessment], call, child, ledger)
    return child


def test_examined_accepted_child_closes_matching_root_lead_without_changing_records() -> (
    None
):
    root, ledger, reviews = setup_reviews()
    seen(root, ledger, reviews)
    assess(root, ledger, reviews, "assigned-task", review())
    answer = "The supported conditional outcome [1, 2]."
    before = copy.deepcopy(reviews.export())
    assert reviews.publication_gap(answer, "law-call", root, ledger) is not None
    assert (
        reviews.publication_gap_for_assembly(
            answer, root, ledger, accepted_owners={"assigned-task"}
        )
        is None
    )
    assert reviews.export() == before
    assert ledger.completely_delivered("law-call") == {1}


def test_unaccepted_child_cannot_discharge_root_publication_obligation() -> None:
    root, ledger, reviews = setup_reviews()
    seen(root, ledger, reviews)
    assess(root, ledger, reviews, "unaccepted-task", review())
    assert (
        reviews.publication_gap_for_assembly(
            "A conclusion [1].", root, ledger, accepted_owners={"accepted-other-task"}
        )
        is not None
    )


def test_one_tasks_not_material_assessment_cannot_close_the_entire_request() -> None:
    root, ledger, reviews = setup_reviews()
    seen(root, ledger, reviews)
    assess(root, ledger, reviews, "first-task", review(status="not_material"))
    answer = "The supported outcomes [1]."
    assert (
        reviews.publication_gap_for_assembly(
            answer, root, ledger, accepted_owners={"first-task", "second-task"}
        )
        is not None
    )
    assess(
        root,
        ledger,
        reviews,
        "second-task",
        review(
            status="not_material",
            effect="The same original excludes the second assigned outcome too.",
        ),
    )
    assert (
        reviews.publication_gap_for_assembly(
            answer, root, ledger, accepted_owners={"first-task", "second-task"}
        )
        is None
    )


@pytest.mark.parametrize("presentation", ["missing", "inline", "cited"])
def test_unresolved_accepted_review_requires_its_exact_uncited_standalone_paragraph(
    presentation: str,
) -> None:
    root, ledger, reviews = setup_reviews()
    seen(root, ledger, reviews)
    gap = "The decision's operative interaction remains unexamined."
    assess(
        root,
        ledger,
        reviews,
        "assigned-task",
        review(status="unresolved", source_role="unknown", witnesses=[], gap=gap),
    )
    prefix = "A supported condition [1]."
    bad = {
        "missing": prefix,
        "inline": prefix + " " + gap,
        "cited": prefix + "\n\n" + gap + " [1].",
    }[presentation]
    assert (
        reviews.publication_gap_for_assembly(
            bad, root, ledger, accepted_owners={"assigned-task"}
        )
        is not None
    )
    assert (
        reviews.publication_gap_for_assembly(
            prefix + "\n\n" + gap, root, ledger, accepted_owners={"assigned-task"}
        )
        is None
    )


def test_examined_child_can_resolve_previously_unresolved_root_interaction() -> None:
    root, ledger, reviews = setup_reviews()
    seen(root, ledger, reviews)
    reviews.apply(
        [
            review(
                status="unresolved",
                source_role="unknown",
                witnesses=[],
                gap="The operative interaction is not yet established.",
            )
        ],
        "law-call",
        root,
        ledger,
    )
    assess(root, ledger, reviews, "assigned-task", review())
    assert (
        reviews.publication_gap_for_assembly(
            "A conditional result [1, 2].",
            root,
            ledger,
            accepted_owners={"assigned-task"},
        )
        is None
    )


def test_changed_candidate_original_cannot_close_root_assembly_gap() -> None:
    root, ledger, reviews = setup_reviews()
    seen(root, ledger, reviews)
    assess(root, ledger, reviews, "assigned-task", review())
    ledger._items[2].text_hash = "a" * 64
    with pytest.raises(ValueError, match="original changed"):
        reviews.publication_gap_for_assembly(
            "A result [1, 2].", root, ledger, accepted_owners={"assigned-task"}
        )


def test_root_cannot_serve_as_accepted_child_owner() -> None:
    root, ledger, reviews = setup_reviews()
    seen(root, ledger, reviews)
    with pytest.raises(ValueError, match="child owners"):
        reviews.publication_gap_for_assembly(
            "A result [1].", root, ledger, accepted_owners={"coordinator"}
        )
