"""Terminal diagnostics target provenance mistakes without loosening source review."""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.legal_source_reviews import (
    LegalSourceReviews,
    RelatedSourceReviewValidationError,
)
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext
from tests.unit.onyx.asv3.test_experimental_workflow import (
    experimental_context,
    terminal_registry,
)
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    lead_id,
    navigation,
    review,
    seen,
    setup_reviews,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record


def opt_in(context: RunContext) -> None:
    context.services.update(research_profile="experimental", experimental_parallel=True)


def test_multiple_reviews_target_exact_second_source_and_remain_atomic() -> None:
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    reviews.record_delivery("law-call", context, navigation("other-decision"), ledger)
    deliver(ledger, "terminal", [1, 2, 3])
    before = reviews.export()
    second = review(lead_id=lead_id("other-decision"))
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply(
            [review(), second], "terminal", context, ledger, detailed_errors=True
        )
    data = caught.value.diagnostic
    assert str(caught.value) == "A review needs fully delivered candidate-source ranges"
    assert data["code"] == "wrong_source"
    assert data["field"] == "_related_source_reviews[1].witnesses[0]"
    assert data["lead_id"] == lead_id("other-decision")
    assert data["source_id"] == "other-decision"
    assert data["actual_source_id"] == "decision"
    assert data["witness"] == {"citation": 2, "start_char": 0, "end_char": 39}
    other = ledger.get(3)
    assert other is not None
    assert data["available_original_witnesses"] == [
        {"citation": 3, "start_char": 0, "end_char": len(other.text)}
    ]
    assert reviews.export() == before


@pytest.mark.parametrize(
    "changes,code,field",
    [
        ({"witnesses": []}, "missing_own_originals", "witnesses"),
        (
            {"status": "not_material", "witnesses": []},
            "missing_own_originals",
            "witnesses",
        ),
        (
            {"source_role": "argument_only"},
            "argument_cannot_close_operative",
            "source_role",
        ),
        ({"status": "unresolved", "gap": ""}, "missing_unresolved_gap", "gap"),
        ({"gap": "An actual unresolved interaction."}, "closed_review_has_gap", "gap"),
        ({"effect": "  "}, "missing_assessment_field", "effect"),
        ({"limitations": "  "}, "missing_assessment_field", "limitations"),
        (
            {"witnesses": [{"citation": 2, "start_char": 0, "end_char": 500}]},
            "invalid_range",
            "witnesses[0]",
        ),
        (
            {"witnesses": [{"citation": 2, "start_char": 10, "end_char": 10}]},
            "invalid_range",
            "witnesses[0]",
        ),
        (
            {"witnesses": [{"citation": 999, "start_char": 0, "end_char": 10}]},
            "unknown_citation",
            "witnesses[0]",
        ),
        (
            {
                "witnesses": [
                    {"citation": 2, "start_char": 0, "end_char": 39},
                    {"citation": 2, "start_char": 0, "end_char": 39},
                ]
            },
            "duplicate_witness",
            "witnesses[1]",
        ),
    ],
)
def test_diagnostic_pinpoints_refusal_without_approving_unsupported_review(
    changes: dict[str, JsonValue], code: str, field: str
) -> None:
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    deliver(ledger, "terminal", [1, 2])
    before = reviews.export()
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply(
            [review(**changes)], "terminal", context, ledger, detailed_errors=True
        )
    assert caught.value.diagnostic["code"] == code
    assert caught.value.diagnostic["field"] == f"_related_source_reviews[0].{field}"
    assert reviews.export() == before


def test_available_witnesses_require_this_call_full_delivery_and_candidate_source() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    deliver(ledger, "previous", [1, 2, 3])
    partial = full_record(ledger, 2)
    item = ledger.get(2)
    assert item is not None
    partial["text"] = item.text[:20]
    ledger.record_delivery(
        "current", "asv3_coordinator", [partial, full_record(ledger, 3)]
    )
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply([review()], "current", context, ledger, detailed_errors=True)
    assert caught.value.diagnostic["code"] == "not_fully_delivered"
    assert caught.value.diagnostic["available_original_witnesses"] == []
    assert caught.value.diagnostic["source_id"] == "decision"
    assert "fully delivered" in str(caught.value)


def test_foreign_owner_diagnostic_cannot_disclose_candidate_or_original_inventory() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    foreign = context.child()
    foreign.services["task_id"] = "other-worker"
    opt_in(foreign)
    deliver(ledger, "terminal", [1, 2, 3])
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply([review()], "terminal", foreign, ledger, detailed_errors=True)
    assert caught.value.diagnostic["code"] == "duplicate_or_foreign_lead"
    assert "source_id" not in caught.value.diagnostic
    assert "available_original_witnesses" not in caught.value.diagnostic


def test_wrong_source_repair_preserves_valid_witnesses_and_supported_answer_body() -> (
    None
):
    context, ledger, reviews = experimental_context()
    opt_in(context)
    seen(context, ledger, reviews)
    deliver(ledger, "terminal", [1, 2, 3])
    context.services["last_model_call_id"] = "terminal"
    observed: list[dict[str, JsonValue]] = []
    registry = terminal_registry(observed)
    body = "Candidate holding [2]; independent governing rule [1]; other original [3]."
    arguments: dict[str, JsonValue] = {
        "answer": body,
        "basis": "originals",
        "_related_source_reviews": [
            review(
                witnesses=[
                    {"citation": 2, "start_char": 0, "end_char": 39},
                    {"citation": 3, "start_char": 0, "end_char": 10},
                ]
            )
        ],
    }
    refused = registry.dispatch(
        CapabilityCall(name="submit_answer", arguments=arguments), context
    )
    diagnostic = cast(dict[str, JsonValue], refused.data["related_source_review_error"])
    assert diagnostic["code"] == "wrong_source"
    assert diagnostic["field"] == "_related_source_reviews[0].witnesses[1]"
    assert observed == []
    arguments["_related_source_reviews"] = [
        review(witnesses=diagnostic["available_original_witnesses"])
    ]
    accepted = registry.dispatch(
        CapabilityCall(name="submit_answer", arguments=arguments), context
    )
    assert accepted.status == OutcomeStatus.FOUND
    assert observed == [{"answer": body, "basis": "originals"}]
    assert reviews.view(context, ledger, {1, 2, 3})["pending_lead_ids"] == []


@pytest.mark.parametrize(
    "profile,parallel", [("normal", True), ("deep", True), ("experimental", False)]
)
def test_explicit_opt_in_cannot_change_other_profile_errors(
    profile: str, parallel: bool
) -> None:
    context, ledger, reviews = setup_reviews()
    context.services.update(research_profile=profile, experimental_parallel=parallel)
    seen(context, ledger, reviews)
    with pytest.raises(ValueError) as caught:
        reviews.apply(
            [review(witnesses=[])], "law-call", context, ledger, detailed_errors=True
        )
    assert type(caught.value) is ValueError
    assert str(caught.value) == "Examined and excluded leads need their own originals"


def test_default_errors_and_successful_checkpoint_are_unchanged() -> None:
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    with pytest.raises(ValueError) as caught:
        reviews.apply([review(witnesses=[])], "law-call", context, ledger)
    assert type(caught.value) is ValueError
    deliver(ledger, "terminal", [1, 2])
    before = reviews.export()
    legacy = LegalSourceReviews(context, "Can the rule be applied?")
    legacy.restore(before, context, "Can the rule be applied?", ledger)
    legacy.apply([review()], "terminal", context, ledger)
    reviews.apply([review()], "terminal", context, ledger, detailed_errors=True)
    assert json.dumps(reviews.export(), sort_keys=True) == json.dumps(
        legacy.export(), sort_keys=True
    )
    restored = LegalSourceReviews(context, "Can the rule be applied?")
    restored.restore(reviews.export(), context, "Can the rule be applied?", ledger)
    assert restored.export() == reviews.export()


@pytest.mark.parametrize("parallel", [False, True])
def test_registry_exposes_scoped_diagnostic_before_terminal_handler(
    parallel: bool,
) -> None:
    context, ledger, reviews = experimental_context()
    context.services["experimental_parallel"] = parallel
    seen(context, ledger, reviews)
    deliver(ledger, "terminal", [1, 2, 3])
    context.services["last_model_call_id"] = "terminal"
    observed: list[dict[str, JsonValue]] = []
    result = terminal_registry(observed).dispatch(
        CapabilityCall(
            name="submit_answer",
            arguments={
                "answer": "Keep all supported details [1][2][3].",
                "basis": "originals",
                "_related_source_reviews": [
                    review(witnesses=[{"citation": 3, "start_char": 0, "end_char": 10}])
                ],
            },
        ),
        context,
    )
    assert result.status == OutcomeStatus.INVALID
    assert observed == []
    assert (
        result.data["detail"]
        == "A review needs fully delivered candidate-source ranges"
    )
    assert result.data["invalid_related_source_review"] is True
    if parallel:
        diagnostic = cast(
            dict[str, JsonValue], result.data["related_source_review_error"]
        )
        assert diagnostic["code"] == "wrong_source"
        assert diagnostic["actual_source_id"] == "other-decision"
        assert diagnostic["available_original_witnesses"] == [
            {"citation": 2, "start_char": 0, "end_char": 39}
        ]
    else:
        assert result.data == {
            "detail": "A review needs fully delivered candidate-source ranges",
            "invalid_related_source_review": True,
        }
