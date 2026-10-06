"""Batch provenance diagnostics avoid serial repairs without admitting any review."""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.harness import Harness, HarnessView
from onyx.asv3.legal_source_reviews import (
    LegalSourceReviews,
    RelatedSourceReviewValidationError,
    annotate_navigation,
)
from onyx.asv3.models import CapabilityCall, Decision, OutcomeStatus, ToolReceipt
from onyx.asv3.native_cache_projection import project_native_originals
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
from tests.unit.onyx.asv3.test_native_model_adapter import turn
from tests.unit.onyx.asv3.test_parallel_review_diagnostics import opt_in
from tests.unit.onyx.asv3.test_shared_originals import full_record, original


def errors(error: RelatedSourceReviewValidationError) -> list[dict[str, JsonValue]]:
    return cast(list[dict[str, JsonValue]], error.diagnostic["validation_errors"])


def test_two_wrong_witnesses_are_reported_together_without_applying_valid_rows() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    ledger.add(
        [original("other-two", "Another foreign original.", source="other-decision")],
        context,
    )
    deliver(ledger, "terminal", [1, 2, 3, 4])
    before = reviews.export()
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply(
            [
                review(
                    witnesses=[
                        {"citation": 2, "start_char": 0, "end_char": 39},
                        {"citation": 3, "start_char": 0, "end_char": 10},
                        {"citation": 4, "start_char": 0, "end_char": 10},
                    ]
                )
            ],
            "terminal",
            context,
            ledger,
            detailed_errors=True,
        )
    diagnostic = caught.value.diagnostic
    rows = errors(caught.value)
    assert diagnostic["code"] == "wrong_source"
    assert diagnostic["field"] == "_related_source_reviews[0].witnesses[1]"
    assert [row["field"] for row in rows] == [
        "_related_source_reviews[0].witnesses[1]",
        "_related_source_reviews[0].witnesses[2]",
    ]
    assert all(row["actual_source_id"] == "other-decision" for row in rows)
    assert all(
        row["available_original_witnesses_ref"] == "available_original_witnesses"
        for row in rows
    )
    assert all(
        "available_original_witnesses" not in row and "instruction" not in row
        for row in rows
    )
    assert json.dumps(diagnostic).count('"available_original_witnesses":') == 1
    assert reviews.export() == before


def test_multiple_leads_duplicates_and_foreign_rows_are_one_atomic_refusal() -> None:
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    reviews.record_delivery("law-call", context, navigation("other-decision"), ledger)
    deliver(ledger, "terminal", [1, 2, 3])
    before = reviews.export()
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply(
            [
                review(witnesses=[{"citation": 3, "start_char": 0, "end_char": 10}]),
                review(lead_id=lead_id("other-decision")),
                review(),
                review(lead_id=lead_id("unseen-source")),
            ],
            "terminal",
            context,
            ledger,
            detailed_errors=True,
        )
    rows = errors(caught.value)
    assert [row["code"] for row in rows] == [
        "wrong_source",
        "wrong_source",
        "duplicate_or_foreign_lead",
        "duplicate_or_foreign_lead",
    ]
    assert [row["review_index"] for row in rows] == [0, 1, 2, 3]
    assert (
        rows[1]["available_original_witnesses_ref"]
        == "additional_original_witnesses[0]"
    )
    assert all(
        "source_id" not in row and "available_original_witnesses_ref" not in row
        for row in rows[2:]
    )
    assert reviews.export() == before


def test_all_independent_witness_restrictions_remain_strict() -> None:
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    ledger.add(
        [
            original(
                "unseen", "A candidate passage not delivered now.", source="decision"
            )
        ],
        context,
    )
    deliver(ledger, "terminal", [1, 2, 3])
    before = reviews.export()
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply(
            [
                review(
                    witnesses=[
                        {"citation": 2, "start_char": 0, "end_char": 39},
                        {"citation": 2, "start_char": 0, "end_char": 39},
                        {"citation": 999, "start_char": 0, "end_char": 10},
                        {"citation": 2, "start_char": 0, "end_char": 400},
                        {"citation": 4, "start_char": 0, "end_char": 10},
                    ]
                )
            ],
            "terminal",
            context,
            ledger,
            detailed_errors=True,
        )
    assert [row["code"] for row in errors(caught.value)] == [
        "duplicate_witness",
        "unknown_citation",
        "invalid_range",
        "not_fully_delivered",
    ]
    assert reviews.export() == before


def test_distinct_owned_leads_share_identical_current_candidate_inventory() -> None:
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    law = original(
        "law-18",
        "Another governing provision.",
        source="law",
        headings=["Example Law", "MADDE 18"],
    )
    assert law.search_doc is not None and law.chunk_id is not None
    law.search_doc.metadata["regulatory_chunk_id"] = law.chunk_id
    ledger.add([law], context)
    deliver(ledger, "second-anchor", [4])
    second_navigation = navigation()
    second_navigation[0]["article_no"] = "18"
    reviews.record_delivery("second-anchor", context, second_navigation, ledger)
    candidates = cast(
        list[dict[str, JsonValue]],
        annotate_navigation(second_navigation)[0]["candidates"],
    )
    second_lead = candidates[0]["lead_id"]
    assert second_lead != lead_id()
    deliver(ledger, "terminal", [1, 2, 3, 4])
    before = reviews.export()
    bad = [{"citation": 3, "start_char": 0, "end_char": 10}]
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply(
            [review(witnesses=bad), review(lead_id=second_lead, witnesses=bad)],
            "terminal",
            context,
            ledger,
            detailed_errors=True,
        )
    diagnostic = caught.value.diagnostic
    assert len(errors(caught.value)) == 2
    assert all(
        row["available_original_witnesses_ref"] == "available_original_witnesses"
        for row in errors(caught.value)
    )
    assert "additional_original_witnesses" not in diagnostic
    assert json.dumps(diagnostic).count('"available_original_witnesses":') == 1
    assert reviews.export() == before


def test_missing_assessment_fields_and_status_gap_are_all_reported() -> None:
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    deliver(ledger, "terminal", [1, 2])
    before = reviews.export()
    with pytest.raises(RelatedSourceReviewValidationError) as caught:
        reviews.apply(
            [review(effect=" ", limitations=" ", status="unresolved", gap="")],
            "terminal",
            context,
            ledger,
            detailed_errors=True,
        )
    assert [row["field"] for row in errors(caught.value)] == [
        "_related_source_reviews[0].effect",
        "_related_source_reviews[0].limitations",
        "_related_source_reviews[0].gap",
    ]
    assert reviews.export() == before


def test_valid_batch_checkpoint_matches_legacy_and_final_retention_still_rejects() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    opt_in(context)
    seen(context, ledger, reviews)
    reviews.record_delivery("law-call", context, navigation("other-decision"), ledger)
    deliver(ledger, "terminal", [1, 2, 3])
    legacy = LegalSourceReviews(context, "Can the rule be applied?")
    legacy.restore(reviews.export(), context, "Can the rule be applied?", ledger)
    other = ledger.get(3)
    assert other is not None
    batch: list[JsonValue] = [
        review(),
        review(
            lead_id=lead_id("other-decision"),
            witnesses=[{"citation": 3, "start_char": 0, "end_char": len(other.text)}],
        ),
    ]
    legacy.apply(batch, "terminal", context, ledger)
    reviews.apply(batch, "terminal", context, ledger, detailed_errors=True)
    assert json.dumps(reviews.export(), ensure_ascii=False) == json.dumps(
        legacy.export(), ensure_ascii=False
    )
    gap = reviews.publication_gap(
        "A broad result alone [1][2][3].", "terminal", context, ledger
    )
    assert gap is not None and gap.status == OutcomeStatus.PARTIAL
    assert gap.data["unretained_examined_source_effects"]


@pytest.mark.parametrize(
    "profile,parallel", [("normal", True), ("deep", True), ("experimental", False)]
)
def test_other_profiles_keep_exact_fail_first_error(
    profile: str, parallel: bool
) -> None:
    context, ledger, reviews = setup_reviews()
    context.services.update(research_profile=profile, experimental_parallel=parallel)
    seen(context, ledger, reviews)
    deliver(ledger, "terminal", [1, 2, 3])
    before = reviews.export()
    with pytest.raises(ValueError) as caught:
        reviews.apply(
            [review(witnesses=[{"citation": 3, "start_char": 0, "end_char": 10}])],
            "terminal",
            context,
            ledger,
            detailed_errors=True,
        )
    assert type(caught.value) is ValueError
    assert str(caught.value) == "A review needs fully delivered candidate-source ranges"
    assert reviews.export() == before


def test_eighty_candidate_ranges_and_all_errors_survive_actual_receipt_projection() -> (
    None
):
    context, ledger, reviews = experimental_context()
    opt_in(context)
    seen(context, ledger, reviews)
    ledger.add(
        [
            original(f"candidate-{n}", "A full candidate passage.", source="decision")
            for n in range(79)
        ],
        context,
    )
    deliver(ledger, "terminal", list(range(1, 83)))
    context.services["last_model_call_id"] = "terminal"
    observed: list[dict[str, JsonValue]] = []
    registry = terminal_registry(observed)
    call = CapabilityCall(
        call_id="refused",
        name="submit_answer",
        arguments={
            "answer": "Preserve the supported answer [1][2][3].",
            "basis": "originals",
            "_related_source_reviews": [
                review(
                    witnesses=[
                        {"citation": 1, "start_char": 0, "end_char": 10},
                        {"citation": 3, "start_char": 0, "end_char": 10},
                    ]
                )
            ],
        },
    )
    before = reviews.export()
    outcome = registry.dispatch(call, context)
    assert outcome.status == OutcomeStatus.INVALID
    assert observed == [] and reviews.export() == before

    def no_model(_: HarnessView) -> Decision:
        raise AssertionError("No additional model decision")

    harness = Harness(
        request="Can the rule be applied?",
        context=context,
        registry=registry,
        decide=no_model,
        evidence=ledger,
    )
    receipt = ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0)
    harness._commit_receipt(receipt)
    message = harness._model_tool_result(receipt)
    native = turn("refused", [])
    native.results = [message]
    before_turn = native.model_dump_json()
    projection = project_native_originals(
        [native],
        [full_record(ledger, n) for n in range(1, 83)],
        ledger,
        compact_identities=True,
    )
    payload = json.loads(projection.turns[0].results[0].content)
    diagnostic = payload["outcome"]["data"]["related_source_review_error"]
    assert len(diagnostic["available_original_witnesses"]) == 80
    assert [row["witness"]["citation"] for row in diagnostic["validation_errors"]] == [
        1,
        3,
    ]
    assert "truncated" not in json.dumps(diagnostic)
    assert native.model_dump_json() == before_turn
    assert reviews.export() == before
