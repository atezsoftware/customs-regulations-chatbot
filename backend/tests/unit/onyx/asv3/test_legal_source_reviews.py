"""Source-lead closure validates acquisition without approving legal interpretation."""

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import (
    LegalSourceReviews,
    annotate_navigation,
)
from onyx.asv3.models import EvidenceItem, RunContext
from tests.unit.onyx.asv3.test_shared_originals import full_record, original


def setup_reviews() -> tuple[RunContext, EvidenceLedger, LegalSourceReviews]:
    context = RunContext(scope={"authorized": "owned"})
    law = original(
        "law-17",
        "The applicable governing provision.",
        source="law",
        headings=["Example Law", "MADDE 17"],
    )
    assert law.search_doc is not None
    assert law.chunk_id is not None
    law.search_doc.metadata["regulatory_chunk_id"] = law.chunk_id
    ledger = EvidenceLedger()
    ledger.add(
        [
            law,
            original(
                "holding", "Only the identified phrase is annulled.", source="decision"
            ),
            original("other", "A different source.", source="other-decision"),
        ],
        context,
    )
    deliver(ledger, "law-call", [1])
    return context, ledger, LegalSourceReviews(context, "Can the rule be applied?")


def deliver(ledger: EvidenceLedger, call: str, citations: list[int]) -> None:
    ledger.record_delivery(
        call, "asv3_coordinator", [full_record(ledger, n) for n in citations]
    )


def navigation(source: str = "decision") -> list[dict[str, JsonValue]]:
    return [
        {
            "anchor_source_id": "law",
            "article_no": "17",
            "qualifier": None,
            "candidates": [
                {
                    "source_id": source,
                    "name": "Court decision concerning Example Law article 17",
                    "candidate_role": "judicial_candidate",
                }
            ],
        }
    ]


def lead_id(source: str = "decision") -> str:
    return str(
        cast(
            list[dict[str, JsonValue]],
            annotate_navigation(navigation(source))[0]["candidates"],
        )[0]["lead_id"]
    )


def review(**changes: JsonValue) -> dict[str, JsonValue]:
    return {
        "lead_id": lead_id(),
        "status": "examined",
        "source_role": "operative_text",
        "effect": "The particular changed wording may affect this outcome.",
        "limitations": "The underlying obligation and applicable dates remain distinct.",
        "witnesses": [{"citation": 2, "start_char": 0, "end_char": 39}],
        "gap": "",
        **changes,
    }


def seen(
    context: RunContext, ledger: EvidenceLedger, reviews: LegalSourceReviews
) -> None:
    reviews.record_delivery("law-call", context, navigation(), ledger)


def test_annotation_is_stable_pure_and_does_not_register_context_exposure() -> None:
    context, ledger, reviews = setup_reviews()
    source = navigation()
    before = copy.deepcopy(source)
    annotated = annotate_navigation(source)
    assert source == before
    assert annotated == LegalSourceReviews.annotate_navigation(source)
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == []
    assert lead_id() != lead_id("other-decision")
    different = navigation()
    different[0]["qualifier"] = "GEÇİCİ"
    assert annotate_navigation(different) != annotated


@pytest.mark.parametrize(
    "failure", ["undelivered", "different_article", "noncanonical"]
)
def test_unfitted_or_ungenuine_anchor_cannot_create_publication_obligation(
    failure: str,
) -> None:
    context, ledger, reviews = setup_reviews()
    source = navigation()
    call = "law-call"
    if failure == "undelivered":
        call = "not-sent"
    elif failure == "different_article":
        source[0]["article_no"] = "18"
    else:
        forged = EvidenceItem(
            source_id="law",
            chunk_id="invented",
            text="A rule.",
            metadata={"heading_path": ["Example Law", "MADDE 17"]},
        )
        ledger = EvidenceLedger()
        ledger.add([forged], context)
        deliver(ledger, call, [1])
    reviews.record_delivery(call, context, source, ledger)
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == []
    assert reviews.publication_gap("A legal answer [1].", call, context, ledger) is None


def test_seen_unread_lead_blocks_publication_even_when_metadata_is_empty() -> None:
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    before = context.budget.snapshot()
    for raw in (None, []):
        gap = reviews.publication_gap(
            "A legal answer [1].", "law-call", context, ledger, raw
        )
        assert gap is not None
        assert gap.data["pending_related_source_review"] is True
        assert (
            cast(list[dict[str, JsonValue]], gap.data["unread_related_sources"])[0][
                "source_id"
            ]
            == "decision"
        )
    reviews.apply([], "law-call", context, ledger)
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == [lead_id()]
    assert reviews.publication_gap("Merhaba!", "law-call", context, ledger) is None
    assert context.budget.snapshot() == before


def test_acquired_passages_are_not_an_examined_holding_and_inline_review_is_previewed() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    deliver(ledger, "answer-call", [1, 2])
    state = reviews.view(context, ledger, {1, 2})
    row = cast(list[dict[str, JsonValue]], state["reviews"])[0]
    assert row["available_original_citations"] == [2]
    assert row["status"] == "pending"
    assert (
        reviews.publication_gap(
            "A conditional result [1][2].", "answer-call", context, ledger, [review()]
        )
        is None
    )
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"] == [lead_id()]
    reviews.apply([review()], "answer-call", context, ledger)
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"] == []
    assert (
        reviews.publication_gap(
            "A conditional result [1][2].", "answer-call", context, ledger
        )
        is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_role": "argument_only"},
        {"source_role": "unknown"},
        {"witnesses": []},
        {"witnesses": [{"citation": 3, "start_char": 0, "end_char": 10}]},
        {"witnesses": [{"citation": 2, "start_char": 9, "end_char": 10000}]},
        {"gap": "The outcome remains unexamined."},
        {"effect": "  "},
        {"limitations": "  "},
    ],
)
def test_invalid_closure_is_atomic(changes: dict[str, JsonValue]) -> None:
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    deliver(ledger, "answer-call", [1, 2, 3])
    before = reviews.export()
    with pytest.raises(ValueError):
        reviews.apply([review(**changes)], "answer-call", context, ledger)
    assert reviews.export() == before


@pytest.mark.parametrize("status", ["examined", "not_material"])
def test_nonempty_closed_review_gap_reports_metadata_defect_without_missing_originals(
    status: str,
) -> None:
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    deliver(ledger, "assessment", [1, 2])
    with pytest.raises(ValueError, match="must leave gap empty"):
        reviews.apply(
            [review(status=status, gap="An open applicability issue remains.")],
            "assessment",
            context,
            ledger,
        )
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"] == [lead_id()]


def test_not_material_also_requires_candidate_original_and_current_full_delivery() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    with pytest.raises(ValueError, match="fully delivered"):
        reviews.apply([review(status="not_material")], "law-call", context, ledger)
    item = ledger.get(2)
    assert item is not None
    record = full_record(ledger, 2)
    record["text"] = item.text[:20]
    ledger.record_delivery("partial-call", "asv3_coordinator", [record])
    with pytest.raises(ValueError, match="fully delivered"):
        reviews.apply([review(status="not_material")], "partial-call", context, ledger)
    deliver(ledger, "complete-call", [2])
    reviews.apply(
        [review(status="not_material", source_role="argument_only")],
        "complete-call",
        context,
        ledger,
    )


@pytest.mark.parametrize(
    "answer_suffix", ["", " [1]", " More claims.", "\nIt also changes another outcome."]
)
def test_unresolved_issue_requires_its_own_exact_uncited_paragraph(
    answer_suffix: str,
) -> None:
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    gap_text = "The effect of the decision on this event date could not be established."
    pending = review(
        status="unresolved", source_role="unknown", witnesses=[], gap=gap_text
    )
    answer = "The independent rule is supported [1].\n\n" + gap_text + answer_suffix
    result = reviews.publication_gap(answer, "law-call", context, ledger, [pending])
    assert (result is None) is (answer_suffix == "")
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == [lead_id()]


def test_empty_unresolved_gap_duplicate_review_and_foreign_owner_cannot_close_lead() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    child = context.child()
    child.services["task_id"] = "worker-one"
    seen(child, ledger, reviews)
    deliver(ledger, "answer-call", [1, 2])
    for raw in (
        [review(status="unresolved", witnesses=[], gap="")],
        [review(), review()],
    ):
        with pytest.raises(ValueError):
            reviews.apply(cast(list[JsonValue], raw), "answer-call", child, ledger)
    with pytest.raises(ValueError, match="this task"):
        reviews.apply([review()], "answer-call", context, ledger)
    assert (
        reviews.publication_gap(
            "Other task answer [1].", "answer-call", context, ledger
        )
        is None
    )
    sibling = context.child()
    sibling.services["task_id"] = "worker-two"
    assert reviews.view(sibling, ledger, {1, 2})["reviews"] == []


def test_concurrent_repeat_exposure_deduplicates_and_preserves_completed_review() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: seen(context, ledger, reviews), range(12)))
    assert len(cast(list[JsonValue], reviews.export()["records"])) == 1
    deliver(ledger, "answer-call", [1, 2])
    reviews.apply([review()], "answer-call", context, ledger)
    seen(context, ledger, reviews)
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"] == []


def test_extra_unchanged_anchor_continuation_preserves_completed_review() -> None:
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    deliver(ledger, "answer-call", [1, 2])
    reviews.apply([review()], "answer-call", context, ledger)
    prior = copy.deepcopy(reviews.export())
    continuation = original(
        "law-17-continuation",
        "A connected operative condition.",
        source="law",
        headings=["Example Law", "MADDE 17"],
    )
    assert continuation.search_doc is not None
    assert continuation.chunk_id is not None
    continuation.search_doc.metadata["regulatory_chunk_id"] = continuation.chunk_id
    number = ledger.add([continuation], context)[0]
    deliver(ledger, "continuation-call", [1, 2, number])
    reviews.record_delivery("continuation-call", context, navigation(), ledger)
    assert reviews.view(context, ledger, {1, 2, number})["pending_lead_ids"] == []
    records = cast(list[dict[str, JsonValue]], reviews.export()["records"])
    prior_records = cast(list[dict[str, JsonValue]], prior["records"])
    assert records[0]["review"] == prior_records[0]["review"]
    assert records[0]["review_hashes"] == prior_records[0]["review_hashes"]
    assert set(cast(dict[str, JsonValue], records[0]["anchor_hashes"])) == {
        "1",
        str(number),
    }
    restored = LegalSourceReviews(context, "Can the rule be applied?")
    restored.restore(reviews.export(), context, "Can the rule be applied?", ledger)
    assert restored.export() == reviews.export()


def test_alternating_fitted_anchor_subsets_preserve_review_without_reopening() -> None:
    context, ledger, reviews = setup_reviews()
    another = original(
        "law-17-second",
        "A second passage of the same governing provision.",
        source="law",
        headings=["Example Law", "MADDE 17"],
    )
    assert another.search_doc is not None
    assert another.chunk_id is not None
    another.search_doc.metadata["regulatory_chunk_id"] = another.chunk_id
    number = ledger.add([another], context)[0]
    seen(context, ledger, reviews)
    deliver(ledger, "answer-call", [1, 2])
    reviews.apply([review()], "answer-call", context, ledger)
    for index, citations in enumerate(([number], [1], [1, number], [number], [1])):
        call = f"fitted-{index}"
        deliver(ledger, call, citations)
        reviews.record_delivery(call, context, navigation(), ledger)
        assert reviews.view(context, ledger, set(citations))["pending_lead_ids"] == []
        assert (
            reviews.publication_gap("Supported rule [1].", call, context, ledger)
            is None
        )
    records = cast(list[dict[str, JsonValue]], reviews.export()["records"])
    assert set(cast(dict[str, JsonValue], records[0]["anchor_hashes"])) == {
        "1",
        str(number),
    }


def test_repeat_exposure_rejects_changed_saved_anchor_atomically() -> None:
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    prior = copy.deepcopy(reviews.export())
    checkpoint = ledger.export()
    rows = cast(list[dict[str, JsonValue]], checkpoint["records"])
    first = cast(dict[str, JsonValue], rows[0]["item"])
    first["text"] = "Changed operative text."
    first["text_hash"] = ""
    checkpoint["deliveries"] = []
    changed = EvidenceLedger()
    changed.restore(checkpoint, context)
    deliver(changed, "changed-call", [1])
    with pytest.raises(ValueError, match="anchor original changed"):
        reviews.record_delivery("changed-call", context, navigation(), changed)
    assert reviews.export() == prior


def test_resume_is_fenced_and_cannot_silently_replace_live_records() -> None:
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    deliver(ledger, "answer-call", [1, 2])
    reviews.apply([review()], "answer-call", context, ledger)
    checkpoint = reviews.export()
    restored = LegalSourceReviews(context, "Can the rule be applied?")
    restored.restore(checkpoint, context, "Can the rule be applied?", ledger)
    assert restored.export() == checkpoint
    for field in ("run_id", "scope_hash", "request_hash"):
        bad = copy.deepcopy(checkpoint)
        bad[field] = "changed"
        with pytest.raises(ValueError, match="checkpoint"):
            restored.restore(bad, context, "Can the rule be applied?", ledger)
        assert restored.export() == checkpoint
    bad = copy.deepcopy(checkpoint)
    cast(list[dict[str, JsonValue]], bad["records"])[0]["review"] = None
    cast(list[dict[str, JsonValue]], bad["records"])[0]["review_hashes"] = {}
    with pytest.raises(ValueError, match="replace live"):
        restored.restore(bad, context, "Can the rule be applied?", ledger)


def test_changed_or_invented_checkpoint_originals_are_rejected_atomically() -> None:
    context, ledger, reviews = setup_reviews()
    seen(context, ledger, reviews)
    deliver(ledger, "answer-call", [1, 2])
    reviews.apply([review()], "answer-call", context, ledger)
    original_checkpoint = reviews.export()
    for target in ("anchor_hashes", "review_hashes", "lead_id"):
        checkpoint = copy.deepcopy(original_checkpoint)
        record = cast(list[dict[str, JsonValue]], checkpoint["records"])[0]
        record[target] = "lead_" + "0" * 64 if target == "lead_id" else {"1": "changed"}
        fresh = LegalSourceReviews(context, "Can the rule be applied?")
        with pytest.raises(ValueError):
            fresh.restore(checkpoint, context, "Can the rule be applied?", ledger)
        assert fresh.export()["records"] == []
