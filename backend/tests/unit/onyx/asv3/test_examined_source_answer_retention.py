"""A selected examined effect must survive in the cited answer, not only its metadata."""

import copy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.legal_source_reviews import (
    LegalSourceReviews,
    _retention_text,
    operative_review_retention_enabled,
)
from onyx.asv3.models import RunContext, ToolOutcome
from tests.unit.onyx.asv3.test_legal_source_assembly_reviews import assess
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    lead_id,
    navigation,
    review,
    seen,
    setup_reviews,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record, original

EFFECT = "The particular changed wording may affect this outcome."
LIMITATIONS = "The underlying obligation and applicable dates remain distinct."


def scope(context: RunContext, *, hosted: bool = False) -> None:
    context.services.update(
        research_profile="experimental", experimental_parallel=not hosted
    )
    if hosted:
        context.services.update(
            serial_session_diagnostics=True,
            lean_native_mode=True,
            task_id="owned-question",
        )


def retained(*, citations: str = "[2]") -> str:
    return f"A supported independent rule [1].\n\n{EFFECT} {citations}\n\n{LIMITATIONS} {citations}"


def omissions(gap: ToolOutcome | None) -> list[dict[str, JsonValue]]:
    assert gap is not None
    assert gap.data["pending_related_source_review"] is True
    assert gap.data["unread_related_sources"] == []
    assert gap.data["undisclosed_related_source_gaps"] == []
    return cast(
        list[dict[str, JsonValue]], gap.data["unretained_examined_source_effects"]
    )


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("same_block", [False, True])
def test_copied_clauses_with_own_citations_pass_without_mutation_or_extra_work(
    hosted: bool, same_block: bool
) -> None:
    context, ledger, reviews = setup_reviews()
    scope(context, hosted=hosted)
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    answer = (
        f"Applied here, {EFFECT} The limit is: {LIMITATIONS} [2]."
        if same_block
        else retained()
    )
    before = (
        copy.deepcopy(reviews.export()),
        ledger.export(),
        context.budget.snapshot(),
    )
    assert (
        reviews.publication_gap(answer, "answer", context, ledger, [review()]) is None
    )
    assert (reviews.export(), ledger.export(), context.budget.snapshot()) == before
    reviews.apply([review()], "answer", context, ledger)
    assert reviews.publication_gap(answer, "answer", context, ledger) is None


@pytest.mark.parametrize(
    ("answer", "missing"),
    [
        (f"{LIMITATIONS} [2].", ["effect"]),
        (f"{EFFECT} [2].", ["limitations"]),
        (f"The changed rule matters [2].\n\n{LIMITATIONS} [2].", ["effect"]),
        (f"{EFFECT} [1].\n\n{LIMITATIONS} [2].", ["effect"]),
        (f"{EFFECT}\n\n{LIMITATIONS}\n\nReferences [2].", ["effect", "limitations"]),
        (f"{EFFECT}\n[2]\n\n{LIMITATIONS} [2].", ["effect"]),
        (f"{EFFECT}\n- [2]\n\n{LIMITATIONS} [2].", ["effect"]),
        (f"# {EFFECT} [2]\n\n{LIMITATIONS} [2].", ["effect"]),
        (f"{EFFECT}\n# Source [2]\n\n{LIMITATIONS} [2].", ["effect"]),
        (f"{EFFECT}\n\n# {LIMITATIONS} [2]", ["effect", "limitations"]),
    ],
)
def test_omission_paraphrase_remote_or_presentation_citation_is_actionable(
    answer: str, missing: list[str]
) -> None:
    context, ledger, reviews = setup_reviews()
    scope(context)
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    row = omissions(
        reviews.publication_gap(answer, "answer", context, ledger, [review()])
    )[0]
    assert row == {
        "lead_id": lead_id(),
        "source_id": "decision",
        "missing_answer_passages": missing,
        "required_inline_citations": [2],
        "undelivered_evidence_numbers": [],
        **({"unbound_evidence_numbers": [2]} if len(missing) == 2 else {}),
    }


@pytest.mark.parametrize("effect", ["[2]", "- [2]", "**[2]**", "# Result [2]"])
def test_metadata_cannot_call_a_citation_or_heading_an_operative_effect(
    effect: str,
) -> None:
    context, ledger, reviews = setup_reviews()
    scope(context)
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    rows = omissions(
        reviews.publication_gap(
            effect + "\n\n" + LIMITATIONS + " [2].",
            "answer",
            context,
            ledger,
            [review(effect=effect)],
        )
    )
    assert rows[0]["missing_answer_passages"] == ["effect"]


def test_separate_passages_bind_their_own_witnesses_and_cover_the_selected_union() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    scope(context)
    number = ledger.add(
        [
            original(
                "holding-limit",
                "The decision applies on the specified date.",
                source="decision",
            )
        ],
        context,
    )[0]
    assert number == 4
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2, 4])
    assessment = review(
        witnesses=[
            {"citation": 2, "start_char": 0, "end_char": 39},
            {"citation": 4, "start_char": 0, "end_char": 20},
        ]
    )
    assert (
        reviews.publication_gap(
            retained(citations="[2, 4]"), "answer", context, ledger, [assessment]
        )
        is None
    )
    assert (
        reviews.publication_gap(
            f"{EFFECT} [2].\n\n{LIMITATIONS} [4].",
            "answer",
            context,
            ledger,
            [assessment],
        )
        is None
    )
    rows = omissions(
        reviews.publication_gap(
            retained(),
            "answer",
            context,
            ledger,
            [assessment],
        )
    )
    assert rows[0]["missing_answer_passages"] == []
    assert rows[0]["required_inline_citations"] == [2, 4]
    assert rows[0]["unbound_evidence_numbers"] == [4]
    rows = omissions(
        reviews.publication_gap(
            f"{EFFECT} [1].\n\n{LIMITATIONS} [2, 4].",
            "answer",
            context,
            ledger,
            [assessment],
        )
    )
    assert rows[0]["missing_answer_passages"] == ["effect"]
    assert "unbound_evidence_numbers" not in rows[0]


def test_each_copied_field_uses_one_substantive_block_as_the_exposed_contract_requires() -> (
    None
):
    context, ledger, reviews = setup_reviews()
    scope(context)
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    passage = f"{EFFECT} [2].\n\n{LIMITATIONS} [2]."
    rows = omissions(
        reviews.publication_gap(
            passage,
            "answer",
            context,
            ledger,
            [review(effect=passage)],
        )
    )
    assert rows[0]["missing_answer_passages"] == ["effect"]


@pytest.mark.parametrize("partial", [False, True])
def test_saved_review_requires_current_full_original_delivery(partial: bool) -> None:
    context, ledger, reviews = setup_reviews()
    scope(context, hosted=True)
    seen(context, ledger, reviews)
    deliver(ledger, "assessment", [1, 2])
    reviews.apply([review()], "assessment", context, ledger)
    deliver(ledger, "current", [1])
    if partial:
        record = full_record(ledger, 2)
        record["text"] = str(record["text"])[:20]
        ledger.record_delivery("current", "asv3_coordinator", [record])
    row = omissions(reviews.publication_gap(retained(), "current", context, ledger))[0]
    assert row["missing_answer_passages"] == []
    assert row["undelivered_evidence_numbers"] == [2]
    with pytest.raises(ValueError, match="fully delivered"):
        reviews.publication_gap(retained(), "current", context, ledger, [review()])


@pytest.mark.parametrize(
    "services",
    [
        {},
        {"research_profile": "experimental", "experimental_parallel": False},
        {"research_profile": "normal", "experimental_parallel": True},
        {"research_profile": "deep", "experimental_parallel": True},
        {"research_profile": "experimental", "experimental_parallel": 1},
        {
            "research_profile": "experimental",
            "experimental_parallel": False,
            "serial_session_diagnostics": True,
            "lean_native_mode": True,
        },
        {
            "research_profile": "experimental",
            "experimental_parallel": False,
            "serial_session_diagnostics": 1,
            "lean_native_mode": True,
            "task_id": "owned",
        },
    ],
)
def test_other_profiles_keep_existing_review_and_assembly_contract(
    services: dict[str, JsonValue],
) -> None:
    context, ledger, reviews = setup_reviews()
    context.services.update(services)
    assert not operative_review_retention_enabled(context)
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    reviews.apply([review()], "answer", context, ledger)
    assert (
        reviews.publication_gap(
            "Existing conditional result [1].", "answer", context, ledger
        )
        is None
    )
    if "task_id" not in services:
        assert (
            reviews.publication_gap_for_assembly(
                "Existing conditional result [1].",
                context,
                ledger,
                accepted_owners=set(),
            )
            is None
        )


def test_hosted_flag_cannot_enable_depth_one_or_missing_task_contract() -> None:
    context, ledger, reviews = setup_reviews()
    scope(context, hosted=True)
    child = context.child()
    assert not operative_review_retention_enabled(child)
    seen(child, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    assert (
        reviews.publication_gap(
            "Existing conditional result [1].", "answer", child, ledger, [review()]
        )
        is None
    )


@pytest.mark.parametrize("hosted", [False, True])
def test_not_material_and_unresolved_contracts_are_unchanged(hosted: bool) -> None:
    context, ledger, reviews = setup_reviews()
    scope(context, hosted=hosted)
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    assert (
        reviews.publication_gap(
            "The independent outcome [1].",
            "answer",
            context,
            ledger,
            [review(status="not_material")],
        )
        is None
    )
    gap = "The interaction remains unexamined for this event date."
    raw = review(status="unresolved", source_role="unknown", witnesses=[], gap=gap)
    assert (
        reviews.publication_gap(
            "The independent outcome [1].\n\n" + gap, "answer", context, ledger, [raw]
        )
        is None
    )


def test_restore_preserves_retention_obligation_and_current_source_fence() -> None:
    context, ledger, reviews = setup_reviews()
    scope(context)
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    reviews.apply([review()], "answer", context, ledger)
    restored = LegalSourceReviews(context, "Can the rule be applied?")
    restored.restore(reviews.export(), context, "Can the rule be applied?", ledger)
    assert restored.export() == reviews.export()
    assert omissions(
        restored.publication_gap("Omitted result [1].", "answer", context, ledger)
    )[0]["missing_answer_passages"] == ["effect", "limitations"]
    assert restored.publication_gap(retained(), "answer", context, ledger) is None
    ledger._items[2].text_hash = "a" * 64
    with pytest.raises(ValueError, match="original changed"):
        restored.publication_gap(retained(), "answer", context, ledger)


def test_assembly_keeps_each_accepted_owners_distinct_examined_effect() -> None:
    root, ledger, reviews = setup_reviews()
    scope(root)
    seen(root, ledger, reviews)
    assess(root, ledger, reviews, "first", review())
    second_effect = "A separate assigned outcome changes under this operative text."
    assess(root, ledger, reviews, "second", review(effect=second_effect))
    before = copy.deepcopy(reviews.export())
    rows = omissions(
        reviews.publication_gap_for_assembly(
            retained(), root, ledger, accepted_owners={"first", "second"}
        )
    )
    assert rows[0]["owner"] == "second"
    assert rows[0]["missing_answer_passages"] == ["effect"]
    assert (
        reviews.publication_gap_for_assembly(
            retained() + "\n\n" + second_effect + " [2].",
            root,
            ledger,
            accepted_owners={"first", "second"},
        )
        is None
    )
    assert reviews.export() == before
    assert ledger.completely_delivered("law-call") == {1}


def test_assembly_own_review_and_unrelated_pending_leads_are_not_skipped() -> None:
    root, ledger, reviews = setup_reviews()
    scope(root)
    seen(root, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    reviews.apply([review()], "answer", root, ledger)
    reviews.record_delivery("law-call", root, navigation("other-decision"), ledger)
    gap = reviews.publication_gap_for_assembly(
        "Only a broad result [1].", root, ledger, accepted_owners=set()
    )
    assert gap is not None
    assert (
        cast(list[dict[str, JsonValue]], gap.data["unread_related_sources"])[0][
            "source_id"
        ]
        == "other-decision"
    )
    assert (
        cast(
            list[dict[str, JsonValue]], gap.data["unretained_examined_source_effects"]
        )[0]["lead_id"]
        == lead_id()
    )
    retained_gap = reviews.publication_gap_for_assembly(
        retained(), root, ledger, accepted_owners=set()
    )
    assert retained_gap is not None
    assert "unretained_examined_source_effects" not in retained_gap.data


def test_foreign_owner_or_foreign_original_cannot_supply_retained_effect() -> None:
    context, ledger, reviews = setup_reviews()
    scope(context)
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2, 3])
    sibling = context.child()
    sibling.services["task_id"] = "foreign"
    with pytest.raises(ValueError, match="this task"):
        reviews.publication_gap(retained(), "answer", sibling, ledger, [review()])
    with pytest.raises(ValueError):
        reviews.publication_gap(
            retained(citations="[3]"),
            "answer",
            context,
            ledger,
            [review(witnesses=[{"citation": 3, "start_char": 0, "end_char": 10}])],
        )


@pytest.mark.parametrize("tuned", [False, True])
def test_quote_typography_retention_is_isolated_to_tuned(tuned: bool) -> None:
    context, ledger, reviews = setup_reviews()
    scope(context)
    if tuned:
        context.services["asv3_workflow_variant"] = "asv3_tuned"
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    effect = "The ‘particular’ wording may affect this outcome."
    limit = "The rule’s dates remain distinct."
    answer = "The 'particular' wording may affect this outcome. [2]\n\nThe rule's dates remain distinct. [2]"
    gap = reviews.publication_gap(
        answer, "answer", context, ledger, [review(effect=effect, limitations=limit)]
    )
    assert (gap is None) is tuned


@pytest.mark.parametrize("tuned", [False, True])
def test_faithful_formatted_passage_keeps_its_local_citations(tuned: bool) -> None:
    context, ledger, reviews = setup_reviews()
    scope(context)
    if tuned:
        context.services["asv3_workflow_variant"] = "asv3_tuned"
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    effect = "The particular wording may affect this outcome."
    limitations = (
        "The underlying obligation remains distinct. "
        "Its application depends on the event date."
    )
    answer = (
        "**The particular wording** may affect this outcome [2].\n\n"
        "**The underlying obligation remains distinct.** "
        "Its application depends on the event date [2]."
    )
    gap = reviews.publication_gap(
        answer,
        "answer",
        context,
        ledger,
        [review(effect=effect, limitations=limitations)],
    )
    assert (gap is None) is tuned


@pytest.mark.parametrize(
    "answer",
    [
        "**The particular wording** cannot affect this outcome [2].",
        "**The particular wording** may affect this outcome [1].",
        "**The particular wording** may affect this outcome.\n\nReferences [2].",
        "# **The particular wording** may affect this outcome [2].",
        "**The particular wording** may affect this outcome.\n[2]",
    ],
)
def test_formatted_retention_rejects_changed_or_unbound_claims(answer: str) -> None:
    context, ledger, reviews = setup_reviews()
    scope(context)
    context.services["asv3_workflow_variant"] = "asv3_tuned"
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    gap = reviews.publication_gap(
        answer + "\n\n" + LIMITATIONS + " [2].",
        "answer",
        context,
        ledger,
        [review(effect="The particular wording may affect this outcome.")],
    )
    assert omissions(gap)[0]["missing_answer_passages"] == ["effect"]


def test_retention_presentation_normalization_preserves_arithmetic_operators() -> None:
    assert _retention_text("The base is x**2 + y**2 [2].") == (
        "The base is x**2 + y**2."
    )


@pytest.mark.parametrize(
    "changed",
    [
        "The 'particular' wording cannot affect this outcome. [2]",
        "The 'particular' wording may affect this outcome. [1]",
    ],
)
def test_typography_normalization_does_not_accept_changed_effect_or_citation(
    changed: str,
) -> None:
    context, ledger, reviews = setup_reviews()
    context.services["asv3_workflow_variant"] = "asv3_tuned"
    seen(context, ledger, reviews)
    deliver(ledger, "answer", [1, 2])
    gap = reviews.publication_gap(
        changed + "\n\n" + LIMITATIONS + " [2]",
        "answer",
        context,
        ledger,
        [review(effect="The ‘particular’ wording may affect this outcome.")],
    )
    assert omissions(gap)[0]["missing_answer_passages"] == ["effect"]
