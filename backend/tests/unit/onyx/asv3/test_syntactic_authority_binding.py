"""Explicit syntax cannot turn another clause's locator into a statute identity."""

import copy
import json

import pytest

from onyx.asv3.authority import native_named_authority_gap, statute_references
from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.models import RunContext
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original
from tests.unit.onyx.asv3.test_shared_originals import full_record


def references(text: str) -> list[tuple[str, str | None, str | None, str | None]]:
    return [
        (ref.number, ref.article, ref.paragraph, ref.clause)
        for ref in statute_references(
            text,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
    ]


@pytest.mark.parametrize(
    "later",
    [
        "hükümlerine göre faiz hesaplanır ve m. 47/1 uyarınca ceza uygulanır",
        "uyarınca ödeme yapılır; m. 47 ayrıca incelenir",
        "[1] ve Uygulama Yönetmeliği m. 47 uygulanır",
        "2026/72 sayılı karar incelenir ve m. 47 uygulanır",
        "| ödeme | m. 47 |",
    ],
)
def test_later_narrative_or_another_instrument_cannot_supply_article(
    later: str,
) -> None:
    assert references(f"8917 sayılı Faaliyet Kanunu {later}.") == [
        ("8917", None, None, None)
    ]


@pytest.mark.parametrize(
    "locator",
    [
        "m. 27",
        "Madde 27",
        "27 nci maddesi",
        "27. maddesi",
        "(m. 27)",
        "(27/2-ç) maddesi",
    ],
)
def test_direct_and_parenthesized_locators_keep_article(locator: str) -> None:
    expected = (
        ("8917", "27", "2", "ç") if "/" in locator else ("8917", "27", None, None)
    )
    assert references(f"8917 sayılı Faaliyet Kanunu’nun {locator} uygulanır.") == [
        expected
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Article 27 of Law No. 8917 applies.", [("8917", "27", None, None)]),
        (
            "27 nci maddesi (8917 sayılı Faaliyet Kanunu) uygulanır.",
            [("8917", "27", None, None)],
        ),
        (
            "8917 sayılı Faaliyet Kanunu m.27 ve m.28 uygulanır.",
            [("8917", "27", None, None), ("8917", "28", None, None)],
        ),
        (
            "8917 sayılı Faaliyet Kanunu m.27, 28 ve 29 maddeleri uygulanır.",
            [
                ("8917", "27", None, None),
                ("8917", "28", None, None),
                ("8917", "29", None, None),
            ],
        ),
        (
            "| 8917 sayılı Faaliyet Kanunu | m.27 | 7251 sayılı Veri Kanunu | m.45 |",
            [("8917", "27", None, None), ("7251", "45", None, None)],
        ),
        (
            "8917 sayılı Faaliyet Kanunu m.27, 7251 sayılı Veri Kanunu m.45.",
            [("8917", "27", None, None), ("7251", "45", None, None)],
        ),
    ],
)
def test_explicit_reverse_lists_and_table_cells_keep_independent_identities(
    text: str, expected: list[tuple[str, str | None, str | None, str | None]]
) -> None:
    assert references(text) == expected


@pytest.mark.parametrize("qualifier", ["geçici", "mükerrer", "ek"])
def test_qualified_article_identity_is_preserved(qualifier: str) -> None:
    parsed = statute_references(
        f"8917 sayılı Faaliyet Kanunu {qualifier} m.27/2-ç uygulanır.",
        syntactic_reference_binding=True,
    )
    assert len(parsed) == 1
    assert (
        parsed[0].qualifier
        == {"geçici": "gecici", "mükerrer": "mukerrer", "ek": "ek"}[qualifier]
    )
    assert (parsed[0].article, parsed[0].paragraph, parsed[0].clause) == (
        "27",
        "2",
        "ç",
    )


def test_inserted_article_is_not_collapsed_to_a_base_article() -> None:
    assert references("8917 sayılı Faaliyet Kanunu m.27/B uygulanır.") == [
        ("8917", "27/B", None, None)
    ]


def test_legacy_parser_result_is_unchanged_without_opt_in() -> None:
    text = "8917 sayılı Faaliyet Kanunu hükümlerine göre faiz ve m.47/1 ceza uygulanır."
    assert [
        (r.number, r.article)
        for r in statute_references(text, strict_reference_boundaries=True)
    ] == [("8917", "47")]
    assert references(text) == [("8917", None, None, None)]


def test_broad_statute_reference_still_requires_its_own_local_original() -> None:
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ"))
    answer = (
        "8917 sayılı Faaliyet Kanunu hükümlerine göre faiz ve m.47 ceza uygulanır [1]."
    )
    gap = native_named_authority_gap(answer, ledger, syntactic_reference_binding=True)
    assert gap is not None
    gaps = gap["named_authority_gaps"]
    assert isinstance(gaps, list) and isinstance(gaps[0], dict)
    assert gaps[0]["article"] is None
    ledger.add([original("8917 sayılı Faaliyet Kanunu", "51")], RunContext())
    assert (
        native_named_authority_gap(
            answer.replace("[1]", "[2]"), ledger, syntactic_reference_binding=True
        )
        is None
    )


def test_formal_names_abbreviations_and_multiple_articles_use_same_binding() -> None:
    ledger = ledger_with(
        original("8917 sayılı Faaliyet Kanunu", "27"),
        original("8917 sayılı Faaliyet Kanunu", "28"),
    )
    for answer in (
        "Faaliyet Kanunu'nun m.27 ve m.28 uygulanır [1][2].",
        "8917 sayılı Faaliyet Kanunu (FK) uygulanır [1].\n\nFK m.27 ve m.28 uygulanır [1][2].",
    ):
        assert (
            native_named_authority_gap(
                answer,
                ledger,
                syntactic_reference_binding=True,
                resolve_defined_abbreviations=True,
            )
            is None
        )
        assert native_named_authority_gap(
            answer.replace("[1][2]", "[1]"),
            ledger,
            syntactic_reference_binding=True,
            resolve_defined_abbreviations=True,
        )


def test_retention_cannot_recreate_a_later_clause_article_and_real_basis_survives_deletion() -> (
    None
):
    context = RunContext(scope={"authorized": "owned"})
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ"))
    ledger.record_delivery("first", "asv3_coordinator", [full_record(ledger, 1)])
    state = AuthorityRequirements(
        context, "Explain the procedure", syntactic_reference_binding=True
    )
    indirect = (
        "8917 sayılı Faaliyet Kanunu hükümlerine göre faiz ve m.47 ceza uygulanır [1]."
    )
    assert state.publication_gap(indirect, "first", context, ledger)
    records = state.export()["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    assert records[0]["article"] is None
    ledger.add([original("8917 sayılı Faaliyet Kanunu", "51")], context)
    ledger.record_delivery("next", "asv3_coordinator", [full_record(ledger, 2)])
    assert (
        state.publication_gap(
            "8917 sayılı Faaliyet Kanunu m.51 faiz düzenler [2].",
            "next",
            context,
            ledger,
        )
        is None
    )
    direct = "7251 sayılı Veri Kanunu m.29 uygulanır [1]."
    assert state.publication_gap(direct, "next", context, ledger)
    assert state.publication_gap("Sonuç uygulanır [2].", "next", context, ledger)
    saved = state.export()
    restored = AuthorityRequirements(
        context, "Explain the procedure", syntactic_reference_binding=True
    )
    restored.restore(saved, context, "Explain the procedure")
    assert restored.publication_gap("Sonuç uygulanır [2].", "next", context, ledger)
    assert restored.export() == saved


@pytest.mark.parametrize("old_policy", [False, True])
def test_checkpoint_policy_change_fails_closed_without_discarding_old_records(
    old_policy: bool,
) -> None:
    context = RunContext(scope={"authorized": "owned"})
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ"))
    state = AuthorityRequirements(
        context, "request", syntactic_reference_binding=old_policy
    )
    assert state.publication_gap(
        "8917 sayılı Faaliyet Kanunu m.27 uygulanır [1].", None, context, ledger
    )
    saved = copy.deepcopy(state.export())
    changed = AuthorityRequirements(
        context, "request", syntactic_reference_binding=not old_policy
    )
    with pytest.raises(ValueError, match="binding policy changed"):
        changed.restore(saved, context, "request")
    assert state.export() == saved
    if not old_policy:
        assert "reference_policy" not in saved


@pytest.mark.parametrize(
    "claim",
    [
        "Article 27 of Law No. 8917 applies [1].",
        "27 nci maddesi (8917 sayılı Faaliyet Kanunu) uygulanır [1].",
        "| 8917 sayılı Faaliyet Kanunu | m.27 | uygulanır [1] |",
    ],
)
def test_retained_explicit_reverse_and_table_anchor_reparses_without_losing_binding(
    claim: str,
) -> None:
    context = RunContext(scope={"authorized": "owned"})
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ"))
    state = AuthorityRequirements(context, "request", syntactic_reference_binding=True)
    assert state.publication_gap(claim, None, context, ledger)
    rows = state.export()["records"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["instrument_number"] == "8917" and rows[0]["article"] == "27"
    assert state.publication_gap("Sonuç uygulanır [1].", None, context, ledger)


def test_opt_in_retention_preserves_full_current_delivery_and_owner_fences() -> None:
    context = RunContext(
        scope={"authorized": "owned"}, services={"task_id": "first-worker"}
    )
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ"))
    state = AuthorityRequirements(context, "request", syntactic_reference_binding=True)
    assert state.publication_gap(
        "8917 sayılı Faaliyet Kanunu m.27 uygulanır [1].", None, context, ledger
    )
    ledger.add([original("8917 sayılı Faaliyet Kanunu", "27")], context)
    assert state.publication_gap(
        "Sonuç uygulanır [2].", "not-delivered", context, ledger
    )
    ledger.record_delivery("delivered", "asv3_researcher", [full_record(ledger, 2)])
    assert (
        state.publication_gap("Sonuç uygulanır [2].", "delivered", context, ledger)
        is None
    )
    context.services["task_id"] = "second-worker"
    assert state.publication_gap("Merhaba", None, context, ledger) is None
    assert state.view(context)["retained_authority_requirements"] == []
    context.services["task_id"] = "first-worker"
    assert state.publication_gap(
        "Sonuç uygulanır [2].", "not-delivered", context, ledger
    )


def test_legacy_checkpoint_serialization_has_exact_original_field_layout() -> None:
    context = RunContext(scope={"authorized": "owned"})
    state = AuthorityRequirements(context, "request")
    saved = state.export()
    expected = {
        key: saved[key]
        for key in (
            "version",
            "run_id",
            "scope_hash",
            "request_hash",
            "records",
            "record_integrity",
        )
    }
    assert json.dumps(saved, ensure_ascii=False) == json.dumps(
        expected, ensure_ascii=False
    )
    restored = AuthorityRequirements(context, "request")
    restored.restore(expected, context, "request")
    assert restored.export() == expected


def test_formal_only_retained_reference_preserves_temporary_article_qualifier() -> None:
    context = RunContext(scope={"authorized": "owned"})
    ledger = ledger_with(original("FAALİYET KANUNU", "28"))
    state = AuthorityRequirements(context, "request", syntactic_reference_binding=True)
    assert state.publication_gap(
        "Faaliyet Kanunu geçici m.27 uygulanır [1].", None, context, ledger
    )
    rows = state.export()["records"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["article"] == "27" and rows[0]["qualifier"] == "gecici"
    ledger.add([original("FAALİYET KANUNU", "27")], context)
    ledger.record_delivery("ordinary", "asv3_researcher", [full_record(ledger, 2)])
    assert state.publication_gap("Sonuç uygulanır [2].", "ordinary", context, ledger)


@pytest.mark.parametrize(
    "continuation",
    [
        ", 15 gün içinde başvuru yapılır",
        " ve 2 taksit halinde ödeme yapılır",
        ", 30 Eylül 2026 tarihinde uygulanır",
        ", 18/5/2026 tarihinde uygulanır",
        " ve 100 TL ödenir",
        ", 20 yüzde oranı uygulanır",
        ", 20% vergi hesaplanır",
        ", [15] kaynakları gösterir",
    ],
)
def test_numbers_in_conditions_are_not_invented_article_list_members(
    continuation: str,
) -> None:
    assert references("8917 sayılı Faaliyet Kanunu m.27" + continuation + ".") == [
        ("8917", "27", None, None)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Law No. 8917: Article 27 of Law No. 7251 applies.",
        "8917 sayılı Faaliyet Kanunu, 27 nci maddesi (7251 sayılı Veri Kanunu) uygulanır.",
    ],
)
def test_explicit_reverse_owner_prevents_binding_same_article_to_prior_instrument(
    text: str,
) -> None:
    assert references(text) == [("8917", None, None, None), ("7251", "27", None, None)]


@pytest.mark.parametrize("numbered", [False, True])
def test_actual_explicit_abbreviation_can_be_retained_and_closed_after_restore(
    numbered: bool,
) -> None:
    context = RunContext(scope={"authorized": "owned"})
    title = "8917 sayılı Faaliyet Kanunu" if numbered else "FAALİYET KANUNU"
    ledger = ledger_with(
        original("Uygulama Tebliği", "3", kind="tebliğ"), original(title, "28")
    )
    state = AuthorityRequirements(context, "request", syntactic_reference_binding=True)
    claim = f"{title} (FK) düzenlemeyi içerir [2].\n\nFK m.27 uygulanır [1]."
    native_gap = native_named_authority_gap(
        claim,
        ledger,
        strict_reference_boundaries=True,
        resolve_defined_abbreviations=True,
        syntactic_reference_binding=True,
    )
    assert native_gap is not None
    assert state.publication_gap(claim, None, context, ledger, native_gap=native_gap)
    saved = state.export()
    restored = AuthorityRequirements(
        context, "request", syntactic_reference_binding=True
    )
    restored.restore(saved, context, "request")
    assert restored.publication_gap("Sonuç uygulanır [1].", None, context, ledger)
    notice = (
        f"{title} (FK) düzenlemeyi içerir [2].\n\nFK m.27 özgün metni incelenemedi."
    )
    assert restored.publication_gap(notice, None, context, ledger) is None
    assert restored.publication_gap(
        "FK m.27 özgün metni incelenemedi.", None, context, ledger
    )
    ledger.add([original(title, "27")], context)
    ledger.record_delivery("full", "asv3_researcher", [full_record(ledger, 3)])
    assert (
        restored.publication_gap(
            f"{title} (FK) düzenlemeyi içerir [2].\n\nFK m.27 uygulanır [3].",
            "full",
            context,
            ledger,
        )
        is None
    )


def test_verified_source_quote_cannot_supply_answer_abbreviation_declaration() -> None:
    context = RunContext(scope={"authorized": "owned"})
    declaration = "8917 sayılı Faaliyet Kanunu (FK)"
    ledger = ledger_with(
        original("Uygulama Tebliği", "3", kind="tebliğ"),
        original("8917 sayılı Faaliyet Kanunu", "28", text=declaration),
    )
    state = AuthorityRequirements(context, "request", syntactic_reference_binding=True)
    assert state.publication_gap(
        "8917 sayılı Faaliyet Kanunu m.27 uygulanır [1].", None, context, ledger
    )
    notice = f'"{declaration}" [2].\n\nFK m.27 özgün metni incelenemedi.'
    assert state.publication_gap(notice, None, context, ledger)
