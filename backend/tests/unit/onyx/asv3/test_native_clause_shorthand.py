"""Canonical clause shorthand keeps inserted articles and physical delivery distinct."""

import copy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import native_named_authority_gap, statute_references
from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from tests.unit.onyx.asv3.test_legal_source_reviews import deliver
from tests.unit.onyx.asv3.test_native_authority import original


def _clause(title: str, article: str, letter: str) -> EvidenceItem:
    item = original(title, article)
    canonical = cast(dict[str, JsonValue], item.metadata["canonical_metadata"])
    canonical["clause_label"] = letter + ")"
    item.chunk_id = f"{item.chunk_id}:{letter}"
    assert item.search_doc is not None
    item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
    return item


def _gaps(gap: dict[str, JsonValue] | None) -> list[dict[str, JsonValue]]:
    assert gap is not None and isinstance(gap["named_authority_gaps"], list)
    return cast(list[dict[str, JsonValue]], gap["named_authority_gaps"])


def _context() -> RunContext:
    return RunContext(
        services={"research_profile": "experimental", "experimental_parallel": True}
    )


@pytest.mark.parametrize(
    "number,name,article,letter",
    [
        ("3065", "Katma Değer Vergisi Kanunu", "21", "a"),
        ("8917", "Faaliyet Kanunu", "41", "b"),
        ("8917", "Faaliyet Kanunu", "41", "ç"),
    ],
)
@pytest.mark.parametrize("numbered", [True, False])
def test_lowercase_actual_clause_matches_own_original_only(
    number: str, name: str, article: str, letter: str, numbered: bool
) -> None:
    ctx, ledger = _context(), EvidenceLedger()
    title = number + " sayılı " + name
    ledger.add([_clause(title, article, letter)], ctx)
    answer = f"{title if numbered else name} m.{article}/{letter} uygulanır [1]."
    assert (
        native_named_authority_gap(
            answer,
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is None
    )
    gap = native_named_authority_gap(
        answer.replace("[1]", ""),
        ledger,
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )
    assert _gaps(gap)[0]["matching_original_evidence"] == [1]


@pytest.mark.parametrize(
    "case",
    ["uppercase", "wrong_clause", "foreign", "missing", "qualifier", "actual_inserted"],
)
def test_ambiguous_or_wrong_identity_stays_open(case: str) -> None:
    ctx, ledger = _context(), EvidenceLedger()
    title = "8917 sayılı Faaliyet Kanunu"
    item = _clause(
        "8918 sayılı Başka Kanunu" if case == "foreign" else title,
        "41",
        "c" if case == "wrong_clause" else "b",
    )
    ledger.add([item], ctx)
    if case == "actual_inserted":
        ledger.add([original(title, "41/B")], ctx)
    answer = f"{title} {'Geçici ' if case == 'qualifier' else ''}m.41/{'B' if case == 'uppercase' else 'b'} uygulanır [1]."
    if case == "missing":
        ledger = EvidenceLedger()
    gap = native_named_authority_gap(
        answer,
        ledger,
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )
    assert gap is not None
    if case == "actual_inserted":
        assert _gaps(gap)[0]["matching_original_evidence"] == [2]
        assert (
            native_named_authority_gap(
                answer.replace("[1]", "[2]"),
                ledger,
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
            )
            is None
        )


@pytest.mark.parametrize("suffix", ["m.41/b ve m.41/B", "m.41/B ve m.41/b"])
def test_mixed_case_locators_do_not_license_inserted_article(suffix: str) -> None:
    ctx, ledger = _context(), EvidenceLedger()
    ledger.add([_clause("8917 sayılı Faaliyet Kanunu", "41", "b")], ctx)
    answer = "8917 sayılı Faaliyet Kanunu " + suffix + " uygulanır [1]."
    refs = statute_references(
        answer, strict_reference_boundaries=True, syntactic_reference_binding=True
    )
    assert {(ref.article, ref.clause_shorthand) for ref in refs} == {
        ("41/B", True),
        ("41/B", False),
    }
    gap = native_named_authority_gap(
        answer,
        ledger,
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )
    assert _gaps(gap)[0]["article"] == "41/B"
    assert _gaps(gap)[0]["matching_original_evidence"] == []
    state = AuthorityRequirements(ctx, answer, syntactic_reference_binding=True)
    assert state.publication_gap(answer, None, ctx, ledger) is not None
    deliver(ledger, "actual", [1])
    assert (
        state.publication_gap("Clause applies [1].", "actual", ctx, ledger) is not None
    )


def test_retained_prior_syntactic_requirement_preserves_snapshot_and_current_delivery() -> (
    None
):
    ctx, ledger = _context(), EvidenceLedger()
    answer = "8917 sayılı Faaliyet Kanunu m.41/b uygulanır [1]."
    state = AuthorityRequirements(ctx, answer, syntactic_reference_binding=True)
    assert state.publication_gap(answer, None, ctx, ledger) is not None
    saved = state.export()
    before = copy.deepcopy(saved)
    assert cast(list[dict[str, JsonValue]], saved["records"])[0]["article"] == "41/B"
    ledger.add([_clause("8917 sayılı Faaliyet Kanunu", "41", "b")], ctx)
    restored = AuthorityRequirements(ctx, answer, syntactic_reference_binding=True)
    restored.restore(saved, ctx, answer, ledger=ledger)
    assert saved == before
    assert restored.publication_gap(answer, "undelivered", ctx, ledger) is not None
    deliver(ledger, "actual", [1])
    assert restored.publication_gap(answer, "actual", ctx, ledger) is None
    assert restored.export() == before


def test_new_normalized_requirement_keeps_specific_clause_dependency() -> None:
    ctx, ledger = _context(), EvidenceLedger()
    title = "8917 sayılı Faaliyet Kanunu"
    ledger.add([_clause(title, "41", "b"), _clause(title, "41", "c")], ctx)
    answer = title + " m.41/b uygulanır [2]."
    state = AuthorityRequirements(ctx, answer, syntactic_reference_binding=True)
    assert state.publication_gap(answer, None, ctx, ledger) is not None
    deliver(ledger, "wrong", [2])
    assert (
        state.publication_gap("Clause condition applies [2].", "wrong", ctx, ledger)
        is not None
    )
    deliver(ledger, "correct", [1])
    assert (
        state.publication_gap("Clause condition applies [1].", "correct", ctx, ledger)
        is None
    )


@pytest.mark.parametrize("selectors", ["m.41/b ve m.42/c", "m.41/b ve m.41/c"])
def test_each_selected_clause_requirement_keeps_its_own_matching_original(
    selectors: str,
) -> None:
    ctx, ledger = _context(), EvidenceLedger()
    title = "8917 sayılı Faaliyet Kanunu"
    second_article = "42" if "42" in selectors else "41"
    ledger.add([_clause(title, "41", "b"), _clause(title, second_article, "c")], ctx)
    answer = title + " " + selectors + " uygulanır."
    state = AuthorityRequirements(ctx, answer, syntactic_reference_binding=True)
    assert state.publication_gap(answer, None, ctx, ledger) is not None
    records = cast(list[dict[str, JsonValue]], state.export()["records"])
    assert {row["article"] for row in records} == {"41/B", second_article + "/C"}
    deliver(ledger, "first", [1])
    gap = state.publication_gap("First condition applies [1].", "first", ctx, ledger)
    assert gap is not None
    unresolved = cast(
        list[dict[str, JsonValue]], gap["retained_authority_requirements"]
    )
    assert {row["article"] for row in unresolved} == {second_article + "/C"}
    deliver(ledger, "both", [1, 2])
    assert (
        state.publication_gap("Both conditions apply [1][2].", "both", ctx, ledger)
        is None
    )


def test_mixed_uncited_initial_references_and_later_inserted_priority() -> None:
    ctx, ledger = _context(), EvidenceLedger()
    title = "8917 sayılı Faaliyet Kanunu"
    ledger.add([_clause(title, "41", "b")], ctx)
    answer = title + " m.41/b ve m.41/B uygulanır."
    state = AuthorityRequirements(ctx, answer, syntactic_reference_binding=True)
    assert state.publication_gap(answer, None, ctx, ledger) is not None
    deliver(ledger, "base", [1])
    assert (
        state.publication_gap("Condition applies [1].", "base", ctx, ledger) is not None
    )
    ledger.add([original(title, "41/B")], ctx)
    deliver(ledger, "inserted", [2])
    assert (
        state.publication_gap(
            "Inserted condition applies [2].", "inserted", ctx, ledger
        )
        is None
    )


def test_declared_abbreviation_preserves_distinct_turkish_clause() -> None:
    ctx, ledger = _context(), EvidenceLedger()
    title = "8917 sayılı Faaliyet Kanunu"
    ledger.add([_clause(title, "41", "ç")], ctx)
    answer = (
        title + " (FK) bu olaya uygulanır [1].\n\nFK m.41/ç özel koşulu belirler [1]."
    )
    assert (
        native_named_authority_gap(
            answer,
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
            resolve_defined_abbreviations=True,
        )
        is None
    )
    assert (
        native_named_authority_gap(
            answer.replace("41/ç", "41/c"),
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
            resolve_defined_abbreviations=True,
        )
        is not None
    )


def test_plain_default_compact_reference_keeps_prior_article_only_diagnostic() -> None:
    ctx, ledger = _context(), EvidenceLedger()
    ledger.add([_clause("8917 sayılı Faaliyet Kanunu", "27", "a")], ctx)
    answer = (
        "8917 sayılı Faaliyet Kanunu m.27/2-b uygulanır ve m.41/c ayrıca incelenir [1]."
    )
    assert native_named_authority_gap(answer, ledger) is None
    uncited = native_named_authority_gap(answer.replace("[1]", ""), ledger)
    assert all("clause" not in row for row in _gaps(uncited))
    assert {row["article"] for row in _gaps(uncited)} == {"27"}


def test_retained_declared_abbreviation_resolves_only_current_owned_original() -> None:
    ctx, ledger = _context(), EvidenceLedger()
    title = "8917 sayılı Faaliyet Kanunu"
    ledger.add(
        [_clause(title, "41", "ç"), original("Uygulama Tebliği", "3", kind="tebliğ")],
        ctx,
    )
    definition = title + " (FK) bu olaya uygulanır [1].\n\n"
    answer = definition + "FK m.41/ç özel koşulu belirler [2]."
    state = AuthorityRequirements(ctx, answer, syntactic_reference_binding=True)
    assert state.publication_gap(answer, None, ctx, ledger) is not None
    before = state.export()
    assert {
        row["article"] for row in cast(list[dict[str, JsonValue]], before["records"])
    } == {"41/Ç"}
    corrected = definition + "FK m.41/ç özel koşulu belirler [1]."
    assert state.publication_gap(corrected, "unseen", ctx, ledger) is not None
    deliver(ledger, "own", [1])
    assert state.publication_gap(corrected, "own", ctx, ledger) is None
    assert state.export() == before
