"""Reference diagnostics explain provenance without discharging publication guards."""

import copy

import pytest
from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.authority_reference_diagnostics import (
    authority_reference_diagnostics,
    source_contained_reference_catalogue,
)
from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.asv3.witnesses import original_witness_text
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original
from tests.unit.onyx.asv3.test_shared_originals import full_record

REFERRAL = "Bu işlem Faaliyet Kanunu esaslarına göre ele alınır."
ANSWER = (
    "7284 sayılı Veri Kanunu m.8, işlemin Faaliyet Kanunu esaslarına göre "
    "ele alınacağını belirtir [1]."
)


def setup() -> tuple[RunContext, EvidenceLedger, AuthorityRequirements]:
    context = RunContext(scope={"authorized": "owned"})
    context.services.update(research_profile="experimental", experimental_parallel=True)
    ledger = ledger_with(
        original("7284 sayılı Veri Kanunu", "8", text=REFERRAL),
        original("8917 sayılı Faaliyet Kanunu", "27"),
    )
    ledger.record_delivery("current", "asv3_coordinator", [full_record(ledger, 1)])
    state = AuthorityRequirements(
        context, "Which procedure applies?", syntactic_reference_binding=True
    )
    return context, ledger, state


def diagnostics(gap: dict[str, JsonValue]) -> list[JsonValue]:
    diagnostic = gap["authority_reference_diagnostics"]
    assert isinstance(diagnostic, dict)
    units = diagnostic["units"]
    assert isinstance(units, list)
    return units


def test_gap_reports_exact_own_current_reference_without_changing_requirements() -> (
    None
):
    context, ledger, state = setup()
    gap = state.publication_gap(ANSWER, "current", context, ledger)
    assert gap is not None and gap["retained_authority_requirements"]
    before = copy.deepcopy(state.export())
    before_ledger = copy.deepcopy(ledger.export())
    rows = diagnostics(gap)
    assert {row["kind"] for row in rows if isinstance(row, dict)} == {
        "named_authority_gaps",
        "retained_authority_requirements",
    }
    for row in rows:
        assert isinstance(row, dict) and row["unit_text"] == ANSWER
        witnesses = row["source_contained_reference_witnesses"]
        assert isinstance(witnesses, list) and len(witnesses) == 1
        witness = witnesses[0]
        assert isinstance(witness, dict)
        assert witness["citation"] == 1
        assert witness["role"] == "source_contained_reference_navigation"
        assert witness["source_quote"] == REFERRAL
        item = ledger.get(1)
        assert item is not None and witness["text_hash"] == item.text_hash
        assert (
            original_witness_text(1, REFERRAL, str(witness["witness_id"])) == REFERRAL
        )
    assert state.export() == before
    assert ledger.export() == before_ledger
    assert state.publication_gap(ANSWER, "current", context, ledger) == gap


@pytest.mark.parametrize("call_id", [None, "new-undelivered", "partial"])
def test_stale_or_partial_delivery_cannot_supply_reference_witness(
    call_id: str | None,
) -> None:
    context, ledger, state = setup()
    record = full_record(ledger, 1)
    record["text"] = REFERRAL[:20]
    ledger.record_delivery("partial", "asv3_coordinator", [record])
    gap = state.publication_gap(ANSWER, call_id, context, ledger)
    assert gap is not None
    assert all(
        row["source_contained_reference_witnesses"] == []
        for row in diagnostics(gap)
        if isinstance(row, dict)
    )


def test_other_fully_delivered_inline_and_noninline_sources_are_not_substituted() -> (
    None
):
    context, ledger, state = setup()
    ledger.add([original("Autre Kaynak", "2", kind="tebliğ", text=REFERRAL)], context)
    ledger.record_delivery("only-foreign", "asv3_coordinator", [full_record(ledger, 3)])
    gap = state.publication_gap(ANSWER, "only-foreign", context, ledger)
    assert gap is not None
    assert all(
        row["source_contained_reference_witnesses"] == []
        for row in diagnostics(gap)
        if isinstance(row, dict)
    )


@pytest.mark.parametrize(
    "defect",
    ["derived", "external", "untrusted", "document_id", "chunk_id", "uncitable"],
)
def test_noncanonical_originals_never_supply_diagnostic_passages(defect: str) -> None:
    item = original("7284 sayılı Veri Kanunu", "8", text=REFERRAL)
    if defect in {"derived", "external", "untrusted"}:
        item.metadata[defect] = True
    elif defect == "uncitable":
        item.search_doc = None
    else:
        assert item.search_doc is not None
        if defect == "document_id":
            item.search_doc.document_id = "foreign-source"
        else:
            item.search_doc.metadata["regulatory_chunk_id"] = "foreign-chunk"
    ledger = ledger_with(item)
    gap: dict[str, JsonValue] = {
        "named_authority_gaps": [
            {
                "unit_id": assertion_inventory(ANSWER)[0]["unit_id"],
                "reference_text": "Faaliyet Kanunu",
                "instrument_number": "8917",
                "article": None,
            }
        ]
    }
    enriched = authority_reference_diagnostics(ANSWER, gap, ledger, {1})
    assert all(
        row["source_contained_reference_witnesses"] == []
        for row in diagnostics(enriched)
        if isinstance(row, dict)
    )
    assert (
        source_contained_reference_catalogue(ledger, [full_record(ledger, 1)]) is None
    )


def test_quotes_only_affect_fresh_native_reference_not_independent_or_retained_rules() -> (
    None
):
    context, ledger, state = setup()
    quoted = f'7284 sayılı Veri Kanunu m.8, "{REFERRAL}" der [1].'
    assert (
        native_named_authority_gap(
            quoted,
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is None
    )
    independent = (
        quoted + "\n\n8917 sayılı Faaliyet Kanunu m.27 uyarınca izin gerekir [1]."
    )
    assert (
        native_named_authority_gap(
            independent,
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is not None
    )
    assert state.publication_gap(ANSWER, "current", context, ledger) is not None
    assert state.publication_gap(quoted, "current", context, ledger) is not None
    direct = "8917 sayılı Faaliyet Kanunu m.27 uyarınca izin gerekir [1]."
    assert state.publication_gap(direct, "current", context, ledger) is not None
    deleted = state.publication_gap("İzin gerekir [1].", "current", context, ledger)
    assert deleted is not None and deleted["retained_authority_requirements"]
    assert "authority_reference_diagnostics" not in deleted


def test_retained_origin_text_requires_the_exact_current_unit_id() -> None:
    _, ledger, _ = setup()
    gap: dict[str, JsonValue] = {
        "retained_authority_requirements": [
            {
                "origin_unit_id": "au0-not-this-block",
                "reference_text": "Faaliyet Kanunu",
                "instrument_number": "8917",
                "article": None,
            }
        ]
    }
    before = copy.deepcopy(gap)
    assert authority_reference_diagnostics(ANSWER, gap, ledger, {1}) is gap
    assert gap == before


@pytest.mark.parametrize(
    "source_reference,article,qualifier,expected",
    [
        ("m.28", "27", None, False),
        ("m.27", "27", None, True),
        ("Geçici Madde 27", "27", None, False),
        ("Geçici Madde 27", "27", "gecici", True),
        ("m.28", None, None, True),
    ],
)
def test_targeted_reference_witness_matches_article_and_qualifier(
    source_reference: str,
    article: str | None,
    qualifier: str | None,
    expected: bool,
) -> None:
    ledger = ledger_with(
        original(
            "7284 sayılı Veri Kanunu",
            "8",
            text=f"8917 sayılı Faaliyet Kanunu {source_reference} esaslarına göre işlem yapılır.",
        )
    )
    gap: dict[str, JsonValue] = {
        "named_authority_gaps": [
            {
                "unit_id": assertion_inventory(ANSWER)[0]["unit_id"],
                "reference_text": "8917 sayılı Faaliyet Kanunu",
                "instrument_number": "8917",
                "article": article,
                "qualifier": qualifier,
            }
        ]
    }
    enriched = authority_reference_diagnostics(ANSWER, gap, ledger, {1})
    rows = diagnostics(enriched)
    assert len(rows) == 1 and isinstance(rows[0], dict)
    assert bool(rows[0]["source_contained_reference_witnesses"]) is expected


@pytest.mark.parametrize("profile", ["plain", "normal", "parallel", "hosted"])
def test_only_trusted_parallel_profiles_receive_diagnostics(profile: str) -> None:
    context, ledger, state = setup()
    if profile == "plain":
        context.services["experimental_parallel"] = False
    elif profile == "normal":
        context.services["research_profile"] = "normal"
    elif profile == "hosted":
        context.services.update(
            experimental_parallel=False,
            serial_session_diagnostics=True,
            lean_native_mode=True,
            task_id="owned-task",
        )
    gap = state.publication_gap(ANSWER, "current", context, ledger)
    assert gap is not None
    assert ("authority_reference_diagnostics" in gap) == (
        profile in {"parallel", "hosted"}
    )


def test_preanswer_catalogue_uses_exact_physical_records_excludes_own_identity_and_text() -> (
    None
):
    _, ledger, _ = setup()
    record = full_record(ledger, 1)
    before = copy.deepcopy(record)
    catalogue = source_contained_reference_catalogue(ledger, [record])
    assert catalogue is not None
    references = catalogue["references"]
    assert isinstance(references, list) and references
    for reference in references:
        assert isinstance(reference, dict)
        assert reference["citation"] == 1
        assert reference["instrument_number"] == "8917"
        assert reference["article"] is None
        assert "source_quote" not in reference and "text" not in reference
        assert (
            original_witness_text(1, REFERRAL, str(reference["witness_id"])) == REFERRAL
        )
    assert record == before
    own = original(
        "7284 sayılı Veri Kanunu", "8", text="7284 sayılı Veri Kanunu m.8 uygulanır."
    )
    own_ledger = ledger_with(own)
    assert (
        source_contained_reference_catalogue(own_ledger, [full_record(own_ledger, 1)])
        is None
    )


def test_same_formal_name_does_not_hide_an_explicit_different_instrument_number() -> (
    None
):
    ledger = ledger_with(
        original(
            "7284 sayılı Faaliyet Kanunu",
            "8",
            text="8917 sayılı Faaliyet Kanunu m.27 esaslarına göre işlem yapılır.",
        ),
        original("8917 sayılı Faaliyet Kanunu", "27"),
    )
    catalogue = source_contained_reference_catalogue(ledger, [full_record(ledger, 1)])
    assert catalogue is not None
    references = catalogue["references"]
    assert isinstance(references, list)
    assert any(
        isinstance(reference, dict) and reference["instrument_number"] == "8917"
        for reference in references
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_id", "foreign-source"),
        ("chunk_id", "foreign-chunk"),
        ("text_hash", "foreign-hash"),
        ("text", "Faaliyet Kanunu"),
        ("start_char", 1),
        ("start_char", False),
        ("start_char", 0.0),
        ("end_char", 1),
        ("end_char", float(len(REFERRAL))),
        ("truncated", True),
        ("truncated", 0),
        ("citable", False),
        ("citation", True),
    ],
)
def test_preanswer_catalogue_refuses_partial_or_foreign_physical_records(
    field: str, value: JsonValue
) -> None:
    _, ledger, _ = setup()
    record = full_record(ledger, 1)
    record[field] = value
    assert source_contained_reference_catalogue(ledger, [record]) is None
