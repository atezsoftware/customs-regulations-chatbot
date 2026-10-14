"""Captured 5309 originals keep local source identity without editing evidence."""

import copy
import json
from pathlib import Path

import pytest

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.cropped_article_bindings import (
    CroppedArticleBindings,
    cropped_identity_ledger,
    cropped_named_authority_gap,
    ordered_source_read_data,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)

FIXTURE = Path(__file__).with_name("fixtures") / "cropped_authority_5309.json"
OPTIONS = {
    "strict_reference_boundaries": True,
    "syntactic_reference_binding": True,
    "strict_numeric_paragraphs": True,
}
CLAIM = "4925 sayılı Kanun m.13’teki kabotaj yasağı Türk plakalı taşıta uygulanmaz [3]."


def captured() -> tuple[list[EvidenceItem], list[str]]:
    raw = json.loads(FIXTURE.read_text())
    items = [EvidenceItem.model_validate(row["item"]) for row in raw["records"]]
    units = [
        row["text"].replace("[255]", "[3]").replace("[378]", "[4]")
        for row in raw["authority_units"]
    ]
    return items, units


def setup(
    items: list[EvidenceItem] | None = None, indices: list[int] | None = None
) -> tuple[RunContext, EvidenceLedger, CroppedArticleBindings, ToolReceipt]:
    current = RunContext(
        run_id="captured-5309",
        scope={"corpus": 15},
        services={"asv3_exp_cropped": True},
    )
    originals = captured()[0] if items is None else items
    indices = [1, 2] if indices is None else indices
    ledger = EvidenceLedger()
    numbers = ledger.add(originals, current)
    state = CroppedArticleBindings(current, "Captured 5309 question")
    current.services["cropped_article_bindings"] = state
    delivered = [originals[index] for index in indices]
    receipt = ToolReceipt(
        call=CapabilityCall(
            name="read_source_range",
            arguments={"source_id": delivered[0].source_id, "start": 27, "limit": 2},
            call_id="captured-ordered-read",
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Captured canonical source range",
            evidence=delivered,
            data=ordered_source_read_data(delivered, current),
        ),
        evidence_ids=[numbers[index] for index in indices],
        elapsed_seconds=0,
    )
    return current, ledger, state, receipt


def test_captured_contiguous_law_and_explicit_lower_attribution_pass_unchanged() -> (
    None
):
    current, ledger, state, receipt = setup()
    units = captured()[1]
    before = copy.deepcopy(ledger.export())
    assert canonical_named_authority_gap(units[1], ledger, **OPTIONS)
    assert state.record_receipt(receipt, current, ledger) == 1
    assert (
        cropped_named_authority_gap("\n\n".join(units), ledger, current, **OPTIONS)
        is None
    )
    identity = cropped_identity_ledger(ledger, current)
    rows = {row["citation"]: row for row in identity.authority_metadata()}
    assert rows[1]["article_no"] == "9"
    assert rows[2]["article_no"] is None  # The heading chunk also contains article 12.
    assert rows[3]["article_no"] == "13"
    assert rows[3]["paragraph_no"] is None
    assert rows[3]["clause_label"] is None  # Stale prior (d) is not inherited.
    assert ledger.export() == before
    assert cropped_named_authority_gap(
        CLAIM.replace("[3]", "[2]"), ledger, current, **OPTIONS
    )
    for wrong in ("m.14", "m.13/1", "geçici m.13", "m.13/d"):
        assert cropped_named_authority_gap(
            CLAIM.replace("m.13", wrong), ledger, current, **OPTIONS
        )


def test_literal_unnumbered_article_can_supply_only_its_own_local_identity() -> None:
    current, ledger, _state, _receipt = setup()
    claim = "5015 sayılı Kanun m.9 uyarınca ithalat için lisans gerekir [1]."
    before = ledger.export()
    assert canonical_named_authority_gap(claim, ledger, **OPTIONS)
    assert cropped_named_authority_gap(claim, ledger, current, **OPTIONS) is None
    assert cropped_named_authority_gap(
        claim.replace("[1]", "[4]"), ledger, current, **OPTIONS
    )
    assert cropped_named_authority_gap(
        claim.replace("m.9", "m.9/1"), ledger, current, **OPTIONS
    )
    assert ledger.export() == before


@pytest.mark.parametrize(
    "case",
    [
        "isolated",
        "gap",
        "new_article",
        "metadata_only",
        "identifier",
        "quoted",
        "amending",
        "conflicting_article",
    ],
)
def test_unnumbered_continuation_requires_actual_unambiguous_adjacent_read(
    case: str,
) -> None:
    items, _units = captured()
    indices = [1, 2]
    if case == "isolated":
        indices = [2]
    elif case == "gap":
        items[2].metadata["position"] = 29
    else:
        raw = items[1].model_dump()
        canonical = raw["metadata"]["canonical_metadata"]
        if case == "new_article":
            raw["text"] = raw["text"].replace("MADDE 13", "MADDE 14")
            canonical["provision_identifiers"] = ["madde 14"]
        elif case == "metadata_only":
            raw["text"] = "Madde 13 kapsamında incelenen bir taşıma söz konusudur."
        elif case == "identifier":
            canonical["provision_identifiers"] = ["madde 14"]
        elif case == "quoted":
            raw["text"] = "“" + raw["text"] + "”"
        elif case == "amending":
            raw["text"] += " Bu madde aşağıdaki şekilde değiştirilmiştir."
        elif case == "conflicting_article":
            canonical["article_no"] = "12"
        raw["text_hash"] = ""
        items[1] = EvidenceItem.model_validate(raw)
    current, ledger, state, receipt = setup(items, indices)
    state.record_receipt(receipt, current, ledger)
    assert cropped_named_authority_gap(CLAIM, ledger, current, **OPTIONS)


@pytest.mark.parametrize("citation", ["", " [4]"])
def test_negative_attribution_never_exempts_a_positive_statutory_claim(
    citation: str,
) -> None:
    current, ledger, _state, _receipt = setup()
    disclaimer = "5015 sayılı Kanun m.9’un bağımsız doğrulaması olarak sunulmamaktadır."
    native_options = {
        key: value
        for key, value in OPTIONS.items()
        if key != "strict_numeric_paragraphs"
    }
    # The negative clause consumes only an explicit source identity and locator.
    assert native_named_authority_gap(disclaimer + citation, ledger, **native_options)
    assert (
        cropped_named_authority_gap(disclaimer + citation, ledger, current, **OPTIONS)
        is None
    )
    positive = "5015 sayılı Kanun m.9 ithalatı yasaklar; " + disclaimer + citation
    assert native_named_authority_gap(positive, ledger, **native_options)
    assert cropped_named_authority_gap(positive, ledger, current, **OPTIONS)
    mixed = (
        "5015 sayılı Kanun m.9 ithalatı yasaklar ve bağımsız doğrulaması olarak sunulmamaktadır."
        + citation
    )
    assert native_named_authority_gap(mixed, ledger, **native_options)
    assert cropped_named_authority_gap(mixed, ledger, current, **OPTIONS)
    trailing = disclaimer.rstrip(".") + " ve ithalatı yasaklar." + citation
    assert cropped_named_authority_gap(trailing, ledger, current, **OPTIONS)
