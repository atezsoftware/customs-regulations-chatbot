"""A captured qualified disclaimer preserves its supported lower-rule explanation."""

import copy
import json
from pathlib import Path

import pytest

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.cropped_article_bindings import cropped_named_authority_gap
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext

FIXTURE = Path(__file__).with_name("fixtures") / "cropped_authority_5315.json"
OPTIONS = {
    "strict_reference_boundaries": True,
    "syntactic_reference_binding": True,
    "strict_numeric_paragraphs": True,
}


def captured() -> tuple[RunContext, EvidenceLedger, str]:
    data = json.loads(FIXTURE.read_text())
    context = RunContext(services={"asv3_exp_cropped": True})
    ledger = EvidenceLedger()
    ledger.add(
        [EvidenceItem.model_validate(row["item"]) for row in data["records"]], context
    )
    text = data["authority_unit"]["text"].replace("[70]", "[1]").replace("[536]", "[2]")
    return context, ledger, text


def test_captured_negative_attribution_preserves_entire_supported_explanation() -> None:
    context, ledger, text = captured()
    before = copy.deepcopy(ledger.export())
    assert canonical_named_authority_gap(text, ledger, **OPTIONS)
    native_options = {
        key: value
        for key, value in OPTIONS.items()
        if key != "strict_numeric_paragraphs"
    }
    assert native_named_authority_gap(text, ledger, **native_options)
    assert cropped_named_authority_gap(text, ledger, context, **OPTIONS) is None
    assert "Doğrulanabilen Yönetmelik m.60/7’nin" in text
    assert "idari yazı kanuni özgün metnin yerine geçmez.[2][1]" in text
    assert ledger.export() == before


@pytest.mark.parametrize(
    "change",
    [
        "prefix",
        "suffix",
        "joined_prefix",
        "joined_suffix",
        "positive_predicate",
        "nonidentity_subject",
    ],
)
def test_qualified_disclaimer_cannot_hide_a_positive_statutory_claim(
    change: str,
) -> None:
    context, ledger, text = captured()
    if change == "prefix":
        text = "4925 sayılı Kanun m.13 Türk plakalı taşıta yasak koymaz; " + text
    elif change == "suffix":
        text += " 4925 sayılı Kanun m.13 Türk plakalı taşıta yasak koymaz [2]."
    elif change == "joined_prefix":
        text = text.replace(
            "m.13’ün bağımsız", "m.13’ün Türk plakalı taşıta izin verdiği ve bağımsız"
        )
    elif change == "joined_suffix":
        text = text.replace(
            "sunmuyorum.", "sunmuyorum ve Türk plakalı taşıta izin verir."
        )
    elif change == "positive_predicate":
        text = text.replace("sunmuyorum.", "sunuyorum.")
    else:
        text = text.replace(
            "4925 sayılı Kanun m.13’ün",
            "4925 sayılı Kanun m.13’ün izin verdiği sonucun",
        )
    assert cropped_named_authority_gap(text, ledger, context, **OPTIONS)
