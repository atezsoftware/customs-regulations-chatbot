"""Whitespace resolution cannot change words, scope, or label validation."""

import json
from hashlib import sha256

import pytest

from onyx.regulatory.labeling.intake_quote_resolution import (
    resolve_intake_labeling_response,
)
from onyx.regulatory.labeling.provider import (
    LabelDefinition,
    TaxonomyDefinition,
    validate_labeling_response,
)


def _taxonomy() -> TaxonomyDefinition:
    return TaxonomyDefinition(
        name="Labels",
        labels=[LabelDefinition(id="customs", name="Customs", description="Documents")],
    )


def _response(quote: str) -> str:
    return json.dumps(
        {
            "labels": [{"label_id": "customs", "evidence_quote": quote}],
            "abstained": False,
        }
    )


@pytest.mark.parametrize(
    "source",
    [
        "Belge\nibraz edilir.",
        "Belge\n   ibraz edilir.",
        "Belge\r\nibraz edilir.",
        "Belge\u00a0ibraz edilir.",
        "Belge\t\n\u2003ibraz edilir.",
    ],
)
def test_whitespace_variant_resolves_to_exact_span_and_preserves_raw_receipt(
    source: str,
) -> None:
    quote = "Belge ibraz edilir."
    canonical = "Önce: " + source + " Sonra."
    raw = _response(quote)
    with pytest.raises(ValueError, match="exact quote"):
        validate_labeling_response(raw, text=canonical, taxonomy=_taxonomy())
    outcome, audits = resolve_intake_labeling_response(
        raw, text=canonical, taxonomy=_taxonomy()
    )
    assert outcome.labels[0].evidence_quote == source
    assert raw == _response(quote)
    assert len(audits) == 1
    audit = audits[0]
    assert audit["method"] == "unicode_whitespace_unique" and audit["version"] == 1
    assert audit["original_quote_sha256"] == sha256(quote.encode()).hexdigest()
    assert audit["resolved_quote_sha256"] == sha256(source.encode()).hexdigest()
    assert audit["source_text_sha256"] == sha256(canonical.encode()).hexdigest()
    assert audit["source_start_char"] == 6 and audit["source_end_char"] == 6 + len(
        source
    )
    assert (
        validate_labeling_response(
            outcome.model_dump_json(), text=canonical, taxonomy=_taxonomy()
        )
        == outcome
    )
    assert set(outcome.labels[0].model_dump()) == {"label_id", "evidence_quote"}


def test_already_exact_quote_remains_unchanged_even_when_repeated() -> None:
    raw = _response("Belge ibraz edilir.")
    expected = validate_labeling_response(
        raw, text="Belge ibraz edilir. Belge ibraz edilir.", taxonomy=_taxonomy()
    )
    assert resolve_intake_labeling_response(
        raw, text="Belge ibraz edilir. Belge ibraz edilir.", taxonomy=_taxonomy()
    ) == (expected, [])


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        ("Belge\nibraz edilir. Belge\tibraz edilir.", "Belge ibraz edilir."),
        ("Belge\nibraz edilir.", "belge ibraz edilir."),
        ("Belge\nibraz edilir.", "Belge ibraz edilir!"),
        ("Belge\nibraz edilir.", "Belge ibraz edilmez."),
        ("TEBLİĞ\nyayımlanır.", "TEBLIĞ yayımlanır."),
        ("Diğer chunk içeriği.", "Belge ibraz edilir."),
        ("Belgeibraz edilir.", "Belge ibraz edilir."),
        ("Belge" + " " * 1024 + "ibraz", "Belge ibraz"),
        ("Belge\nibraz", " "),
        ("Belge\nibraz", "z" * 1025),
    ],
)
def test_ambiguous_or_nonwhitespace_or_invalid_length_fails(
    text: str, quote: str
) -> None:
    with pytest.raises(ValueError):
        resolve_intake_labeling_response(
            _response(quote), text=text, taxonomy=_taxonomy()
        )


@pytest.mark.parametrize("change", ["unknown", "duplicate", "abstained", "schema"])
def test_full_original_validator_remains_authoritative(change: str) -> None:
    payload = json.loads(_response("Belge ibraz"))
    if change == "unknown":
        payload["labels"][0]["label_id"] = "missing"
    elif change == "duplicate":
        payload["labels"].append(payload["labels"][0].copy())
    elif change == "abstained":
        payload["abstained"] = True
    else:
        payload["labels"][0]["unsupported"] = True
    with pytest.raises(ValueError):
        resolve_intake_labeling_response(
            json.dumps(payload), text="Belge\nibraz", taxonomy=_taxonomy()
        )
