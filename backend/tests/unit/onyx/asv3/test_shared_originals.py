"""Recorded provision sharing must preserve source, temporal and delivery boundaries."""

import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.shared_originals import related_provision_originals
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc


def original(
    chunk: str,
    text: str,
    *,
    source: str = "authorized-statute",
    headings: list[str] | None = None,
    metadata: dict[str, JsonValue] | None = None,
    citable: bool = True,
) -> EvidenceItem:
    return EvidenceItem(
        source_id=source,
        chunk_id=chunk,
        text=text,
        metadata={
            "heading_path": headings or ["Statute", "MADDE 17"],
            "read_as_of_date": "2026-10-05",
            "version_unknown": True,
            **(metadata or {}),
        },
        search_doc=SearchDoc(
            document_id=source,
            chunk_ind=0,
            semantic_identifier="Statute",
            blurb=text,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={},
            match_highlights=[],
        )
        if citable
        else None,
    )


def recorded(items: list[EvidenceItem]) -> EvidenceLedger:
    ledger = EvidenceLedger()
    ledger.add(items, RunContext())
    return ledger


def full_record(ledger: EvidenceLedger, citation: int) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        json.loads(ledger.serialize_records([citation], required=[citation]))[0],
    )


def test_selected_rule_receives_already_read_exception_and_later_step_verbatim() -> (
    None
):
    items = [
        original("rule", "An application may be approved."),
        original(
            "exception",
            "Approval requires a signed certificate AND the authority's consent.",
            headings=["Statute", "MADDE 17", "(2) Conditions"],
        ),
        original(
            "later-step",
            "After approval the applicant must submit a completion declaration.",
            headings=["Statute", "MADDE 17", "(3) Settlement"],
        ),
        original("other-article", "Unrelated rule.", headings=["Statute", "MADDE 18"]),
        original("other-law", "A different statute's rule.", source="another-statute"),
    ]
    ledger = recorded(items)
    before = ledger.export()
    own_original = full_record(ledger, 1)
    shared = related_provision_originals(ledger, [own_original])
    assert [record["citation"] for record in shared] == [2, 3]
    assert [record["text"] for record in shared] == [items[1].text, items[2].text]
    assert all(record["truncated"] is False for record in shared)
    assert all(
        record["text_hash"] == items[i + 1].text_hash for i, record in enumerate(shared)
    )
    assert related_provision_originals(ledger, [own_original, *shared]) == []
    assert ledger.export() == before


@pytest.mark.parametrize(
    "changed",
    [
        {"read_as_of_date": "2025-01-01"},
        {"validity_start": "2025-01-01"},
        {"validity_end": "2025-12-31"},
        {"version_unknown": False},
        {"source_sha256": "another-source-revision"},
        {"publication_revision_id": "older-revision"},
        {"version": "older"},
    ],
)
def test_same_article_does_not_share_another_captured_version_or_date(
    changed: dict[str, JsonValue],
) -> None:
    ledger = recorded(
        [
            original("current", "Current rule."),
            original("old", "Old exception.", metadata=changed),
        ]
    )
    assert related_provision_originals(ledger, [full_record(ledger, 1)]) == []


@pytest.mark.parametrize(
    "headings",
    [
        ["Statute", "GEÇİCİ MADDE 17"],
        ["Statute", "MÜKERRER MADDE 17"],
        ["Statute", "MADDE 17/A"],
        ["Statute", "EK 2", "MADDE 17"],
        ["Statute", "MADDE 18", "MADDE 17 applies to this procedure"],
    ],
)
def test_same_number_does_not_cross_qualifier_suffix_annex_or_reference_scope(
    headings: list[str],
) -> None:
    ledger = recorded(
        [
            original("own", "Own rule."),
            original("unrelated", "Other rule.", headings=headings),
        ]
    )
    assert related_provision_originals(ledger, [full_record(ledger, 1)]) == []


@pytest.mark.parametrize(
    "changed",
    [
        {"derived": True},
        {"external": True},
        {"heading_path": []},
        {"truncated": True},
        {"canonical_metadata": {"truncated": True}},
    ],
)
def test_unproven_noncanonical_original_cannot_seed_or_join_a_bundle(
    changed: dict[str, JsonValue],
) -> None:
    ledger = recorded(
        [
            original("canonical", "Canonical rule."),
            original("unproven", "Derived output.", metadata=changed),
        ]
    )
    assert related_provision_originals(ledger, [full_record(ledger, 1)]) == []
    assert related_provision_originals(ledger, [], citation_numbers=[2]) == []


def test_non_citable_item_and_foreign_ledger_do_not_supply_originals() -> None:
    ledger = recorded(
        [
            original("canonical", "Canonical rule."),
            original("navigation", "Navigation only.", citable=False),
        ]
    )
    assert related_provision_originals(ledger, [full_record(ledger, 1)]) == []
    foreign = recorded(
        [
            original("canonical", "Changed foreign text."),
            original("foreign-exception", "Foreign exception."),
        ]
    )
    assert related_provision_originals(foreign, [full_record(ledger, 1)]) == []


@pytest.mark.parametrize(
    "changed",
    [
        {"citation": True},
        {"citation": 999},
        {"source_id": "another-statute"},
        {"chunk_id": "another-chunk"},
        {"text_hash": "wrong-hash"},
        {"text": "Fabricated rule."},
        {"start_char": True},
        {"end_char": 1},
        {"end_char": True},
        {"total_chars": 999},
        {"total_chars": True},
    ],
)
def test_changed_record_identity_or_text_cannot_select_sibling_originals(
    changed: dict[str, JsonValue],
) -> None:
    ledger = recorded(
        [
            original("rule", "Recorded rule."),
            original("exception", "Recorded exception."),
        ]
    )
    record = {**full_record(ledger, 1), **changed}
    assert related_provision_originals(ledger, [record]) == []


def test_partial_range_is_not_widened_and_explicit_citation_can_select_full_original() -> (
    None
):
    ledger = recorded(
        [
            original("rule", "Start. Selected clause. Original tail."),
            original("exception", "The exception remains in the original."),
        ]
    )
    own = full_record(ledger, 1)
    own.update({"text": "Selected clause.", "start_char": 7, "truncated": True})
    assert [
        record["citation"] for record in related_provision_originals(ledger, [own])
    ] == [2]
    shared = related_provision_originals(ledger, [own], citation_numbers=[1])
    assert [record["citation"] for record in shared] == [1, 2]
    assert shared[0]["text"] == full_record(ledger, 1)["text"]


def test_explicit_candidate_citation_selects_only_its_already_recorded_provision() -> (
    None
):
    ledger = recorded(
        [
            original("rule", "Recorded rule."),
            original("exception", "Recorded exception."),
            original("unrelated", "Other article.", headings=["Statute", "MADDE 23"]),
        ]
    )
    result = related_provision_originals(ledger, [], citation_numbers=[1, 1, 999, True])
    assert [record["citation"] for record in result] == [1, 2]
    assert related_provision_originals(ledger, []) == []


def test_metadata_inventory_is_detached_and_unrelated_text_is_never_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = recorded(
        [
            original("rule", "Recorded rule."),
            original("exception", "Recorded exception."),
            original(
                "unrelated",
                "An unrelated large text." * 500,
                headings=["Statute", "MADDE 23"],
            ),
        ]
    )
    own = full_record(ledger, 1)
    inventory = ledger.provision_metadata()
    assert all("text" not in record for record in inventory)
    metadata = inventory[0]["metadata"]
    assert isinstance(metadata, dict)
    metadata["heading_path"] = ["Invented heading"]
    assert ledger.provision_metadata()[0]["metadata"] != metadata
    copied: list[int] = []
    original_get = ledger.get

    def observe_get(number: int) -> EvidenceItem | None:
        copied.append(number)
        return original_get(number)

    monkeypatch.setattr(ledger, "get", observe_get)
    result = related_provision_originals(ledger, [own])
    assert [record["citation"] for record in result] == [2]
    assert copied == [1, 2]
