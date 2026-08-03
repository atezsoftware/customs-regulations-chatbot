from __future__ import annotations

import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from scripts.generate_basel_status_artifacts import (
    ANNEX_VII_ADOPTION_DATE,
    EU_MEMBERS,
    EXPECTED_ANNEX_MEMBER_COUNT,
    EXPECTED_STATE_COUNT,
    EXPRESS_ANNEX_VII_STATES,
    OECD_MEMBERS,
    AnnexClassification,
    AnnexRecord,
    ArtifactInputs,
    ArtifactOutputs,
    ArtifactTimestamps,
    StatusRecord,
    build_annex_records,
    build_status_records,
    render_annex_artifact,
    render_status_artifact,
    validate_annex_records,
    validate_artifact_paths,
    validate_status_records,
)

from onyx.regulatory.chunker import RegulatoryChunker


def _status_record(name: str, iso3: str, iso2: str = "ZZ") -> StatusRecord:
    return StatusRecord(
        name=name,
        iso2=iso2,
        iso3=iso3,
        untc_name=name,
        signature_date=None,
        convention_instrument_date=None,
        convention_action=None,
        convention_effective_date=None,
        convention_party=False,
        ban_consent=False,
        ban_instrument_date=None,
        ban_action=None,
        ban_effective_date=None,
    )


def _synthetic_state_universe() -> list[StatusRecord]:
    member_codes = sorted(
        set(OECD_MEMBERS) | set(EU_MEMBERS) | set(EXPRESS_ANNEX_VII_STATES)
    )
    records = [
        _status_record(
            f"Member state {iso3}",
            iso3,
            f"{chr(ord('A') + index // 26)}{chr(ord('A') + index % 26)}",
        )
        for index, iso3 in enumerate(member_codes)
    ]

    needed_nonmembers = EXPECTED_STATE_COUNT - len(records)
    for index in range(needed_nonmembers):
        identifier_index = len(member_codes) + index
        first = chr(ord("A") + identifier_index // 26)
        second = chr(ord("A") + identifier_index % 26)
        records.append(
            _status_record(
                f"Non-member state {index:03d}",
                f"Z{first}{second}",
                f"{first}{second}",
            )
        )
    return records


def test_annex_classification_is_materialized_for_every_declared_state() -> None:
    records = build_annex_records(_synthetic_state_universe())

    validate_annex_records(
        records,
        expected_state_count=EXPECTED_STATE_COUNT,
        expected_member_count=EXPECTED_ANNEX_MEMBER_COUNT,
    )
    assert len(records) == EXPECTED_STATE_COUNT
    assert (
        sum(record.classification == AnnexClassification.MEMBER for record in records)
        == EXPECTED_ANNEX_MEMBER_COUNT
    )
    assert (
        sum(
            record.classification == AnnexClassification.NOT_MEMBER
            for record in records
        )
        == EXPECTED_STATE_COUNT - EXPECTED_ANNEX_MEMBER_COUNT
    )


def test_annex_basis_and_qualifying_date_are_classification_safe() -> None:
    statuses = [
        _status_record("OECD state", "AAA", "AA"),
        _status_record("EU state", "BBB", "BB"),
        _status_record("Express state", "CCC", "CC"),
        _status_record("Outside state", "DDD", "DD"),
    ]
    records = build_annex_records(
        statuses,
        oecd_members={"AAA": date(1980, 1, 1)},
        eu_members={"BBB": date(2004, 5, 1)},
        express_states=frozenset({"CCC"}),
    )
    by_code = {record.iso3: record for record in records}

    assert by_code["AAA"].classification == AnnexClassification.MEMBER
    assert by_code["AAA"].qualifying_since == ANNEX_VII_ADOPTION_DATE
    assert by_code["BBB"].classification == AnnexClassification.MEMBER
    assert by_code["BBB"].qualifying_since == date(2004, 5, 1)
    assert by_code["CCC"].classification == AnnexClassification.MEMBER
    assert by_code["CCC"].qualifying_since == ANNEX_VII_ADOPTION_DATE

    outside = by_code["DDD"]
    assert outside.classification == AnnexClassification.NOT_MEMBER
    assert outside.oecd_date is None
    assert outside.eu_date is None
    assert outside.express_liechtenstein is False
    assert outside.qualifying_since is None


def test_negative_record_is_explicit_evidence_not_a_missing_result() -> None:
    record = AnnexRecord(
        name="Outside state",
        iso2="OS",
        iso3="OUT",
        classification=AnnexClassification.NOT_MEMBER,
        oecd_date=None,
        eu_date=None,
        express_liechtenstein=False,
        qualifying_since=None,
    )

    rendered = render_annex_artifact([record], "2026-08-03T00:00:00+03:00")

    assert "[COUNTRY:OUT]" in rendered
    assert "Current Annex VII classification: NOT MEMBER" in rendered
    assert "OECD member: NO" in rendered
    assert "EU member: NO" in rendered
    assert "expressly named Liechtenstein: NO" in rendered
    assert "timestamped derived set-complement" in rendered
    assert "not inferred from absence in search results" in rendered
    assert "Annex VII qualifying since:" not in rendered


def test_unknown_category_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the declared state universe"):
        build_annex_records(
            [_status_record("Known state", "AAA")],
            oecd_members={"ZZZ": date(2000, 1, 1)},
            eu_members={},
            express_states=frozenset(),
        )


def test_untc_column_drift_fails_closed() -> None:
    with pytest.raises(ValueError, match="4 columns; expected exactly 3"):
        build_status_records(
            [("State", "", "", "unexpected column")],
            [],
            {},
            {},
        )


def test_positive_status_records_require_complete_source_facts() -> None:
    complete = replace(
        _status_record("Complete state", "CMP"),
        convention_party=True,
        convention_instrument_date=date(2000, 1, 1),
        convention_action="ratification",
        convention_effective_date=date(2000, 4, 1),
        ban_consent=True,
        ban_instrument_date=date(2020, 1, 1),
        ban_action="acceptance",
        ban_effective_date=date(2020, 4, 1),
    )

    validate_status_records([complete], expected_state_count=1)

    with pytest.raises(ValueError, match="Convention party CMP.*incomplete"):
        validate_status_records([replace(complete, convention_effective_date=None)])
    with pytest.raises(ValueError, match="Ban participant CMP.*incomplete"):
        validate_status_records([replace(complete, ban_action=None)])


def test_status_records_reject_contradictory_facts_and_identifiers() -> None:
    nonparty = _status_record("Non-party", "NON", "NP")
    with pytest.raises(ValueError, match="Convention non-party NON.*contradictory"):
        validate_status_records(
            [replace(nonparty, convention_action="accession")],
            expected_state_count=1,
        )
    with pytest.raises(ValueError, match="Ban non-participant NON.*contradictory"):
        validate_status_records(
            [replace(nonparty, ban_instrument_date=date(2020, 1, 1))],
            expected_state_count=1,
        )
    with pytest.raises(ValueError, match="ISO-3 identifiers must be three"):
        validate_status_records([replace(nonparty, iso3="bad")])


def test_status_date_order_allows_retroactive_state_succession() -> None:
    successor = replace(
        _status_record("Successor state", "SUC", "SU"),
        convention_party=True,
        convention_instrument_date=date(2006, 10, 23),
        convention_action="succession",
        convention_effective_date=date(2006, 6, 3),
    )

    validate_status_records([successor], expected_state_count=1)
    with pytest.raises(ValueError, match="effective date precedes deposit"):
        validate_status_records(
            [replace(successor, convention_action="accession")],
            expected_state_count=1,
        )


def test_artifact_paths_must_not_alias_inputs_or_each_other(tmp_path: Path) -> None:
    inputs = ArtifactInputs(
        convention_xml=tmp_path / "convention.xml",
        ban_xml=tmp_path / "ban.xml",
        convention_json=tmp_path / "convention.json",
        ban_json=tmp_path / "ban.json",
    )

    with pytest.raises(ValueError, match="status output.*Convention XML input"):
        validate_artifact_paths(
            inputs,
            ArtifactOutputs(
                status=inputs.convention_xml,
                annex=tmp_path / "annex.md",
            ),
        )
    with pytest.raises(ValueError, match="Annex output.*status output"):
        validate_artifact_paths(
            inputs,
            ArtifactOutputs(
                status=tmp_path / "same.md",
                annex=tmp_path / "same.md",
            ),
        )


def test_artifact_timestamps_are_ordered_timezone_aware_iso_values() -> None:
    ArtifactTimestamps(
        retrieved_at="2026-08-03T00:00:00+03:00",
        status_as_at="2026-08-02T17:00:00-04:00 (EDT)",
    )

    with pytest.raises(ValueError, match="status_as_at must be an ISO 8601"):
        ArtifactTimestamps(
            retrieved_at="2026-08-03T00:00:00+03:00",
            status_as_at="not-a-timestamp",
        )
    with pytest.raises(ValueError, match="retrieved_at must include a UTC offset"):
        ArtifactTimestamps(
            retrieved_at="2026-08-03T00:00:00",
            status_as_at="2026-08-02T17:00:00-04:00",
        )
    with pytest.raises(ValueError, match="cannot precede"):
        ArtifactTimestamps(
            retrieved_at="2026-08-02T16:59:59-04:00",
            status_as_at="2026-08-02T17:00:00-04:00",
        )


def test_annex_artifact_keeps_one_bounded_chunk_per_country() -> None:
    records = build_annex_records(_synthetic_state_universe())
    artifact = render_annex_artifact(records, "2026-08-03T00:00:00+03:00")

    chunked = RegulatoryChunker().chunk_text(
        artifact,
        source_path="basel-annex-vii.md",
        source_file="basel-annex-vii.md",
        parser="markdown",
    )
    marker_pattern = re.compile(r"\[COUNTRY:([A-Z]{3})\]")
    marker_chunks = [
        marker_pattern.findall(chunk.text)
        for chunk in chunked.chunks
        if marker_pattern.search(chunk.text)
    ]

    assert len(marker_chunks) == EXPECTED_STATE_COUNT
    assert all(len(markers) == 1 for markers in marker_chunks)
    assert len({markers[0] for markers in marker_chunks}) == EXPECTED_STATE_COUNT
    assert (
        max(
            len(chunk.text)
            for chunk in chunked.chunks
            if marker_pattern.search(chunk.text)
        )
        < 2400
    )
    assert chunked.warnings == []


def test_status_artifact_keeps_one_bounded_chunk_per_country() -> None:
    records = _synthetic_state_universe()
    organization = replace(
        _status_record("European Union", "EUE", "EU"),
        convention_party=True,
        convention_instrument_date=date(1994, 2, 7),
        convention_action="approval",
        convention_effective_date=date(1994, 5, 8),
        ban_consent=True,
        ban_instrument_date=date(1997, 9, 30),
        ban_action="approval",
        ban_effective_date=date(2019, 12, 5),
    )
    timestamps = ArtifactTimestamps(
        retrieved_at="2026-08-03T00:00:00+03:00",
        status_as_at="2026-08-02T17:00:00-04:00 (EDT)",
    )
    artifact = render_status_artifact(records, organization, timestamps)

    chunked = RegulatoryChunker().chunk_text(
        artifact,
        source_path="basel-status.md",
        source_file="basel-status.md",
        parser="markdown",
    )
    marker_pattern = re.compile(r"\[COUNTRY:([A-Z]{3})\]")
    marker_chunks = [
        marker_pattern.findall(chunk.text)
        for chunk in chunked.chunks
        if marker_pattern.search(chunk.text)
    ]

    assert len(marker_chunks) == EXPECTED_STATE_COUNT
    assert all(len(markers) == 1 for markers in marker_chunks)
    assert len({markers[0] for markers in marker_chunks}) == EXPECTED_STATE_COUNT
    assert (
        max(
            len(chunk.text)
            for chunk in chunked.chunks
            if marker_pattern.search(chunk.text)
        )
        < 2400
    )
    assert chunked.warnings == []
