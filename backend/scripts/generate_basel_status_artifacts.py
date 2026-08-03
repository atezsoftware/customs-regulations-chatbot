"""Generate reproducible Basel country-status reference artifacts.

The command consumes frozen UN Treaty Collection and Basel Secretariat source
snapshots. It does not fetch the network. Both outputs are timestamped Markdown
files whose country headings are intentionally atomic for ``RegulatoryChunker``.

Example:

    python backend/scripts/generate_basel_status_artifacts.py \
        --convention-xml /tmp/untc-convention.xml \
        --ban-xml /tmp/untc-ban.xml \
        --convention-json /tmp/basel-convention.json \
        --ban-json /tmp/basel-ban.json \
        --status-output /tmp/basel-status.md \
        --annex-output /tmp/basel-annex-vii.md \
        --retrieved-at 2026-08-02T05:23:39+03:00 \
        --status-as-at "2026-08-01T11:15:30-04:00 (EDT)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import cast
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as etree

EXPECTED_STATE_COUNT = 195
EXPECTED_ANNEX_MEMBER_COUNT = 44
ANNEX_VII_ADOPTION_DATE = date(1995, 9, 22)

UNTC_CONVENTION_URL = (
    "https://treaties.un.org/pages/ViewDetails.aspx?chapter=27&clang=_en&"
    "mtdsg_no=XXVII-3&src=TREATY"
)
UNTC_CONVENTION_XML_URL = (
    "https://treaties.un.org/doc/Publication/MTDSG/Volume%20II/"
    "Chapter%20XXVII/XXVII-3.en.xml"
)
UNTC_BAN_URL = (
    "https://treaties.un.org/pages/ViewDetails.aspx?chapter=27&clang=_en&"
    "mtdsg_no=XXVII-3-a&src=TREATY"
)
UNTC_BAN_XML_URL = (
    "https://treaties.un.org/doc/Publication/MTDSG/Volume%20II/"
    "Chapter%20XXVII/XXVII-3-a.en.xml"
)
BASEL_CONVENTION_STATUS_URL = (
    "https://www.basel.int/Countries/StatusofRatifications/"
    "PartiesSignatories/tabid/4499/Default.aspx"
)
BASEL_BAN_STATUS_URL = (
    "https://www.basel.int/Countries/StatusofRatifications/"
    "BanAmendment/tabid/1344/Default.aspx"
)
BASEL_TEXT_URL = (
    "https://www.basel.int/Portals/4/download.aspx?"
    "e=UNEP-CHW-IMPL-CONVTEXT-2025.English.pdf"
)
BASEL_FAQ_URL = (
    "https://www.basel.int/Implementation/LegalMatters/BanAmendment/"
    "QuestionsandAnswers/tabid/3596/ItemId/2952/Default.aspx"
)
UN_MEMBER_URL = "https://www.un.org/en/about-us/member-states"
UN_M49_URL = "https://unstats.un.org/unsd/methodology/m49/"
UN_TERMS_URL = "https://www.un.org/en/about-us/terms-of-use"
OECD_URL = "https://www.oecd.org/en/about/legal.html"
OECD_TERMS_URL = "https://www.oecd.org/en/about/terms-conditions.html"
COUNCIL_URL = (
    "https://www.consilium.europa.eu/en/policies/how-enlargement-works/"
    "timeline-accession-eu-member-states/"
)
COUNCIL_COPYRIGHT_URL = "https://www.consilium.europa.eu/en/about-site/copyright/"


def _dates(values: Mapping[str, str]) -> dict[str, date]:
    return {iso3: date.fromisoformat(value) for iso3, value in values.items()}


# Keys are ISO-3 rather than display names so a source-side rename cannot alter
# a country's classification. Dates are the official category-membership dates.
OECD_MEMBERS: Mapping[str, date] = _dates(
    {
        "AUS": "1971-06-07",
        "AUT": "1961-09-30",
        "BEL": "1961-09-30",
        "CAN": "1961-09-30",
        "CHL": "2010-05-07",
        "COL": "2020-04-28",
        "CRI": "2021-05-25",
        "CZE": "1995-12-21",
        "DNK": "1961-09-30",
        "EST": "2010-12-09",
        "FIN": "1969-01-28",
        "FRA": "1961-09-30",
        "DEU": "1961-09-30",
        "GRC": "1961-09-30",
        "HUN": "1996-05-07",
        "ISL": "1961-09-30",
        "IRL": "1961-09-30",
        "ISR": "2010-09-07",
        "ITA": "1962-03-29",
        "JPN": "1964-04-28",
        "KOR": "1996-12-12",
        "LVA": "2016-07-01",
        "LTU": "2018-07-05",
        "LUX": "1961-12-07",
        "MEX": "1994-05-18",
        "NLD": "1961-11-13",
        "NZL": "1973-05-29",
        "NOR": "1961-09-30",
        "POL": "1996-11-22",
        "PRT": "1961-09-30",
        "SVK": "2000-12-14",
        "SVN": "2010-07-21",
        "ESP": "1961-09-30",
        "SWE": "1961-09-30",
        "CHE": "1961-09-30",
        "TUR": "1961-09-30",
        "GBR": "1961-09-30",
        "USA": "1961-09-30",
    }
)

EU_MEMBERS: Mapping[str, date] = _dates(
    {
        "AUT": "1995-01-01",
        "BEL": "1958-01-01",
        "BGR": "2007-01-01",
        "HRV": "2013-07-01",
        "CYP": "2004-05-01",
        "CZE": "2004-05-01",
        "DNK": "1973-01-01",
        "EST": "2004-05-01",
        "FIN": "1995-01-01",
        "FRA": "1958-01-01",
        "DEU": "1958-01-01",
        "GRC": "1981-01-01",
        "HUN": "2004-05-01",
        "IRL": "1973-01-01",
        "ITA": "1958-01-01",
        "LVA": "2004-05-01",
        "LTU": "2004-05-01",
        "LUX": "1958-01-01",
        "MLT": "2004-05-01",
        "NLD": "1958-01-01",
        "POL": "2004-05-01",
        "PRT": "1986-01-01",
        "ROU": "2007-01-01",
        "SVK": "2004-05-01",
        "SVN": "2004-05-01",
        "ESP": "1986-01-01",
        "SWE": "1995-01-01",
    }
)

EXPRESS_ANNEX_VII_STATES = frozenset({"LIE"})

# These current UN members were absent from the frozen UNTC participant table.
# ISO identifiers come from UN M49. A future source refresh may make an entry
# redundant; build_status_records adds it only while the state remains absent.
ABSENT_UN_MEMBERS: Mapping[str, tuple[str, str]] = {
    "Fiji": ("FJ", "FJI"),
    "South Sudan": ("SS", "SSD"),
    "Timor-Leste": ("TL", "TLS"),
}


@dataclass(frozen=True)
class ArtifactInputs:
    convention_xml: Path
    ban_xml: Path
    convention_json: Path
    ban_json: Path


@dataclass(frozen=True)
class ArtifactOutputs:
    status: Path
    annex: Path


@dataclass(frozen=True)
class ArtifactTimestamps:
    retrieved_at: str
    status_as_at: str

    def __post_init__(self) -> None:
        if not self.retrieved_at.strip() or not self.status_as_at.strip():
            raise ValueError("Artifact timestamps cannot be blank.")
        retrieved_at = _aware_iso_timestamp(self.retrieved_at, "retrieved_at")
        status_timestamp = re.sub(r"\s+\([^()]+\)\s*$", "", self.status_as_at).strip()
        status_as_at = _aware_iso_timestamp(status_timestamp, "status_as_at")
        if retrieved_at < status_as_at:
            raise ValueError("retrieved_at cannot precede status_as_at.")


def _aware_iso_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO 8601 timestamp.") from error
    if parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC offset.")
    return parsed


@dataclass(frozen=True)
class BaselServiceRecord:
    name: str
    iso2: str
    iso3: str
    effective_date: date | None


@dataclass(frozen=True)
class StatusRecord:
    name: str
    iso2: str
    iso3: str
    untc_name: str | None
    signature_date: date | None
    convention_instrument_date: date | None
    convention_action: str | None
    convention_effective_date: date | None
    convention_party: bool
    ban_consent: bool
    ban_instrument_date: date | None
    ban_action: str | None
    ban_effective_date: date | None


class AnnexClassification(str, Enum):
    MEMBER = "MEMBER"
    NOT_MEMBER = "NOT MEMBER"


@dataclass(frozen=True)
class AnnexRecord:
    name: str
    iso2: str
    iso3: str
    classification: AnnexClassification
    oecd_date: date | None
    eu_date: date | None
    express_liechtenstein: bool
    qualifying_since: date | None


UntcRow = tuple[str, ...]


def _fold_name(value: str) -> str:
    value = re.sub(r"<superscript>.*?</superscript>", "", value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.casefold().replace("st.", "saint")
    value = value.replace("czech republic", "czechia")
    value = value.replace("naoero", "nauru")
    value = re.sub(r"\(the\)", "", value)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _entry_text(element: Element) -> str:
    value = "".join(element.itertext()).strip()
    value = re.sub(r"<superscript>.*?</superscript>", "", value)
    return " ".join(value.split())


def parse_untc_rows(path: Path) -> list[UntcRow]:
    document = etree.parse(path)
    rows: list[UntcRow] = []
    for row in document.findall(".//Participants//Tbody//Row"):
        rows.append(tuple(_entry_text(entry) for entry in row.findall("./Entry")))
    if not rows:
        raise ValueError(f"No UNTC participant rows found in {path}.")
    return rows


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Basel service row has no non-empty {key!r} value.")
    return value


def _optional_brs_date(value: object) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Basel service effective date must be a string or null.")
    return datetime.strptime(value, "%d/%m/%Y").date()


def load_basel_service_records(path: Path) -> dict[str, BaselServiceRecord]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Basel service snapshot {path} is not a JSON object.")
    raw_values = payload.get("value")
    if not isinstance(raw_values, list):
        raise ValueError(f"Basel service snapshot {path} has no value list.")

    records: dict[str, BaselServiceRecord] = {}
    for raw_row in raw_values:
        if not isinstance(raw_row, dict):
            raise ValueError(
                f"Basel service snapshot {path} contains a non-object row."
            )
        row = cast(Mapping[str, object], raw_row)
        record = BaselServiceRecord(
            name=_required_string(row, "name"),
            iso2=_required_string(row, "brs_iso2a"),
            iso3=_required_string(row, "brs_iso3a"),
            effective_date=_optional_brs_date(row.get("brs_entryintoforce")),
        )
        key = _fold_name(record.name)
        if key in records:
            raise ValueError(f"Duplicate Basel service participant {record.name!r}.")
        records[key] = record
    return records


def _untc_date(value: str) -> date | None:
    match = re.search(r"(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})", value)
    if match is None:
        return None
    return datetime.strptime(" ".join(match.groups()), "%d %b %Y").date()


def _untc_action(value: str, *, ban: bool = False) -> str | None:
    if _untc_date(value) is None:
        return None
    code_match = re.search(r"\d{4}\s*(AA|A|a|c|d)?\s*$", value)
    code = code_match.group(1) if code_match else None
    labels: Mapping[str | None, str] = {
        "AA": "approval",
        "A": "acceptance",
        "a": "accession",
        "c": "formal confirmation",
        "d": "succession",
        None: "ratification",
    }
    if ban and code == "a":
        raise ValueError("Ban Amendment table unexpectedly contains accession.")
    return labels[code]


def _rows_by_name(
    rows: Sequence[UntcRow], *, expected_columns: int
) -> dict[str, UntcRow]:
    result: dict[str, UntcRow] = {}
    for row in rows:
        if len(row) != expected_columns:
            raise ValueError(
                f"UNTC participant row has {len(row)} columns; expected exactly "
                f"{expected_columns}."
            )
        key = _fold_name(row[0])
        if key in result:
            raise ValueError(f"Duplicate UNTC participant {row[0]!r}.")
        result[key] = row
    return result


def build_status_records(
    convention_rows: Sequence[UntcRow],
    ban_rows: Sequence[UntcRow],
    convention_service: Mapping[str, BaselServiceRecord],
    ban_service: Mapping[str, BaselServiceRecord],
) -> tuple[list[StatusRecord], StatusRecord]:
    convention_by_name = _rows_by_name(convention_rows, expected_columns=3)
    ban_by_name = _rows_by_name(ban_rows, expected_columns=2)
    if set(convention_by_name) != set(convention_service):
        raise ValueError("UNTC and Basel Convention participant sets do not match.")
    if set(ban_by_name) != set(ban_service):
        raise ValueError("UNTC and Basel Ban participant sets do not match.")
    if not set(ban_by_name).issubset(convention_by_name):
        raise ValueError("A Ban participant is absent from the Convention snapshot.")

    records: list[StatusRecord] = []
    organization: StatusRecord | None = None
    for key, service_record in convention_service.items():
        convention_row = convention_by_name[key]
        signature_date = _untc_date(convention_row[1])
        convention_instrument_date = _untc_date(convention_row[2])
        ban_row = ban_by_name.get(key)
        ban_service_record = ban_service.get(key)
        if (ban_row is None) != (ban_service_record is None):
            raise ValueError(
                f"Ban source mismatch for participant {service_record.name!r}."
            )

        record = StatusRecord(
            name=service_record.name,
            iso2=service_record.iso2,
            iso3=service_record.iso3,
            untc_name=convention_row[0],
            signature_date=signature_date,
            convention_instrument_date=convention_instrument_date,
            convention_action=_untc_action(convention_row[2]),
            convention_effective_date=service_record.effective_date,
            convention_party=convention_instrument_date is not None,
            ban_consent=ban_row is not None,
            ban_instrument_date=_untc_date(ban_row[1]) if ban_row else None,
            ban_action=_untc_action(ban_row[1], ban=True) if ban_row else None,
            ban_effective_date=(
                ban_service_record.effective_date if ban_service_record else None
            ),
        )
        if record.name == "European Union":
            organization = record
        else:
            records.append(record)

    present_names = {_fold_name(record.name) for record in records}
    for name, (iso2, iso3) in ABSENT_UN_MEMBERS.items():
        if _fold_name(name) in present_names:
            continue
        records.append(
            StatusRecord(
                name=name,
                iso2=iso2,
                iso3=iso3,
                untc_name=None,
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
        )

    if organization is None:
        raise ValueError("European Union organization record is missing.")
    _require_valid_unique_identifiers(records)
    return sorted(records, key=lambda record: record.name), organization


def _require_valid_unique_identifiers(
    records: Sequence[StatusRecord | AnnexRecord],
) -> None:
    iso2_codes = [record.iso2 for record in records]
    iso3_codes = [record.iso3 for record in records]
    if any(re.fullmatch(r"[A-Z]{2}", code) is None for code in iso2_codes):
        raise ValueError("Country ISO-2 identifiers must be two uppercase letters.")
    if any(re.fullmatch(r"[A-Z]{3}", code) is None for code in iso3_codes):
        raise ValueError("Country ISO-3 identifiers must be three uppercase letters.")
    if len(iso2_codes) != len(set(iso2_codes)):
        raise ValueError("Country ISO-2 identifiers must be unique.")
    if len(iso3_codes) != len(set(iso3_codes)):
        raise ValueError("Country ISO-3 identifiers must be unique.")


def validate_status_records(
    records: Sequence[StatusRecord],
    *,
    organization: StatusRecord | None = None,
    expected_state_count: int | None = None,
) -> None:
    participants = [*records]
    if organization is not None:
        participants.append(organization)
    _require_valid_unique_identifiers(participants)
    if expected_state_count is not None and len(records) != expected_state_count:
        raise ValueError(
            f"Expected {expected_state_count} state status records; got {len(records)}."
        )

    for record in participants:
        if record.convention_party and any(
            value is None
            for value in (
                record.convention_instrument_date,
                record.convention_action,
                record.convention_effective_date,
            )
        ):
            raise ValueError(
                f"Convention party {record.iso3} has incomplete positive source facts."
            )
        if not record.convention_party and any(
            value is not None
            for value in (
                record.convention_instrument_date,
                record.convention_action,
                record.convention_effective_date,
            )
        ):
            raise ValueError(
                f"Convention non-party {record.iso3} has contradictory positive "
                "source facts."
            )
        if record.ban_consent and any(
            value is None
            for value in (
                record.ban_instrument_date,
                record.ban_action,
                record.ban_effective_date,
            )
        ):
            raise ValueError(
                f"Ban participant {record.iso3} has incomplete positive source facts."
            )
        if record.ban_consent and not record.convention_party:
            raise ValueError(
                f"Ban participant {record.iso3} is not a Convention party."
            )
        if not record.ban_consent and any(
            value is not None
            for value in (
                record.ban_instrument_date,
                record.ban_action,
                record.ban_effective_date,
            )
        ):
            raise ValueError(
                f"Ban non-participant {record.iso3} has contradictory positive "
                "source facts."
            )
        if (
            record.signature_date is not None
            and record.convention_instrument_date is not None
            and record.signature_date > record.convention_instrument_date
        ):
            raise ValueError(
                f"Convention signature follows instrument deposit for {record.iso3}."
            )
        if (
            record.convention_instrument_date is not None
            and record.convention_effective_date is not None
            and record.convention_instrument_date > record.convention_effective_date
            and record.convention_action != "succession"
        ):
            raise ValueError(
                f"Convention effective date precedes deposit for {record.iso3}."
            )
        if (
            record.ban_instrument_date is not None
            and record.ban_effective_date is not None
            and record.ban_instrument_date > record.ban_effective_date
            and record.ban_action != "succession"
        ):
            raise ValueError(f"Ban effective date precedes deposit for {record.iso3}.")


def build_annex_records(
    status_records: Sequence[StatusRecord],
    *,
    oecd_members: Mapping[str, date] = OECD_MEMBERS,
    eu_members: Mapping[str, date] = EU_MEMBERS,
    express_states: frozenset[str] = EXPRESS_ANNEX_VII_STATES,
) -> list[AnnexRecord]:
    _require_valid_unique_identifiers(status_records)
    status_codes = {record.iso3 for record in status_records}
    category_codes = set(oecd_members) | set(eu_members) | set(express_states)
    unknown_codes = sorted(category_codes - status_codes)
    if unknown_codes:
        raise ValueError(
            "Annex VII category contains ISO-3 identifiers outside the declared "
            f"state universe: {', '.join(unknown_codes)}."
        )

    records: list[AnnexRecord] = []
    for status_record in status_records:
        iso3 = status_record.iso3
        oecd_date = oecd_members.get(iso3)
        eu_date = eu_members.get(iso3)
        express_liechtenstein = iso3 in express_states
        membership_starts = [
            value for value in (oecd_date, eu_date) if value is not None
        ]
        if express_liechtenstein:
            membership_starts.append(ANNEX_VII_ADOPTION_DATE)

        is_member = bool(membership_starts)
        qualifying_since = (
            max(ANNEX_VII_ADOPTION_DATE, min(membership_starts))
            if membership_starts
            else None
        )
        records.append(
            AnnexRecord(
                name=status_record.name,
                iso2=status_record.iso2,
                iso3=iso3,
                classification=(
                    AnnexClassification.MEMBER
                    if is_member
                    else AnnexClassification.NOT_MEMBER
                ),
                oecd_date=oecd_date,
                eu_date=eu_date,
                express_liechtenstein=express_liechtenstein,
                qualifying_since=qualifying_since,
            )
        )

    _require_valid_unique_identifiers(records)
    return sorted(records, key=lambda record: record.name)


def validate_annex_records(
    records: Sequence[AnnexRecord],
    *,
    expected_state_count: int | None = None,
    expected_member_count: int | None = None,
) -> None:
    _require_valid_unique_identifiers(records)
    if expected_state_count is not None and len(records) != expected_state_count:
        raise ValueError(
            f"Expected {expected_state_count} Annex VII state records; got {len(records)}."
        )

    member_count = 0
    for record in records:
        has_basis = bool(
            record.oecd_date or record.eu_date or record.express_liechtenstein
        )
        if record.classification == AnnexClassification.MEMBER:
            member_count += 1
            if not has_basis or record.qualifying_since is None:
                raise ValueError(f"Annex VII member {record.iso3} has no basis/date.")
        elif has_basis or record.qualifying_since is not None:
            raise ValueError(
                f"Annex VII non-member {record.iso3} has a positive basis/date."
            )

    if expected_member_count is not None and member_count != expected_member_count:
        raise ValueError(
            f"Expected {expected_member_count} Annex VII members; got {member_count}."
        )


def _md_link(label: str, url: str) -> str:
    return f"[{label}]({url})"


def _render_status_record(record: StatusRecord, timestamps: ArtifactTimestamps) -> str:
    if record.untc_name is None:
        normalization = (
            "UNTC participant label: none (the state is absent from the current "
            "participant table)"
        )
    elif record.untc_name == record.name:
        normalization = f"UNTC participant label: {record.name}"
    else:
        normalization = f"UNTC participant label: {record.untc_name}"

    signature = (
        record.signature_date.isoformat() if record.signature_date else "none recorded"
    )
    if record.convention_party:
        convention = (
            f"PARTY; {record.convention_action} instrument deposited "
            f"{record.convention_instrument_date}; effective "
            f"{record.convention_effective_date}"
        )
    elif record.signature_date:
        convention = (
            "SIGNATORY ONLY; no consent-to-be-bound instrument or effective date"
        )
    else:
        convention = (
            "NEITHER PARTY NOR SIGNATORY; timestamped negative classification "
            "from the declared UNTC/UN state universe"
        )

    if record.ban_consent:
        ban = (
            f"CONSENT DEPOSITED / BOUND; {record.ban_action} instrument deposited "
            f"{record.ban_instrument_date}; effective {record.ban_effective_date}"
        )
    else:
        ban = (
            "NO CONSENT INSTRUMENT RECORDED / NOT BOUND at the stated status "
            "timestamp; no instrument or effective date"
        )

    extra_source = (
        f"; {_md_link('UN member universe', UN_MEMBER_URL)} and "
        f"{_md_link('UN M49 normalization', UN_M49_URL)}"
        if record.untc_name is None
        else ""
    )
    return (
        f"[COUNTRY:{record.iso3}] **{record.name}** — normalized identifiers: "
        f"ISO 3166-1 alpha-2 `{record.iso2}`, alpha-3 `{record.iso3}`; "
        f"{normalization}. Convention signature date: {signature}. Basel "
        f"Convention status: {convention}. Ban Amendment status: {ban}. Status "
        f"timestamp: {timestamps.status_as_at}; retrieved: "
        f"{timestamps.retrieved_at}. Sources: "
        f"{_md_link('UNTC XXVII-3', UNTC_CONVENTION_URL)}; "
        f"{_md_link('UNTC XXVII-3-a', UNTC_BAN_URL)}; effective-date and ISO "
        f"enrichment cross-checked against the "
        f"{_md_link('Basel Convention status interface', BASEL_CONVENTION_STATUS_URL)} "
        f"and {_md_link('Ban status interface', BASEL_BAN_STATUS_URL)}"
        f"{extra_source}."
    )


def render_status_artifact(
    records: Sequence[StatusRecord],
    organization: StatusRecord,
    timestamps: ArtifactTimestamps,
) -> str:
    lines = [
        "# Basel Convention and Ban Amendment: current state status",
        "",
        f"Retrieved at: {timestamps.retrieved_at}",
        "",
        f"UN Treaty Collection status as at: {timestamps.status_as_at}",
        "",
        "## Scope and interpretation",
        "",
        (
            "This artifact contains one atomic record for each state in its "
            "declared 195-state universe: the 193 current UN member states, plus "
            "the Cook Islands and the State of Palestine because they occur as "
            "state participants in UNTC XXVII-3. States absent from the UNTC "
            "participant table remain explicit timestamped negative records. The "
            "European Union is reported separately as a regional economic "
            "integration organization, not counted as a state."
        ),
        "",
        "## Sources and date semantics",
        "",
        (
            f"Authority of record: {_md_link('UNTC XXVII-3 live status', UNTC_CONVENTION_URL)} "
            f"and {_md_link('official XML', UNTC_CONVENTION_XML_URL)} for the "
            f"Convention; {_md_link('UNTC XXVII-3-a live status', UNTC_BAN_URL)} "
            f"and {_md_link('official XML', UNTC_BAN_XML_URL)} for the Ban "
            "Amendment. Per-state effective dates and ISO identifiers are "
            "cross-checked against the public "
            f"{_md_link('Basel Convention status interface', BASEL_CONVENTION_STATUS_URL)} "
            f"and {_md_link('Ban interface', BASEL_BAN_STATUS_URL)}."
        ),
        "",
        "## Normalization",
        "",
        (
            "Canonical display names and ISO-2/ISO-3 keys use the Basel "
            "Secretariat status service, with the original UNTC participant label "
            "preserved in every record. Join country records by ISO-3, not by "
            "display-name spelling."
        ),
        "",
        "## Licensing, attribution, and inference",
        "",
        (
            "This artifact reproduces no source page wholesale; it contains "
            "attributed factual extractions and clearly labeled negative "
            "classifications. UNTC reuse remains subject to the "
            f"{_md_link('United Nations terms of use', UN_TERMS_URL)}. The "
            f"{_md_link('consolidated Basel text', BASEL_TEXT_URL)} remains the "
            "legal source and live status data must be refreshed before a "
            "time-sensitive decision."
        ),
        "",
        "# State records",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## Country — {record.name} ({record.iso2} / {record.iso3})",
                "",
                _render_status_record(record, timestamps),
                "",
            ]
        )

    lines.extend(
        [
            "# Non-state participant",
            "",
            "## Organization — European Union (EU / EUE)",
            "",
            (
                "[ORGANIZATION:EUE] **European Union** — entity type: regional "
                "economic integration organization; `EU` / `EUE` are Basel "
                "Secretariat service identifiers, not ISO state codes. Basel "
                f"Convention: PARTY; {organization.convention_action} instrument "
                f"deposited {organization.convention_instrument_date}; effective "
                f"{organization.convention_effective_date}. Ban Amendment: CONSENT "
                f"DEPOSITED / BOUND; {organization.ban_action} instrument deposited "
                f"{organization.ban_instrument_date}; effective "
                f"{organization.ban_effective_date}. Status timestamp: "
                f"{timestamps.status_as_at}; retrieved: {timestamps.retrieved_at}. "
                f"Sources: {_md_link('UNTC XXVII-3', UNTC_CONVENTION_URL)}; "
                f"{_md_link('UNTC XXVII-3-a', UNTC_BAN_URL)}; "
                f"{_md_link('Basel status interface', BASEL_CONVENTION_STATUS_URL)}."
            ),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_annex_record(record: AnnexRecord, retrieved_at: str) -> str:
    common = (
        f"[COUNTRY:{record.iso3}] **{record.name}** — normalized identifiers: "
        f"ISO 3166-1 alpha-2 `{record.iso2}`, alpha-3 `{record.iso3}`. Current "
        f"Annex VII classification: {record.classification.value}"
    )
    source_links = [_md_link("Basel Annex VII rule", BASEL_FAQ_URL)]
    if record.classification == AnnexClassification.NOT_MEMBER:
        source_links.extend(
            [
                _md_link("OECD member list", OECD_URL),
                _md_link("Council EU accession timeline", COUNCIL_URL),
            ]
        )
        return (
            f"{common} (timestamped derived set-complement). Current qualifying-"
            "basis checks: OECD member: NO; EU member: NO; expressly named "
            "Liechtenstein: NO. This atomic negative classification materializes "
            "the result of checking the state against the complete declared "
            "category sets at the snapshot timestamp; it is not inferred from "
            f"absence in search results. Retrieved: {retrieved_at}. Sources: "
            f"{'; '.join(source_links)}."
        )

    bases: list[str] = []
    if record.oecd_date:
        bases.append(
            f"OECD member; OECD Convention effective {record.oecd_date.isoformat()}"
        )
        source_links.append(_md_link("OECD member list", OECD_URL))
    if record.eu_date:
        bases.append(f"EU member; accession {record.eu_date.isoformat()}")
        source_links.append(_md_link("Council EU accession timeline", COUNCIL_URL))
    if record.express_liechtenstein:
        bases.append("Liechtenstein, named expressly in Annex VII")
    if record.qualifying_since is None:
        raise ValueError(f"Annex VII member {record.iso3} has no qualifying date.")
    return (
        f"{common} (derived union classification). Current qualifying basis: "
        f"{' + '.join(bases)}. Annex VII qualifying since: "
        f"{record.qualifying_since.isoformat()} (derived as the later of Annex VII "
        "adoption on 1995-09-22 and the earliest qualifying current-category "
        "membership; this is not an Article 4A treaty-effective date). Retrieved: "
        f"{retrieved_at}. Sources: {'; '.join(source_links)}."
    )


def render_annex_artifact(records: Sequence[AnnexRecord], retrieved_at: str) -> str:
    member_count = sum(
        record.classification == AnnexClassification.MEMBER for record in records
    )
    nonmember_count = len(records) - member_count
    overlap_count = sum(
        record.oecd_date is not None and record.eu_date is not None
        for record in records
    )
    lines = [
        "# Basel Convention Annex VII: current exhaustive state classification",
        "",
        f"Retrieved at: {retrieved_at}",
        "",
        "## Rule and scope",
        "",
        (
            "Annex VII is category-based, not a frozen country schedule: it covers "
            "Parties and other States that are members of the OECD or the European "
            "Community, plus Liechtenstein. The Basel Secretariat confirms that "
            "the European Union is the EC's legal successor. This artifact computes "
            "the current category as `OECD members ∪ EU members ∪ {Liechtenstein}` "
            f"and materializes one atomic classification for all {len(records)} "
            f"states in the declared universe: {member_count} MEMBER and "
            f"{nonmember_count} NOT MEMBER records. The negative records are "
            "timestamped derived set-complements, not conclusions drawn from a "
            "missing search result. Annex VII classification remains independent "
            "of Convention party status and Ban Amendment consent; join the "
            "companion status artifact by ISO-3 when those axes are needed."
        ),
        "",
        "## Official sources and update dates",
        "",
        (
            f"Legal definition and interpretation: {_md_link('Basel Secretariat Ban Amendment FAQ', BASEL_FAQ_URL)} "
            f"and the {_md_link('Basel Convention consolidated text', BASEL_TEXT_URL)}. "
            f"OECD membership: {_md_link('OECD Legal Framework', OECD_URL)}. EU "
            "membership and accession dates: "
            f"{_md_link('Council of the EU accession timeline', COUNCIL_URL)}. The "
            "living category sources must be refreshed before a time-sensitive "
            "decision."
        ),
        "",
        "## Date semantics",
        "",
        (
            "`OECD Convention effective` and `EU accession` are official category-"
            "membership dates. `Annex VII qualifying since` appears only on MEMBER "
            "records and is capped no earlier than Annex VII's adoption on "
            "1995-09-22. It is not the date on which Article 4A became binding for "
            "a state; that also depends on Convention party status, Ban consent, "
            "and the amendment's state-specific effective date. Current source "
            f"cardinalities are 38 OECD members, 27 EU members, and {overlap_count} "
            "states in both categories."
        ),
        "",
        "## Licensing, attribution, and inference",
        "",
        (
            "This artifact copies no source page wholesale. It records attributed "
            "membership facts and transparent union/set-complement classifications. "
            f"Basel/UN material remains subject to the {_md_link('UN terms', UN_TERMS_URL)}; "
            f"OECD material to the {_md_link('OECD terms', OECD_TERMS_URL)}; and "
            "Council material to its "
            f"{_md_link('copyright notice', COUNCIL_COPYRIGHT_URL)}."
        ),
        "",
        "# Annex VII state records",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## Country — {record.name} ({record.iso2} / {record.iso3})",
                "",
                _render_annex_record(record, retrieved_at),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def generate_artifacts(
    inputs: ArtifactInputs, timestamps: ArtifactTimestamps
) -> tuple[str, str]:
    status_records, organization = build_status_records(
        parse_untc_rows(inputs.convention_xml),
        parse_untc_rows(inputs.ban_xml),
        load_basel_service_records(inputs.convention_json),
        load_basel_service_records(inputs.ban_json),
    )
    validate_status_records(
        status_records,
        organization=organization,
        expected_state_count=EXPECTED_STATE_COUNT,
    )
    annex_records = build_annex_records(status_records)
    validate_annex_records(
        annex_records,
        expected_state_count=EXPECTED_STATE_COUNT,
        expected_member_count=EXPECTED_ANNEX_MEMBER_COUNT,
    )
    return (
        render_status_artifact(status_records, organization, timestamps),
        render_annex_artifact(annex_records, timestamps.retrieved_at),
    )


def validate_artifact_paths(
    inputs: ArtifactInputs,
    outputs: ArtifactOutputs,
) -> None:
    named_paths = (
        ("Convention XML input", inputs.convention_xml),
        ("Ban XML input", inputs.ban_xml),
        ("Convention JSON input", inputs.convention_json),
        ("Ban JSON input", inputs.ban_json),
        ("status output", outputs.status),
        ("Annex output", outputs.annex),
    )
    labels_by_path: dict[Path, str] = {}
    for label, path in named_paths:
        resolved = path.resolve()
        conflicting_label = labels_by_path.get(resolved)
        if conflicting_label is not None:
            raise ValueError(
                f"{label} resolves to the same path as {conflicting_label}: {resolved}."
            )
        labels_by_path[resolved] = label


def _write_artifact(path: Path, content: str) -> dict[str, str | int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    encoded = content.encode("utf-8")
    return {
        "path": str(path),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convention-xml", type=Path, required=True)
    parser.add_argument("--ban-xml", type=Path, required=True)
    parser.add_argument("--convention-json", type=Path, required=True)
    parser.add_argument("--ban-json", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--annex-output", type=Path, required=True)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument("--status-as-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    inputs = ArtifactInputs(
        convention_xml=args.convention_xml,
        ban_xml=args.ban_xml,
        convention_json=args.convention_json,
        ban_json=args.ban_json,
    )
    outputs = ArtifactOutputs(status=args.status_output, annex=args.annex_output)
    validate_artifact_paths(inputs, outputs)
    timestamps = ArtifactTimestamps(
        retrieved_at=args.retrieved_at,
        status_as_at=args.status_as_at,
    )
    status_content, annex_content = generate_artifacts(inputs, timestamps)
    for result in (
        _write_artifact(outputs.status, status_content),
        _write_artifact(outputs.annex, annex_content),
    ):
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
