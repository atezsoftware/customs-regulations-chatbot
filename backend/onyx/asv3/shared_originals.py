"""Share recorded siblings of a selected provision, without source acquisition."""

import json
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import model_evidence_metadata
from onyx.regulatory.heading_path import parse_regulatory_article_heading


class _ProvisionIdentity(NamedTuple):
    source_id: str
    enclosing_scope: tuple[str, ...]
    article_no: str
    qualifier: str | None
    version: str


_VERSION_FIELDS = (
    "source_sha256",
    "publication_revision",
    "publication_revision_id",
    "revision_id",
    "revision",
    "version",
    "document_date",
    "validity_start",
    "validity_end",
    "read_as_of_date",
    "version_unknown",
)


def _provision_identity(record: dict[str, JsonValue]) -> _ProvisionIdentity | None:
    source_id, metadata = record.get("source_id"), record.get("metadata")
    if (
        not isinstance(source_id, str)
        or not source_id
        or record.get("citable") is not True
        or record.get("truncated") is True
        or not isinstance(metadata, dict)
        or metadata.get("derived") is True
        or metadata.get("external") is True
    ):
        return None
    headings = metadata.get("heading_path")
    if not isinstance(headings, list):
        return None
    scope: list[str] = []
    for heading in headings:
        if not isinstance(heading, str):
            return None
        parsed = parse_regulatory_article_heading(heading)
        if parsed is not None:
            # The first structural article owns the passage; descendant prose
            # may start with a cross-reference to a different article.
            version = json.dumps(
                {key: metadata.get(key) for key in _VERSION_FIELDS},
                sort_keys=True,
                ensure_ascii=False,
            )
            return _ProvisionIdentity(
                source_id, tuple(scope), parsed.article_no, parsed.qualifier, version
            )
        scope.append(" ".join(heading.split()))
    return None


def related_provision_originals(
    ledger: EvidenceLedger,
    current_originals: Sequence[dict[str, JsonValue]],
    *,
    citation_numbers: Iterable[int] = (),
) -> list[dict[str, JsonValue]]:
    """Return new full originals sharing a selected canonical provision/version.

    The ledger must be the current run's authorized, revalidated ledger. Explicit
    citations select already recorded originals; topic words and global source
    titles never select evidence. A selected partial range remains partial unless
    its citation is explicitly requested.
    """
    selected_numbers = {number for number in citation_numbers if type(number) is int}
    if not current_originals and not selected_numbers:
        return []
    inventory = ledger.provision_metadata()
    by_number = {
        number: record
        for record in inventory
        if type(number := record.get("citation")) is int
    }
    identities: set[_ProvisionIdentity] = set()
    present: set[int] = set()
    complete: set[int] = set()
    for record in current_originals:
        number, text, start = (
            record.get("citation"),
            record.get("text"),
            record.get("start_char", 0),
        )
        known = by_number.get(number) if type(number) is int else None
        if (
            type(number) is not int
            or not isinstance(text, str)
            or not text
            or type(start) is not int
            or start < 0
            or known is None
            or record.get("source_id") != known.get("source_id")
            or record.get("chunk_id") != known.get("chunk_id")
            or record.get("text_hash") != known.get("text_hash")
            or (
                "end_char" in record
                and (
                    type(record.get("end_char")) is not int
                    or record.get("end_char") != start + len(text)
                )
            )
        ):
            continue
        item = ledger.get(number)
        if (
            item is None
            or item.text[start : start + len(text)] != text
            or (
                "total_chars" in record
                and (
                    type(record.get("total_chars")) is not int
                    or record.get("total_chars") != len(item.text)
                )
            )
        ):
            continue
        present.add(number)
        if start == 0 and text == item.text:
            complete.add(number)
        if (identity := _provision_identity(known)) is not None:
            identities.add(identity)
    for number in selected_numbers:
        if (known := by_number.get(number)) is not None:
            if (identity := _provision_identity(known)) is not None:
                identities.add(identity)
    if not identities:
        return []
    originals: list[dict[str, JsonValue]] = []
    for number, record in by_number.items():
        if number in complete or (number in present and number not in selected_numbers):
            continue
        if _provision_identity(record) not in identities:
            continue
        item = ledger.get(number)
        if item is None or item.text_hash != record.get("text_hash"):
            continue
        originals.append(
            {
                "citation": number,
                "source_id": item.source_id,
                "chunk_id": item.chunk_id,
                "text_hash": item.text_hash,
                "text": item.text,
                "truncated": False,
                "citable": item.search_doc is not None,
                "metadata": model_evidence_metadata(item.metadata),
            }
        )
    return originals


def delivered_provision_navigation(
    complete_originals: Sequence[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    """Index fitted full chunks by their own structural provision and version.

    Full chunks do not establish complete provision closure. The caller supplies
    only canonical ranges physically retained in the current decision.
    """
    grouped: dict[_ProvisionIdentity, list[int]] = {}
    for record in complete_originals:
        identity = _provision_identity(record)
        citation = record.get("citation")
        if identity is not None and type(citation) is int:
            numbers = grouped.setdefault(identity, [])
            if citation not in numbers:
                numbers.append(citation)
    return [
        {
            "source_id": identity.source_id,
            "enclosing_scope": list(identity.enclosing_scope),
            "article_no": identity.article_no,
            "qualifier": identity.qualifier,
            "version": json.loads(identity.version),
            "available_full_original_citations": sorted(numbers),
        }
        for identity, numbers in grouped.items()
    ]
