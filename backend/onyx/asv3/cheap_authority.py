"""Canonical identity projections; witnesses never approve legal application."""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import (
    _authority_aliases,
    _gap_masking_aliases,
    _matching_native_rows,
    _named_native_references,
    _native_original_rows,
    _normalized_native_reference,
    _original_gap_references,
    _precise_original_gap,
    _without_verified_quotes,
    folded,
    native_named_authority_gap,
)
from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, model_evidence_metadata
from onyx.context.search.models import SearchDoc
from onyx.regulatory.heading_path import parse_regulatory_article_heading

CHEAP_AUTHORITY_POLICY = "cheap-canonical-authority-v1"
_ORDINALS = {
    "birinci": "1",
    "ikinci": "2",
    "üçüncü": "3",
    "dördüncü": "4",
    "beşinci": "5",
}
_POSSESSIVE = r"(?P<owner>\bKanunu)['’]nun\s+"
_TITLED_ARTICLE = re.compile(
    _POSSESSIVE
    + r'(?:\*\*)?[“"][^“”"\n.;|]{1,80}[”"]\s+başlıklı\s+'
    + r"(?P<article>\d{1,4}(?:/[A-ZÇĞİÖŞÜ])?)\.\s*maddesinin"
    + r"(?:\s+(?P<ordinal>birinci|ikinci|üçüncü|dördüncü|beşinci)\s+fıkrası)?(?:\*\*)?",
    re.IGNORECASE,
)
_QUOTED_ARTICLE_HEADING = re.compile(
    _POSSESSIVE
    + r'özgün\s+metninde\s+(?:\*\*)?[“"]'
    + r"(?P<qualifier>GEÇİCİ\s+|MÜKERRER\s+|EK\s+)?MADDE\s+"
    + r"(?P<article>\d{1,4}(?:/[A-ZÇĞİÖŞÜ])?)"
    + r'[”"](?:\*\*)?\s+başlığıyla\s+başlayan\s+hüküm',
    re.IGNORECASE,
)
_AMENDING_PASSAGE = re.compile(
    r"\b(?:asagidaki sekilde|degistirilmistir|eklenmistir|yururlukten kaldirilmistir)\b"
)
_OWN_ARTICLE_OPENING = re.compile(
    r"^(?P<heading>\*\*(?:(?:GEÇİCİ|MÜKERRER|EK)\s+)?MADDE\s+"
    r"\d{1,4}(?:/[A-ZÇĞİÖŞÜ])?\s*\.?\s*[-–—:]?\*\*)"
    r"\s*[.\-–—:]*\s*(?:\(\d{1,3}\)|\d{1,3}\.)\s+\S",
    re.IGNORECASE,
)


def _bound_reference_text(answer: str) -> str:
    """Normalize only explicit local possessive links, without changing the answer."""

    def titled(match: re.Match[str]) -> str:
        ordinal = _ORDINALS.get((match["ordinal"] or "").casefold())
        return (
            match["owner"]
            + " m."
            + match["article"]
            + ("/" + ordinal if ordinal else "")
        )

    def headed(match: re.Match[str]) -> str:
        return (
            match["owner"] + " " + (match["qualifier"] or "") + "m." + match["article"]
        )

    return _QUOTED_ARTICLE_HEADING.sub(headed, _TITLED_ARTICLE.sub(titled, answer))


@dataclass(frozen=True)
class ArticleIdentityWitness:
    citation: int
    source_id: str
    chunk_id: str
    text_hash: str
    heading_start: int
    heading_end: int
    heading_hash: str
    article_no: str
    qualifier: str | None
    previous_article_no: str | None


def _own_article_witness(
    citation: int, item: EvidenceItem, row: dict[str, JsonValue]
) -> ArticleIdentityWitness | None:
    metadata = model_evidence_metadata(item.metadata)
    if (
        folded(str(metadata.get("document_type") or ""))
        not in {"kanun", "law", "act", "statute"}
        or metadata.get("untrusted")
        or metadata.get("truncated")
        or not item.chunk_id
    ):
        return None
    canonical = item.metadata.get("canonical_metadata")
    canonical = canonical if isinstance(canonical, dict) else item.metadata
    if canonical.get("truncated") or canonical.get("untrusted"):
        return None
    stripped = item.text.lstrip()
    lines = stripped.splitlines()
    if not lines:
        return None
    first = lines[0]
    opening = _OWN_ARTICLE_OPENING.match(first)
    plain = first.replace("**", "").replace("__", "")
    heading = parse_regulatory_article_heading(plain)
    identifiers = metadata.get("provision_identifiers")
    if (
        opening is None
        or heading is None
        or heading.is_reverse
        or not isinstance(identifiers, list)
        or not any(
            isinstance(identifier, str)
            and (parsed := parse_regulatory_article_heading(identifier)) is not None
            and (parsed.article_no, parsed.qualifier)
            == (heading.article_no, heading.qualifier)
            for identifier in identifiers
        )
        or _AMENDING_PASSAGE.search(folded(item.text))
        or any(
            parse_regulatory_article_heading(line.replace("**", "").replace("__", ""))
            for line in lines[1:]
        )
    ):
        return None
    previous = row.get("article_no")
    if previous is not None and not isinstance(previous, str):
        return None
    for field in ("target_article_no", "target_id"):
        if canonical.get(field) is not None:
            return None
    original_path = canonical.get("original_heading_path")
    if isinstance(original_path, list) and any(
        isinstance(part, str)
        and (parsed := parse_regulatory_article_heading(part)) is not None
        and (parsed.article_no, parsed.qualifier)
        != (heading.article_no, heading.qualifier)
        for part in original_path
    ):
        return None
    if previous is not None and previous != heading.article_no:
        # An inserted article can retain its base article's navigation path.
        inserted = re.fullmatch(r"(\d{1,4})/[A-ZÇĞİÖŞÜ]", heading.article_no)
        if (
            inserted is None
            or previous != inserted[1]
            or row.get("article_qualifier") is not None
            or heading.qualifier is not None
            or row.get("clause_label") is not None
            or canonical.get("asv3_navigation_position_authority") is not None
            or not any(
                isinstance(identifier, str)
                and (parsed := parse_regulatory_article_heading(identifier)) is not None
                and parsed.article_no == previous
                and parsed.qualifier is None
                for identifier in identifiers
            )
        ):
            return None
    elif previous is not None and row.get("article_qualifier") != heading.qualifier:
        return None
    start = len(item.text) - len(stripped)
    return ArticleIdentityWitness(
        citation=citation,
        source_id=item.source_id,
        chunk_id=item.chunk_id,
        text_hash=item.text_hash,
        heading_start=start,
        heading_end=start + opening.end("heading"),
        heading_hash=hashlib.sha256(opening["heading"].encode()).hexdigest(),
        article_no=heading.article_no,
        qualifier=heading.qualifier,
        previous_article_no=previous,
    )


@dataclass(frozen=True)
class ArticleParagraphBinding:
    citation: int
    source_id: str
    chunk_id: str
    text_hash: str
    article_no: str
    qualifier: str | None
    paragraph_no: str


class _IdentityLedger(EvidenceLedger):
    """Read-only projection; canonical items and physical deliveries stay owned."""

    def __init__(
        self,
        ledger: EvidenceLedger,
        article_bindings: dict[int, ArticleParagraphBinding] | None = None,
    ) -> None:
        super().__init__()
        self._source = ledger
        self.witnesses: dict[int, ArticleIdentityWitness] = {}
        self.article_bindings = dict(article_bindings or {})
        for number, binding in self.article_bindings.items():
            item = ledger.get(number)
            if (
                binding.citation != number
                or item is None
                or item.identity
                != (binding.source_id, binding.chunk_id, binding.text_hash)
            ):
                raise ValueError("Changed ordered article binding")
        for row in _native_original_rows(ledger):
            citation = row.get("citation")
            item = ledger.get(citation) if type(citation) is int else None
            if item is not None and type(citation) is int:
                witness = _own_article_witness(citation, item, row)
                if witness is not None:
                    self.witnesses[citation] = witness

    def get(self, number: int) -> EvidenceItem | None:
        return self._source.get(number)

    def citation_mapping(self) -> dict[int, SearchDoc]:
        return self._source.citation_mapping()

    def completely_delivered(self, call_id: str) -> set[int]:
        return self._source.completely_delivered(call_id)

    def authority_metadata(self) -> list[dict[str, JsonValue]]:
        result = []
        for row in self._source.authority_metadata():
            citation = row.get("citation")
            witness = self.witnesses.get(citation) if type(citation) is int else None
            binding = (
                self.article_bindings.get(citation) if type(citation) is int else None
            )
            if binding is not None:
                result.append(
                    {
                        **row,
                        "article_no": binding.article_no,
                        "article_qualifier": binding.qualifier,
                        "paragraph_no": binding.paragraph_no,
                    }
                )
                continue
            result.append(
                {
                    **row,
                    "article_no": witness.article_no,
                    "article_qualifier": witness.qualifier,
                }
                if witness is not None
                else row
            )
        return result


def canonical_identity_ledger(
    ledger: EvidenceLedger,
    *,
    article_bindings: dict[int, ArticleParagraphBinding] | None = None,
) -> EvidenceLedger:
    """Project verified own-article identity without changing originals or receipts."""
    if isinstance(ledger, _IdentityLedger) and article_bindings is None:
        return ledger
    return _IdentityLedger(ledger, article_bindings)


def normalize_canonical_authority_references(answer: str) -> str:
    """Bind explicit possessive locators without changing the published answer."""
    return _bound_reference_text(answer)


def _recovered_paragraphs(item: EvidenceItem) -> set[str]:
    opening = _OWN_ARTICLE_OPENING.match(item.text.lstrip().splitlines()[0])
    if opening is None:
        return set()
    first = re.match(
        r"\s*[.\-–—:]*\s*(?:\((\d{1,3})\)|(\d{1,3})\.)\s+\S",
        item.text.lstrip()[opening.end("heading") :],
    )
    if first is None:
        return set()
    first_number = first[1] or first[2]
    metadata = model_evidence_metadata(item.metadata)
    declared = metadata.get("paragraph_no")
    if declared is not None and str(declared) != first_number:
        return set()
    return {
        first_number,
        *re.findall(r"(?m)^\s*\((\d{1,3})\)\s+\S", item.text),
    }


def _recovered_paragraph_gaps(
    answer: str,
    ledger: _IdentityLedger,
    *,
    strict_reference_boundaries: bool,
    syntactic_reference_binding: bool,
) -> list[dict[str, JsonValue]]:
    rows = _native_original_rows(ledger)
    aliases = _authority_aliases(
        ledger,
        rows,
        strict_reference_boundaries=strict_reference_boundaries,
        syntactic_reference_binding=syntactic_reference_binding,
    )
    gaps: list[dict[str, JsonValue]] = []
    masking_aliases = (
        _gap_masking_aliases(rows, {}) if syntactic_reference_binding else {}
    )
    for unit in assertion_inventory(answer):
        if unit["presentation_only"]:
            continue
        cited = set(unit["evidence_numbers"])
        if not cited and (
            _original_gap_references(
                unit["text"], aliases, masking_aliases=masking_aliases
            )
            is not None
            if syntactic_reference_binding
            else _precise_original_gap(unit["text"], aliases)
        ):
            continue
        seen: set[tuple[str | None, ...]] = set()
        for reference, name in _named_native_references(
            _without_verified_quotes(unit["text"], cited, ledger),
            aliases,
            strict_reference_boundaries=strict_reference_boundaries,
            syntactic_reference_binding=syntactic_reference_binding,
        ):
            if reference.paragraph is None:
                continue
            if syntactic_reference_binding:
                reference = _normalized_native_reference(reference, name, rows)
            key = (
                reference.number,
                name if not reference.number else None,
                reference.article,
                reference.paragraph,
                reference.qualifier,
            )
            if key in seen:
                continue
            seen.add(key)
            matching = _matching_native_rows(
                reference, name, rows, clause_specific=syntactic_reference_binding
            )
            if not any(
                number in ledger.witnesses or number in ledger.article_bindings
                for number in matching
            ):
                continue
            proven: list[int] = []
            for number in matching:
                item = ledger.get(number)
                binding = ledger.article_bindings.get(number)
                if binding is not None:
                    if binding.paragraph_no == reference.paragraph:
                        proven.append(number)
                elif number not in ledger.witnesses or (
                    item is not None
                    and reference.paragraph in _recovered_paragraphs(item)
                ):
                    proven.append(number)
            if cited.intersection(proven):
                continue
            entry: dict[str, JsonValue] = {
                "unit_id": unit["unit_id"],
                "instrument_number": reference.number or None,
                "reference_text": reference.reference_text,
                "article": reference.article,
                "paragraph": reference.paragraph,
                "inline_evidence": sorted(cited),
                "matching_original_evidence": proven,
            }
            if entry not in gaps:
                gaps.append(entry)
    return gaps


def canonical_named_authority_gap(
    answer: str,
    ledger: EvidenceLedger,
    *,
    strict_reference_boundaries: bool = False,
    syntactic_reference_binding: bool = False,
    resolve_defined_abbreviations: bool = False,
    strict_numeric_paragraphs: bool = False,
) -> dict[str, JsonValue] | None:
    """Check source identity; recovered headings never establish unread paragraphs."""
    identity = canonical_identity_ledger(ledger)
    assert isinstance(identity, _IdentityLedger)
    normalized = normalize_canonical_authority_references(answer)
    gap: dict[str, JsonValue] | None = native_named_authority_gap(
        normalized,
        identity,
        strict_reference_boundaries=strict_reference_boundaries,
        syntactic_reference_binding=syntactic_reference_binding,
        resolve_defined_abbreviations=resolve_defined_abbreviations,
    )
    if strict_numeric_paragraphs:
        paragraph_gaps = _recovered_paragraph_gaps(
            normalized,
            identity,
            strict_reference_boundaries=strict_reference_boundaries,
            syntactic_reference_binding=syntactic_reference_binding,
        )
        if paragraph_gaps:
            if gap is None:
                gap = {
                    "named_authority_gaps": [],
                    "citation_format": "[n]",
                    "instruction": (
                        "A recovered own-article heading identifies only its actual literal "
                        "paragraphs. Read the precise missing paragraph or disclose its source "
                        "gap without asserting its effect; a heading never proves article closure."
                    ),
                }
            else:
                gap = copy.deepcopy(gap)
            entries = gap["named_authority_gaps"]
            assert isinstance(entries, list)
            for entry in paragraph_gaps:
                if entry not in entries:
                    entries.append(entry)
    if gap is not None and normalized != answer:
        originals, parsed = assertion_inventory(answer), assertion_inventory(normalized)
        if len(originals) != len(parsed):
            raise ValueError("Canonical authority normalization changed answer units")
        units = {
            current["unit_id"]: original["unit_id"]
            for original, current in zip(originals, parsed, strict=True)
        }
        gap = copy.deepcopy(gap)
        entries = gap.get("named_authority_gaps")
        assert isinstance(entries, list)
        for entry in entries:
            assert isinstance(entry, dict) and isinstance(entry.get("unit_id"), str)
            entry["unit_id"] = units[cast(str, entry["unit_id"])]
    return gap


class _StoredPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    policy: Literal["cheap-canonical-authority-v1"]
    requirements: dict[str, JsonValue]


class CheapAuthorityPolicy:
    def __init__(self, context: RunContext, request: str) -> None:
        self._check_cheap(context)
        self._requirements = AuthorityRequirements(
            context, request, syntactic_reference_binding=True
        )

    @staticmethod
    def _check_cheap(context: RunContext) -> None:
        if context.services.get("asv3_exp_cheap") is not True:
            raise ValueError("Cheap authority policy requires ASv3Exp Cheap")

    def identity_ledger(self, ledger: EvidenceLedger) -> EvidenceLedger:
        return canonical_identity_ledger(ledger)

    def identity_witnesses(self, ledger: EvidenceLedger) -> list[dict[str, JsonValue]]:
        return [
            cast(dict[str, JsonValue], asdict(witness))
            for witness in _IdentityLedger(ledger).witnesses.values()
        ]

    def publication_gap(
        self,
        answer: str,
        model_call_id: str | None,
        context: RunContext,
        ledger: EvidenceLedger,
        *,
        validated_delivered: set[int] | None = None,
    ) -> dict[str, JsonValue] | None:
        self._check_cheap(context)
        identity_ledger = self.identity_ledger(ledger)
        gap = canonical_named_authority_gap(
            answer,
            identity_ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
            resolve_defined_abbreviations=True,
        )
        return self._requirements.publication_gap(
            answer,
            model_call_id,
            context,
            identity_ledger,
            native_gap=gap,
            validated_delivered=validated_delivered,
        )

    def view(self, context: RunContext) -> dict[str, JsonValue]:
        self._check_cheap(context)
        return self._requirements.view(context)

    def export(self) -> dict[str, JsonValue]:
        return {
            "version": 1,
            "policy": CHEAP_AUTHORITY_POLICY,
            "requirements": self._requirements.export(),
        }

    def restore(
        self, snapshot: dict[str, JsonValue], context: RunContext, request: str
    ) -> None:
        self._check_cheap(context)
        if type(snapshot.get("version")) is not int:
            raise ValueError("Invalid Cheap authority policy version")
        saved = _StoredPolicy.model_validate(snapshot)
        if saved.version != 1 or saved.policy != CHEAP_AUTHORITY_POLICY:
            raise ValueError("Cheap authority policy changed")
        self._requirements.restore(saved.requirements, context, request)
