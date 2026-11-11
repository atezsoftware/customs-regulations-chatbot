"""Prove source-derived locators locally instead of asking the model to repeat them."""

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from onyx.regulatory.chunker import _ordered_clause_text
from onyx.regulatory.document_intake import metadata

CORE_LOCATORS = frozenset({"article_no", "paragraph_no", "clause_label"})
STRUCTURAL_FIELDS = CORE_LOCATORS | {"article_title", "section_title", "annex_title"}


class VerifiedStructuralFact(metadata.LegalMetadataFact):
    verification_origin: Literal["source_structure"] = Field(
        default="source_structure", exclude=True
    )


def positive_locator_evidence(
    field: str, value: object, entry: metadata.MetadataEvidence
) -> bool:
    if field == "article_no":
        if not isinstance(value, str):
            return False
        claimed = metadata._article_locator_identity(value, locator_value=True)
        quoted = (
            metadata._article_locator_identity(entry.text)
            if entry.scope == "heading"
            else metadata._body_article_identity(entry.text)
        )
        return claimed is not None and quoted == claimed
    if field == "clause_label":
        return (
            isinstance(value, str)
            and metadata.clause_body_identity(entry.text) is not None
            and metadata.clause_body_identity(entry.text)
            == metadata._clause_value(value)
        )
    if field == "paragraph_no":
        if not isinstance(value, str):
            return False
        if _ordered_clause_text(entry.text) is not None:
            return False
        markers = metadata._paragraph_markers(entry.text)
        if not markers and entry.scope == "heading":
            markers = metadata._paragraph_markers(entry.text, numbered=True)
        return markers == {value.strip().strip("()")}
    return True


def explicit_own_start_rules(
    chunks: Sequence[metadata.MetadataChunk],
) -> list[tuple[str, str]]:
    """Require representation of unambiguous observed own-commencement dates."""
    rules = set()
    for chunk in chunks:
        for line in chunk.text.splitlines():
            if not re.match(
                r"^\s*(?:\(\d+\)\s*)?bu\s+(?:kanun|yonetmelik|teblig|karar)\s+[^.;]{0,160}\btarihinde\s+yururluge\s+girer\b",
                metadata._fold(line),
            ):
                continue
            dates = metadata.quoted_dates(line)
            if len(dates) == 1:
                rules.add((chunk.chunk_id, next(iter(dates)).isoformat()))
    return sorted(rules)


def explicit_target_start_rules(
    chunks: Sequence[metadata.MetadataChunk],
) -> dict[str, str]:
    rules = {}
    for chunk in chunks:
        dates = set()
        for line in chunk.text.splitlines():
            if re.match(
                r"^\s*(?:madde\s+\d+\s*[-–—]\s*)?(?:\(\d+\)\s*)?bu\s+(?:madde|fikra|bent|hukum)\s+[^.;]{0,160}\btarihinde\s+yururluge\s+girer\b",
                metadata._fold(line),
            ):
                dates.update(metadata.quoted_dates(line))
        if len(dates) == 1:
            rules[chunk.chunk_id] = next(iter(dates)).isoformat()
    return rules


def verified_structural_metadata(
    chunks: Sequence[metadata.MetadataChunk],
) -> dict[str, dict[str, metadata.LegalMetadataFact]]:
    catalog = metadata.build_metadata_evidence_catalog(chunks)
    by_source: dict[str, list[metadata.MetadataEvidence]] = {}
    for entry in catalog:
        by_source.setdefault(entry.source_chunk_id, []).append(entry)
    verified: dict[str, dict[str, metadata.LegalMetadataFact]] = {}
    for chunk in chunks:
        selected: dict[str, metadata.LegalMetadataFact] = {}
        for field, value in chunk.structural_metadata.items():
            if (
                field not in STRUCTURAL_FIELDS
                or not isinstance(value, str)
                or not value.strip()
            ):
                continue
            candidates = sorted(
                by_source.get(chunk.chunk_id, []),
                key=lambda e: (e.scope != "heading", len(e.text)),
            )
            for entry in candidates:
                if not positive_locator_evidence(field, value, entry):
                    continue
                if field == "clause_label" and metadata.clause_body_identity(
                    entry.text
                ) != metadata._clause_value(value):
                    continue
                if field not in CORE_LOCATORS and (
                    entry.scope != "heading"
                    or metadata._fold(" ".join(value.split()))
                    != metadata._fold(" ".join(entry.text.split()))
                ):
                    continue
                fact = metadata.LegalMetadataFact.model_validate(
                    {
                        "field": field,
                        "value": value,
                        "source_chunk_id": entry.source_chunk_id,
                        "evidence_quote": entry.text,
                        "basis": "explicit",
                    }
                )
                candidate = metadata.LegalMetadataResult(
                    chunks=[
                        metadata.ChunkLegalMetadata(
                            chunk_id=chunk.chunk_id, fields=[fact]
                        )
                    ]
                )
                try:
                    metadata.validate_metadata_response(
                        candidate.model_dump_json(),
                        chunks=chunks,
                        target_ids=[chunk.chunk_id],
                    )
                except ValueError:
                    continue
                selected[field] = fact
                break
        verified[chunk.chunk_id] = selected
    return verified
