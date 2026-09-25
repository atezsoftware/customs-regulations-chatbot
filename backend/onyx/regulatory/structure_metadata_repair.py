"""Exact source evidence for repairing structural metadata without rechunking."""

import re
from collections import defaultdict
from hashlib import sha256
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from onyx.db.regulatory_writer_publication import OwnedWriterInputs
    from onyx.document_index.publication_models import FileOwnership
    from onyx.regulatory.writer_publication_models import WriterPublicationManifest

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.chunker import (
    ATOMIC_CHUNK_VARIANT,
    REGULATORY_CHUNKER_CODE_VERSION,
    RegulatoryChunk,
    RegulatoryChunker,
)

STRUCTURE_FIELDS = (
    "article_no",
    "article_title",
    "paragraph_no",
    "clause_label",
    "subclause_label",
    "appendix_label",
)


class StructureRepairChange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    before: AnnexCanonicalSnapshot
    after: AnnexCanonicalSnapshot
    source_start: int
    source_end: int
    lineage_ids: list[str] = Field(default_factory=list)


class StructureRepairPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_sha256: str
    parser_version: str = REGULATORY_CHUNKER_CODE_VERSION
    changes: list[StructureRepairChange] = Field(default_factory=list)
    unresolved: dict[str, str] = Field(default_factory=dict)
    unchanged: list[str] = Field(default_factory=list)


def plan_structure_repair(
    rows: list[AnnexCanonicalSnapshot],
    *,
    markdown: str,
    source_file: str,
    history: list[AnnexCanonicalSnapshot] | None = None,
) -> StructureRepairPlan:
    """Accept only one complete atomic source block for each selected stored text."""
    if len({row.user_file_id for row in rows}) > 1:
        raise ValueError("Structure repair requires one source file")
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("Structure repair requires unique canonical identities")
    ancestors = {row.id: row for row in [*(history or []), *rows]}
    parsed = RegulatoryChunker().chunk_text(markdown, source_file=source_file)
    by_text: dict[str, list[RegulatoryChunk]] = defaultdict(list)
    for chunk in parsed.chunks:
        if chunk.metadata.chunk_variant == ATOMIC_CHUNK_VARIANT:
            by_text[chunk.text].append(chunk)
    plan = StructureRepairPlan(source_sha256=sha256(markdown.encode()).hexdigest())
    for row in rows:
        if (
            row.status != "active"
            or row.metadata.get("chunk_variant") == "hierarchical_aggregate"
            or row.metadata.get("bound_to_regulatory_chunk_id")
            or row.chunk_type == "image"
        ):
            plan.unresolved[row.id] = "requires_derived_or_historical_source_evidence"
            continue
        proof_row = row
        lineage = [row.id]
        matches = by_text.get(proof_row.text, [])
        while len(matches) != 1 or not proof_row.text:
            previous = ancestors.get(proof_row.supersedes_chunk_id or "")
            if (
                proof_row.source != "amendment"
                or previous is None
                or previous.id in lineage
                or previous.user_file_id != row.user_file_id
                or previous.superseded_by_chunk_id != proof_row.id
                or previous.status != "superseded"
                or previous.position != proof_row.position
                or previous.validity_end_date is None
                or proof_row.validity_start_date is None
                or previous.validity_end_date > proof_row.validity_start_date
            ):
                break
            proof_row = previous
            lineage.append(previous.id)
            matches = by_text.get(proof_row.text, [])
        if len(matches) != 1 or not proof_row.text:
            plan.unresolved[row.id] = "source_text_not_unique"
            continue
        metadata = matches[0].metadata
        start, end = metadata.source_start_char, metadata.source_end_char
        # The parser joins blank lines; prove the entire span without changing
        # any non-whitespace character in the stored canonical text.
        if " ".join(markdown[start:end].split()) != " ".join(proof_row.text.split()):
            plan.unresolved[row.id] = "source_span_not_exact"
            continue
        if metadata.article_no is None and metadata.appendix_label is None:
            plan.unresolved[row.id] = "source_structure_not_explicit"
            continue
        if len(lineage) > 1 and not _preserves_explicit_unit(row.text, matches[0]):
            plan.unresolved[row.id] = "amended_unit_identity_changed"
            continue
        repaired = dict(row.metadata)
        for key in STRUCTURE_FIELDS:
            if len(lineage) > 1 and key == "article_title":
                continue
            value = getattr(metadata, key)
            if value is None:
                if repaired.get(key) is not None:
                    repaired.pop(key)
            else:
                repaired[key] = value
        heading_path = list(metadata.heading_path)
        if len(lineage) > 1:
            from onyx.regulatory.amendments.draft_integrity import (
                reconcile_existing_heading_path,
            )
            from onyx.regulatory.provision_identity import article_identity

            for index, heading in enumerate(heading_path):
                if article_identity(heading) == metadata.article_no:
                    current = next(
                        (
                            value
                            for value in row.heading_path
                            if article_identity(value) == metadata.article_no
                        ),
                        None,
                    )
                    if current is not None:
                        heading_path[index] = current
                    break
            heading_path = reconcile_existing_heading_path(
                heading_path,
                amended_text=row.text,
                chunk_type=metadata.chunk_type,
                article_no=metadata.article_no,
                paragraph_no=metadata.paragraph_no,
                clause_label=metadata.clause_label,
                subclause_label=metadata.subclause_label,
            )
        if "heading_path" in repaired or row.heading_path != heading_path:
            heading_values: list[JsonValue] = list(heading_path)
            repaired["heading_path"] = heading_values
        after = row.model_copy(
            update={
                "chunk_type": metadata.chunk_type,
                "heading_path": heading_path,
                "metadata": repaired,
            }
        )
        if after == row:
            plan.unchanged.append(row.id)
        else:
            plan.changes.append(
                StructureRepairChange(
                    before=row,
                    after=after,
                    source_start=start,
                    source_end=end,
                    lineage_ids=lineage if len(lineage) > 1 else [],
                )
            )
    return plan


def _preserves_explicit_unit(text: str, original: RegulatoryChunk) -> bool:
    from onyx.regulatory.provision_identity import (
        article_identity,
        canonical_clause_label,
    )

    clean = re.sub(r"(?:\*\*|__|`|<[^>]+>)", "", text).strip()
    header = re.match(
        r"^(?:(?:EK|GEÇİCİ|GECICI|MÜKERRER|MUKERRER)\s+)?MADDE\s+\d+[a-z]?\s*[-–—:]?\s*",
        clean,
        re.IGNORECASE,
    )
    if header is not None:
        if article_identity(header.group()) != original.metadata.article_no:
            return False
        clean = clean[header.end() :]
    kind = original.metadata.chunk_type
    if kind == "paragraph":
        match = re.match(r"^(?:\((\d+)\)|(\d+)[.)])", clean)
        return (
            match is not None
            and (match.group(1) or match.group(2)) == original.metadata.paragraph_no
        )
    if kind == "clause":
        match = re.match(
            r"^(?:\(([a-zçğıöşü])\)|([a-zçğıöşü])\))", clean, re.IGNORECASE
        )
        return (
            match is not None
            and canonical_clause_label(match.group(1) or match.group(2))
            == original.metadata.clause_label
        )
    return kind == "article" and header is not None


def prepare_structure_metadata_manifest(
    owner: "FileOwnership",
    inputs: "OwnedWriterInputs",
    *,
    plan: StructureRepairPlan,
    markdown: str,
) -> "WriterPublicationManifest":
    """Freeze a verified metadata transformation with identical text and vectors."""
    import json
    from uuid import uuid4

    from onyx.document_index.publication_models import (
        ObservedPublicationProjection,
        SourceHeadingRepair,
        merge_publication_indexes,
        publication_digest,
    )
    from onyx.regulatory.writer_publication_models import WriterPublicationManifest

    if owner.user_file_id != inputs.file.id:
        raise ValueError("Structure repair escaped its owned source")
    by_id = {row.id: row for row in inputs.canonical}
    selected_ids = (
        [change.before.id for change in plan.changes]
        + list(plan.unresolved)
        + plan.unchanged
    )
    if not set(selected_ids).issubset(by_id):
        raise ValueError("Structure repair canonical baseline disappeared")
    current = plan_structure_repair(
        [by_id[identifier] for identifier in selected_ids],
        markdown=markdown,
        source_file=inputs.file.name,
        history=inputs.canonical,
    )
    if current != plan:
        raise ValueError("Structure repair source proof or canonical baseline changed")
    changes = {change.before.id: change for change in plan.changes}
    if not changes:
        raise ValueError("Structure repair has no proven metadata changes")
    updated = []
    revisions = {}
    covered: set[str] = set()
    for previous in inputs.bindings:
        source = json.loads(previous.projection.source_json)
        identifier = source["regulatory_chunk_id"]
        change = changes.get(identifier)
        if change is None:
            updated.append(previous)
            revisions[previous.id] = inputs.revisions[previous.id]
            continue
        if previous.representation_text != change.before.text:
            raise ValueError(
                "Structure repair requires exact representation source evidence"
            )
        covered.add(identifier)
        new_id = uuid4()
        fields = {
            **previous.projection.model_dump(),
            "context_projection_id": str(new_id),
        }
        if isinstance(previous.projection, ObservedPublicationProjection):
            retained = previous.projection.heading_repair
            fields["heading_repair"] = SourceHeadingRepair(
                canonical_chunk_id=identifier,
                original_heading_present=retained.original_heading_present
                if retained
                else "heading_path" in source,
                original_heading_path=retained.original_heading_path
                if retained
                else source.get("heading_path"),
                corrected_heading_path=change.after.heading_path,
                source_sha256=plan.source_sha256,
                canonical_before_sha256=publication_digest(
                    change.before.model_dump(mode="json")
                ),
                canonical_after_sha256=publication_digest(
                    change.after.model_dump(mode="json")
                ),
                parser_version=plan.parser_version,
                source_start=change.source_start,
                source_end=change.source_end,
            ).model_dump()
        source["heading_path"] = list(change.after.heading_path)
        fields["source_json"] = json.dumps(source)
        projection = type(previous.projection).model_validate(fields)
        representation_metadata = dict(previous.representation_metadata)
        for key in (*STRUCTURE_FIELDS, "heading_path"):
            if key in change.after.metadata:
                representation_metadata[key] = change.after.metadata[key]
            else:
                representation_metadata.pop(key, None)
        updated.append(
            previous.model_copy(
                update={
                    "id": new_id,
                    "projection": projection,
                    "representation_metadata": representation_metadata,
                }
            )
        )
    if covered != set(changes):
        raise ValueError(
            "Structure repair requires the complete existing index baseline"
        )
    indexes = [
        merge_publication_indexes(
            [
                binding.index
                for binding in inputs.bindings
                if binding.index.index_uuid == index_uuid
            ]
        )
        for index_uuid in sorted(
            {binding.index.index_uuid for binding in inputs.bindings}
        )
    ]
    return WriterPublicationManifest(
        id=uuid4(),
        scope=owner.scope,
        user_file_id=owner.user_file_id,
        kind="metadata",
        index_state_sha256=inputs.index_state_sha256,
        canonical_before_sha256=publication_digest(
            [row.model_dump(mode="json") for row in inputs.canonical]
        ),
        canonical_after=[
            changes[row.id].after if row.id in changes else row
            for row in inputs.canonical
        ],
        indexes=indexes,
        previous_binding_ids=[binding.id for binding in inputs.bindings],
        bindings=updated,
        canonical_revisions=revisions,
        structure_repair=plan.model_dump(mode="json"),
    )


def prepare_owned_structure_metadata(
    owner: "FileOwnership",
    client: "Elasticsearch",
    plan: StructureRepairPlan,
) -> "WriterPublicationManifest":
    """Verify the stored source and actual index before any metadata publication."""
    import json

    from onyx.db.regulatory_publication import PublicationStore
    from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.publication_models import matches_indexed_evidence
    from onyx.file_store.file_store import get_default_file_store

    inputs = load_owned_writer_inputs(owner)
    if inputs.file.file_type not in {"text/markdown", "text/plain"}:
        raise ValueError("Structure repair requires the original canonical text source")
    with get_default_file_store().read_file(inputs.file.file_id, mode="rb") as stream:
        content = stream.read(4 * 1024 * 1024 + 1)
    if len(content) > 4 * 1024 * 1024:
        raise ValueError("Structure source exceeds the bounded repair budget")
    manifest = prepare_structure_metadata_manifest(
        owner,
        inputs,
        plan=plan,
        markdown=content.decode("utf-8"),
    )
    reservations = PublicationStore(owner.scope).reservations(owner)
    for index in manifest.indexes:
        expected = {
            binding.projection.ordinal: binding
            for binding in inputs.bindings
            if binding.index.index_uuid == index.index_uuid
        }
        actuals = FencedPublicationIndex(client, index).inventory_evidence(reservations)
        if {json.loads(actual.source_json)["chunk_index"] for actual in actuals} != set(
            expected
        ):
            raise ValueError(
                "Structure repair indexed inventory differs from qualified baseline"
            )
        for actual in actuals:
            previous = expected[json.loads(actual.source_json)["chunk_index"]]
            if not matches_indexed_evidence(previous.projection, actual):
                raise ValueError(
                    "Structure repair actual source differs from qualified evidence"
                )
    return manifest
