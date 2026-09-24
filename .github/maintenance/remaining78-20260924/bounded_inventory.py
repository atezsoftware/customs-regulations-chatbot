"""Stream vector-bearing inventories without retaining their decoded payloads."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from scripts.prepare_regulatory_publication_baselines import read_file_inventory

from onyx.document_index.publication_models import (
    IndexedProjectionEvidence,
    PublicationIndexSnapshot,
    publication_list_digest,
)


@dataclass
class CompactInventory:
    count: int = 0
    source_sha256: str = ""
    current_evidence: bool = True
    headings: dict[str, list[str]] = field(default_factory=dict)
    full_hits: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[IndexedProjectionEvidence] = field(default_factory=list)


def ordered_hits(
    client: Elasticsearch, file_id: UUID, *, index_name: str, index_uuid: str
) -> Iterator[dict[str, Any]]:
    def check_index() -> None:
        info = client.indices.get(index=index_name)
        if (
            set(info) != {index_name}
            or info[index_name]["settings"]["index"]["uuid"] != index_uuid
        ):
            raise ValueError("DEV physical index identity changed")

    check_index()
    previous = -1
    for hit in scan(
        client,
        index=index_name,
        size=16,
        preserve_order=True,
        query={
            "seq_no_primary_term": True,
            "sort": [{"chunk_index": "asc"}],
            "query": {"term": {"document_id": str(file_id)}},
        },
    ):
        ordinal = hit["_source"].get("chunk_index")
        if type(ordinal) is not int or ordinal <= previous:
            raise ValueError("inventory ordinal is invalid, duplicate or unordered")
        previous = ordinal
        yield hit
    check_index()


def read_compact_inventory(
    client: Elasticsearch,
    file_id: UUID,
    *,
    index_name: str,
    index_uuid: str,
    excluded_ids: frozenset[str] = frozenset(),
    retain_hits: bool = False,
    evidence_index: PublicationIndexSnapshot | None = None,
    qualified_only: bool = False,
) -> CompactInventory:
    result = CompactInventory()

    def sources() -> Iterator[dict[str, Any]]:
        for hit in ordered_hits(
            client, file_id, index_name=index_name, index_uuid=index_uuid
        ):
            source = hit["_source"]
            identifier = source["regulatory_chunk_id"]
            result.count += 1
            result.headings[identifier] = source.get("heading_path") or []
            result.current_evidence &= (source.get("publication_evidence") or {}).get(
                "index", {}
            ).get("index_uuid") == index_uuid
            if retain_hits:
                result.full_hits.append(hit)
            if evidence_index is not None and (
                not qualified_only
                or (source.get("publication_evidence") or {})
                .get("index", {})
                .get("index_uuid")
                == index_uuid
            ):
                result.evidence.extend(
                    read_file_inventory(
                        client,
                        evidence_index,
                        "public",
                        file_id,
                        hits=[hit],
                        verify_index=False,
                    )
                )
            if identifier not in excluded_ids:
                yield {
                    key: value
                    for key, value in source.items()
                    if not key.startswith("publication_")
                }

    result.source_sha256 = publication_list_digest(sources())
    return result
