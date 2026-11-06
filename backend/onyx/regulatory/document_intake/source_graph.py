"""Explicit ownership for ZIP documents, separate from their navigation links."""

from collections.abc import Collection, Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.markdown_draft_models import FrozenMarkdownDraft

ZIP_INPUT_KIND = "legislation_zip_v1"
MARKDOWN_DRAFT_INPUT_KIND = "legislation_markdown_draft_v1"
OWNERSHIP_INPUT_KINDS = {ZIP_INPUT_KIND, MARKDOWN_DRAFT_INPUT_KIND}
AssetHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class IntakeGraph(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: Literal[1]
    root_hashes: list[AssetHash] = Field(min_length=1)
    primary_parent_by_hash: dict[AssetHash, AssetHash]

    def parents_for(self, asset_hash: str) -> tuple[str, ...]:
        parent = self.primary_parent_by_hash.get(asset_hash)
        return (parent,) if parent is not None else ()

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("The ZIP intake graph version must be an integer")
        return value


def parse_intake_graph(
    manifest: Mapping[str, object],
    *,
    input_kind: object,
    asset_hashes: Collection[str],
    links: Sequence[SourceLink],
) -> IntakeGraph | None:
    """Require a complete source-backed forest only for the explicit ZIP route."""
    if input_kind not in OWNERSHIP_INPUT_KINDS:
        if "intake_graph" in manifest:
            raise ValueError("An intake ownership graph requires the ZIP input route")
        return None
    if "intake_graph" not in manifest:
        raise ValueError("The ZIP source manifest has no intake ownership graph")
    graph = IntakeGraph.model_validate(manifest["intake_graph"])
    hashes = set(asset_hashes)
    roots = set(graph.root_hashes)
    parents = graph.primary_parent_by_hash
    if (
        not hashes
        or len(roots) != len(graph.root_hashes)
        or roots.intersection(parents)
        or roots.union(parents) != hashes
        or any(parent not in hashes for parent in parents.values())
    ):
        raise ValueError("The ZIP intake graph must cover each document exactly once")
    if input_kind == MARKDOWN_DRAFT_INPUT_KIND:
        _require_markdown_upload_edges(manifest, hashes, parents, links)
    source_edges = {(link.parent_asset_hash, link.target_asset_hash) for link in links}
    for child, parent in parents.items():
        if child == parent or (parent, child) not in source_edges:
            raise ValueError("Every ZIP parent must have an actual source link")
        visited: set[str] = set()
        current = child
        while current not in roots:
            if current in visited or current not in parents:
                raise ValueError("The ZIP intake graph must be a rooted forest")
            visited.add(current)
            current = parents[current]
    return graph


def _require_markdown_upload_edges(
    manifest: Mapping[str, object],
    hashes: set[str],
    parents: dict[str, str],
    links: Sequence[SourceLink],
) -> None:
    uploaded = FrozenMarkdownDraft.model_validate(manifest.get("markdown_draft"))
    documents = uploaded.documents
    by_id = {row.id: row for row in documents}
    if (
        len(by_id) != len(documents)
        or {row.sha256 for row in documents} != hashes
        or len(documents) != len(hashes)
    ):
        raise ValueError(
            "The Markdown upload tree must cover every source exactly once"
        )
    expected_parents: dict[str, str] = {}
    for row in documents:
        parent_id = row.parent_document_id
        if parent_id is None:
            continue
        if parent_id not in by_id:
            raise ValueError("The Markdown attachment parent is missing")
        parent_hash = by_id[parent_id].sha256
        expected_parents[row.sha256] = parent_hash
        provenance = f"markdown-draft:{parent_id}:{row.id}"
        if not any(
            link.parent_asset_hash == parent_hash
            and link.target_asset_hash == row.sha256
            and link.kind == "embedded"
            and link.source_field == provenance
            and link.original_url is None
            and link.requested_url is None
            for link in links
        ):
            raise ValueError(
                "The Markdown attachment has no authenticated upload provenance"
            )
    if expected_parents != parents:
        raise ValueError(
            "The Markdown ownership graph differs from its frozen upload tree"
        )
