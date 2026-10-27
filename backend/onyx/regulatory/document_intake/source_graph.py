"""Explicit ownership for ZIP documents, separate from their navigation links."""

from collections.abc import Collection, Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from onyx.regulatory.amendments.annexes.models import SourceLink

ZIP_INPUT_KIND = "legislation_zip_v1"
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
    if input_kind != ZIP_INPUT_KIND:
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
