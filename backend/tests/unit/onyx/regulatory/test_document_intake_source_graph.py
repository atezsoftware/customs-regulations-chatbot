"""Ownership edges must be explicit and independently source-backed."""

from typing import Any

import pytest

from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.source_graph import (
    ZIP_INPUT_KIND,
    parse_intake_graph,
)

A, B, C = "a" * 64, "b" * 64, "c" * 64


def _link(parent: str, child: str) -> SourceLink:
    return SourceLink(
        parent_asset_hash=parent,
        target_asset_hash=child,
        source_field="zip:link",
        label="EK",
        kind="embedded",
    )


def test_explicit_forest_ignores_navigation_backlinks() -> None:
    graph = parse_intake_graph(
        {
            "intake_graph": {
                "version": 1,
                "root_hashes": [A, C],
                "primary_parent_by_hash": {B: A},
            }
        },
        input_kind=ZIP_INPUT_KIND,
        asset_hashes=[B, C, A],
        links=[_link(A, B), _link(B, A), _link(C, A)],
    )
    assert graph is not None
    assert graph.parents_for(A) == ()
    assert graph.parents_for(B) == (A,)
    assert graph.parents_for(C) == ()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate-root",
        "unknown-root",
        "missing-vertex",
        "root-with-parent",
        "unknown-parent",
        "self",
        "cycle",
        "unbacked",
        "version-bool",
        "extra",
    ],
)
def test_invalid_graph_is_rejected_without_guessing_a_parent(change: str) -> None:
    graph: dict[str, Any] = {
        "version": 1,
        "root_hashes": [A],
        "primary_parent_by_hash": {B: A, C: B},
    }
    links = [_link(A, B), _link(B, C), _link(C, B)]
    manifest: dict[str, Any] = {"intake_graph": graph}
    if change == "missing":
        manifest = {}
    elif change == "duplicate-root":
        graph["root_hashes"] = [A, A]
    elif change == "unknown-root":
        graph["root_hashes"] = ["d" * 64]
    elif change == "missing-vertex":
        graph["primary_parent_by_hash"] = {B: A}
    elif change == "root-with-parent":
        graph["primary_parent_by_hash"] = {A: B, B: A, C: B}
    elif change == "unknown-parent":
        graph["primary_parent_by_hash"] = {B: "d" * 64, C: B}
    elif change == "self":
        graph["primary_parent_by_hash"] = {B: B, C: B}
    elif change == "cycle":
        graph["primary_parent_by_hash"] = {B: C, C: B}
    elif change == "unbacked":
        links = []
    elif change == "version-bool":
        graph["version"] = True
    else:
        graph["unexpected"] = "authority"
    with pytest.raises(ValueError):
        parse_intake_graph(
            manifest, input_kind=ZIP_INPUT_KIND, asset_hashes=[A, B, C], links=links
        )


@pytest.mark.parametrize("kind", [None, "url", "legislation_markdown_v1"])
def test_existing_routes_have_no_new_graph_or_single_root_policy_change(
    kind: object,
) -> None:
    assert (
        parse_intake_graph(
            {}, input_kind=kind, asset_hashes=[A, B], links=[_link(A, B)]
        )
        is None
    )
    with pytest.raises(ValueError, match="ZIP input route"):
        parse_intake_graph(
            {
                "intake_graph": {
                    "version": 1,
                    "root_hashes": [A, B],
                    "primary_parent_by_hash": {},
                }
            },
            input_kind=kind,
            asset_hashes=[A, B],
            links=[],
        )
