from datetime import date

import pytest

from onyx.regulatory.amendment_dependents import rebuild_amendment_dependents
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.selective_impact import source_ids
from tests.unit.onyx.regulatory.annexes.test_publication_timeline import snapshot


def aggregate(identifier: str, members: list[str]) -> AnnexCanonicalSnapshot:
    return snapshot(identifier, None, None).model_copy(
        update={
            "chunk_type": "hierarchical_aggregate",
            "metadata": {
                "chunk_variant": "hierarchical_aggregate",
                "hierarchy_root_path": ["EK-1"],
                "source_regulatory_chunk_ids": members,
            },
        }
    )


def test_derived_versions_follow_two_dates_and_transitive_sources_without_retroactivity() -> (
    None
):
    start, early, late = date(2020, 1, 1), date(2027, 1, 1), date(2028, 1, 1)
    a, b = snapshot("a", start, None), snapshot("b", start, None)
    parent = aggregate("parent", ["a", "b"])
    ancestor = aggregate("ancestor", ["parent"])
    untouched = snapshot("unrelated", None, None)
    before = [a, b, parent, ancestor, untouched]
    after = [
        a.model_copy(
            update={"validity_end_date": early, "superseded_by_chunk_id": "a2"}
        ),
        b.model_copy(
            update={"validity_end_date": late, "superseded_by_chunk_id": "b2"}
        ),
        parent.model_copy(update={"validity_end_date": early}),
        ancestor,
        untouched,
        snapshot("a2", early, None).model_copy(update={"supersedes_chunk_id": "a"}),
        snapshot("b2", late, None).model_copy(update={"supersedes_chunk_id": "b"}),
    ]
    result = rebuild_amendment_dependents(before, after)
    by_id = {r.id: r for r in result}
    assert by_id["parent"].validity_end_date == early
    assert by_id["ancestor"].validity_end_date == early
    assert by_id["unrelated"] == untouched
    versions = sorted(
        (r for r in result if r.supersedes_chunk_id == "parent"),
        key=lambda r: r.validity_start_date or date.min,
    )
    assert [(r.validity_start_date, r.validity_end_date) for r in versions] == [
        (early, late),
        (late, None),
    ]
    assert [r.metadata["source_regulatory_chunk_ids"] for r in versions] == [
        ["a2", "b"],
        ["a2", "b2"],
    ]
    parents = {r.id for r in versions}
    uppers = [r for r in result if r.supersedes_chunk_id == "ancestor"]
    assert len(uppers) == 2
    assert {source_ids(r)[0] for r in uppers} == parents
    assert rebuild_amendment_dependents(before, after) == result


def test_dependency_cycle_never_produces_a_partial_derived_result() -> None:
    source = snapshot("source", None, None)
    a, b = aggregate("a", ["source", "b"]), aggregate("b", ["a"])
    with pytest.raises(ValueError, match="cyclic"):
        rebuild_amendment_dependents(
            [source, a, b], [source.model_copy(update={"text": "new"}), a, b]
        )


def test_image_companion_rebinds_parent_and_preserves_image_caption() -> None:
    source = snapshot("source", None, None).model_copy(
        update={"text": "original provision"}
    )
    image = snapshot("image", None, None).model_copy(
        update={
            "text": "original provision\nImage caption",
            "metadata": {
                "chunk_variant": "image_companion",
                "bound_to_regulatory_chunk_id": "source",
            },
        }
    )
    boundary = date(2027, 1, 1)
    successor = snapshot("new", boundary, None).model_copy(
        update={"supersedes_chunk_id": "source"}
    )
    after = [
        source.model_copy(update={"validity_end_date": boundary}),
        image,
        successor,
    ]
    result = rebuild_amendment_dependents([source, image], after)
    updated = next(r for r in result if r.supersedes_chunk_id == "image")
    assert updated.text == "new\nImage caption"
    assert updated.metadata["bound_to_regulatory_chunk_id"] == "new"
    assert updated.validity_start_date == boundary


def test_recovered_membership_survives_history_split() -> None:
    from onyx.regulatory.amendment_projection_impact import analyze_amendment_impact
    from onyx.regulatory.chunker import hierarchical_aggregate_text

    a, b = snapshot("a", None, None), snapshot("b", None, None)
    parent = aggregate("parent", []).model_copy(
        update={"text": hierarchical_aggregate_text("EK-1", [a.text, b.text])}
    )
    boundary = date(2027, 1, 1)
    before = [a, b, parent]
    after = [
        a.model_copy(update={"validity_end_date": boundary}),
        b,
        parent,
        snapshot("a2", boundary, None).model_copy(update={"supersedes_chunk_id": "a"}),
    ]
    rebuilt = rebuild_amendment_dependents(before, after)
    historical = next(r for r in rebuilt if r.id == parent.id)
    assert historical.metadata["source_regulatory_chunk_ids"] == ["a", "b"]
    report = analyze_amendment_impact(
        before=before, after=rebuilt, bindings=[], llm=None
    )
    assert not report.unresolved


@pytest.mark.parametrize("related", [False, True])
def test_missing_source_only_blocks_a_consumer_reached_from_changed_source(
    related: bool,
) -> None:
    from onyx.regulatory.amendment_projection_impact import find_source_consumers

    source = snapshot("source", None, None)
    parent = aggregate("parent", ["missing", "source"] if related else ["missing"])
    before = [source, parent]
    after = [source.model_copy(update={"text": "changed"}), parent]
    if related:
        with pytest.raises(ValueError, match="source missing"):
            find_source_consumers(before, {source.id})
        with pytest.raises(ValueError, match="source missing"):
            rebuild_amendment_dependents(before, after)
    else:
        assert find_source_consumers(before, {source.id}) == []
        assert rebuild_amendment_dependents(before, after) == after


def test_temporary_source_absence_restores_derived_consumer_after_window() -> None:
    a, b = snapshot("a", None, None), snapshot("b", None, None)
    parent = aggregate("parent", ["a", "b"])
    early, late = date(2027, 1, 1), date(2028, 1, 1)
    restored = a.model_copy(
        update={
            "id": "restored",
            "validity_start_date": late,
            "supersedes_chunk_id": "a",
        }
    )
    result = rebuild_amendment_dependents(
        [a, b, parent],
        [a.model_copy(update={"validity_end_date": early}), b, parent, restored],
    )
    successors = sorted(
        (r for r in result if r.supersedes_chunk_id == parent.id),
        key=lambda r: r.validity_start_date or date.min,
    )
    assert [(r.validity_start_date, r.validity_end_date) for r in successors] == [
        (early, late),
        (late, None),
    ]
    assert [r.metadata["source_regulatory_chunk_ids"] for r in successors] == [
        ["b"],
        ["restored", "b"],
    ]
