from datetime import date

import pytest

from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeItemDraft,
    PreparedContextView,
)
from onyx.regulatory.amendments.annexes.selective_impact import (
    dependency_impact,
    local_source_rows,
)
from tests.unit.onyx.regulatory.annexes.test_publication_timeline import draft, snapshot


def test_reverse_transitive_membership_deduplicates_shared_consumers() -> None:
    old, new, untouched = [snapshot(i, None, None) for i in ("old", "new", "untouched")]
    parent = snapshot("parent", None, None).model_copy(
        update={"metadata": {"source_regulatory_chunk_ids": ["old", "untouched"]}}
    )
    grandparent = snapshot("grandparent", None, None).model_copy(
        update={"metadata": {"source_regulatory_chunk_ids": ["parent", "old"]}}
    )
    changed_parent = parent.model_copy(
        update={"metadata": {"source_regulatory_chunk_ids": ["new", "untouched"]}}
    )
    changed_grandparent = grandparent.model_copy(
        update={"metadata": {"source_regulatory_chunk_ids": ["parent", "new"]}}
    )
    item = AnnexChangeItemDraft(
        operation="replace",
        old_chunk_ids=["old"],
        new_chunks=[new],
        old_positions=[0],
        new_positions=[0],
    )
    impact = dependency_impact(
        before=[old, untouched, parent, grandparent],
        after=[new, untouched, changed_parent, changed_grandparent],
        items=[item],
        contexts=PreparedContextView(),
        contextual_ids=set(),
    )
    assert impact.affected_ids == ["grandparent", "new", "old", "parent"]
    assert impact.unchanged_ids == ["untouched"]
    assert not impact.unresolved
    assert {
        r.id
        for r in local_source_rows(
            [new, untouched, changed_parent, changed_grandparent], "grandparent"
        )
    } == {"new", "untouched", "parent", "grandparent"}


def test_unknown_context_is_not_unchanged_and_no_generation_is_attempted() -> None:
    before = snapshot("old", None, None)
    after = snapshot("new", None, None)
    other = snapshot("legacy", None, None)
    impact = dependency_impact(
        before=[before, other],
        after=[after, other],
        items=[
            AnnexChangeItemDraft(
                operation="replace",
                old_chunk_ids=["old"],
                new_chunks=[after],
                old_positions=[0],
                new_positions=[0],
            )
        ],
        contexts=PreparedContextView(),
        contextual_ids={"legacy"},
    )
    assert impact.unresolved == {"legacy": ["context source ranges unavailable"]}
    assert "legacy" not in impact.unchanged_ids


def test_inserted_window_membership_is_detected_without_an_old_id() -> None:
    before = snapshot("aggregate", None, None)
    inserted = snapshot("inserted", None, None)
    after = before.model_copy(
        update={"metadata": {"source_regulatory_chunk_ids": ["inserted"]}}
    )
    impact = dependency_impact(
        before=[before],
        after=[after, inserted],
        items=[
            AnnexChangeItemDraft(
                operation="insert",
                old_chunk_ids=[],
                new_chunks=[inserted],
                old_positions=[],
                new_positions=[0],
            )
        ],
        contexts=PreparedContextView(),
        contextual_ids=set(),
    )
    assert impact.affected_ids == ["aggregate", "inserted"]


def test_scope_mismatch_rejected() -> None:
    old = snapshot("old", None, None)
    with pytest.raises(ValueError, match="file scope"):
        dependency_impact(
            before=[old],
            after=[old.model_copy(update={"user_file_id": "other"})],
            items=[],
            contexts=PreparedContextView(),
            contextual_ids=set(),
        )


def test_legacy_review_roundtrip_keeps_its_frozen_payload_shape() -> None:
    original = draft(temporary=False)
    payload = original.model_dump(mode="json")
    assert "impact_strategy" not in payload
    assert "selection_parent_id" not in payload
    assert type(original).model_validate(payload).model_dump(mode="json") == payload
    assert original.effective_date == date(2026, 1, 1)


def test_atomic_units_join_remove_and_insert_but_not_shared_consumers() -> None:
    from onyx.regulatory.amendments.annexes.selective_impact import review_units

    removal = AnnexChangeItemDraft(
        operation="remove",
        old_chunk_ids=["old"],
        new_chunks=[],
        old_positions=[0],
        new_positions=[],
    )
    insertion = AnnexChangeItemDraft(
        operation="insert",
        old_chunk_ids=[],
        new_chunks=[snapshot("new", None, None)],
        old_positions=[],
        new_positions=[0],
        insertion_after_chunk_id="old",
    )
    separate = AnnexChangeItemDraft(
        operation="replace",
        old_chunk_ids=["other"],
        new_chunks=[snapshot("new-other", None, None)],
        old_positions=[1],
        new_positions=[1],
    )
    assert review_units([removal, insertion, separate]) == [[0, 1], [2]]
    shared = separate.model_copy(update={"new_positions": [0]})
    assert review_units([removal, insertion, shared]) == [[0, 1, 2]]


def test_position_views_keep_same_day_and_historical_order_separate() -> None:
    from onyx.db.regulatory_annex_publication import effective_positions
    from onyx.regulatory.amendments.annexes.models import AnnexPositionView

    first, second = date(2026, 1, 1), date(2026, 2, 1)
    views = [
        AnnexPositionView(
            effective_start=None,
            effective_end=first,
            positions={"old": 0, "unchanged": 1},
        ),
        AnnexPositionView(
            effective_start=first,
            effective_end=None,
            positions={"new": 0, "unchanged": 2},
        ),
        AnnexPositionView(
            effective_start=first,
            effective_end=second,
            positions={"new": 0, "unchanged": 2},
        ),
        AnnexPositionView(
            effective_start=second,
            effective_end=None,
            positions={"new": 0, "unchanged": 3},
        ),
    ]
    assert effective_positions(views, date(2025, 12, 31)) == {"old": 0, "unchanged": 1}
    assert effective_positions(views, first)["unchanged"] == 2
    assert effective_positions(views, second)["unchanged"] == 3


def test_legacy_membership_recovery_requires_exact_unique_same_version_sources() -> (
    None
):
    from onyx.regulatory.amendments.annexes.selective_impact import (
        recover_source_membership,
    )

    a, b = [
        snapshot(i, None, None).model_copy(
            update={"text": i, "heading_path": ["Root", i], "position": pos}
        )
        for pos, i in enumerate(("a", "b"))
    ]
    aggregate = snapshot("aggregate", None, None).model_copy(
        update={
            "text": "Root\n\na\n\nb",
            "metadata": {
                "chunk_variant": "hierarchical_aggregate",
                "hierarchy_root_path": ["Root"],
            },
        }
    )
    assert recover_source_membership([a, b, aggregate]) == {"aggregate": ["a", "b"]}
    for invalid in (["deleted-a", "deleted-b"], ["a"]):
        broken = aggregate.model_copy(
            update={
                "metadata": {
                    **aggregate.metadata,
                    "source_regulatory_chunk_ids": invalid,
                }
            }
        )
        assert recover_source_membership([a, b, broken]) == {"aggregate": ["a", "b"]}
    assert not recover_source_membership(
        [a, b.model_copy(update={"user_file_id": "other"}), aggregate]
    )
    assert not recover_source_membership(
        [a, b.model_copy(update={"validity_start_date": date(2026, 1, 1)}), aggregate]
    )
    duplicate = a.model_copy(update={"id": "duplicate", "position": 2})
    one = aggregate.model_copy(update={"text": "Root\n\na"})
    assert not recover_source_membership([a, duplicate, one])


@pytest.mark.parametrize("caption", [False, True])
def test_image_source_recovery_requires_exact_text_and_recorded_split_lineage(
    caption: bool,
) -> None:
    from onyx.regulatory.amendments.annexes.selective_impact import (
        recover_source_membership,
    )

    first = snapshot("part-1", None, None).model_copy(
        update={
            "text": "first part",
            "metadata": {
                "oversized_split": {
                    "source_chunk_id": "old-parent",
                    "part": 1,
                    "parts": 2,
                }
            },
        }
    )
    second = first.model_copy(
        update={
            "id": "part-2",
            "text": "second part",
            "metadata": {
                "oversized_split": {
                    "source_chunk_id": "old-parent",
                    "part": 2,
                    "parts": 2,
                },
            },
        }
    )
    image = first.model_copy(
        update={
            "id": "image",
            "metadata": {
                "chunk_variant": "image_companion",
                "bound_to_regulatory_chunk_id": "old-parent",
                "image_file_id": "asset-1",
                "image_alt": "stored caption",
            },
            "text": first.text + ("\n\n[Görsel: stored caption]" if caption else ""),
        }
    )
    assert recover_source_membership([first, second, image]) == {"image": ["part-1"]}
    assert recover_source_membership([second, image]) == {}
    assert (
        recover_source_membership(
            [first, second, first.model_copy(update={"id": "duplicate"}), image]
        )
        == {}
    )
    assert (
        recover_source_membership(
            [first.model_copy(update={"metadata": {}}), second, image]
        )
        == {}
    )


def test_final_ingest_source_integrity_rejects_dangling_and_wrong_aggregate_text() -> (
    None
):
    from onyx.regulatory.amendments.annexes.selective_impact import (
        validate_canonical_source_integrity,
    )
    from onyx.regulatory.chunker import hierarchical_aggregate_text

    row = snapshot("source", None, None)
    aggregate = row.model_copy(
        update={
            "id": "aggregate",
            "text": hierarchical_aggregate_text("EK-1", [row.text]),
            "metadata": {
                "chunk_variant": "hierarchical_aggregate",
                "hierarchy_root_path": ["EK-1"],
                "source_regulatory_chunk_ids": [row.id],
            },
        }
    )
    validate_canonical_source_integrity([row, aggregate])
    with pytest.raises(ValueError, match="membership"):
        validate_canonical_source_integrity(
            [row, aggregate.model_copy(update={"text": "wrong text"})]
        )
    with pytest.raises(ValueError, match="membership"):
        validate_canonical_source_integrity([aggregate])
