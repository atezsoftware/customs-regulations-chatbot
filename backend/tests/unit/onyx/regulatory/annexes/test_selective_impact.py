from datetime import date
from hashlib import sha256

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


def test_recorded_source_identity_retains_root_member_after_position_shift() -> None:
    from onyx.regulatory.amendments.annexes.selective_impact import (
        recover_source_membership,
    )

    root = snapshot("root-current", None, None).model_copy(
        update={"text": "Root", "position": 11, "heading_path": ["Root"]}
    )
    child = root.model_copy(
        update={"id": "child-current", "text": "Child", "position": 12}
    )
    recorded = [
        "rc_" + sha256(f"{root.user_file_id}:{order}:{body}".encode()).hexdigest()[:40]
        for order, body in [(10, "Root"), (11, "Child")]
    ]
    aggregate = root.model_copy(
        update={
            "id": "aggregate",
            "text": "Root\n\nChild",
            "metadata": {
                "chunk_variant": "hierarchical_aggregate",
                "hierarchy_root_path": ["Root"],
                "source_chunk_orders": [10, 11],
                "source_regulatory_chunk_ids": recorded,
            },
        }
    )
    assert recover_source_membership([root, child, aggregate]) == {
        "aggregate": ["root-current", "child-current"]
    }
    duplicate = root.model_copy(update={"id": "ambiguous-root", "position": 13})
    duplicate_child = child.model_copy(update={"id": "ambiguous-child", "position": 14})
    assert not recover_source_membership(
        [root, child, duplicate, duplicate_child, aggregate]
    )
    # A repeated heading alone is not a second matching ordered source window.
    assert recover_source_membership([root, child, duplicate, aggregate]) == {
        "aggregate": ["root-current", "child-current"]
    }


@pytest.mark.parametrize(
    "parent_start,parent_end,valid",
    [
        (None, None, True),
        (date(2020, 1, 1), None, True),
        (None, date(2027, 1, 1), True),
        (date(2026, 1, 1), date(2026, 9, 9), True),
        (date(2026, 1, 2), None, False),
        (None, date(2026, 9, 8), False),
    ],
)
def test_historical_aggregate_requires_parent_covering_entire_legal_window(
    parent_start: date | None, parent_end: date | None, valid: bool
) -> None:
    from onyx.regulatory.amendments.annexes.selective_impact import (
        aggregate_membership_is_valid,
    )

    parent = snapshot("Root text", parent_start, parent_end)
    aggregate = parent.model_copy(
        update={
            "id": "historical-aggregate",
            "validity_start_date": date(2026, 1, 1),
            "validity_end_date": date(2026, 9, 9),
            "metadata": {
                "chunk_variant": "hierarchical_aggregate",
                "hierarchy_root_path": ["Root"],
                "source_regulatory_chunk_ids": [parent.id],
            },
        }
    )
    assert aggregate_membership_is_valid(aggregate, {parent.id: parent}) is valid


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
def test_image_source_recovery_uses_exact_text_and_available_split_lineage(
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
    assert recover_source_membership(
        [first.model_copy(update={"metadata": {}}), second, image]
    ) == {"image": ["part-1"]}
    assert recover_source_membership(
        [
            first,
            first.model_copy(update={"id": "unrecorded-copy", "metadata": {}}),
            image,
        ]
    ) == {"image": ["part-1"]}


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
