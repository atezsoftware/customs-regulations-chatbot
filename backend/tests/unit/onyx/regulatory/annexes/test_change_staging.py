from datetime import date
from typing import TYPE_CHECKING, Literal

import pytest

from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalPatch,
    AnnexCanonicalSnapshot,
    AnnexPatchPlan,
)

if TYPE_CHECKING:
    from onyx.db.models import RegulatoryChunk
    from onyx.regulatory.amendments.annexes.models import PreparedContextView


def baseline() -> list[AnnexCanonicalSnapshot]:
    return [
        AnnexCanonicalSnapshot(
            id="one",
            user_file_id="00000000-0000-0000-0000-000000000001",
            chunk_type="table",
            status="active",
            projection_ordinal=0,
            supersedes_chunk_id=None,
            superseded_by_chunk_id=None,
            position=0,
            text="old",
            heading_path=["EK-1"],
            metadata={},
            source="indexed",
            validity_start_date=None,
            validity_end_date=None,
        )
    ]


@pytest.mark.parametrize(
    "operation,new_text,expected_count",
    [("replace", "new", 1), ("insert", "added", 1), ("remove", None, 0)],
)
def test_staged_ids_are_frozen_without_empty_delete_chunks(
    operation: str, new_text: str | None, expected_count: int
) -> None:
    from onyx.regulatory.amendments.annexes.staging import stage_canonical_items

    rows = baseline()
    patch = AnnexCanonicalPatch.model_validate(
        dict(
            old_chunk_id=None if operation == "insert" else "one",
            old_text=None if operation == "insert" else "old",
            new_text=new_text,
            old_positions=[] if operation == "insert" else [0],
            new_positions=[] if operation == "remove" else [0],
            operation=operation,
        )
    )
    plan = AnnexPatchPlan(
        baseline_sha256="baseline",
        comparison_sha256="comparison",
        effective_date=date(2026, 9, 10),
        patches=[patch],
        direct_canonical_changes=[],
        metadata_only=[],
        retire_history=[],
        unchanged=[],
        issues=[],
        ready=True,
    )
    staged = stage_canonical_items(plan=plan, baseline_scope=rows)
    assert len(staged) == 1 and len(staged[0].new_chunks) == expected_count
    if staged[0].new_chunks:
        chunk = staged[0].new_chunks[0]
        assert chunk.id != rows[0].id
        assert chunk.text == new_text
        assert chunk.validity_start_date == date(2026, 9, 10)
    assert rows[0].text == "old" and rows[0].validity_end_date is None


def test_staging_rejects_invented_old_id() -> None:
    from onyx.regulatory.amendments.annexes.staging import stage_canonical_items

    patch = AnnexCanonicalPatch(
        old_chunk_id="foreign",
        old_text="old",
        new_text="new",
        old_positions=[0],
        new_positions=[0],
        operation="replace",
    )
    plan = AnnexPatchPlan(
        baseline_sha256="baseline",
        comparison_sha256="comparison",
        effective_date=date(2026, 9, 10),
        patches=[patch],
        direct_canonical_changes=[],
        metadata_only=[],
        retire_history=[],
        unchanged=[],
        issues=[],
        ready=True,
    )
    with pytest.raises(ValueError, match="scope"):
        stage_canonical_items(plan=plan, baseline_scope=baseline())


def test_insert_requires_explicit_anchor_in_multi_chunk_scope() -> None:
    from onyx.regulatory.amendments.annexes.staging import stage_canonical_items

    rows = baseline()
    rows.append(
        rows[0].model_copy(
            update={"id": "other", "position": 1, "projection_ordinal": 1}
        )
    )
    plan = AnnexPatchPlan(
        baseline_sha256="baseline",
        comparison_sha256="comparison",
        effective_date=date(2026, 9, 10),
        patches=[
            AnnexCanonicalPatch(
                old_chunk_id=None,
                old_text=None,
                new_text="insert",
                old_positions=[],
                new_positions=[1],
                operation="insert",
            )
        ],
        direct_canonical_changes=[],
        metadata_only=[],
        retire_history=[],
        unchanged=[],
        issues=[],
        ready=True,
    )
    with pytest.raises(ValueError, match="anchor"):
        stage_canonical_items(plan=plan, baseline_scope=rows)
    items = stage_canonical_items(
        plan=plan, baseline_scope=rows, insertion_after_chunk_id="one"
    )
    assert items[0].insertion_after_chunk_id == "one"
    assert items[0].new_chunks[0].heading_path == rows[0].heading_path


def test_physical_merge_stages_multi_chunk_lineage_as_one_item() -> None:
    from hashlib import sha256

    from onyx.regulatory.amendments.annexes.models import (
        AnnexComparison,
        AnnexCoverage,
        AnnexDifference,
        AnnexElementReference,
        AnnexLocator,
    )
    from onyx.regulatory.amendments.annexes.staging import stage_canonical_items

    rows = baseline()
    rows.append(
        rows[0].model_copy(
            update={
                "id": "two",
                "text": "second",
                "position": 1,
                "projection_ordinal": 1,
            }
        )
    )
    comparison = AnnexComparison(
        old_source_sha256="a",
        new_source_sha256="b",
        old_snapshot_sha256="old",
        new_snapshot_sha256="new",
        changes=[
            AnnexDifference(
                operation="merge",
                old=[
                    AnnexElementReference(
                        position=index, text=row.text, locator=AnnexLocator()
                    )
                    for index, row in enumerate(rows)
                ],
                new=[
                    AnnexElementReference(
                        position=0, text="combined", locator=AnnexLocator()
                    )
                ],
                explanation="merged rows",
            )
        ],
        coverage=AnnexCoverage(
            old_positions=[0, 1],
            new_positions=[0],
            old_pages=[],
            new_pages=[],
            method="native_structure",
        ),
        issues=[],
        ready=True,
    )
    plan = AnnexPatchPlan(
        baseline_sha256="baseline",
        comparison_sha256=sha256(comparison.model_dump_json().encode()).hexdigest(),
        effective_date=date(2026, 9, 10),
        patches=[
            AnnexCanonicalPatch(
                old_chunk_id="one",
                old_text="old",
                new_text="combined",
                old_positions=[0],
                new_positions=[0],
                operation="replace",
            ),
            AnnexCanonicalPatch(
                old_chunk_id="two",
                old_text="second",
                new_text=None,
                old_positions=[1],
                new_positions=[],
                operation="remove",
            ),
        ],
        direct_canonical_changes=["one", "two"],
        metadata_only=[],
        retire_history=["two"],
        unchanged=[],
        issues=[],
        ready=True,
    )
    items = stage_canonical_items(plan=plan, baseline_scope=rows, comparison=comparison)
    assert len(items) == 1 and items[0].operation == "merge"
    assert items[0].old_chunk_ids == ["one", "two"]
    assert [chunk.text for chunk in items[0].new_chunks] == ["combined"]


def test_staged_validation_binds_each_replacement_to_its_canonical_old_text() -> None:
    from onyx.regulatory.amendments.annexes.staging import (
        stage_canonical_items,
        validate_staged_items,
    )

    rows = baseline()
    rows.append(
        rows[0].model_copy(
            update={
                "id": "two",
                "text": "second",
                "position": 1,
                "projection_ordinal": 1,
            }
        )
    )
    plan = AnnexPatchPlan(
        baseline_sha256="baseline",
        comparison_sha256="comparison",
        effective_date=date(2026, 9, 10),
        patches=[
            AnnexCanonicalPatch(
                old_chunk_id=row.id,
                old_text=row.text,
                new_text=f"new {row.text}",
                old_positions=[index],
                new_positions=[index],
                operation="replace",
            )
            for index, row in enumerate(rows)
        ],
        direct_canonical_changes=[row.id for row in rows],
        metadata_only=[],
        retire_history=[],
        unchanged=[],
        issues=[],
        ready=True,
    )
    items = stage_canonical_items(plan=plan, baseline_scope=rows)
    validate_staged_items(plan=plan, baseline_scope=rows, items=items)
    swapped = [
        item.model_copy(update={"old_chunk_ids": items[1 - index].old_chunk_ids})
        for index, item in enumerate(items)
    ]
    with pytest.raises(ValueError, match="staged"):
        validate_staged_items(plan=plan, baseline_scope=rows, items=swapped)
    forged = plan.model_copy(
        update={
            "patches": [
                plan.patches[0].model_copy(update={"old_text": "forged"}),
                plan.patches[1],
            ]
        }
    )
    with pytest.raises(ValueError, match="scope"):
        validate_staged_items(plan=forged, baseline_scope=rows, items=items)


def _prepared_view(
    rows: list["RegulatoryChunk"],
    day: date,
    path: "Literal['normal', 'durable']" = "normal",
) -> "PreparedContextView":
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        canonical_dependency_ids,
        context_hash,
        effective_context_rows,
        freeze_context_source_snapshot,
    )
    from onyx.regulatory.amendments.annexes.models import (
        FrozenContextProjection,
        PreparedContextView,
    )
    from onyx.regulatory.indexing_jobs.contextual import _row_block
    from onyx.regulatory.projection import _row_context_text

    projections = []
    snapshots = {}
    for row in effective_context_rows(rows, day):
        snapshot = freeze_context_source_snapshot(
            rows=rows,
            target=row,
            reference_date=day,
            generation_path=path,
            row_text=_row_context_text if path == "normal" else _row_block,
        )
        snapshots[snapshot.sha256] = snapshot
        projections.append(
            FrozenContextProjection(
                canonical_chunk_id=row.id,
                source_snapshot_sha256=snapshot.sha256,
                generation_path=path,
                request_hashes=[],
                embedding_input_sha256=context_hash([row.text]),
                embedding_config_sha256=context_hash({"model": "fixture"}),
                embedding_config={"model": "fixture"},
                embedding_texts=[row.text],
                canonical_text_sha256=context_hash(row.text),
                metadata_sha256=context_hash(
                    [
                        row.chunk_metadata,
                        row.heading_path,
                        row.position,
                        row.validity_start_date,
                        row.validity_end_date,
                    ]
                ),
                validity_start=day,
                validity_end=row.validity_end_date,
                canonical_dependency_ids=canonical_dependency_ids(row),
            )
        )
    return PreparedContextView(
        projections=projections, snapshots=list(snapshots.values())
    )


@pytest.mark.parametrize("path", ["normal", "durable"])
def test_context_validation_distinguishes_omitted_consumers_from_true_empty_retirement(
    path: "Literal['normal', 'durable']",
) -> None:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
        validate_complete_context_view,
    )
    from onyx.regulatory.amendments.annexes.models import (
        AnnexChangeItemDraft,
        PreparedContextView,
    )
    from onyx.regulatory.amendments.annexes.staging import (
        canonical_snapshot_rows,
        prepare_staged_candidate_rows,
    )

    day = date(2026, 9, 10)
    scope = baseline()
    rows = canonical_snapshot_rows(scope)
    old = _prepared_view(rows, day, path)
    validate_complete_context_view(rows=rows, view=old, as_of_date=day)
    with pytest.raises(ValueError, match="coverage"):
        validate_complete_context_view(
            rows=rows, view=PreparedContextView(), as_of_date=day
        )
    omitted_range = old.model_copy(
        update={
            "snapshots": [old.snapshots[0].model_copy(update={"ordered_ranges": []})]
        }
    )
    with pytest.raises(ValueError, match="range"):
        validate_complete_context_view(rows=rows, view=omitted_range, as_of_date=day)
    removed = prepare_staged_candidate_rows(
        baseline_scope=scope,
        items=[
            AnnexChangeItemDraft(
                operation="remove",
                old_chunk_ids=[scope[0].id],
                new_chunks=[],
                old_positions=[0],
                new_positions=[],
            )
        ],
        effective_date=day,
    )
    validate_complete_context_view(
        rows=removed, view=PreparedContextView(), as_of_date=day
    )
    impact = compare_context_views(
        old=old, new=PreparedContextView(), direct_canonical_changes=[]
    )
    assert impact.ready and impact.retire_history == [scope[0].id]
    expired = canonical_snapshot_rows(
        [scope[0].model_copy(update={"validity_end_date": day})]
    )
    validate_complete_context_view(
        rows=expired, view=PreparedContextView(), as_of_date=day
    )


def test_complete_candidate_keeps_non_annex_consumers_and_rewrites_aggregate_sources() -> (
    None
):
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        validate_complete_context_view,
    )
    from onyx.regulatory.amendments.annexes.models import AnnexChangeItemDraft
    from onyx.regulatory.amendments.annexes.staging import prepare_staged_candidate_rows

    scope = baseline()
    scope.extend(
        [
            scope[0].model_copy(
                update={
                    "id": "outside",
                    "position": 1,
                    "projection_ordinal": 1,
                    "heading_path": ["Article 2"],
                    "text": "unchanged other provision",
                }
            ),
            scope[0].model_copy(
                update={
                    "id": "aggregate",
                    "position": 2,
                    "projection_ordinal": 2,
                    "text": "old aggregate",
                    "metadata": {
                        "chunk_variant": "hierarchical_aggregate",
                        "hierarchy_root_path": ["EK-1"],
                        "source_regulatory_chunk_ids": ["one"],
                    },
                }
            ),
        ]
    )
    candidate = scope[0].model_copy(
        update={
            "id": "prospective",
            "text": "new",
            "projection_ordinal": 3,
            "validity_start_date": date(2026, 9, 10),
        }
    )
    rows = prepare_staged_candidate_rows(
        baseline_scope=scope,
        items=[
            AnnexChangeItemDraft(
                operation="replace",
                old_chunk_ids=["one"],
                new_chunks=[candidate],
                old_positions=[0],
                new_positions=[0],
            )
        ],
        effective_date=date(2026, 9, 10),
    )
    aggregate = next(row for row in rows if row.id == "aggregate")
    assert aggregate.chunk_metadata["source_regulatory_chunk_ids"] == ["prospective"]
    assert "new" in aggregate.text and "old" not in aggregate.text
    assert scope[2].text == "old aggregate" and scope[0].validity_end_date is None
    prepared = _prepared_view(rows, date(2026, 9, 10))
    assert {projection.canonical_chunk_id for projection in prepared.projections} == {
        "prospective",
        "outside",
        "aggregate",
    }
    omitted = prepared.model_copy(
        update={
            "projections": [
                projection
                for projection in prepared.projections
                if projection.canonical_chunk_id != "outside"
            ]
        }
    )
    with pytest.raises(ValueError, match="coverage"):
        validate_complete_context_view(
            rows=rows, view=omitted, as_of_date=date(2026, 9, 10)
        )


def test_candidate_insertion_uses_explicit_anchor_without_mutating_baseline() -> None:
    from onyx.regulatory.amendments.annexes.models import AnnexChangeItemDraft
    from onyx.regulatory.amendments.annexes.staging import prepare_staged_candidate_rows

    scope = baseline()
    scope.append(
        scope[0].model_copy(
            update={"id": "next", "position": 1, "projection_ordinal": 1}
        )
    )
    inserted = scope[0].model_copy(update={"id": "inserted", "projection_ordinal": 2})
    rows = prepare_staged_candidate_rows(
        baseline_scope=scope,
        items=[
            AnnexChangeItemDraft(
                operation="insert",
                old_chunk_ids=[],
                new_chunks=[inserted],
                old_positions=[],
                new_positions=[1],
                insertion_after_chunk_id="one",
            )
        ],
        effective_date=date(2026, 9, 10),
    )
    assert [(row.id, row.position) for row in rows] == [
        ("one", 0),
        ("inserted", 1),
        ("next", 2),
    ]
    assert [row.position for row in scope] == [0, 1]


@pytest.mark.parametrize("depth", [1, 3])
def test_pure_delete_retires_empty_aggregate_chain_and_preserves_history(
    depth: int,
) -> None:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
        validate_complete_context_view,
    )
    from onyx.regulatory.amendments.annexes.models import AnnexChangeItemDraft
    from onyx.regulatory.amendments.annexes.staging import (
        canonical_snapshot_rows,
        prepare_staged_candidate_rows,
    )

    scope = baseline()
    for index in range(depth):
        scope.append(
            scope[0].model_copy(
                update={
                    "id": f"aggregate-{index}",
                    "position": index + 1,
                    "projection_ordinal": index + 1,
                    "text": f"historical aggregate {index}",
                    "metadata": {
                        "chunk_variant": "hierarchical_aggregate",
                        "hierarchy_root_path": ["EK-1"],
                        "source_regulatory_chunk_ids": [scope[-1].id],
                    },
                }
            )
        )
    day = date(2026, 9, 10)
    old = _prepared_view(canonical_snapshot_rows(scope), day)
    items = [
        AnnexChangeItemDraft(
            operation="remove",
            old_chunk_ids=["one"],
            new_chunks=[],
            old_positions=[0],
            new_positions=[],
        )
    ]
    rows = prepare_staged_candidate_rows(
        baseline_scope=scope, items=items, effective_date=day
    )
    assert len(rows) == len(scope)
    for original, row in zip(scope, rows, strict=True):
        assert row.id == original.id and row.text == original.text
        assert row.chunk_metadata == original.metadata
        assert row.status == "superseded" and row.validity_end_date == day
        assert original.status == "active" and original.validity_end_date is None
    prepared = _prepared_view(rows, day)
    assert not prepared.projections and not prepared.snapshots
    validate_complete_context_view(rows=rows, view=prepared, as_of_date=day)
    impact = compare_context_views(old=old, new=prepared, direct_canonical_changes=[])
    assert impact.ready and impact.retire_history == sorted(row.id for row in scope)


@pytest.mark.parametrize(
    "sources,root,error",
    [
        ([], ["EK-1"], "aggregate_provenance_unavailable"),
        (["unknown"], ["EK-1"], "aggregate_source_unavailable"),
        (["one", "unknown"], ["EK-1"], "aggregate_source_unavailable"),
        (["one"], [], "aggregate_provenance_unavailable"),
    ],
)
def test_aggregate_retirement_does_not_hide_unknown_provenance(
    sources: list[str], root: list[str], error: str
) -> None:
    from onyx.regulatory.amendments.annexes.models import AnnexChangeItemDraft
    from onyx.regulatory.amendments.annexes.staging import prepare_staged_candidate_rows

    scope = baseline()
    scope.append(
        scope[0].model_copy(
            update={
                "id": "aggregate",
                "position": 1,
                "projection_ordinal": 1,
                "metadata": {
                    "chunk_variant": "hierarchical_aggregate",
                    "hierarchy_root_path": root,
                    "source_regulatory_chunk_ids": sources,
                },
            }
        )
    )
    with pytest.raises(ValueError, match=error):
        prepare_staged_candidate_rows(
            baseline_scope=scope,
            items=[
                AnnexChangeItemDraft(
                    operation="remove",
                    old_chunk_ids=["one"],
                    new_chunks=[],
                    old_positions=[0],
                    new_positions=[],
                )
            ],
            effective_date=date(2026, 9, 10),
        )
