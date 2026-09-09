from datetime import date

import pytest

from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalPatch,
    AnnexCanonicalSnapshot,
    AnnexPatchPlan,
)


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
