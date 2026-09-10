from datetime import date
from hashlib import sha256
from importlib import import_module
from uuid import uuid4

import pytest

from onyx.regulatory.amendments.annexes.models import (
    AnnexAfterWindowAuthority,
    AnnexCanonicalSnapshot,
    AnnexChangeDraft,
    AnnexChangeItemDraft,
)
from onyx.regulatory.amendments.models import DateResolution


def snapshot(
    identifier: str, start: date | None, end: date | None
) -> AnnexCanonicalSnapshot:
    return AnnexCanonicalSnapshot(
        id=identifier,
        user_file_id=str(FILE_ID),
        chunk_type="appendix",
        status="active",
        projection_ordinal=0,
        supersedes_chunk_id=None,
        superseded_by_chunk_id=None,
        position=0,
        text=identifier,
        heading_path=["EK-1"],
        metadata={},
        source="indexed",
        validity_start_date=start,
        validity_end_date=end,
    )


FILE_ID = uuid4()


def draft(*, temporary: bool = True, scheduled: bool = False) -> AnnexChangeDraft:
    old = snapshot("old", date(2020, 1, 1), date(2027, 1, 1) if scheduled else None)
    rows = [old]
    if scheduled:
        rows[0] = old.model_copy(update={"superseded_by_chunk_id": "scheduled"})
        rows.append(
            snapshot("scheduled", date(2027, 1, 1), None).model_copy(
                update={"supersedes_chunk_id": "old", "projection_ordinal": 1}
            )
        )
    new = snapshot("new", date(2026, 1, 1), old.validity_end_date).model_copy(
        update={"supersedes_chunk_id": "old", "projection_ordinal": 2}
    )
    return AnnexChangeDraft(
        after_window_authority=AnnexAfterWindowAuthority(
            kind="restore_predecessor",
            effective_date=date(2026, 7, 1),
            source_text_sha256=sha256(
                "EK-1 için 01.07.2026 tarihinde önceki hükümler yeniden uygulanır.".encode()
            ).hexdigest(),
            source_start=0,
            source_end=len(
                "EK-1 için 01.07.2026 tarihinde önceki hükümler yeniden uygulanır."
            ),
            source_quote="EK-1 için 01.07.2026 tarihinde önceki hükümler yeniden uygulanır.",
            predecessor_ids=["old"],
            successor_ids=[],
        )
        if temporary
        else None,
        submitted_source_text="EK-1 için 01.07.2026 tarihinde önceki hükümler yeniden uygulanır.",
        instruction_indices=[0],
        instruction_texts=["temporary change"],
        annex_label="ek:1",
        user_file_id=FILE_ID,
        effective_date=date(2026, 1, 1),
        baseline_scope=rows,
        date_resolution=DateResolution(
            effective_start_date="2026-01-01",
            effective_end_date="2026-07-01" if temporary else None,
            rationale="explicit source window",
        ),
        items=[
            AnnexChangeItemDraft(
                operation="replace",
                old_chunk_ids=["old"],
                new_chunks=[new],
                old_positions=[0],
                new_positions=[0],
            )
        ],
    )


def planner():
    try:
        module = import_module("onyx.regulatory.amendments.annexes.publication")
    except ModuleNotFoundError:
        pytest.fail("annex temporal publication planner is missing")
    return module.prepare_legal_publication_timeline


def test_temporary_window_restores_prior_law_and_preserves_scheduled_successor():
    prepared = planner()(draft(scheduled=True))
    rows = {row.id: row for row in prepared.canonical_rows}
    assert rows["old"].validity_end_date == date(2026, 1, 1)
    assert rows["new"].validity_end_date == date(2026, 7, 1)
    assert rows["scheduled"] == draft(scheduled=True).baseline_scope[1]
    restored = [
        row for row in rows.values() if row.id not in ("old", "new", "scheduled")
    ]
    assert len(restored) == 1
    assert (
        restored[0].text,
        restored[0].validity_start_date,
        restored[0].validity_end_date,
    ) == ("old", date(2026, 7, 1), date(2027, 1, 1))
    assert prepared == planner()(draft(scheduled=True))
    assert prepared.restoration_predecessors == {restored[0].id: "old"}


def test_permanent_change_ends_at_already_scheduled_successor():
    prepared = planner()(draft(temporary=False, scheduled=True))
    assert len(prepared.canonical_rows) == 3
    assert next(
        row for row in prepared.canonical_rows if row.id == "new"
    ).validity_end_date == date(2027, 1, 1)


def test_source_only_keeps_legal_rows_identical():
    source = draft().model_copy(
        update={"items": [], "source_only_canonical_ids": ["old"]}
    )
    assert planner()(source).canonical_rows == source.baseline_scope


def test_ambiguous_scheduled_overlap_is_rejected_before_publication():
    source = draft(scheduled=True)
    source = source.model_copy(
        update={
            "baseline_scope": [
                source.baseline_scope[0],
                source.baseline_scope[1].model_copy(
                    update={"validity_start_date": date(2026, 6, 1)}
                ),
            ]
        }
    )
    with pytest.raises(ValueError, match="scheduled"):
        planner()(source)


def test_temporary_end_date_alone_is_not_restoration_authority():
    source = draft().model_copy(update={"after_window_authority": None})
    with pytest.raises(ValueError, match="after-window authority"):
        planner()(source)


@pytest.mark.parametrize(
    "quote",
    [
        "EK-2 için 01.07.2026 tarihinde önceki hükümler yeniden uygulanır.",
        "EK-1 için 01.08.2026 tarihinde önceki hükümler yeniden uygulanır.",
        "EK-1 için 01.07.2026 tarihinde önceki hükümler yeniden uygulanmayacaktır.",
    ],
)
def test_after_window_authority_requires_exact_scope_date_and_noncontradiction(
    quote: str,
) -> None:
    source = draft()
    assert source.after_window_authority is not None
    authority = source.after_window_authority.model_copy(
        update={
            "source_quote": quote,
            "source_start": 0,
            "source_end": len(quote),
            "source_text_sha256": sha256(quote.encode()).hexdigest(),
        }
    )
    source = source.model_copy(
        update={"submitted_source_text": quote, "after_window_authority": authority}
    )
    with pytest.raises(ValueError, match="after-window"):
        planner()(source)


def test_explicit_cessation_does_not_restore_old_law() -> None:
    from onyx.regulatory.amendments.annexes.publication import (
        resolve_after_window_authority,
    )

    source = draft().model_copy(
        update={
            "submitted_source_text": "EK-1 01.07.2026 tarihinde yürürlükten kalkar.",
            "after_window_authority": None,
        }
    )
    resolved = resolve_after_window_authority(source)
    assert resolved.after_window_authority is not None
    assert resolved.after_window_authority.kind == "cessation"
    timeline = planner()(resolved)
    assert not timeline.restoration_predecessors
    assert all(
        row.validity_end_date <= date(2026, 7, 1) for row in timeline.canonical_rows
    )


def test_exact_scheduled_successor_is_the_after_window_authority() -> None:
    from onyx.regulatory.amendments.annexes.publication import (
        resolve_after_window_authority,
    )

    source = draft(scheduled=True).model_copy(
        update={
            "after_window_authority": None,
            "submitted_source_text": "temporary",
            "date_resolution": DateResolution(
                effective_start_date="2026-01-01",
                effective_end_date="2027-01-01",
                rationale="source",
            ),
        }
    )
    resolved = resolve_after_window_authority(source)
    assert resolved.after_window_authority is not None
    assert resolved.after_window_authority.kind == "scheduled_successor"
    timeline = planner()(resolved)
    assert not timeline.restoration_predecessors
    assert (
        next(row for row in timeline.canonical_rows if row.id == "scheduled")
        == source.baseline_scope[1]
    )
