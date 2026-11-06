from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import AmendmentBatch, AmendmentProposal, RegulatoryChunk
from onyx.db.regulatory_amendments import approve_amendment_proposal
from onyx.document_index.publication_models import FileOwnership, PublicationScope

FILE_ID = UUID("00000000-0000-0000-0000-000000000123")
START = date(2025, 1, 1)
EFFECTIVE = date(2026, 10, 16)


def _snapshot(row: RegulatoryChunk) -> dict[str, Any]:
    return {
        "id": row.id,
        "user_file_id": str(FILE_ID),
        "position": row.position,
        "text": row.text,
        "chunk_type": row.chunk_type,
        "heading_path": list(row.heading_path),
        "metadata": deepcopy(row.chunk_metadata),
        "status": "active",
        "source": "indexed",
        "validity_start_date": "2025-01-01",
        "validity_end_date": None,
        "supersedes_chunk_id": None,
        "superseded_by_chunk_id": None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.fixture
def approval_case(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Session, AmendmentProposal, FileOwnership, list[RegulatoryChunk]]:
    from onyx.db import (
        regulatory_amendments,
        regulatory_annexes,
        regulatory_publication,
    )

    rows = [
        RegulatoryChunk(
            id=identifier,
            user_file_id=FILE_ID,
            text=body,
            position=position,
            projection_ordinal=position,
            chunk_type="free_text" if label is None else "clause",
            heading_path=["Source", "EK IV"] + ([body] if label else []),
            chunk_metadata={"appendix_label": "EK IV", "clause_label": label},
            status="active",
            source="indexed",
            validity_start_date=START,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        for identifier, position, body, label in [
            ("intro", 100, "Ölçümler standartlara göre yapılır.", None),
            ("a", 101, "a) Değişmeyen koşul.", "a"),
            ("c", 102, "c) Eski ölçüm yöntemi.", "c"),
            ("d", 103, "d) Eski koşul.", "d"),
        ]
    ]
    snapshots = {row.id: _snapshot(row) for row in rows}
    snapshots["intro"]["descendant_snapshots"] = [
        deepcopy(snapshots[row.id]) for row in rows[1:]
    ]
    changes = []
    for identifier, body, instruction in [
        (
            "intro",
            "Ölçümler standartlara göre yapılır. Kahve makineleri son döngüden sonra ölçülür.",
            "Ek IV'ün birinci cümlesinden sonra aşağıdaki cümle eklenmiştir: “Kahve makineleri son döngüden sonra ölçülür.”",
        ),
        (
            "c",
            "c) Yeni ölçüm yöntemi.",
            "Ek IV'ün (c) bendi aşağıdaki şekilde değiştirilmiştir: “c) Yeni ölçüm yöntemi.”",
        ),
        ("d", "d) (Mülga)", "Ek IV'ün (d) bendi yürürlükten kaldırılmıştır."),
    ]:
        snapshot = snapshots[identifier]
        changes.append(
            {
                "old_chunk_id": identifier,
                "old_chunk_snapshot": deepcopy(snapshot),
                "instruction_texts": [instruction],
                "new_chunk_draft": {
                    "user_file_id": str(FILE_ID),
                    "position": snapshot["position"],
                    "text": body,
                    "chunk_type": snapshot["chunk_type"],
                    "heading_path": snapshot["heading_path"],
                    "metadata": snapshot["metadata"],
                    "effective_start_date": "2026-10-16",
                    "effective_end_date": None,
                },
            }
        )
    proposal = AmendmentProposal(
        id=42,
        batch_id=10,
        instruction_index=0,
        instruction_text=changes[0]["instruction_texts"][0],
        status="approving",
        chunk_changes=changes,
        old_chunk_snapshot=deepcopy(snapshots["intro"]),
        new_chunk_draft=deepcopy(changes[0]["new_chunk_draft"]),
    )
    session = MagicMock(spec=Session)

    def select_one(statement: Any) -> AmendmentProposal | AmendmentBatch | int:
        if list(statement.selected_columns.keys()) == ["batch_id"]:
            return 10
        if statement.column_descriptions[0]["entity"] is AmendmentBatch:
            return AmendmentBatch(id=10)
        return proposal

    session.scalar.side_effect = select_one

    def select_rows(statement: Any) -> MagicMock:
        params = statement.compile().params
        identifier = params["id_1"]
        if isinstance(identifier, list):
            result = [row for row in rows if row.id in identifier]
        else:
            parent = next(row for row in rows if row.id == identifier)
            result = [
                row
                for row in rows
                if row.user_file_id == parent.user_file_id
                and row.status == "active"
                and row.id != parent.id
                and row.position > parent.position
            ]
        scalar_result = MagicMock()
        scalar_result.all.return_value = sorted(result, key=lambda row: row.position)
        scalar_result.__iter__.side_effect = lambda: iter(result)
        return scalar_result

    session.scalars.side_effect = select_rows
    session.get.side_effect = lambda _model, identifier: next(
        (row for row in rows if row.id == identifier), None
    )
    session.add.side_effect = lambda row: (
        rows.append(row)
        if isinstance(row, RegulatoryChunk) and row not in rows
        else None
    )
    store = MagicMock()
    store.allocate_in_session.side_effect = range(1000, 1100)
    monkeypatch.setattr(
        regulatory_publication, "PublicationStore", lambda _scope: store
    )
    monkeypatch.setattr(
        regulatory_annexes, "copy_annex_chunk_links", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        regulatory_amendments,
        "supersede_hierarchical_aggregates_referencing_chunk",
        lambda *_a, **_kw: None,
    )
    owner = FileOwnership(
        scope=PublicationScope(
            tenant_id="public", environment="dev", database_identity="test-db"
        ),
        user_file_id=FILE_ID,
        owner_id=FILE_ID,
        fencing_token=1,
        expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    return session, proposal, owner, rows


@pytest.mark.parametrize("child_first", [False, True])
def test_partial_parent_and_children_version_only_reviewed_targets(
    approval_case: tuple[
        Session, AmendmentProposal, FileOwnership, list[RegulatoryChunk]
    ],
    child_first: bool,
) -> None:
    session, proposal, owner, rows = approval_case
    if child_first:
        proposal.chunk_changes = list(reversed(proposal.chunk_changes))
    unchanged = _snapshot(rows[1])
    result = approve_amendment_proposal(session, proposal, publication_owner=owner)
    assert len(result.new_chunks) == 3
    assert {row.supersedes_chunk_id for row in result.new_chunks} == {"intro", "c", "d"}
    assert {row.supersedes_chunk_id: row.text for row in result.new_chunks} == {
        "intro": "Ölçümler standartlara göre yapılır. Kahve makineleri son döngüden sonra ölçülür.",
        "c": "c) Yeni ölçüm yöntemi.",
        "d": "d) (Mülga)",
    }
    for old in rows[:4]:
        if old.id == "a":
            assert _snapshot(old) == unchanged
            assert old.status == "active"
            assert old.validity_end_date is None
            assert old.superseded_by_chunk_id is None
        else:
            assert old.status == "superseded"
            assert old.validity_end_date == EFFECTIVE
            assert old.superseded_by_chunk_id in proposal.applied_new_chunk_ids
    assert all(row.validity_start_date == EFFECTIVE for row in result.new_chunks)
    resumed = approve_amendment_proposal(session, proposal, publication_owner=owner)
    assert [row.id for row in resumed.new_chunks] == [
        row.id for row in result.new_chunks
    ]
    assert len(rows) == 7


@pytest.mark.parametrize("change", ["added", "removed", "text", "metadata"])
def test_changed_descendant_scope_rejects_before_writing(
    approval_case: tuple[
        Session, AmendmentProposal, FileOwnership, list[RegulatoryChunk]
    ],
    change: str,
) -> None:
    session, proposal, owner, rows = approval_case
    if change == "added":
        extra = RegulatoryChunk(
            id="new-child",
            user_file_id=FILE_ID,
            position=104,
            text="e) New.",
            chunk_type="clause",
            heading_path=["Source", "EK IV", "e) New."],
            chunk_metadata={"appendix_label": "EK IV", "clause_label": "e"},
            status="active",
        )
        rows.append(extra)
    elif change == "removed":
        rows[1].status = "superseded"
    elif change == "text":
        rows[1].text = "a) Concurrent edit."
    else:
        rows[1].chunk_metadata = {**rows[1].chunk_metadata, "paragraph_no": "2"}
    with pytest.raises(ValueError):
        approve_amendment_proposal(session, proposal, publication_owner=owner)
    assert len(rows) == (5 if change == "added" else 4)
    assert rows[0].status == "active"
    assert rows[0].superseded_by_chunk_id is None
    assert not proposal.applied_new_chunk_ids


@pytest.mark.parametrize("change", ["missing_snapshot", "repeal", "appendix_identity"])
def test_parent_preservation_requires_reviewed_scope_and_stable_identity(
    approval_case: tuple[
        Session, AmendmentProposal, FileOwnership, list[RegulatoryChunk]
    ],
    change: str,
) -> None:
    session, proposal, owner, rows = approval_case
    changes = deepcopy(proposal.chunk_changes)
    parent = changes[0]
    if change == "missing_snapshot":
        parent["old_chunk_snapshot"].pop("descendant_snapshots")
    elif change == "repeal":
        parent["instruction_texts"] = ["Ek IV yürürlükten kaldırılmıştır."]
        parent["new_chunk_draft"]["text"] = "(Mülga)"
    else:
        parent["new_chunk_draft"]["metadata"]["appendix_label"] = "EK V"
    proposal.chunk_changes = changes
    with pytest.raises(ValueError):
        approve_amendment_proposal(session, proposal, publication_owner=owner)
    assert len(rows) == 4
    assert all(row.status == "active" for row in rows)
