"""Approved siblings can move reviewed boundaries without authorizing other drift."""

from datetime import date
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    DocumentSet,
    RegulatoryChunk,
    UserFile,
)
from onyx.db.regulatory_amendment_order import (
    load_amendment_order,
)
from onyx.db.regulatory_amendments import (
    approve_amendment_proposal,
    finalize_amendment_proposal_projection,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.regulatory.amendments.insertion_order import plan_insertion
from onyx.regulatory.amendments.pipeline import _chunk_to_review_dict
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
)

_START = date(2023, 2, 3)


class SiblingReview(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    file: UserFile
    batch: AmendmentBatch
    rows: dict[str, RegulatoryChunk]
    proposals: dict[str, AmendmentProposal]
    authority: OwnedAuthority


def _batch(session: Session, group: DocumentSet, file: UserFile) -> AmendmentBatch:
    batch = AmendmentBatch(
        document_set_id=group.id,
        raw_text="Reviewed amendments.",
        user_file_ids=[str(file.id)],
    )
    session.add(batch)
    session.flush()
    return batch


def _review(session: Session, monkeypatch: pytest.MonkeyPatch) -> SiblingReview:
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    rows: dict[str, RegulatoryChunk] = {}
    for position, (key, paragraph, clause, text) in enumerate(
        [
            ("p2", "2", None, "MADDE 6 - (2) Manevra kuralları."),
            ("d", "2", "d", "d) Eski gündüz manevrası kuralı."),
            ("g", "2", "g", "g) Korunan manevra kuralı."),
            ("p3", "3", None, "(3) Tren teşkili kuralları."),
            ("c", "3", "c", "c) Eski emniyet vagonu kuralı."),
            ("p4", "4", None, "(4) Korunan sevk kuralları."),
        ]
    ):
        row = _chunk(session, file, position, text)
        row.chunk_type = "clause" if clause else "paragraph"
        row.chunk_metadata = {"article_no": "6", "paragraph_no": paragraph}
        if clause:
            row.chunk_metadata = {**row.chunk_metadata, "clause_label": clause}
        row.heading_path = ["Yönetmelik", "MADDE 6", f"({paragraph})", text]
        row.validity_start_date = date(2015, 7, 16)
        rows[key] = row
    session.flush()
    batch = _batch(session, group, file)
    proposals: dict[str, AmendmentProposal] = {}
    operations = [
        ("repeal", "2", "d", "d) (Mülga:RG-3/2/2023-32093)", rows["d"]),
        ("ğ", "2", "ğ", "ğ) Yeni aydınlatma kuralı.", None),
        ("h", "2", "h", "h) Yeni zorunlu manevra kuralı.", None),
        ("replace", "3", "c", "c) Yeni emniyet vagonu kuralı.", rows["c"]),
        ("ç", "3", "ç", "ç) Yeni koruyucu mesafe kuralı.", None),
    ]
    for index, (key, paragraph, clause, text, old) in enumerate(operations):
        paragraph_word = "ikinci" if paragraph == "2" else "üçüncü"
        prefix = f"MADDE 1- Aynı Yönetmeliğin 6 ncı maddesinin {paragraph_word} "
        operation = "fıkrasına aşağıdaki bent eklenmiştir."
        if old is not None:
            command = (
                "yürürlükten kaldırılmıştır" if key == "repeal" else "değiştirilmiştir"
            )
            operation = f"fıkrasının ({clause}) bendi {command}."
        instruction = prefix + operation + f"\n“{text}”"
        draft: dict[str, Any] = {
            "user_file_id": str(file.id),
            "position": old.position if old is not None else 0,
            "text": text,
            "chunk_type": "clause",
            "heading_path": ["Yönetmelik", "MADDE 6", f"({paragraph})", text],
            "metadata": {
                "article_no": "6",
                "paragraph_no": paragraph,
                "clause_label": clause,
            },
            "effective_start_date": _START.isoformat(),
            "effective_end_date": None,
        }
        if old is None:
            order = plan_insertion(
                load_amendment_order(session, file.id),
                article_no="6",
                paragraph_no=paragraph,
                clause_label=clause,
            )
            draft.update(
                position=order.position, insertion_order=order.model_dump(mode="json")
            )
        proposal = AmendmentProposal(
            batch_id=batch.id,
            instruction_index=index,
            instruction_text=instruction,
            old_chunk_id=old.id if old is not None else None,
            old_chunk_snapshot=_chunk_to_review_dict(old) if old is not None else {},
            new_chunk_draft=draft,
            status="approving",
        )
        session.add(proposal)
        proposals[key] = proposal
    session.flush()
    authority = OwnedAuthority(file.id)
    monkeypatch.setattr(PublicationStore, "lock_owned_snapshot", lambda *_: None)
    monkeypatch.setattr(
        PublicationStore,
        "allocate_in_session",
        lambda _store, _session, owner, key: authority.allocate(owner, key),
    )
    return SiblingReview(
        file=file, batch=batch, rows=rows, proposals=proposals, authority=authority
    )


def _approve(session: Session, review: SiblingReview, key: str) -> RegulatoryChunk:
    proposal = review.proposals[key]
    result = approve_amendment_proposal(
        session, proposal, publication_owner=review.authority.owner
    )
    assert finalize_amendment_proposal_projection(
        session, proposal_id=proposal.id, succeeded=True
    )
    session.flush()
    return result.new_chunk


def test_sibling_approvals_rebase_additions_and_preserve_canonical_history(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = source_session
    review = _review(session, monkeypatch)
    original = {
        key: (row.id, row.text, row.projection_ordinal)
        for key, row in review.rows.items()
    }
    results = {
        key: _approve(session, review, key)
        for key in ("repeal", "ğ", "replace", "h", "ç")
    }
    session.expire_all()
    active = [
        row for row in load_amendment_order(session, review.file.id) if row.active
    ]
    expected = {**review.rows, **results}
    assert [row.id for row in active] == [
        expected[key].id
        for key in ("p2", "repeal", "g", "ğ", "h", "p3", "replace", "ç", "p4")
    ]
    assert [row.position for row in active] == list(range(len(active)))
    for key, row in review.rows.items():
        assert (row.id, row.text, row.projection_ordinal) == original[key]
        if key in {"d", "c"}:
            successor = results["repeal" if key == "d" else "replace"]
            assert row.status == "superseded" and row.validity_end_date == _START
            assert row.superseded_by_chunk_id == successor.id
            assert successor.supersedes_chunk_id == row.id
            assert successor.position == row.position
    for key in ("ğ", "h", "ç"):
        row = results[key]
        assert row.supersedes_chunk_id is None and row.validity_start_date == _START
        assert row.chunk_metadata["clause_label"] == key
        assert row.text == review.proposals[key].new_chunk_draft["text"]
    proposal = review.proposals["h"]
    proposal.status = "approving"
    resumed = approve_amendment_proposal(
        session, proposal, publication_owner=review.authority.owner
    )
    assert resumed.new_chunk.id == results["h"].id
    assert len(load_amendment_order(session, review.file.id)) == len(original) + len(
        results
    )
