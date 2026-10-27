"""Corpus-scale callers resolve review identity once, including a negative lookup."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy.orm import Session

from onyx.db.document_intake_review import (
    file_attachment_relations,
    parent_labeling_contexts,
)


def test_large_ordinary_label_scope_performs_one_negative_identity_query() -> None:
    session = MagicMock(spec=Session)
    session.execute.return_value = []
    assert parent_labeling_contexts(session, [uuid4() for _ in range(12_000)]) == {}
    session.execute.assert_called_once()
    session.get.assert_not_called()
    session.scalars.assert_not_called()


def test_empty_label_page_does_not_query_review_identity() -> None:
    session = MagicMock(spec=Session)
    assert parent_labeling_contexts(session, []) == {}
    session.execute.assert_not_called()


def test_parent_advisory_contains_identity_and_review_reason_without_operative_content() -> (
    None
):
    session = MagicMock(spec=Session)
    child_id, parent_id, decision_id = uuid4(), uuid4(), uuid4()
    session.execute.return_value = [
        (
            child_id,
            SimpleNamespace(
                id=decision_id,
                existing_parent_file_id=parent_id,
                existing_parent_snapshot={
                    "display_name": "Başvurular Hakkında Tebliğ",
                    "source_url": "https://example.gov/main",
                    "canonical_sha256": "a" * 64,
                    "publication_epoch": 25,
                    "text": "Current amended operative text must never become annex source evidence",
                    "validity_start_date": "2030-01-01",
                },
                reason="Original annex identifies this existing legal instrument",
            ),
        )
    ]
    context = parent_labeling_contexts(cast(Session, session), [child_id])[child_id]
    assert context["decision_id"] == str(decision_id)
    assert context["user_file_id"] == str(parent_id)
    assert context["display_name"] == "Başvurular Hakkında Tebliğ"
    assert (
        context["reviewer_reason"]
        == "Original annex identifies this existing legal instrument"
    )
    assert "text" not in context and "validity_start_date" not in context
    assert (
        context["authority"]
        == "reviewed_parent_identity_only_not_operative_text_or_validity_evidence"
    )
    session.execute.assert_called_once()


def test_unrelated_legacy_file_navigation_uses_one_negative_relation_query() -> None:
    session = MagicMock(spec=Session)
    session.execute.return_value = []
    assert file_attachment_relations(session, 15, uuid4()) == {
        "existing_parents": [],
        "attachments": [],
    }
    session.execute.assert_called_once()
    session.get.assert_not_called()
    session.scalars.assert_not_called()
