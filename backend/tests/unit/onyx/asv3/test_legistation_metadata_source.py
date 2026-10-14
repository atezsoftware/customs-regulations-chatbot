"""File descriptions remain optional and behind existing source fences."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus as corpus
from onyx.db.models import User
from onyx.regulatory.legistation_metadata import (
    LegistationMetadata,
    parse_legistation_metadata,
)


@pytest.mark.parametrize(
    "value",
    [None, {}, {"other": "unrelated storage field"}, "bad", [], {"validity_status": 7}],
)
def test_absent_or_malformed_annotations_do_not_break_reads(value: object) -> None:
    assert parse_legistation_metadata(value) is None


def test_selected_fields_are_lossless_and_optional() -> None:
    fields = {
        "document_type": "Karar",
        "subject": "İthalat",
        "official_title": "9366 sayılı Karar",
        "official_gazette_date": "31.12.2024",
        "validity_source": None,
        "validity_status": "Mülga",
        "repeal_note": "Repeal by 9515 (from 01.02.2025)",
    }
    metadata = parse_legistation_metadata({**fields, "other": "not imported"})
    assert metadata is not None
    assert metadata.model_dump() == fields
    assert LegistationMetadata(validity_status="Mülga").repeal_note is None


def test_annotation_is_loaded_in_existing_fenced_select_not_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eligible, denied, unpublished, absent = (uuid4() for _ in range(4))
    annotation = {
        "validity_status": "Mülga",
        "repeal_note": "Repeal by 9515 (from 01.02.2025)",
    }
    records = [
        SimpleNamespace(
            id=identifier,
            name=f"folder/{identifier}.md",
            file_id=f"physical-{identifier}",
            legistation_metadata=annotation if identifier != absent else None,
        )
        for identifier in (eligible, denied, unpublished, absent)
    ]
    session = MagicMock()
    session.execute.return_value.all.return_value = records
    monkeypatch.setattr(corpus, "_validate_filters", MagicMock())
    monkeypatch.setattr(corpus, "observe_publication_read", lambda: object())
    monkeypatch.setattr(corpus, "get_acl_for_user", lambda *_: {"owner"})
    access = {
        str(row.id): SimpleNamespace(
            to_acl=lambda identifier=row.id: (
                {"other"} if identifier == denied else {"owner"}
            )
        )
        for row in records
    }
    monkeypatch.setattr(corpus, "get_access_for_user_files", lambda *_: access)
    monkeypatch.setattr(
        corpus,
        "filter_publication_read",
        lambda _observation, sources, _key: [
            source for source in sources if source.id != unpublished
        ],
    )
    parser = MagicMock(wraps=parse_legistation_metadata)
    monkeypatch.setattr(corpus, "parse_legistation_metadata", parser)
    sources, more = corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=IndexFilters(access_control_list=[]),
    )
    assert not more
    assert [source.id for source in sources] == [eligible, absent]
    assert sources[0].legistation_metadata == LegistationMetadata(**annotation)
    assert sources[1] == corpus.CorpusSource(
        absent, f"folder/{absent}.md", f"physical-{absent}"
    )
    assert parser.call_count == 2
    session.execute.assert_called_once()
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert (
        "LEFT OUTER JOIN file_record ON file_record.file_id = user_file.file_id" in sql
    )
    assert "file_record.file_metadata" in sql
    session.add.assert_not_called()
    session.commit.assert_not_called()
