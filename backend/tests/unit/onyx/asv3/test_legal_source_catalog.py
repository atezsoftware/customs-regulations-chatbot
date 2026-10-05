"""Alternative title identities remain inside the existing catalogue access fence."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import operators

from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus
from onyx.db.models import User


def catalog_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    session = MagicMock()
    monkeypatch.setattr(asv3_corpus, "_validate_filters", lambda *_args: None)
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", lambda: object())
    monkeypatch.setattr(
        asv3_corpus, "get_acl_for_user", lambda *_args: {"user:allowed"}
    )
    return session


def test_alternative_queries_are_one_union_with_shared_acl_and_publication_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = catalog_session(monkeypatch)
    named, numbered, denied, unpublished = (uuid4() for _ in range(4))
    session.execute.return_value.all.return_value = [
        SimpleNamespace(
            id=identifier, name=f"Source {identifier}", file_id=str(identifier)
        )
        for identifier in (named, numbered, denied, unpublished)
    ]
    # Default arguments freeze each fake ACL row's identity.
    access = {
        str(identifier): SimpleNamespace(
            to_acl=lambda identifier=identifier: (
                {"user:allowed"} if identifier != denied else {"user:other"}
            )
        )
        for identifier in (named, numbered, denied, unpublished)
    }
    monkeypatch.setattr(asv3_corpus, "get_access_for_user_files", lambda *_args: access)
    publication = MagicMock(
        side_effect=lambda _read, rows, _key: [
            row for row in rows if row.id != unpublished
        ]
    )
    monkeypatch.setattr(asv3_corpus, "filter_publication_read", publication)
    results, more = asv3_corpus.find_related_sources(
        session,
        user=cast(User, object()),
        filters=IndexFilters(
            access_control_list=[],
            asv3_document_set_id=73,
            forced_document_set=[asv3_corpus.PC_CORPUS_NAME],
        ),
        query_variants=("Gümrük Kanunu 241", "4458 sayılı Kanun 241"),
    )
    assert [row.id for row in results] == [named, numbered] and more is False
    assert session.execute.call_count == 1
    statement = session.execute.call_args.args[0]
    union = statement._where_criteria[-1]
    assert union.operator is operators.or_
    assert len(list(union.clauses)) == 2
    compiled = statement.compile(dialect=postgresql.dialect())
    assert all(
        term in compiled.params.values()
        for term in {"%gumruk%", "%kanunu%", "%4458%", "%sayili%", "%kanun%", "%241%"}
    )
    assert 73 in compiled.params.values()
    assert "is_deleting IS false" in str(compiled)
    observed_rows = publication.call_args.args[1]
    assert [row.id for row in observed_rows] == [named, numbered, unpublished]


def test_regular_source_query_does_not_gain_alternative_match_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = catalog_session(monkeypatch)
    session.execute.return_value.all.return_value = []
    monkeypatch.setattr(asv3_corpus, "get_access_for_user_files", lambda *_args: {})
    monkeypatch.setattr(
        asv3_corpus, "filter_publication_read", lambda _read, rows, _key: rows
    )
    asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=IndexFilters(access_control_list=[]),
        query="2026/72 Gümrük Kanunu",
    )
    statement = session.execute.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    assert session.execute.call_count == 1
    assert (
        "%gumruk%" in compiled.params.values()
        and "%kanunu%" in compiled.params.values()
    )
    assert "(^|[^0-9])2026[/_. -]+0*72([^0-9]|$)" in compiled.params.values()
    assert not any(
        getattr(condition, "operator", None) is operators.or_
        for condition in statement._where_criteria[-3:]
    )


@pytest.mark.parametrize("queries", [(), ("Gümrük Kanunu", "")])
def test_empty_related_query_cannot_turn_the_union_into_a_broad_catalog_scan(
    queries: tuple[str, ...],
) -> None:
    session = MagicMock()
    with pytest.raises(ValueError, match="explicit nonempty"):
        asv3_corpus.find_related_sources(
            session,
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            query_variants=queries,
        )
    session.execute.assert_not_called()
