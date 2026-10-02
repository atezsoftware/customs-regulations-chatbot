"""Mandatory corpus identity remains an AND fence around caller knowledge scopes."""

from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus
from onyx.db.models import User


def scope_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    session = MagicMock()
    corpus = SimpleNamespace(id=73, name=asv3_corpus.PC_CORPUS_NAME)
    session.scalars.return_value = [corpus]
    monkeypatch.setattr(
        "shared_configs.contextvars.get_current_tenant_id", lambda: "tenant-A"
    )
    monkeypatch.setattr(
        asv3_corpus, "get_document_set_by_id_for_user", lambda *_args, **_kwargs: corpus
    )
    monkeypatch.setattr(
        asv3_corpus,
        "filter_document_set_names_by_user_access",
        lambda _session, names, _user: set(names),
    )
    return session


def test_frontend_user_file_scope_pins_tenant_identity_and_preserves_narrowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = scope_session(monkeypatch)
    filters = IndexFilters(
        access_control_list=[],
        source_type=[DocumentSource.USER_FILE],
        regulatory_chunks_only=True,
        as_of_date=date(2025, 12, 31),
        document_set=["Additional subset"],
        attached_document_ids=[str(uuid4())],
        project_id_filter=9,
    )
    original = filters.model_dump()
    bound = asv3_corpus.resolve_pc_corpus_scope(
        session, user=cast(User, object()), filters=filters
    )
    assert bound.asv3_document_set_id == 73
    assert bound.forced_document_set == [asv3_corpus.PC_CORPUS_NAME]
    assert bound.tenant_id == "tenant-A"
    assert bound.source_type == [DocumentSource.USER_FILE]
    assert bound.document_set == filters.document_set
    assert bound.attached_document_ids == filters.attached_document_ids
    assert bound.project_id_filter == 9 and bound.as_of_date == filters.as_of_date
    assert filters.model_dump() == original
    assert IndexFilters.model_validate(bound.model_dump(mode="json")) == bound
    query = session.scalars.call_args.args[0].compile(dialect=postgresql.dialect())
    assert asv3_corpus.PC_CORPUS_NAME in query.params.values()
    assert "is_deleting IS false" in str(query)


@pytest.mark.parametrize(
    "sources", [[DocumentSource.FILE], [DocumentSource.USER_FILE, DocumentSource.FILE]]
)
def test_file_connector_alias_never_broadens_user_file_corpus(
    sources: list[DocumentSource], monkeypatch: pytest.MonkeyPatch
) -> None:
    session = scope_session(monkeypatch)
    with pytest.raises(asv3_corpus.CorpusScopeUnavailable, match="USER_FILE"):
        asv3_corpus.resolve_pc_corpus_scope(
            session,
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[], source_type=sources),
        )
    session.scalars.assert_not_called()


@pytest.mark.parametrize(
    "defect", ["missing", "denied", "wrong_tenant", "wrong_id", "disjoint_forced"]
)
def test_scope_resolution_failures_never_fall_back_to_broad_corpus(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = scope_session(monkeypatch)
    filters = IndexFilters(access_control_list=[])
    if defect == "missing":
        session.scalars.return_value = []
    elif defect == "denied":
        monkeypatch.setattr(
            asv3_corpus,
            "get_document_set_by_id_for_user",
            lambda *_args, **_kwargs: None,
        )
    elif defect == "wrong_tenant":
        filters.tenant_id = "tenant-B"
    elif defect == "wrong_id":
        filters.asv3_document_set_id = 15
    else:
        filters.forced_document_set = ["Other corpus"]
    with pytest.raises((PermissionError, asv3_corpus.CorpusScopeUnavailable)):
        asv3_corpus.resolve_pc_corpus_scope(
            session, user=cast(User, object()), filters=filters
        )
    session.execute.assert_not_called()


def test_pinned_membership_is_and_fence_outside_attachment_or_scope() -> None:
    source_id = uuid4()
    filters = IndexFilters(
        access_control_list=[],
        asv3_document_set_id=73,
        forced_document_set=[asv3_corpus.PC_CORPUS_NAME],
        document_set=["Disjoint"],
        attached_document_ids=[str(source_id)],
    )
    query = asv3_corpus._source_statement(filters).compile(dialect=postgresql.dialect())
    sql = str(query)
    assert "document_set.id =" in sql and "is_deleting IS false" in sql
    assert " OR user_file.id IN" in sql
    assert " AND user_file.id IN" in sql
    assert 73 in query.params.values()


@pytest.mark.parametrize("defect", ["unbound", "renamed", "denied"])
def test_every_read_revalidates_mandatory_scope_before_source_sql(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = scope_session(monkeypatch)
    filters = IndexFilters(
        access_control_list=[],
        asv3_document_set_id=73,
        forced_document_set=[asv3_corpus.PC_CORPUS_NAME],
        source_type=[DocumentSource.USER_FILE],
    )
    if defect == "unbound":
        filters.asv3_document_set_id = None
    elif defect == "renamed":
        monkeypatch.setattr(
            asv3_corpus,
            "get_document_set_by_id_for_user",
            lambda *_args, **_kwargs: SimpleNamespace(id=73, name="Moved corpus"),
        )
    else:
        monkeypatch.setattr(
            asv3_corpus,
            "get_document_set_by_id_for_user",
            lambda *_args, **_kwargs: None,
        )
    with pytest.raises((PermissionError, asv3_corpus.CorpusScopeUnavailable)):
        asv3_corpus.find_sources(session, user=cast(User, object()), filters=filters)
    session.execute.assert_not_called()
