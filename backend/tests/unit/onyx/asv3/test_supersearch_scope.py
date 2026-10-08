"""Check mandatory PC identity and authoritative aggregate membership fences."""

import json
from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus, supersearch
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from tests.unit.onyx.asv3.test_corpus_scope import scope_session


def test_supersearch_binds_actual_corpus_identity_and_preserves_caller_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = scope_session(monkeypatch)
    manager = MagicMock()
    manager.__enter__.return_value = session
    monkeypatch.setattr(supersearch, "get_session_with_current_tenant", lambda: manager)
    filters = IndexFilters(
        access_control_list=["restricted-ACL"],
        as_of_date=date(2024, 7, 1),
        document_set=["Other subset"],
        attached_document_ids=[str(uuid4())],
        forced_document_set=[asv3_corpus.PC_CORPUS_NAME, "Other corpus"],
        regulatory_label_search_enabled=False,
        regulatory_label_run_ids=(uuid4(),),
    )
    actual = supersearch.bind_supersearch_pc_scope(
        user=cast(User, object()),
        filters=filters,
        document_set_names_override=[asv3_corpus.PC_CORPUS_NAME, "Other corpus"],
    )
    assert actual.asv3_document_set_id == 73
    assert actual.forced_document_set == [asv3_corpus.PC_CORPUS_NAME]
    assert actual.source_type == [DocumentSource.USER_FILE]
    assert actual.access_control_list == ["restricted-ACL"]
    assert actual.as_of_date == filters.as_of_date
    assert actual.document_set == ["Other subset"]
    assert actual.attached_document_ids == filters.attached_document_ids
    assert actual.regulatory_label_search_enabled is False
    assert actual.regulatory_label_run_ids == filters.regulatory_label_run_ids
    query = asv3_corpus._source_statement(actual).compile(dialect=postgresql.dialect())
    assert 73 in query.params.values()
    assert "document_set.id =" in str(query)
    assert " AND user_file.id IN" in str(query)
    assert filters.asv3_document_set_id is None


@pytest.mark.parametrize("override", [[], ["Other corpus"], ["PC Kulliyati"]])
@pytest.mark.parametrize("filter_kind", ["override", "forced"])
def test_disjoint_operator_scope_fails_before_db_access(
    override: list[str], filter_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_factory = MagicMock()
    monkeypatch.setattr(supersearch, "get_session_with_current_tenant", session_factory)
    filters = IndexFilters(
        access_control_list=[],
        forced_document_set=override if filter_kind == "forced" else None,
    )
    with pytest.raises(OnyxError, match="does not intersect") as rejection:
        supersearch.bind_supersearch_pc_scope(
            user=cast(User, object()),
            filters=filters,
            document_set_names_override=override if filter_kind == "override" else None,
        )
    assert rejection.value.error_code is OnyxErrorCode.INVALID_INPUT
    session_factory.assert_not_called()


@pytest.mark.parametrize(
    "defect", ["missing", "ambiguous", "denied", "wrong_id", "wrong_tenant", "source"]
)
def test_supersearch_missing_or_inaccessible_corpus_never_broadens(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = scope_session(monkeypatch)
    manager = MagicMock()
    manager.__enter__.return_value = session
    monkeypatch.setattr(supersearch, "get_session_with_current_tenant", lambda: manager)
    filters = IndexFilters(access_control_list=[])
    if defect == "missing":
        session.scalars.return_value = []
    elif defect == "ambiguous":
        session.scalars.return_value = [SimpleNamespace(id=73), SimpleNamespace(id=74)]
    elif defect == "denied":
        monkeypatch.setattr(
            asv3_corpus,
            "get_document_set_by_id_for_user",
            lambda *_args, **_kwargs: None,
        )
    elif defect == "wrong_id":
        filters.asv3_document_set_id = 15
    elif defect == "wrong_tenant":
        filters.tenant_id = "tenant-B"
    else:
        filters.source_type = [DocumentSource.WEB]
    with pytest.raises(
        OnyxError, match="Supersearch PC Külliyatı scope is unavailable"
    ):
        supersearch.bind_supersearch_pc_scope(
            user=cast(User, object()), filters=filters
        )
    session.execute.assert_not_called()


def aggregate_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    session = MagicMock()
    monkeypatch.setattr(supersearch, "require_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supersearch, "qualified_file_ids", lambda *_args: frozenset())
    monkeypatch.setattr(supersearch, "observe_publication_read", MagicMock)
    monkeypatch.setattr(supersearch, "require_publication_files", lambda *_args: None)
    return session


def resolve_centers(
    session: MagicMock, source_id: UUID, as_of_date: date | None = None
) -> dict[str, tuple[str, ...]]:
    return supersearch.resolve_supersearch_center_ids(
        session,
        user=cast(User, object()),
        filters=IndexFilters(access_control_list=[], as_of_date=as_of_date),
        source_id=source_id,
        center_ids=("aggregate", "ordinary", "stale"),
        check_active=lambda: None,
    )


def test_aggregate_membership_comes_from_scoped_stored_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = aggregate_session(monkeypatch)
    source_id = uuid4()
    aggregate = SimpleNamespace(
        id="aggregate",
        chunk_type="hierarchical_aggregate",
        chunk_metadata={
            "source_regulatory_chunk_ids": ["child-a", "child-b", "child-a"]
        },
    )
    ordinary = SimpleNamespace(id="ordinary", chunk_type="article", chunk_metadata={})
    session.execute.side_effect = [
        [aggregate, ordinary],
        [SimpleNamespace(id=value) for value in ("child-a", "child-b", "ordinary")],
    ]
    actual = resolve_centers(session, source_id, date(2025, 7, 1))
    assert actual == {
        "aggregate": ("child-a", "child-b"),
        "ordinary": ("ordinary",),
        "stale": (),
    }
    for call in session.execute.call_args_list:
        compiled = call.args[0].compile(dialect=postgresql.dialect())
        assert source_id in compiled.params.values()
        assert "regulatory_chunk.user_file_id =" in str(compiled)
        assert "validity_start_date" in str(compiled) and "validity_end_date" in str(
            compiled
        )


@pytest.mark.parametrize("members", [None, [], ["child", 1], ["outside-source"]])
def test_aggregate_missing_or_cross_source_members_fail_closed(
    members: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = aggregate_session(monkeypatch)
    session.execute.side_effect = [
        [
            SimpleNamespace(
                id="aggregate",
                chunk_type="hierarchical_aggregate",
                chunk_metadata={"source_regulatory_chunk_ids": members},
            )
        ],
        [],
    ]
    with pytest.raises(asv3_corpus.CorpusScopeUnavailable):
        resolve_centers(session, uuid4())


def test_aggregate_read_authorizes_source_before_any_chunk_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = aggregate_session(monkeypatch)
    gate = MagicMock(side_effect=PermissionError("Outside PC corpus"))
    monkeypatch.setattr(supersearch, "require_source", gate)
    with pytest.raises(PermissionError, match="Outside PC"):
        resolve_centers(session, uuid4())
    session.execute.assert_not_called()


def temporal_center(
    identifier: str,
    *,
    chunk_type: str = "article",
    role: str = "canonical",
    members: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        projection=SimpleNamespace(
            source_json=json.dumps(
                {
                    "regulatory_chunk_id": identifier,
                    "chunk_type": chunk_type,
                    "source_regulatory_chunk_ids": members,
                }
            )
        ),
        representation_metadata={},
        derived_role=role,
    )


@pytest.mark.parametrize("child_type", ["article", "hierarchical_aggregate", "missing"])
def test_qualified_aggregate_uses_dated_accepted_atomic_bindings(
    child_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = aggregate_session(monkeypatch)
    source_id = uuid4()
    monkeypatch.setattr(
        supersearch, "qualified_file_ids", lambda *_args: frozenset((source_id,))
    )
    snapshot = object()
    monkeypatch.setattr(
        supersearch, "resolve_source_query_index", lambda *_args: snapshot
    )
    aggregate = temporal_center(
        "aggregate",
        chunk_type="hierarchical_aggregate",
        role="aggregate",
        members=["child"],
    )
    children = (
        []
        if child_type == "missing"
        else [temporal_center("child", chunk_type=child_type)]
    )
    reader = MagicMock(
        side_effect=[
            (binding for binding in [aggregate]),
            (binding for binding in children),
        ]
    )
    monkeypatch.setattr(supersearch, "iter_public_temporal_bindings", reader)
    if child_type == "article":
        actual = resolve_centers(session, source_id, date(2024, 7, 1))
        assert actual["aggregate"] == ("child",)
    else:
        with pytest.raises(
            asv3_corpus.CorpusScopeUnavailable, match="membership is missing"
        ):
            resolve_centers(session, source_id, date(2024, 7, 1))
    for call in reader.call_args_list:
        assert call.args[1] == source_id
        assert call.kwargs["index"] is snapshot
        assert call.kwargs["as_of_date"] == date(2024, 7, 1)
    assert reader.call_args_list[1].kwargs["canonical_chunk_ids"] == ("child",)
    session.execute.assert_not_called()


def test_aggregate_revalidates_publication_after_membership_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = aggregate_session(monkeypatch)
    session.execute.side_effect = [
        [SimpleNamespace(id="ordinary", chunk_type="article", chunk_metadata={})]
    ]
    publication_gate = MagicMock(side_effect=PermissionError("Publication was revoked"))
    monkeypatch.setattr(supersearch, "require_publication_files", publication_gate)
    with pytest.raises(PermissionError, match="Publication was revoked"):
        resolve_centers(session, uuid4())
    assert session.execute.call_count == 1
    publication_gate.assert_called_once()
