"""Formal title fallback preserves exact identity and existing corpus fences."""

from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql

from onyx.asv3 import corpus_tools
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource, NumberedLawTitle
from onyx.db.models import User
from onyx.document_index.publication_models import PublicationIndexSnapshot


def filters() -> IndexFilters:
    return IndexFilters(
        access_control_list=[],
        asv3_document_set_id=73,
        forced_document_set=[asv3_corpus.PC_CORPUS_NAME],
        document_set=["Requested subset"],
        project_id_filter=9,
        as_of_date=date(2026, 5, 18),
    )


def source(name: str = "Kanunlar/faaliyet_kanunu_.md") -> CorpusSource:
    identifier = uuid4()
    return CorpusSource(identifier, name, str(identifier))


def chunk(
    candidate: CorpusSource,
    root: str = "8237 SAYILI FAALİYET KANUNU",
    kind: str | None = "kanun",
) -> CorpusChunk:
    return CorpusChunk(
        "own-canonical-chunk",
        candidate.id,
        "8237 sayılı Faaliyet Kanunu is quoted here, not an identity field.",
        0,
        0,
        (root,),
        {"document_type": kind},
        None,
        None,
        "active",
    )


def catalogue(
    monkeypatch: pytest.MonkeyPatch,
    pages: list[list[CorpusSource]],
    *,
    denied: set[UUID] | None = None,
    unpublished: set[UUID] | None = None,
) -> MagicMock:
    session = MagicMock()
    session.execute.side_effect = [
        SimpleNamespace(all=lambda page=page: page) for page in pages
    ]
    monkeypatch.setattr(asv3_corpus, "_validate_filters", MagicMock())
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", lambda: object())
    monkeypatch.setattr(asv3_corpus, "get_acl_for_user", lambda *_args: {"user:owner"})
    denied_ids = denied or set()
    unpublished_ids = unpublished or set()
    access = {
        str(row.id): SimpleNamespace(
            to_acl=lambda identifier=row.id: (
                {"user:other"} if identifier in denied_ids else {"user:owner"}
            )
        )
        for page in pages
        for row in page
    }
    monkeypatch.setattr(asv3_corpus, "get_access_for_user_files", lambda *_args: access)
    monkeypatch.setattr(
        asv3_corpus,
        "filter_publication_read",
        lambda _read, rows, _key: [
            row for row in rows if row.id not in unpublished_ids
        ],
    )
    return session


@pytest.mark.parametrize(
    "query",
    [
        "8237 sayılı Faaliyet Kanunu",
        " 8237 SAYILI  FAALİYET KANUN ",
        "08237 sayili Faaliyet Kanunu",
    ],
)
def test_full_formal_title_retains_number_and_official_title(query: str) -> None:
    assert asv3_corpus._numbered_law_title(query) == NumberedLawTitle(
        "8237", "faaliyet kanun"
    )


@pytest.mark.parametrize(
    "query",
    [
        "8237 sayılı Kanun",
        "Faaliyet Kanunu",
        "8237 sayılı Faaliyet Kanunu m. 27",
        "8237 sayılı Faaliyet Yönetmeliği",
    ],
)
def test_nonformal_or_provision_queries_do_not_trigger_fallback(query: str) -> None:
    assert asv3_corpus._numbered_law_title(query) is None


def test_default_resolution_never_relaxes_numbered_filename_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = catalogue(monkeypatch, [[]])
    identity_read = MagicMock()
    monkeypatch.setattr(asv3_corpus, "_source_has_numbered_law_identity", identity_read)
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query="8237 sayılı Faaliyet Kanunu",
    ) == ([], False)
    assert session.execute.call_count == 1
    identity_read.assert_not_called()
    compiled = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert all(
        term in compiled.params.values()
        for term in ("%8237%", "%sayili%", "%faaliyet%", "%kanunu%")
    )


def test_strict_title_match_never_replaced_by_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = source("Kanunlar/8237_sayili_faaliyet_kanunu.md")
    session = catalogue(monkeypatch, [[strict]])
    identity_read = MagicMock()
    monkeypatch.setattr(asv3_corpus, "_source_has_numbered_law_identity", identity_read)
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query="8237 sayılı Faaliyet Kanunu",
        allow_numbered_title_fallback=True,
    ) == ([strict], False)
    identity_read.assert_not_called()
    assert session.execute.call_count == 1


def test_fallback_keeps_acl_publication_scope_and_exact_canonical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid, wrong_number, denied, unpublished = (source() for _ in range(4))
    session = catalogue(
        monkeypatch,
        [[], [valid, wrong_number, denied, unpublished]],
        denied={denied.id},
        unpublished={unpublished.id},
    )
    monkeypatch.setattr(asv3_corpus, "resolve_source_query_index", lambda *_args: None)
    readable = {
        valid.id: [chunk(valid)],
        wrong_number.id: [chunk(wrong_number, root="8918 SAYILI FAALİYET KANUNU")],
    }
    read = MagicMock(
        side_effect=lambda _session, **kwargs: (
            next(row for row in (valid, wrong_number) if row.id == kwargs["source_id"]),
            readable[kwargs["source_id"]],
            False,
        )
    )
    monkeypatch.setattr(asv3_corpus, "read_source_chunks", read)
    scope = filters()
    original_scope = scope.model_dump()
    results, more = asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=scope,
        query="8237 sayılı Faaliyet Kanunu",
        allow_numbered_title_fallback=True,
    )
    assert results == [valid] and more is False
    assert {call.kwargs["source_id"] for call in read.call_args_list} == {
        valid.id,
        wrong_number.id,
    }
    assert all(call.kwargs["filters"] is scope for call in read.call_args_list)
    assert all(call.kwargs["limit"] == 2 for call in read.call_args_list)
    assert scope.model_dump() == original_scope
    for call in session.execute.call_args_list:
        compiled = call.args[0].compile(dialect=postgresql.dialect())
        assert 73 in compiled.params.values() and 9 in compiled.params.values()
        assert ["Requested subset"] in compiled.params.values()
        assert [asv3_corpus.PC_CORPUS_NAME] in compiled.params.values()
        assert "is_deleting IS false" in str(compiled)
    fallback = session.execute.call_args_list[-1].args[0]
    compiled = fallback.compile(dialect=postgresql.dialect())
    assert "%8237%" not in compiled.params.values()
    assert "%sayili%" not in compiled.params.values()
    assert all(term in compiled.params.values() for term in ("%faaliyet%", "%kanun%"))
    assert all(
        "regexp_replace" in str(condition)
        for condition in fallback._where_criteria[-2:]
    )


@pytest.mark.parametrize(
    "root,kind",
    [
        ("8237 SAYILI BAŞKA KANUN", "kanun"),
        ("8918 SAYILI FAALİYET KANUNU", "kanun"),
        ("8237 SAYILI FAALİYET KANUNU", "genelge"),
        ("8237 SAYILI FAALİYET KANUNU", None),
        ("Faaliyet Genelgesi", "kanun"),
    ],
)
def test_other_identity_or_quoted_law_never_substitutes_for_own_metadata(
    root: str,
    kind: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = source()
    monkeypatch.setattr(asv3_corpus, "resolve_source_query_index", lambda *_args: None)
    monkeypatch.setattr(
        asv3_corpus,
        "read_source_chunks",
        lambda *_args, **_kwargs: (
            candidate,
            [chunk(candidate, root=root, kind=kind)],
            False,
        ),
    )
    assert not asv3_corpus._source_has_numbered_law_identity(
        MagicMock(),
        user=cast(User, object()),
        filters=filters(),
        source=candidate,
        identity=NumberedLawTitle("8237", "faaliyet kanun"),
    )


def test_identity_read_uses_verified_query_index_and_preserves_date_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = source()
    index = cast(PublicationIndexSnapshot, object())
    monkeypatch.setattr(asv3_corpus, "resolve_source_query_index", lambda *_args: index)
    read = MagicMock(return_value=(candidate, [chunk(candidate)], True))
    monkeypatch.setattr(asv3_corpus, "read_source_chunks", read)
    scope = filters()
    assert asv3_corpus._source_has_numbered_law_identity(
        MagicMock(),
        user=cast(User, object()),
        filters=scope,
        source=candidate,
        identity=NumberedLawTitle("8237", "faaliyet kanun"),
    )
    assert read.call_args.kwargs["query_indexes"] == {candidate.id: index}
    assert read.call_args.kwargs["filters"].as_of_date == date(2026, 5, 18)


def test_conflicting_own_filename_number_cannot_be_replaced_by_root_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = source("Kanunlar/8918_sayili_faaliyet_kanunu.md")
    monkeypatch.setattr(asv3_corpus, "resolve_source_query_index", lambda *_args: None)
    monkeypatch.setattr(
        asv3_corpus,
        "read_source_chunks",
        lambda *_args, **_kwargs: (candidate, [chunk(candidate)], False),
    )
    assert not asv3_corpus._source_has_numbered_law_identity(
        MagicMock(),
        user=cast(User, object()),
        filters=filters(),
        source=candidate,
        identity=NumberedLawTitle("8237", "faaliyet kanun"),
    )


def test_publication_identity_barrier_is_not_relabelled_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        asv3_corpus,
        "resolve_source_query_index",
        MagicMock(
            side_effect=asv3_corpus.CorpusScopeUnavailable(
                "Ambiguous publication index"
            )
        ),
    )
    with pytest.raises(asv3_corpus.CorpusScopeUnavailable, match="Ambiguous"):
        asv3_corpus._source_has_numbered_law_identity(
            MagicMock(),
            user=cast(User, object()),
            filters=filters(),
            source=source(),
            identity=NumberedLawTitle("8237", "faaliyet kanun"),
        )


def test_later_strict_empty_page_cannot_switch_to_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = source("Kanunlar/8237_sayili_faaliyet_kanunu.md")
    session = catalogue(monkeypatch, [[], [strict]])
    read = MagicMock()
    monkeypatch.setattr(asv3_corpus, "_source_has_numbered_law_identity", read)
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        offset=20,
        query="8237 sayılı Faaliyet Kanunu",
        allow_numbered_title_fallback=True,
    ) == ([], False)
    assert session.execute.call_count == 2
    read.assert_not_called()


def test_fallback_pages_keep_raw_candidate_offset_and_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, next_candidate = source(), source()
    session = catalogue(monkeypatch, [[], [], [candidate, next_candidate]])
    monkeypatch.setattr(
        asv3_corpus,
        "_source_has_numbered_law_identity",
        lambda *_args, **_kwargs: False,
    )
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        offset=1,
        limit=1,
        query="8237 sayılı Faaliyet Kanunu",
        allow_numbered_title_fallback=True,
    ) == ([], True)
    statement = session.execute.call_args_list[-1].args[0]
    assert statement._limit_clause.value == 2
    assert statement._offset_clause.value == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_broker_threads_only_explicit_opt_in(
    enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    manager = MagicMock()
    manager.__enter__.return_value = session
    monkeypatch.setattr(
        corpus_tools, "get_session_with_current_tenant", lambda: manager
    )
    lookup = MagicMock(return_value=([], False))
    monkeypatch.setattr(corpus_tools, "find_sources", lookup)
    broker = corpus_tools.CorpusBroker(
        cast(User, object()), filters(), allow_numbered_title_fallback=enabled
    )
    broker.sources("8237 sayılı Faaliyet Kanunu", RunContext())
    assert lookup.call_args.kwargs["allow_numbered_title_fallback"] is enabled


def test_multiple_verified_source_ids_remain_uncitable_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = source(), source()
    session = catalogue(monkeypatch, [[], [first, second]])
    manager = MagicMock()
    manager.__enter__.return_value = session
    monkeypatch.setattr(
        corpus_tools, "get_session_with_current_tenant", lambda: manager
    )
    monkeypatch.setattr(
        asv3_corpus,
        "_source_has_numbered_law_identity",
        lambda *_args, **_kwargs: True,
    )
    broker = corpus_tools.CorpusBroker(
        cast(User, object()), filters(), allow_numbered_title_fallback=True
    )
    resolve = next(
        spec
        for spec in corpus_tools.build_corpus_specs(broker)
        if spec.name == "resolve_source"
    )
    outcome = resolve.handler({"query": "8237 sayılı Faaliyet Kanunu"}, RunContext())
    assert outcome.status is OutcomeStatus.AMBIGUOUS
    assert outcome.evidence == [] and outcome.data["absence_proven"] is False
    assert outcome.data["sources"] == [
        {"source_id": str(candidate.id), "name": candidate.name}
        for candidate in (first, second)
    ]
