from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from scripts import regulatory_import

from onyx.db.enums import UserFileStatus


def _args(paths: list[Path], **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "paths": paths,
        "user_email": "admin@example.com",
        "project_id": 7,
        "project_name": None,
        "tenant_id": "public",
        "recursive": True,
        "allow_duplicate": False,
        "manifest": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _session_factory(
    *sessions: MagicMock,
) -> Callable[[], AbstractContextManager[MagicMock]]:
    session_iterator = iter(sessions)

    @contextmanager
    def factory() -> Iterator[MagicMock]:
        yield next(session_iterator)

    return factory


def _patch_successful_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: regulatory_import.ImportTarget,
) -> MagicMock:
    monkeypatch.setattr(
        regulatory_import, "ensure_document_import_available", MagicMock()
    )
    monkeypatch.setattr(
        regulatory_import.SqlEngine,
        "scoped_engine",
        MagicMock(return_value=nullcontext()),
    )
    monkeypatch.setattr(
        regulatory_import,
        "get_session_with_current_tenant",
        MagicMock(return_value=nullcontext(MagicMock())),
    )
    monkeypatch.setattr(
        regulatory_import,
        "_resolve_import_target",
        MagicMock(return_value=target),
    )
    monkeypatch.setattr(
        regulatory_import,
        "_ensure_database_schema_current",
        MagicMock(),
    )
    elasticsearch_target = _elasticsearch_target()
    monkeypatch.setattr(
        regulatory_import,
        "_resolve_elasticsearch_target",
        MagicMock(return_value=elasticsearch_target),
    )
    ensure_elasticsearch_ready = MagicMock()
    monkeypatch.setattr(
        regulatory_import, "_ensure_elasticsearch_ready", ensure_elasticsearch_ready
    )
    monkeypatch.setattr(
        regulatory_import,
        "_active_file_with_name_exists",
        MagicMock(return_value=False),
    )
    return ensure_elasticsearch_ready


def _elasticsearch_target() -> regulatory_import.ElasticsearchTarget:
    return regulatory_import.ElasticsearchTarget(
        index_name="danswer_chunk_test",
        embedding_dimension=768,
        tenant_id="public",
        multitenant=False,
    )


def test_resolve_paths_expands_user_and_honors_recursive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    source_dir = home / "sources"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    top_level = source_dir / "top.md"
    nested = nested_dir / "nested.pdf"
    top_level.write_text("top", encoding="utf-8")
    nested.write_bytes(b"nested")
    monkeypatch.setenv("HOME", str(home))

    assert regulatory_import._resolve_paths([Path("~/sources")], recursive=True) == [
        nested.resolve(),
        top_level.resolve(),
    ]
    assert regulatory_import._resolve_paths([Path("~/sources")], recursive=False) == [
        top_level.resolve()
    ]


def test_database_schema_preflight_requires_exact_head_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    monkeypatch.setattr(
        regulatory_import,
        "_get_application_alembic_heads",
        MagicMock(return_value=frozenset({"head-a", "head-b"})),
    )
    database_heads = MagicMock(return_value=frozenset({"head-b", "head-a"}))
    monkeypatch.setattr(
        regulatory_import,
        "get_database_alembic_heads",
        database_heads,
    )

    regulatory_import._ensure_database_schema_current(session)

    database_heads.assert_called_once_with(session)


def test_database_schema_preflight_reports_image_and_database_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        regulatory_import,
        "_get_application_alembic_heads",
        MagicMock(return_value=frozenset({"image-head"})),
    )
    monkeypatch.setattr(
        regulatory_import,
        "get_database_alembic_heads",
        MagicMock(return_value=frozenset({"database-head"})),
    )

    with pytest.raises(ValueError, match="image=image-head; database=database-head"):
        regulatory_import._ensure_database_schema_current(MagicMock())


def test_elasticsearch_preflight_is_read_only_and_validates_current_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _elasticsearch_target()
    expected_schema = {"properties": {"regulatory_chunk_id": {"type": "keyword"}}}
    document_schema = MagicMock(return_value=expected_schema)
    monkeypatch.setattr(
        regulatory_import.DocumentSchema,
        "get_document_schema",
        document_schema,
    )
    client = MagicMock()
    client.ping.return_value = True
    client.index_exists.return_value = True
    client.validate_index.return_value = True
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    client_context.__exit__.return_value = None
    client_factory = MagicMock(return_value=client_context)
    monkeypatch.setattr(regulatory_import, "ElasticsearchIndexClient", client_factory)

    regulatory_import._ensure_elasticsearch_ready(target)

    document_schema.assert_called_once_with(
        target.embedding_dimension,
        target.multitenant,
    )
    client_factory.assert_called_once_with(
        index_name=target.index_name,
        emit_metrics=False,
    )
    client.ping.assert_called_once_with()
    client.index_exists.assert_called_once_with()
    client.validate_index.assert_called_once_with(expected_schema)
    assert not client.method_calls or all(
        call[0] in {"ping", "index_exists", "validate_index"}
        for call in client.method_calls
    )


def test_projection_visibility_requires_exact_chunk_identities() -> None:
    target = regulatory_import.ElasticsearchTarget(
        index_name="chunks",
        embedding_dimension=768,
        tenant_id="tenant-a",
        multitenant=True,
    )
    identities = [
        regulatory_import.RegulatoryProjectionIdentity("reg-a", 0),
        regulatory_import.RegulatoryProjectionIdentity("reg-b", 1),
    ]
    client = MagicMock()
    client.count_by_query.return_value = 2
    client.search_for_document_ids.return_value = ["os-a", "os-b"]

    assert regulatory_import._projection_is_visible(
        client,
        document_id="file-a",
        identities=identities,
        target=target,
    )

    count_query = client.count_by_query.call_args.args[0]
    assert {
        "term": {regulatory_import.TENANT_ID_FIELD_NAME: {"value": "tenant-a"}}
    } in count_query["query"]["bool"]["filter"]
    search_query = client.search_for_document_ids.call_args.args[0]
    assert search_query["_source"] is False
    assert search_query["size"] == 2
    identity_filters = [
        clause["bool"]["filter"] for clause in search_query["query"]["bool"]["should"]
    ]
    assert {
        "term": {regulatory_import.REGULATORY_CHUNK_ID_FIELD_NAME: {"value": "reg-a"}}
    } in identity_filters[0]
    assert {
        "term": {regulatory_import.CHUNK_INDEX_FIELD_NAME: {"value": 0}}
    } in identity_filters[0]

    client.search_for_document_ids.return_value = ["os-a"]
    assert not regulatory_import._projection_is_visible(
        client,
        document_id="file-a",
        identities=identities,
        target=target,
    )


def test_projection_visibility_refreshes_once_and_polls_with_a_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _elasticsearch_target()
    identities = [regulatory_import.RegulatoryProjectionIdentity("reg-a", 0)]
    client = MagicMock()
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    client_context.__exit__.return_value = None
    monkeypatch.setattr(
        regulatory_import,
        "ElasticsearchIndexClient",
        MagicMock(return_value=client_context),
    )
    visibility = MagicMock(side_effect=[False, True])
    monkeypatch.setattr(regulatory_import, "_projection_is_visible", visibility)
    sleep = MagicMock()
    monkeypatch.setattr(regulatory_import.time, "sleep", sleep)

    regulatory_import._ensure_projection_visible(
        document_id="file-a",
        identities=identities,
        target=target,
    )

    client.refresh_index.assert_called_once_with()
    assert visibility.call_count == 2
    sleep.assert_called_once_with(
        regulatory_import._ELASTICSEARCH_VISIBILITY_INTERVAL_S
    )


def test_projection_visibility_fails_after_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _elasticsearch_target()
    identities = [regulatory_import.RegulatoryProjectionIdentity("reg-a", 0)]
    client = MagicMock()
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    client_context.__exit__.return_value = None
    monkeypatch.setattr(
        regulatory_import,
        "ElasticsearchIndexClient",
        MagicMock(return_value=client_context),
    )
    visibility = MagicMock(return_value=False)
    monkeypatch.setattr(regulatory_import, "_projection_is_visible", visibility)
    monkeypatch.setattr(regulatory_import, "_ELASTICSEARCH_VISIBILITY_ATTEMPTS", 3)
    monkeypatch.setattr(regulatory_import.time, "sleep", MagicMock())

    with pytest.raises(ValueError, match="after 3 checks"):
        regulatory_import._ensure_projection_visible(
            document_id="file-a",
            identities=identities,
            target=target,
        )

    client.refresh_index.assert_called_once_with()
    assert visibility.call_count == 3


def test_run_rejects_duplicate_batch_names_before_opening_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "regulation.pdf").write_bytes(b"first")
    (second_dir / "regulation.pdf").write_bytes(b"second")
    ensure_capability = MagicMock()
    scoped_engine = MagicMock(return_value=nullcontext())
    monkeypatch.setattr(
        regulatory_import, "ensure_document_import_available", ensure_capability
    )
    monkeypatch.setattr(regulatory_import.SqlEngine, "scoped_engine", scoped_engine)

    with pytest.raises(ValueError, match="duplicate destination file names"):
        regulatory_import.run(_args([first_dir, second_dir]))

    ensure_capability.assert_called_once_with()
    scoped_engine.assert_not_called()


def test_schema_mismatch_aborts_before_target_resolution_or_file_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "regulation.pdf"
    source.write_bytes(b"content")
    monkeypatch.setattr(
        regulatory_import, "ensure_document_import_available", MagicMock()
    )
    monkeypatch.setattr(
        regulatory_import.SqlEngine,
        "scoped_engine",
        MagicMock(return_value=nullcontext()),
    )
    monkeypatch.setattr(
        regulatory_import,
        "get_session_with_current_tenant",
        MagicMock(return_value=nullcontext(MagicMock())),
    )
    schema_preflight = MagicMock(side_effect=ValueError("schema mismatch"))
    monkeypatch.setattr(
        regulatory_import,
        "_ensure_database_schema_current",
        schema_preflight,
    )
    resolve_target = MagicMock()
    import_one = MagicMock()
    monkeypatch.setattr(regulatory_import, "_resolve_import_target", resolve_target)
    monkeypatch.setattr(regulatory_import, "_import_one_file", import_one)

    with pytest.raises(ValueError, match="schema mismatch"):
        regulatory_import.run(_args([source]))

    schema_preflight.assert_called_once()
    resolve_target.assert_not_called()
    import_one.assert_not_called()


def test_existing_duplicate_aborts_preflight_before_any_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "regulation.pdf"
    source.write_bytes(b"content")
    target = regulatory_import.ImportTarget(user_id=uuid4(), project_id=7)
    ensure_elasticsearch_ready = _patch_successful_preflight(monkeypatch, target=target)
    monkeypatch.setattr(
        regulatory_import,
        "_active_file_with_name_exists",
        MagicMock(return_value=True),
    )
    import_one = MagicMock()
    monkeypatch.setattr(regulatory_import, "_import_one_file", import_one)
    original_tenant = regulatory_import.CURRENT_TENANT_ID_CONTEXTVAR.get()

    with pytest.raises(ValueError, match="Active files with the same name"):
        regulatory_import.run(_args([source]))

    ensure_elasticsearch_ready.assert_not_called()
    import_one.assert_not_called()
    assert regulatory_import.CURRENT_TENANT_ID_CONTEXTVAR.get() == original_tenant


def test_manifest_writability_is_checked_before_database_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "regulation.pdf"
    source.write_bytes(b"content")
    ensure_capability = MagicMock()
    scoped_engine = MagicMock(return_value=nullcontext())
    monkeypatch.setattr(
        regulatory_import, "ensure_document_import_available", ensure_capability
    )
    monkeypatch.setattr(regulatory_import.SqlEngine, "scoped_engine", scoped_engine)
    monkeypatch.setattr(
        regulatory_import,
        "_prepare_manifest_path",
        MagicMock(side_effect=PermissionError("output is read-only")),
    )

    with pytest.raises(PermissionError, match="read-only"):
        regulatory_import.run(
            _args([source], manifest=tmp_path / "output" / "manifest.json")
        )

    ensure_capability.assert_called_once_with()
    scoped_engine.assert_not_called()


def test_manifest_is_written_atomically_as_json(tmp_path: Path) -> None:
    manifest = regulatory_import._prepare_manifest_path(
        tmp_path / "output" / "manifest.json"
    )
    results = [
        regulatory_import.ImportResult(
            path="/imports/regulation.pdf",
            status="completed",
            user_file_id=str(uuid4()),
            chunk_count=3,
            detail=None,
        )
    ]

    regulatory_import._write_manifest(manifest, results)

    assert json.loads(manifest.read_text(encoding="utf-8")) == results
    assert not list(manifest.parent.glob(f".{manifest.name}.*"))


def test_main_returns_distinct_nonzero_code_for_preflight_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        regulatory_import,
        "run",
        MagicMock(side_effect=ValueError("database is not ready")),
    )

    exit_code = regulatory_import.main(
        [
            "source.pdf",
            "--user-email",
            "admin@example.com",
            "--project-id",
            "7",
        ]
    )

    assert exit_code == 2
    assert "Importer preflight failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status", "recorded_chunk_count", "postgres_chunk_count"),
    [
        (UserFileStatus.FAILED, 1, 1),
        (UserFileStatus.COMPLETED, 0, 0),
        (UserFileStatus.COMPLETED, 2, 1),
    ],
)
def test_import_one_file_rejects_failed_empty_or_inconsistent_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: UserFileStatus,
    recorded_chunk_count: int,
    postgres_chunk_count: int,
) -> None:
    source = tmp_path / "regulation.pdf"
    source.write_bytes(b"content")
    user_id = uuid4()
    user_file_id = uuid4()
    target = regulatory_import.ImportTarget(user_id=user_id, project_id=19)
    persisted_user = SimpleNamespace(id=user_id)
    created_file = SimpleNamespace(id=user_file_id)
    categorized = SimpleNamespace(
        rejected_files=[],
        user_files=[created_file],
        indexable_files=[created_file],
    )
    creation_session = MagicMock()
    creation_session.get.return_value = persisted_user
    verification_session = MagicMock()
    verification_session.get.return_value = SimpleNamespace(
        status=status,
        chunk_count=recorded_chunk_count,
    )
    create_user_files = MagicMock(return_value=categorized)
    process_user_file_impl = MagicMock()
    chunks = [
        SimpleNamespace(id=f"regulatory-{index}", position=index)
        for index in range(postgres_chunk_count)
    ]
    monkeypatch.setattr(
        regulatory_import,
        "get_session_with_current_tenant",
        _session_factory(creation_session, verification_session),
    )
    monkeypatch.setattr(regulatory_import, "create_user_files", create_user_files)
    monkeypatch.setattr(
        regulatory_import, "process_user_file_impl", process_user_file_impl
    )
    monkeypatch.setattr(
        regulatory_import, "get_chunks_for_file", MagicMock(return_value=chunks)
    )

    result = regulatory_import._import_one_file(
        source,
        target=target,
        tenant_id="tenant_one",
        elasticsearch_target=_elasticsearch_target(),
    )

    assert result["status"] == "failed"
    assert result["chunk_count"] == postgres_chunk_count
    assert result["detail"] is not None
    assert "Post-import verification failed" in result["detail"]
    creation_session.get.assert_called_once_with(regulatory_import.User, target.user_id)
    create_call = create_user_files.call_args.kwargs
    assert create_call["project_id"] == target.project_id
    assert create_call["user"] is persisted_user
    process_user_file_impl.assert_called_once_with(
        user_file_id=str(user_file_id),
        tenant_id="tenant_one",
        redis_locking=False,
    )


def test_import_one_file_succeeds_only_with_completed_matching_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "regulation.md"
    source.write_text("content", encoding="utf-8")
    target = regulatory_import.ImportTarget(user_id=uuid4(), project_id=23)
    user_file_id = uuid4()
    persisted_user = SimpleNamespace(id=target.user_id)
    created_file = SimpleNamespace(id=user_file_id)
    categorized = SimpleNamespace(
        rejected_files=[],
        user_files=[created_file],
        indexable_files=[created_file],
    )
    creation_session = MagicMock()
    creation_session.get.return_value = persisted_user
    verification_session = MagicMock()
    verification_session.get.return_value = SimpleNamespace(
        status=UserFileStatus.COMPLETED,
        chunk_count=2,
    )
    monkeypatch.setattr(
        regulatory_import,
        "get_session_with_current_tenant",
        _session_factory(creation_session, verification_session),
    )
    monkeypatch.setattr(
        regulatory_import,
        "create_user_files",
        MagicMock(return_value=categorized),
    )
    process_user_file_impl = MagicMock()
    monkeypatch.setattr(
        regulatory_import, "process_user_file_impl", process_user_file_impl
    )
    monkeypatch.setattr(
        regulatory_import,
        "get_chunks_for_file",
        MagicMock(
            return_value=[
                SimpleNamespace(id="regulatory-0", position=0),
                SimpleNamespace(id="regulatory-1", position=1),
            ]
        ),
    )
    ensure_projection_visible = MagicMock()
    monkeypatch.setattr(
        regulatory_import,
        "_ensure_projection_visible",
        ensure_projection_visible,
    )

    result = regulatory_import._import_one_file(
        source,
        target=target,
        tenant_id="tenant_one",
        elasticsearch_target=_elasticsearch_target(),
    )

    assert result == {
        "path": str(source),
        "status": "completed",
        "user_file_id": str(user_file_id),
        "chunk_count": 2,
        "detail": None,
    }
    process_user_file_impl.assert_called_once_with(
        user_file_id=str(user_file_id),
        tenant_id="tenant_one",
        redis_locking=False,
    )
    ensure_projection_visible.assert_called_once()
    visibility_call = ensure_projection_visible.call_args.kwargs
    assert visibility_call["document_id"] == str(user_file_id)
    assert visibility_call["identities"] == [
        regulatory_import.RegulatoryProjectionIdentity("regulatory-0", 0),
        regulatory_import.RegulatoryProjectionIdentity("regulatory-1", 1),
    ]
    assert visibility_call["target"] == _elasticsearch_target()


def test_import_one_file_reports_elasticsearch_projection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "regulation.md"
    source.write_text("content", encoding="utf-8")
    target = regulatory_import.ImportTarget(user_id=uuid4(), project_id=23)
    user_file_id = uuid4()
    persisted_user = SimpleNamespace(id=target.user_id)
    created_file = SimpleNamespace(id=user_file_id)
    categorized = SimpleNamespace(
        rejected_files=[],
        user_files=[created_file],
        indexable_files=[created_file],
    )
    creation_session = MagicMock()
    creation_session.get.return_value = persisted_user
    verification_session = MagicMock()
    verification_session.get.return_value = SimpleNamespace(
        status=UserFileStatus.COMPLETED,
        chunk_count=1,
    )
    monkeypatch.setattr(
        regulatory_import,
        "get_session_with_current_tenant",
        _session_factory(creation_session, verification_session),
    )
    monkeypatch.setattr(
        regulatory_import,
        "create_user_files",
        MagicMock(return_value=categorized),
    )
    monkeypatch.setattr(regulatory_import, "process_user_file_impl", MagicMock())
    monkeypatch.setattr(
        regulatory_import,
        "get_chunks_for_file",
        MagicMock(return_value=[SimpleNamespace(id="regulatory-0", position=0)]),
    )
    monkeypatch.setattr(
        regulatory_import,
        "_ensure_projection_visible",
        MagicMock(side_effect=ValueError("not visible")),
    )

    result = regulatory_import._import_one_file(
        source,
        target=target,
        tenant_id="tenant_one",
        elasticsearch_target=_elasticsearch_target(),
    )

    assert result["status"] == "failed"
    assert result["user_file_id"] == str(user_file_id)
    assert result["chunk_count"] == 1
    assert result["detail"] == (
        "Elasticsearch post-import verification failed: not visible"
    )


def test_run_returns_nonzero_when_import_verification_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "regulation.pdf"
    source.write_bytes(b"content")
    target = regulatory_import.ImportTarget(user_id=uuid4(), project_id=7)
    _patch_successful_preflight(monkeypatch, target=target)
    import_one = MagicMock(
        return_value=regulatory_import.ImportResult(
            path=str(source),
            status="failed",
            user_file_id=str(uuid4()),
            chunk_count=0,
            detail="Post-import verification failed",
        )
    )
    monkeypatch.setattr(regulatory_import, "_import_one_file", import_one)

    assert regulatory_import.run(_args([source])) == 1
    import_one.assert_called_once_with(
        source.resolve(),
        target=target,
        tenant_id="public",
        elasticsearch_target=_elasticsearch_target(),
    )


def test_import_one_file_reloads_user_in_active_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "regulation.pdf"
    source.write_bytes(b"content")
    target = regulatory_import.ImportTarget(user_id=uuid4(), project_id=7)
    creation_session = MagicMock()
    creation_session.get.return_value = None
    create_user_files = MagicMock()
    monkeypatch.setattr(
        regulatory_import,
        "get_session_with_current_tenant",
        _session_factory(creation_session),
    )
    monkeypatch.setattr(regulatory_import, "create_user_files", create_user_files)

    with pytest.raises(ValueError, match="disappeared after preflight"):
        regulatory_import._import_one_file(
            source,
            target=target,
            tenant_id="tenant_one",
            elasticsearch_target=_elasticsearch_target(),
        )

    create_user_files.assert_not_called()
