"""Fixed source-package diagnostic guards without services or provider requests."""

import contextlib
import datetime
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ValidationError
from scripts import regulatory_annex_dev_cutover as runner


def report() -> dict[str, Any]:
    return dict(
        stage="source861",
        status="scope_refused",
        database_read_only=True,
        reproduction=False,
        attempt_count=0,
        http_request_count=0,
        page_index=0,
    )


def scope_objects() -> tuple[Any, Any, Any]:
    run = SimpleNamespace(
        release_sha=runner.SOURCE861_RUNTIME,
        run_id=UUID("5950d8bc-5dac-443b-b143-8494d0b082d1"),
        file_id=UUID("a3e6717c-ac94-4a05-a04c-16302f0f2f75"),
        document_set_id=23,
        package_id=UUID("4ddd28f3-a9a5-44de-be92-f871fb17469f"),
        user_id=UUID(int=9),
        batch_id=None,
        phase="cleaned",
        name="private run",
    )
    package = SimpleNamespace(
        id=run.package_id,
        document_set_id=23,
        environment="dev",
        created_by=run.user_id,
        idempotency_key="annex-canary-" + str(run.run_id),
        created_at=datetime.datetime.fromisoformat("2026-09-11T15:56:39.875587+00:00"),
        status="failed",
        issues=[{"code": "parse_failed", "locator": "PRIVATE"}],
        asset_count=0,
        manifest_file_id=None,
        manifest_sha256=None,
        input_file_id="saved",
        input_spec={"mime_type": "application/pdf"},
    )
    scope = SimpleNamespace(is_public=False, user_id=run.user_id, name=run.name)
    return run, package, scope


@pytest.mark.parametrize(
    "target,field,value",
    [
        (0, "release_sha", "a" * 40),
        (0, "file_id", UUID(int=1)),
        (0, "batch_id", 49),
        (0, "phase", "running"),
        (1, "created_by", UUID(int=2)),
        (1, "document_set_id", 24),
        (1, "environment", "test"),
        (1, "idempotency_key", "other"),
        (2, "is_public", True),
        (2, "name", "other"),
    ],
)
def test_exact_retained_scope_refuses_foreign_state(
    target: int, field: str, value: Any
) -> None:
    objects = scope_objects()
    assert runner.source861_scope_matches(*objects)
    setattr(objects[target], field, value)
    assert not runner.source861_scope_matches(*objects)


def test_stored_concrete_issue_prevents_any_original_or_model_read() -> None:
    from onyx.db import amendment_sources, document_set, regulatory_annex_acceptance
    from onyx.db.engine import sql_engine
    from onyx.file_store import file_store
    from onyx.llm import factory

    run, package, scope = scope_objects()
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                "os.environ",
                {
                    "POSTGRES_DB": "customs-regulations-dev",
                    "REGULATORY_ANNEX_ENVIRONMENT": "dev",
                    "PGOPTIONS": "-c default_transaction_read_only=on",
                },
            )
        )
        engine = stack.enter_context(
            patch.object(sql_engine.SqlEngine, "scoped_engine")
        )
        stack.enter_context(patch.object(sql_engine, "get_session_with_current_tenant"))
        stack.enter_context(
            patch.object(runner, "load_source861_canary", return_value=run, create=True)
        )
        stack.enter_context(
            patch.object(amendment_sources, "get_source_package", return_value=package)
        )
        stack.enter_context(
            patch.object(document_set, "get_document_set_by_id", return_value=scope)
        )
        stack.enter_context(
            patch.object(
                regulatory_annex_acceptance,
                "verify_dev_configuration",
                return_value={"vision_provider": "vertex_ai"},
            )
        )
        store = stack.enter_context(patch.object(file_store, "get_default_file_store"))
        llm = stack.enter_context(patch.object(factory, "get_default_llm_with_vision"))
        result = report()
        runner.reproduce_source861(result)
    assert result["status"] == "stored_issues_only"
    assert result["issues"] == ["parse_failed"]
    assert "PRIVATE" not in json.dumps(result)
    store.assert_not_called()
    llm.assert_not_called()
    assert (
        engine.call_args.kwargs["connect_args"]["options"]
        == "-c default_transaction_read_only=on"
    )
    runner.validate_source861_report(result)


def test_safe_failure_preserves_known_code_and_schema_without_values() -> None:
    class Shape(BaseModel):
        elements: list[int]

    result = report()
    try:
        Shape.model_validate({"elements": ["PRIVATE-CREDENTIAL"]})
    except ValidationError as error:
        runner.source861_failure(result, error)
    assert result["schema_errors"] == [{"loc": ["elements", 0], "type": "int_parsing"}]
    assert "PRIVATE-CREDENTIAL" not in json.dumps(result)
    runner.validate_source861_report(result)
    runner.source861_failure(result, ValueError("pdf_table_cells_overlap"))
    assert result["failure_code"] == "pdf_table_cells_overlap"
    runner.source861_failure(result, ValueError("PRIVATE MESSAGE"))
    assert result["failure_code"] == "unknown"
    assert "PRIVATE MESSAGE" not in json.dumps(result)


@pytest.mark.parametrize(
    "key,value",
    [
        ("attempt_count", 13),
        ("http_request_count", 13),
        ("page_index", 5),
        ("prompt", "PRIVATE"),
        ("issues", ["PRIVATE"]),
        ("database_read_only", False),
    ],
)
def test_output_rejects_unbounded_or_unexpected_fields(key: str, value: Any) -> None:
    result = report()
    result[key] = value
    with pytest.raises(runner.CutoverRefusal):
        runner.validate_source861_report(result)


def test_fixed_child_roundtrip_and_early_routing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = Mock(spec=runner.Driver)
    driver.sha = runner.SOURCE861_RUNTIME
    driver.get.return_value = {"data": {"sha": driver.sha, "phase": "released"}}
    driver.pods.return_value = [
        {
            "metadata": {"name": "worker-pod"},
            "spec": {
                "containers": [
                    {"name": "worker", "image": runner.REPOSITORY + ":" + driver.sha}
                ]
            },
        }
    ]
    driver.command.return_value = json.dumps(report())
    with (
        patch.object(runner, "require_release_runs"),
        patch.object(runner, "verify_frontend"),
        patch.object(runner, "diagnose_batch44_logs") as obsolete,
    ):
        runner.diagnose_release(driver, "b" * 40)
    obsolete.assert_not_called()
    args = driver.command.call_args.args[0]
    assert "PGOPTIONS" in args[-3] and "read_only=on" in args[-3]
    assert args[-2] == "source861-diagnostic"
    program = args[-1]
    compile(program, "fixed-diagnostic", "exec")
    assert "session.get(" in program and "session.commit(" not in program
    assert "signal.alarm(220)" in program
    assert "MAX_PACKAGE_SECONDS" in program and "page_attempts >= 3" in program
    assert driver.command.call_args.kwargs["timeout"] == 240
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == report()


def test_dal_only_reads_fixed_key() -> None:
    from onyx.db.regulatory_annex_acceptance_diagnostic import load_source861_canary

    session = Mock()
    session.get.return_value = None
    assert load_source861_canary(session) is None
    assert (
        session.get.call_args.args[1]
        == "regulatory_annex_acceptance:" + runner.SOURCE861_RUNTIME
    )
    assert len(session.method_calls) == 1


@pytest.mark.parametrize("mode", ["hash_mismatch", "provider_refused", "prepared"])
def test_generic_failure_requires_original_hash_and_vertex_before_preparation(
    mode: str,
) -> None:
    import hashlib
    import time
    from io import BytesIO
    from pathlib import Path

    from onyx.db import amendment_sources, document_set, regulatory_annex_acceptance
    from onyx.db.engine import sql_engine
    from onyx.file_store import file_store
    from onyx.llm import factory
    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes import sources

    content = (
        Path(sources.__file__)
        .with_name("acceptance_fixtures")
        .joinpath("new.pdf")
        .read_bytes()
    )
    expected = "c5a20c8fd76d3dbd4983ceeaacc44eef217716427adec8716475ac5a753a0557"
    assert hashlib.sha256(content).hexdigest() == expected
    run, package, scope = scope_objects()
    package.issues = [{"code": "acquisition_failed"}]
    asset = SimpleNamespace(sha256=expected)
    llm = Mock()
    llm.config.model_provider = "vertex_ai"
    llm.config.model_dump_json.return_value = '{"test_model":true}'
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                "os.environ",
                {
                    "POSTGRES_DB": "customs-regulations-dev",
                    "REGULATORY_ANNEX_ENVIRONMENT": "dev",
                    "PGOPTIONS": "-c default_transaction_read_only=on",
                },
            )
        )
        stack.enter_context(patch.object(sql_engine.SqlEngine, "scoped_engine"))
        stack.enter_context(patch.object(sql_engine, "get_session_with_current_tenant"))
        stack.enter_context(
            patch.object(runner, "load_source861_canary", return_value=run, create=True)
        )
        stack.enter_context(
            patch.object(amendment_sources, "get_source_package", return_value=package)
        )
        stack.enter_context(
            patch.object(document_set, "get_document_set_by_id", return_value=scope)
        )
        stack.enter_context(
            patch.object(
                regulatory_annex_acceptance,
                "verify_dev_configuration",
                return_value={
                    "vision_provider": "openrouter"
                    if mode == "provider_refused"
                    else "vertex_ai"
                },
            )
        )
        store = stack.enter_context(patch.object(file_store, "get_default_file_store"))
        store.return_value.read_file.return_value = BytesIO(
            b"wrong" if mode == "hash_mismatch" else content
        )
        model = stack.enter_context(
            patch.object(factory, "get_default_llm_with_vision", return_value=llm)
        )
        acquire = stack.enter_context(
            patch.object(
                sources,
                "acquire_source_package",
                return_value=SimpleNamespace(
                    status="ready", issues=[], links=[], assets=[asset]
                ),
            )
        )
        prepare = stack.enter_context(
            patch.object(
                pdf_vision,
                "prepare_pdf_source",
                return_value=SimpleNamespace(
                    pdf_vision=SimpleNamespace(transcript_sha256="a" * 64)
                ),
            )
        )
        result = report()
        before = time.monotonic()
        runner.reproduce_source861(result)
    runner.validate_source861_report(result)
    if mode == "prepared":
        assert (
            result["status"] == "reproduction_passed" and result["reproduction"] is True
        )
        assert acquire.call_args.kwargs["content"] == content
        assert prepare.call_args.args == (asset,)
        assert prepare.call_args.kwargs["llm"] is llm
        assert (
            before + 179
            <= prepare.call_args.kwargs["deadline"]
            <= time.monotonic() + 180
        )
        assert type(prepare.call_args.kwargs["store"]).__name__ == "MemoryEvidenceStore"
    else:
        assert result["status"] == (
            "scope_refused" if mode == "hash_mismatch" else mode
        )
        prepare.assert_not_called()
        acquire.assert_not_called()
        model.assert_not_called()
