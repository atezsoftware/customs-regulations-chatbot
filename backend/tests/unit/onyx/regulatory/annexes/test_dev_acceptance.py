"""Fixed release probes refuse the wrong environment before touching services."""

import importlib.util
import shutil
from pathlib import Path

import pytest


def test_fixed_dev_entrypoint_is_packaged() -> None:
    assert (
        importlib.util.find_spec("onyx.regulatory.amendments.annexes.dev_acceptance")
        is not None
    )


@pytest.mark.parametrize(
    "database,environment,machine",
    [
        ("annex_local", "dev", "x86_64"),
        ("customs-regulations-dev", "test", "x86_64"),
        ("customs-regulations-dev", "dev", "arm64"),
    ],
)
def test_scope_refuses_non_dev_or_non_amd64_runtime(
    database: str, environment: str, machine: str
) -> None:
    from onyx.regulatory.amendments.annexes.dev_acceptance import validate_scope

    with pytest.raises(ValueError):
        validate_scope(database=database, environment=environment, machine=machine)


def test_fixture_manifest_verifies_every_packaged_byte() -> None:
    from onyx.regulatory.amendments.annexes.dev_acceptance import load_fixtures

    fixtures = load_fixtures()
    assert set(fixtures) == {"old.pdf", "new.pdf", "page.png"}
    assert fixtures["old.pdf"].startswith(b"%PDF-")
    assert fixtures["page.png"].startswith(b"\x89PNG\r\n\x1a\n")


def test_packaged_fixture_corruption_refuses_before_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.regulatory.amendments.annexes import dev_acceptance

    original = Path(dev_acceptance.__file__).with_name("acceptance_fixtures")
    copied = tmp_path / "acceptance_fixtures"
    shutil.copytree(original, copied)
    (copied / "old.pdf").write_bytes(b"corrupted")
    monkeypatch.setattr(dev_acceptance, "__file__", str(tmp_path / "dev_acceptance.py"))
    with pytest.raises(ValueError, match="fixed_fixture_hash_mismatch"):
        dev_acceptance.load_fixtures()


def test_disabled_canary_refuses_before_mutating_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments.annexes import acceptance_canary, config

    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", False)
    with pytest.raises(ValueError, match="explicit_creation_activation"):
        acceptance_canary.run_canary("a" * 40)


def test_failed_batch_refuses_without_polling_until_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from unittest.mock import Mock
    from uuid import uuid4

    import httpx

    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.regulatory.amendments.annexes import acceptance_canary

    run = CanaryRun(
        release_sha="a" * 40, user_id=uuid4(), document_set_id=3, batch_id=4
    )
    request = Mock(side_effect=[[], [{"id": 4, "status": "failed"}]])
    monkeypatch.setattr(acceptance_canary, "request_json", request)
    sleep = Mock()
    monkeypatch.setattr(acceptance_canary.time, "sleep", sleep)
    with httpx.Client() as client:
        with pytest.raises(ValueError, match="fictional_amendment_batch_failed"):
            acceptance_canary.wait_review(client, run, time.monotonic() + 600)
    sleep.assert_not_called()


def test_failed_canary_cleans_files_revokes_token_and_returns_owned_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock
    from uuid import uuid4

    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.regulatory.amendments.annexes import acceptance_canary, config

    run = CanaryRun(release_sha="a" * 40, user_id=uuid4())
    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    monkeypatch.setattr(acceptance_canary, "reserve_canary", Mock(return_value=run))
    monkeypatch.setattr(
        acceptance_canary, "issue_canary_token", Mock(return_value="memory-only")
    )
    monkeypatch.setattr(acceptance_canary, "request_json", Mock())
    monkeypatch.setattr(acceptance_canary, "save_canary", Mock())
    monkeypatch.setattr(
        acceptance_canary,
        "bootstrap_original",
        Mock(side_effect=RuntimeError("private detail")),
    )
    cleanup = Mock()
    revoke = Mock()
    monkeypatch.setattr(acceptance_canary, "cleanup_canary", cleanup)
    monkeypatch.setattr(acceptance_canary, "revoke_canary_token", revoke)
    report = acceptance_canary.run_canary(run.release_sha)
    assert report["status"] == "failed"
    assert report["canary"]["run_id"] == str(run.run_id)
    assert report["canary"]["evidence"]["acceptance_passed"] is False
    assert "private detail" not in str(report)
    assert "memory-only" not in str(report)
    cleanup.assert_called_once_with(run)
    revoke.assert_called_once_with(run)


def test_preflight_prints_failed_calibration_once_before_gate_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    import sys
    from unittest.mock import Mock

    from onyx.db import regulatory_annex_acceptance
    from onyx.regulatory.amendments.annexes import (
        acceptance_calibration,
        dev_acceptance,
    )

    monkeypatch.setenv("POSTGRES_DB", "customs-regulations-dev")
    monkeypatch.setenv("REGULATORY_ANNEX_ENVIRONMENT", "dev")
    monkeypatch.setenv("ANNEX_ACCEPTANCE_RELEASE_SHA", "a" * 40)
    monkeypatch.setattr(sys, "argv", ["dev_acceptance", "preflight"])
    monkeypatch.setattr(dev_acceptance.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        regulatory_annex_acceptance,
        "verify_dev_configuration",
        Mock(return_value={"database": "customs-regulations-dev"}),
    )
    monkeypatch.setattr(
        dev_acceptance, "native_parser_probe", Mock(return_value={"pdf_pages": 4})
    )
    calibration = Mock(
        return_value={
            "status": "failed",
            "cases": [{"supported": False}],
            "attempt_count": 1,
        }
    )
    monkeypatch.setattr(acceptance_calibration, "run_native_calibration", calibration)
    with pytest.raises(SystemExit) as result:
        dev_acceptance.main()
    assert result.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["calibration"]["cases"] == [{"supported": False}]
    calibration.assert_called_once_with()


@pytest.mark.parametrize("kind", ["chat", "markdown"])
def test_lost_creation_response_recovers_from_precommitted_intent(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock
    from uuid import uuid4

    import httpx

    from onyx.db import regulatory_annex_acceptance as ownership
    from onyx.regulatory.amendments.annexes import acceptance_canary as canary

    run = ownership.CanaryRun(release_sha="a" * 40, user_id=uuid4(), document_set_id=42)
    durable: dict[str, object] = {}
    created: list[object] = []

    def save(value: ownership.CanaryRun) -> None:
        durable.update(value.model_dump(mode="json"))

    def server_committed_response_lost(*_args: object, **_kwargs: object) -> None:
        if kind == "markdown":
            assert _args[2] == "/manage/admin/document-set/42/file/upload"
        persisted = ownership.CanaryRun.model_validate(durable)
        assert len(persisted.creation_intents) == 1
        assert persisted.creation_intents[0].kind == kind
        assert persisted.creation_intents[0].artifact_id is None
        created.append(uuid4())
        raise httpx.ReadTimeout("response lost after commit")

    monkeypatch.setattr(canary, "save_canary", save)
    monkeypatch.setattr(ownership, "save_canary", save)
    monkeypatch.setattr(canary, "request_json", server_committed_response_lost)
    with httpx.Client() as client:
        with pytest.raises(httpx.ReadTimeout):
            if kind == "chat":
                canary.create_canary_chat(client, run, purpose="dated 2026-09-09")
            else:
                canary.upload_canary_markdown(client, run)
    recovered = ownership.CanaryRun.model_validate(durable)
    monkeypatch.setattr(ownership, "matching_creation_ids", Mock(return_value=created))
    assert ownership.recover_creation_intents(recovered) is True
    assert recovered.creation_intents[0].artifact_id == created[0]
    identifiers = recovered.chat_ids if kind == "chat" else recovered.markdown_file_ids
    assert identifiers == created


def test_unresolved_creation_never_claims_cleaned_or_zero_live_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock
    from uuid import uuid4

    from onyx.db import regulatory_annex_acceptance as ownership
    from onyx.db import regulatory_writer_publication
    from onyx.regulatory.amendments.annexes import acceptance_canary as canary

    run = ownership.CanaryRun(release_sha="a" * 40, user_id=uuid4())
    monkeypatch.setattr(canary, "recover_creation_intents", Mock(return_value=False))
    monkeypatch.setattr(canary, "save_canary", Mock())
    monkeypatch.setattr(
        regulatory_writer_publication, "writer_file_exists", Mock(return_value=False)
    )
    monkeypatch.setattr(
        canary, "canary_file_state", Mock(return_value={"file_exists": False})
    )
    monkeypatch.setattr(canary, "index_evidence", Mock(return_value=[]))
    monkeypatch.setattr(canary, "cleanup_empty_canary_scope", Mock())
    monkeypatch.setattr(canary, "retained_canary_audit", Mock(return_value=[]))
    with pytest.raises(ValueError, match="canary_creation_unresolved"):
        canary.cleanup_canary(run)
    assert run.phase == "cleanup_incomplete"
    assert run.evidence["cleanup_complete"] is False
    assert "cleanup_live_projections" not in run.evidence


@pytest.mark.parametrize("matches_count", [0, 2])
def test_uncertain_creation_requires_exactly_one_owned_match(
    matches_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock
    from uuid import uuid4

    from onyx.db import regulatory_annex_acceptance as ownership

    run = ownership.CanaryRun(release_sha="a" * 40, user_id=uuid4())
    run.creation_intents = [
        ownership.CreationIntent(
            kind="markdown", marker="ANNEXCANARY" + run.run_id.hex + ".md"
        )
    ]
    monkeypatch.setattr(
        ownership,
        "matching_creation_ids",
        Mock(return_value=[uuid4() for _ in range(matches_count)]),
    )
    monkeypatch.setattr(ownership, "save_canary", Mock())
    assert ownership.recover_creation_intents(run) is False
    assert run.creation_intents[0].artifact_id is None
    assert run.markdown_file_ids == []


def test_retained_source_scope_cannot_hide_remaining_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, Mock
    from uuid import uuid4

    from onyx.db import regulatory_annex_acceptance as ownership
    from onyx.file_store import file_store

    run = ownership.CanaryRun(release_sha="a" * 40, user_id=uuid4(), document_set_id=42)
    session = MagicMock()
    session.get.return_value = SimpleNamespace(
        id=42,
        name=run.name,
        user_id=run.user_id,
        is_public=False,
        user_files=[object()],
    )
    session.scalar.return_value = uuid4()
    context = MagicMock()
    context.__enter__.return_value = session
    monkeypatch.setattr(
        ownership, "get_session_with_current_tenant", Mock(return_value=context)
    )
    monkeypatch.setattr(file_store, "get_default_file_store", Mock(return_value=Mock()))
    with pytest.raises(ValueError, match="canary_cleanup_scope_still_has_files"):
        ownership.cleanup_empty_canary_scope(run)
    assert "retained_source_scope" not in run.evidence


@pytest.mark.parametrize("include_header", [True, False])
def test_canary_retains_actual_explicit_vision_roles(
    include_header: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock
    from uuid import uuid4

    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.regulatory.amendments.annexes import acceptance_canary as canary

    run = CanaryRun(release_sha="a" * 40, user_id=uuid4())
    monkeypatch.setattr(canary, "save_canary", Mock())
    elements = [
        {
            "kind": "table_cell",
            "extraction_method": "vision",
            "table_role": role,
            "locator": {"page": 1},
        }
        for role in (["column_header", "data"] if include_header else ["data"])
    ]
    review = {
        "review_payload": {
            "old_extraction": {"source_sha256": "b" * 64, "elements": []},
            "new_extraction": {"source_sha256": "c" * 64, "elements": elements},
        }
    }
    if not include_header:
        with pytest.raises(
            ValueError, match="canary_explicit_vision_header_and_data_required"
        ):
            canary.record_vision_roles(run, review)
    else:
        canary.record_vision_roles(run, review)
        assert [item["table_role"] for item in run.vision_roles] == [
            "column_header",
            "data",
        ]
        assert [item["original_position"] for item in run.vision_roles] == [0, 1]
        assert all(item["source_sha256"] == "c" * 64 for item in run.vision_roles)


@pytest.mark.parametrize("mapping_fault", [None, "missing", "duplicate", "parent"])
def test_two_png_role_receipts_bind_exact_originals(
    mapping_fault: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import hashlib
    import json
    from pathlib import Path
    from unittest.mock import Mock
    from uuid import uuid4

    from scripts.regulatory_annex_dev_cutover import emit_acceptance_report

    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.regulatory.amendments.annexes import acceptance_canary as canary

    extraction = json.loads(
        (Path(__file__).parent / "fixtures/two_png_role_extraction.json").read_text()
    )
    view = extraction["evidence_view"]
    mapping = next(
        item for item in view["element_mappings"] if item["view_position"] == 18
    )
    if mapping_fault == "missing":
        view["element_mappings"].remove(mapping)
    elif mapping_fault == "duplicate":
        view["element_mappings"].append(mapping.copy())
    elif mapping_fault == "parent":
        mapping["parent_index"] = -1
    run = CanaryRun(release_sha="a" * 40, user_id=uuid4())
    monkeypatch.setattr(canary, "save_canary", Mock())
    review = {
        "review_payload": {
            "old_extraction": {"source_sha256": "b" * 64, "elements": []},
            "new_extraction": extraction,
        }
    }
    if mapping_fault:
        with pytest.raises(ValueError, match="canary_vision_role_mapping_invalid"):
            canary.record_vision_roles(run, review)
        return
    canary.record_vision_roles(run, review)
    headers = [
        item for item in run.vision_roles if item["table_role"] == "column_header"
    ]
    assert [item["source_sha256"] for item in headers] == [
        "5768ac5e2765ba93b207ed8c719a9b5fd95a2a0c7736a1611382f9e26a41c091"
    ] * 3 + ["6786fff866181da78feb6834133dce01b5cfe0b0a2319e92111cb5838f554789"] * 3
    assert [item["original_position"] for item in headers] == [3, 4, 5, 3, 4, 5]
    assert [item["view_position"] for item in headers] == [3, 4, 5, 18, 19, 20]
    for receipt in run.vision_roles:
        original = next(
            item
            for item in view["element_mappings"]
            if item["view_position"] == receipt["view_position"]
        )
        assert (
            json.loads(str(receipt["original_locator"])) == original["original_locator"]
        )
        assert (
            receipt["original_locator_sha256"]
            == hashlib.sha256(str(receipt["original_locator"]).encode()).hexdigest()
        )
        assert receipt["view_sha256"] == view["sha256"]
    emit_acceptance_report(
        json.dumps(
            {
                "phase": "canary",
                "status": "passed",
                "release_sha_metadata": run.release_sha,
                "canary": run.model_dump(mode="json"),
            }
        ),
        "canary",
        run.release_sha,
    )
    assert (
        json.loads(capsys.readouterr().out)["canary"]["vision_roles"]
        == run.vision_roles
    )


def test_markdown_probe_uses_registered_upload_list_and_index_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from typing import Any
    from unittest.mock import Mock
    from uuid import uuid4

    import httpx
    from fastapi.routing import APIRoute

    from onyx.db.regulatory_annex_acceptance import CanaryRun
    from onyx.regulatory.amendments.annexes import acceptance_canary as canary
    from onyx.server.features.document_set.api import router

    run = CanaryRun(release_sha="a" * 40, user_id=uuid4(), document_set_id=42)
    file = {
        "id": str(uuid4()),
        "file_id": str(uuid4()),
        "name": "ANNEXCANARY" + run.run_id.hex + ".md",
        "chat_file_type": "plain_text",
    }
    operations: list[str] = []
    indexed = False

    def request(_client: httpx.Client, method: str, path: str, **_kwargs: Any) -> Any:
        nonlocal indexed
        if "/document-set/" in path:
            routes = [
                route
                for route in router.routes
                if isinstance(route, APIRoute)
                and method in route.methods
                and route.path_regex.fullmatch(path)
            ]
            assert len(routes) == 1, (method, path)
            operations.append(method + " " + routes[0].path)
            if path.endswith("/file/upload"):
                return {"rejected_files": [], "user_files": [file]}
            if path.endswith("/index"):
                indexed = True
                return {}
            return [{**file, "status": "COMPLETED" if indexed else "CHUNKED"}]
        if path == "/tool":
            return [{"id": 1, "display_name": "Internal Search"}]
        if path == "/chat/create-chat-session":
            return {"chat_session_id": str(uuid4())}
        assert path == "/chat/send-chat-message"
        assert _kwargs["json"]["forced_tool_id"] == 1
        assert _kwargs["json"]["internal_search_filters"] == {
            "document_set": [run.name]
        }
        assert "ANNEXCANARY" not in _kwargs["json"]["message"]
        return {
            "answer": "ANNEXCANARY" + run.run_id.hex,
            "top_documents": [{"document_id": file["id"]}],
            "citation_info": [{"citation_num": 1}],
        }

    monkeypatch.setattr(canary, "request_json", request)
    monkeypatch.setattr(canary, "save_canary", Mock())
    monkeypatch.setattr(canary.time, "sleep", Mock())
    with httpx.Client() as client:
        canary.markdown_canary(client, run, time.monotonic() + 10)
    assert len(operations) == 4
    assert indexed
    assert run.evidence["ordinary_markdown_upload_index_chat"] is True
