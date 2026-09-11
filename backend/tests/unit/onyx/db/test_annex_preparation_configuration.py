import hashlib
import json
from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from onyx.db import regulatory_annex_changes as changes
from onyx.db.models import AmendmentBatch, SearchSettings, UserFile
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexProjectionAccess,
)
from onyx.utils.sensitive import SensitiveAccessError, SensitiveValue


def credential(value: str) -> SensitiveValue[str]:
    cipher = Fernet(Fernet.generate_key())
    return SensitiveValue(
        encrypted_bytes=cipher.encrypt(value.encode()),
        decrypt_fn=lambda data: cipher.decrypt(data).decode(),
    )


@pytest.fixture
def mapped_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, UserFile, SearchSettings]:
    from onyx.db import search_settings

    file = UserFile(id=uuid4(), name="fixture document")
    settings = SearchSettings(
        id=1, model_name="fixture", model_dim=3072, reduced_dimension=1024
    )
    session = MagicMock(spec=Session)
    session.get.return_value = file
    monkeypatch.setattr(
        search_settings, "get_current_search_settings", lambda _session: settings
    )
    return session, file, settings


def test_mapped_sensitive_credential_is_stable_private_and_rotation_sensitive(
    mapped_configuration: tuple[MagicMock, UserFile, SearchSettings],
    capsys: pytest.CaptureFixture[str],
) -> None:
    session, file, settings = mapped_configuration
    secret = "fictional-rerank-secret-one"
    settings.rerank_api_key = credential(secret)
    first = changes.capture_preparation_configuration(session, user_file_id=file.id)
    settings.rerank_api_key = credential(secret)
    assert (
        changes.capture_preparation_configuration(session, user_file_id=file.id)
        == first
    )
    settings.rerank_api_key = credential("fictional-rerank-secret-two")
    rotated = changes.capture_preparation_configuration(session, user_file_id=file.id)
    assert rotated["search_settings"] != first["search_settings"]
    assert rotated["user_file"] == first["user_file"]
    assert rotated["runtime_policy"] == first["runtime_policy"]
    assert all(len(value) == 64 for value in [*first.values(), *rotated.values()])
    captured = capsys.readouterr()
    assert secret not in json.dumps([first, rotated]) + captured.out + captured.err
    with pytest.raises(SensitiveAccessError):
        str(settings.rerank_api_key)
    with pytest.raises(SensitiveAccessError):
        context_hash(settings.rerank_api_key)


def test_non_sensitive_mapped_hashes_remain_exactly_legacy(
    mapped_configuration: tuple[MagicMock, UserFile, SearchSettings],
) -> None:
    session, file, settings = mapped_configuration
    actual = changes.capture_preparation_configuration(session, user_file_id=file.id)
    for name, row in (("user_file", file), ("search_settings", settings)):
        assert actual[name] == context_hash(
            {
                column.key: getattr(row, column.key)
                for column in inspect(type(row)).columns
                if column.key not in ("created_at", "updated_at", "last_accessed_at")
            }
        )
    from onyx.db.regulatory_annex_publication import publication_input_scope_hash

    access = projection_access()
    legacy_rows = [
        {
            column.key: getattr(row, column.key)
            for column in inspect(type(row)).columns
            if column.key not in ("created_at", "updated_at", "last_accessed_at")
        }
        for row in (file, settings)
    ]
    assert publication_input_scope_hash(file, [settings], access) == context_hash(
        [legacy_rows[0], [legacy_rows[1]], access.model_dump(mode="json")]
    )
    settings.model_dim = 1024
    assert (
        changes.capture_preparation_configuration(session, user_file_id=file.id)[
            "search_settings"
        ]
        != actual["search_settings"]
    )


@pytest.mark.parametrize("rotate", [False, True])
def test_prepared_guard_rejects_rotation_before_staging(
    mapped_configuration: tuple[MagicMock, UserFile, SearchSettings],
    monkeypatch: pytest.MonkeyPatch,
    rotate: bool,
) -> None:
    from onyx.regulatory.amendments.annexes import evidence, patch_plan, staging

    session, file, settings = mapped_configuration
    settings.rerank_api_key = credential("fictional-first")
    frozen = changes.capture_preparation_configuration(session, user_file_id=file.id)
    batch = AmendmentBatch(raw_text="source", document_set_id=1, created_by=None)
    batch.source_text_sha256 = hashlib.sha256(batch.raw_text.encode()).hexdigest()
    package = MagicMock(created_by=None, manifest_sha256="manifest")
    draft = MagicMock(spec=AnnexChangeDraft)
    for name in (
        "source_package_id",
        "baseline",
        "old_extraction",
        "new_extraction",
        "comparison",
        "patch_plan",
        "impact",
        "baseline_context",
        "evidence",
    ):
        setattr(draft, name, MagicMock())
    draft.items = []
    draft.baseline_scope = []
    draft.insertion_after_chunk_id = None
    draft.user_file_id = file.id
    draft.batch_id = None
    draft.source_text_sha256 = batch.source_text_sha256
    draft.source_manifest_sha256 = package.manifest_sha256
    draft.preparation_configuration = frozen
    draft.effective_date = date(2026, 9, 11)
    for name in (
        "submitted_source_text",
        "date_resolution",
        "publication",
        "raw_new_extraction",
        "new_evidence_remapping",
    ):
        setattr(draft, name, None)
    draft.source_only_canonical_ids = []
    draft.new_extraction.evidence_view = None
    monkeypatch.setattr(
        changes, "require_ready_source_package", lambda *_args, **_kwargs: package
    )
    monkeypatch.setattr(changes, "list_source_assets", lambda *_args: [])
    monkeypatch.setattr(
        patch_plan, "prepare_annex_patch", lambda **_kwargs: draft.patch_plan
    )
    monkeypatch.setattr(evidence, "validate_compared_evidence", lambda **_kwargs: None)
    reached_staging = MagicMock(side_effect=RuntimeError("staging reached"))
    monkeypatch.setattr(staging, "validate_staged_items", reached_staging)
    settings.rerank_api_key = credential(
        "fictional-rotated" if rotate else "fictional-first"
    )
    with pytest.raises(
        ValueError if rotate else RuntimeError,
        match="prepared file or search configuration changed"
        if rotate
        else "staging reached",
    ):
        changes.validate_prepared_annex_change(
            session, batch=batch, draft=draft, environment="dev"
        )
    assert reached_staging.call_count == (0 if rotate else 1)
    assert draft.preparation_configuration == frozen


def projection_access() -> "AnnexProjectionAccess":
    from onyx.access.models import DocumentAccess
    from onyx.regulatory.amendments.annexes.models import AnnexProjectionAccess

    return AnnexProjectionAccess(
        access=DocumentAccess.build(
            user_emails=["fixture@example.com"],
            user_groups=[],
            external_user_emails=[],
            external_user_group_ids=[],
            is_public=False,
        ),
        project_ids=[1],
        persona_ids=[2],
        document_sets=["fixture"],
    )


def test_publication_mapped_fingerprint_preserves_logical_secrets_order_and_acl(
    mapped_configuration: tuple[MagicMock, UserFile, SearchSettings],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from onyx.db.regulatory_annex_publication import publication_input_scope_hash

    session, file, settings = mapped_configuration
    access = projection_access()
    future = SearchSettings(id=2, model_name="future", model_dim=3072)
    secret = "fictional-private-reranker"
    settings.rerank_api_key = credential(secret)
    prepared = changes.capture_preparation_configuration(session, user_file_id=file.id)
    published = publication_input_scope_hash(file, [settings, future], access)

    def legacy_columns(row: UserFile | SearchSettings) -> dict[str, object]:
        return {
            column.key: secret
            if column.key == "rerank_api_key" and row is settings
            else getattr(row, column.key)
            for column in inspect(type(row)).columns
            if column.key not in ("created_at", "updated_at", "last_accessed_at")
        }

    assert published == context_hash(
        [
            legacy_columns(file),
            [legacy_columns(settings), legacy_columns(future)],
            access.model_dump(mode="json"),
        ]
    )
    settings.rerank_api_key = credential(secret)
    assert publication_input_scope_hash(file, [future, settings], access) == published
    assert (
        changes.capture_preparation_configuration(session, user_file_id=file.id)
        == prepared
    )
    settings.rerank_api_key = credential("fictional-rotated")
    assert publication_input_scope_hash(file, [settings, future], access) != published
    assert (
        changes.capture_preparation_configuration(session, user_file_id=file.id)
        != prepared
    )
    settings.rerank_api_key = credential(secret)
    assert (
        publication_input_scope_hash(
            file, [settings, future], access.model_copy(update={"project_ids": [3]})
        )
        != published
    )
    future.model_dim = 1024
    assert publication_input_scope_hash(file, [settings, future], access) != published
    captured = capsys.readouterr()
    assert secret not in published + json.dumps(prepared) + captured.out + captured.err
    with pytest.raises(SensitiveAccessError):
        context_hash(settings.rerank_api_key)


@pytest.mark.parametrize("boundary", ["baseline", "runtime"])
@pytest.mark.parametrize("rotate", [False, True])
def test_publication_guards_reject_credential_rotation(
    mapped_configuration: tuple[MagicMock, UserFile, SearchSettings],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    rotate: bool,
) -> None:
    from contextlib import nullcontext

    from onyx.db import regulatory_annex_execution as execution
    from onyx.db import regulatory_annex_publication as publication

    session, file, settings = mapped_configuration
    access = projection_access()
    settings.rerank_api_key = credential("fictional-first")
    prepared = MagicMock()
    prepared.input_scope_sha256 = publication.publication_input_scope_hash(
        file, [settings], access
    )
    prepared.scope.tenant_id = "public"
    prepared.indexes = []
    draft = MagicMock(spec=AnnexChangeDraft)
    draft.user_file_id = file.id
    monkeypatch.setattr(
        publication,
        "load_annex_publication_inputs",
        lambda *_args: (file, [settings], access),
    )
    monkeypatch.setattr(
        execution, "get_session_with_tenant", lambda **_kwargs: nullcontext(session)
    )
    monkeypatch.setattr(changes, "lock_annex_preparation_scope", lambda *_args: None)
    monkeypatch.setattr(execution, "PublicationStore", MagicMock())
    monkeypatch.setattr(
        "onyx.db.regulatory_physical_indexes.validate_physical_index_snapshots",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_amendments.get_batch", lambda *_args: MagicMock()
    )
    retained_guard = MagicMock()
    monkeypatch.setattr(changes, "validate_prepared_annex_change", retained_guard)
    settings.rerank_api_key = credential(
        "fictional-rotated" if rotate else "fictional-first"
    )

    def check() -> object:
        if boundary == "baseline":
            return execution.validate_database_baseline(session, draft, prepared)
        return execution.load_validated_runtime_settings(draft, prepared)

    if rotate:
        with pytest.raises(
            ValueError, match="publication file/ACL/index configuration changed"
        ):
            check()
        retained_guard.assert_not_called()
    else:
        result = check()
        if boundary == "runtime":
            assert result == [settings]
        else:
            retained_guard.assert_called_once()
