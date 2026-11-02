"""Larger source tables must preserve rows without changing prose or retries."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake
from onyx.db import document_intake_edit_budget as edit_budget
from onyx.db.models import RegulatoryChunk, User, UserFile
from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.models import (
    IntakePreparationPolicy,
    IntakeSourceAsset,
)
from onyx.regulatory.document_intake.preparation import build_intake_document
from shared_configs.enums import EmbeddingProvider
from tests.unit.onyx.regulatory.whole_table_policy_helpers import _context


def test_legacy_document_keeps_optional_null_link_fields_in_its_receipt_input() -> None:
    context = _context()
    asset = IntakeSourceAsset(
        id=context.asset_id,
        package_id=context.package_id,
        sha256=context.asset_sha256,
        file_id="retained",
        text_file_id=None,
        text_sha256=None,
        mime_type="text/markdown",
        display_name="annex.md",
        byte_count=10,
    )
    link = SourceLink(
        parent_asset_hash="b" * 64,
        target_asset_hash=asset.sha256,
        source_field="href",
        label="Ek",
        kind="url",
    )
    document = build_intake_document(
        package_id=context.package_id,
        asset=asset,
        user_file_id=uuid4(),
        text="MADDE 1 — Özgün belge sunulur.",
        links=[link],
    )
    assert document.doc_metadata is not None
    stored = document.doc_metadata["regulatory_document_intake"]
    assert stored["source_links"] == [link.model_dump(mode="json")]
    assert (
        "parent_url" in stored["source_links"][0]
        and stored["source_links"][0]["parent_url"] is None
    )
    assert "preparation_policy" not in stored


@pytest.mark.parametrize(
    ("provider", "model", "contextual", "eligible"),
    [
        ("google", "gemini-embedding-2", True, True),
        ("google", "gemini-embedding-2", False, False),
        ("google", "gemini-embedding-001", True, False),
        ("openrouter", "gemini-embedding-2", True, False),
        (None, "gemini-embedding-2", True, False),
    ],
)
def test_admission_freezes_policy_only_for_the_exact_active_model(
    monkeypatch: pytest.MonkeyPatch,
    provider: str | None,
    model: str,
    contextual: bool,
    eligible: bool,
) -> None:
    owner = cast(User, SimpleNamespace(id=uuid4()))
    package = SimpleNamespace(id=uuid4(), created_by=owner.id)
    monkeypatch.setattr(
        document_intake, "resolve_intake_target", lambda *_args: SimpleNamespace(id=15)
    )
    monkeypatch.setattr(
        document_intake,
        "require_ready_source_package",
        lambda *_args, **_kwargs: package,
    )
    monkeypatch.setattr(
        document_intake,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: None,
    )
    settings = MagicMock(
        return_value=SimpleNamespace(
            provider_type=provider, model_name=model, enable_contextual_rag=True
        )
    )
    monkeypatch.setattr(
        document_intake,
        "effective_contextual_rag_enabled",
        lambda _settings: contextual,
    )
    monkeypatch.setattr(document_intake, "get_current_search_settings", settings)
    session = MagicMock(spec=Session)
    session.scalar.return_value = None
    row, created = document_intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=46,
        user=owner,
        environment="dev",
    )
    assert created
    assert (row.preparation_policy is not None) is eligible
    session.scalar.return_value = row
    settings.reset_mock()
    again, created = document_intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=46,
        user=owner,
        environment="dev",
    )
    assert not created and again is row
    settings.assert_not_called()


@pytest.mark.parametrize(
    ("chunk_type", "repeats", "accepted"),
    [
        ("table", 300, True),
        ("table", 1000, False),
        ("paragraph", 300, False),
    ],
)
def test_review_edit_enforces_the_same_frozen_table_and_body_budgets(
    monkeypatch: pytest.MonkeyPatch, chunk_type: str, repeats: int, accepted: bool
) -> None:
    policy = IntakePreparationPolicy().model_dump(mode="json")
    settings = SimpleNamespace(
        provider_type=EmbeddingProvider.GOOGLE, model_name="gemini-embedding-2"
    )
    monkeypatch.setattr(
        edit_budget, "get_current_search_settings", lambda _session: settings
    )
    monkeypatch.setattr(
        edit_budget, "effective_contextual_rag_enabled", lambda _settings: True
    )
    monkeypatch.setattr(
        edit_budget,
        "_embedding_tokenizer",
        lambda _settings: TiktokenTokenizer("text-embedding-3-large"),
    )
    generation = edit_budget.compute_regulatory_chunk_generation_hash(
        embedding_provider=EmbeddingProvider.GOOGLE,
        embedding_model_name=settings.model_name,
        enable_contextual_rag=True,
        intake_chunking_version="4",
        intake_preparation_policy=policy,
    )
    file_id = uuid4()
    row = RegulatoryChunk(
        id="whole-table",
        user_file_id=file_id,
        text="| 0101 | " + "menşeli " * repeats + "| İmalat |",
        chunk_type=chunk_type,
        chunk_metadata={
            "intake_chunking_version": "4",
            "intake_preparation_policy": policy,
            "chunk_variant": "atomic",
        },
    )
    files = {
        file_id: cast(
            UserFile, SimpleNamespace(regulatory_chunk_generation_hash=generation)
        )
    }
    if accepted:
        edit_budget.validate_intake_edit_budgets(
            MagicMock(spec=Session), files, {file_id: [row]}
        )
    else:
        with pytest.raises(ValueError, match="content limit"):
            edit_budget.validate_intake_edit_budgets(
                MagicMock(spec=Session), files, {file_id: [row]}
            )
