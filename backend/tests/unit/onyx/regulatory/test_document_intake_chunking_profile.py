"""Scoped chunking identities remain compatible with unversioned generations."""

import json
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db.enums import RegulatoryIndexingStage, UserFileStatus
from onyx.db.models import RegulatoryChunk
from onyx.regulatory.document_intake.chunking_profile import (
    INTAKE_CHUNKING_CODE_VERSION,
    documents_intake_chunking_version,
    intake_chunking_version,
)
from onyx.regulatory.document_intake.models import IntakeSourceContext
from onyx.regulatory.indexing import (
    _as_indexing_document,
    _group_indexing_documents_by_file,
)
from onyx.regulatory.indexing_jobs import configuration, preparation
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
from shared_configs.enums import EmbeddingProvider
from tests.unit.onyx.regulatory.indexing_jobs.test_configuration import (
    _install_admin_configuration,
)


def _document(*, version: str | None = None, legacy_context: bool = False) -> Document:
    context = IntakeSourceContext(
        package_id=uuid4(),
        asset_id=uuid4(),
        asset_sha256="a" * 64,
        parent_asset_hashes=[],
        parent_asset_ids=[],
        parent_titles=[],
        source_links=[],
    ).model_dump(mode="json")
    if legacy_context:
        del context["chunking_version"]
    elif version is not None:
        context["chunking_version"] = version
    return Document(
        id=str(uuid4()),
        source=DocumentSource.USER_FILE,
        semantic_identifier="test.md",
        metadata={},
        sections=[TextSection(text="MADDE 1 — Belge ibraz edilir.", link=None)],
        doc_metadata={"regulatory_document_intake": context}
        if version is not None or legacy_context
        else {},
    )


def test_unversioned_source_documents_keep_legacy_profile() -> None:
    plain, old_intake = _document(), _document(legacy_context=True)
    assert intake_chunking_version(plain) is None
    assert intake_chunking_version(old_intake) is None
    assert documents_intake_chunking_version([plain, old_intake]) is None
    assert documents_intake_chunking_version([]) is None


@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_explicit_profile_requires_supported_complete_context_and_uniform_documents(
    version: str,
) -> None:
    first, second = _document(version=version), _document(version=version)
    assert intake_chunking_version(first) == version
    assert documents_intake_chunking_version([first, second]) == version
    with pytest.raises(ValueError, match="Mixed"):
        documents_intake_chunking_version([first, _document()])
    with pytest.raises(ValueError, match="Mixed"):
        documents_intake_chunking_version(
            [first, _document(version="2" if version == "1" else "1")]
        )
    with pytest.raises(ValueError):
        intake_chunking_version(_document(version="4"))
    invalid = first.model_copy(
        update={
            "doc_metadata": {"regulatory_document_intake": {"chunking_version": "1"}}
        }
    )
    with pytest.raises(ValueError):
        intake_chunking_version(invalid)
    malformed = first.model_copy(
        update={"doc_metadata": {"regulatory_document_intake": "1"}}
    )
    with pytest.raises(ValueError, match="verified source context"):
        intake_chunking_version(malformed)


def test_file_section_grouping_rejects_mixed_profiles_before_first_metadata_wins() -> (
    None
):
    legacy, versioned = _document(), _document(version="1")
    versioned.id = legacy.id
    for documents in ([legacy, versioned], [versioned, legacy]):
        with pytest.raises(ValueError, match="Mixed"):
            _group_indexing_documents_by_file(
                [_as_indexing_document(document) for document in documents]
            )
    grouped = _group_indexing_documents_by_file(
        [_as_indexing_document(legacy), _as_indexing_document(legacy)]
    )
    assert len(grouped) == 1
    assert len(grouped[0].processed_sections or []) == 2


@pytest.mark.parametrize("contextual", [False, True])
@pytest.mark.parametrize("provider", [None, EmbeddingProvider.OPENROUTER])
def test_default_generation_hash_is_byte_exact_original_payload(
    contextual: bool, provider: EmbeddingProvider | None
) -> None:
    payload = {
        "identity_version": "regulatory-chunk-generation-v1",
        "chunker_code_version": configuration.REGULATORY_CHUNKER_CODE_VERSION,
        "indexing_generation_code_version": configuration.REGULATORY_INDEXING_GENERATION_CODE_VERSION,
        "embedding_provider": provider.value if provider is not None else "local",
        "embedding_model_name": "openai/text-embedding-3-large",
        "enable_contextual_rag": contextual,
        "regulatory_max_chunk_chars": configuration.REGULATORY_MAX_CHUNK_CHARS,
        "regulatory_contextual_initial_max_chunk_chars": configuration.REGULATORY_CONTEXTUAL_INITIAL_MAX_CHUNK_CHARS,
        "regulatory_contextual_min_max_chunk_chars": configuration.REGULATORY_CONTEXTUAL_MIN_MAX_CHUNK_CHARS,
        "document_embedding_context_size": configuration.DOC_EMBEDDING_CONTEXT_SIZE,
        "blurb_size": configuration.BLURB_SIZE,
        "default_contextual_reserved_tokens": configuration.DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
        "minimum_contextual_reserved_tokens": configuration.MIN_CONTEXTUAL_RAG_RESERVED_TOKENS,
    }
    expected = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    actual = configuration.compute_regulatory_chunk_generation_hash(
        embedding_provider=provider,
        embedding_model_name="openai/text-embedding-3-large",
        enable_contextual_rag=contextual,
    )
    assert actual == expected
    assert (
        configuration.compute_regulatory_chunk_generation_hash(
            embedding_provider=provider,
            embedding_model_name="openai/text-embedding-3-large",
            enable_contextual_rag=contextual,
            intake_chunking_version=None,
        )
        == expected
    )
    assert (
        configuration.compute_regulatory_chunk_generation_hash(
            embedding_provider=provider,
            embedding_model_name="openai/text-embedding-3-large",
            enable_contextual_rag=contextual,
            intake_chunking_version="1",
        )
        != expected
    )
    version_one_payload = {**payload, "intake_chunking_version": "1"}
    version_one_hash = sha256(
        json.dumps(version_one_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert (
        configuration.compute_regulatory_chunk_generation_hash(
            embedding_provider=provider,
            embedding_model_name="openai/text-embedding-3-large",
            enable_contextual_rag=contextual,
            intake_chunking_version="1",
        )
        == version_one_hash
    )
    assert configuration.compute_regulatory_chunk_generation_hash(
        embedding_provider=provider,
        embedding_model_name="openai/text-embedding-3-large",
        enable_contextual_rag=contextual,
        intake_chunking_version="2",
    ) not in {expected, version_one_hash}


@pytest.mark.parametrize("version", [None, "1", "2"])
def test_old_and_intake_snapshots_validate_their_own_generation_without_rechunking(
    monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    _install_admin_configuration(monkeypatch)
    session = cast(Session, SimpleNamespace())
    snapshot = configuration.resolve_regulatory_indexing_snapshot(
        session, intake_chunking_version=version
    )
    payload = snapshot.model_dump(mode="json")
    if version is None:
        del payload["intake_chunking_version"]
    restored = RegulatoryIndexingConfigSnapshot.model_validate(payload)
    assert restored.intake_chunking_version == version
    configuration.validate_snapshot_for_stage(
        session, restored, RegulatoryIndexingStage.EMBEDDING
    )
    other_version = "1" if version is None else None
    drifted = restored.model_copy(update={"intake_chunking_version": other_version})
    with pytest.raises(
        configuration.RegulatoryIndexingConfigurationError, match="generation"
    ):
        configuration.validate_snapshot_for_stage(
            session, drifted, RegulatoryIndexingStage.EMBEDDING
        )


@pytest.mark.parametrize(
    ("file_profile", "row_profiles", "expected_profile"),
    [
        (None, [None], None),
        ("1", ["1", None], "1"),
        ("1", [None, None], "1"),
        ("2", ["2", None], "2"),
        ("2", [None, None], "2"),
    ],
)
def test_chunk_row_snapshot_uses_file_identity_after_amendment_insertions(
    monkeypatch: pytest.MonkeyPatch,
    file_profile: str | None,
    row_profiles: list[str | None],
    expected_profile: str | None,
) -> None:
    captured = _prepare_rows_without_database(
        monkeypatch, file_profile=file_profile, row_profiles=row_profiles
    )
    assert captured["snapshot"].intake_chunking_version == expected_profile
    assert captured["snapshot"].chunk_generation_hash == captured["file_hash"]


@pytest.mark.parametrize(
    ("file_profile", "row_profiles"),
    [
        (None, ["1"]),
        ("1", ["2"]),
        ("1", ["1", "2"]),
        ("2", ["1"]),
        ("2", ["2", "3"]),
        ("unknown", [None]),
    ],
)
def test_chunk_row_snapshot_rejects_unknown_or_conflicting_generation_markers(
    monkeypatch: pytest.MonkeyPatch,
    file_profile: str | None,
    row_profiles: list[str | None],
) -> None:
    with pytest.raises(ValueError, match="generation|inconsistent"):
        _prepare_rows_without_database(
            monkeypatch, file_profile=file_profile, row_profiles=row_profiles
        )


def _prepare_rows_without_database(
    monkeypatch: pytest.MonkeyPatch,
    *,
    file_profile: str | None,
    row_profiles: list[str | None],
) -> dict[str, Any]:
    _install_admin_configuration(monkeypatch)
    file_id, job_id = uuid4(), uuid4()
    generation = (
        "0" * 64
        if file_profile == "unknown"
        else configuration.compute_regulatory_chunk_generation_hash(
            embedding_provider=EmbeddingProvider.OPENROUTER,
            embedding_model_name="openai/text-embedding-3-large",
            intake_chunking_version=file_profile,
        )
    )
    file = SimpleNamespace(
        status=UserFileStatus.CHUNKED, regulatory_chunk_generation_hash=generation
    )
    session = cast(Session, SimpleNamespace(get=lambda *_args: file))
    rows = [
        cast(
            RegulatoryChunk,
            SimpleNamespace(chunk_metadata={"intake_chunking_version": profile}),
        )
        for profile in row_profiles
    ]
    monkeypatch.setattr(
        preparation, "get_chunks_for_file_snapshot", lambda *_args: rows
    )
    monkeypatch.setattr(
        preparation, "regulatory_chunks_content_hash", lambda *_args: "a" * 64
    )
    captured: dict[str, Any] = {"file_hash": generation}

    def create_job(*_args: object, **kwargs: Any) -> SimpleNamespace:
        snapshot = kwargs["snapshot"]
        captured["snapshot"] = snapshot
        return SimpleNamespace(
            id=job_id,
            content_hash="a" * 64,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            lease_generation=0,
        )

    monkeypatch.setattr(
        preparation, "_create_owned_regulatory_indexing_job", create_job
    )
    monkeypatch.setattr(
        preparation, "claim_regulatory_indexing_job", lambda *_args, **_kwargs: False
    )
    assert (
        preparation.prepare_regulatory_indexing_job_from_chunks(
            file_id, "tenant", session
        )
        == job_id
    )
    return captured


def test_new_source_context_uses_current_version_and_unknown_hash_fails() -> None:
    metadata = _document(version="1").doc_metadata
    assert metadata is not None
    source = IntakeSourceContext.model_validate(metadata["regulatory_document_intake"])
    fresh = IntakeSourceContext.model_validate(
        source.model_dump(exclude={"chunking_version"})
    )
    assert fresh.chunking_version == INTAKE_CHUNKING_CODE_VERSION == "3"
    with pytest.raises(ValueError, match="not supported"):
        configuration.compute_regulatory_chunk_generation_hash(
            embedding_provider=None,
            embedding_model_name="embedding",
            intake_chunking_version="4",
        )


@pytest.mark.parametrize("version", ["1", "2", "3"])
def test_canonical_writer_passes_frozen_profile_to_intake_chunker(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    from onyx.regulatory.document_intake import chunking
    from onyx.regulatory.indexing import RegulatoryIndexingChunker

    captured: dict[str, Any] = {}

    class Chunker:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def chunk_text(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(chunks=[])

    monkeypatch.setattr(chunking, "IntakeRegulatoryChunker", Chunker)
    writer = SimpleNamespace(
        _regulatory_chunker=SimpleNamespace(max_chunk_chars=1000),
        tokenizer=object(),
        chunk_token_limit=512,
        enable_contextual_rag=False,
    )
    RegulatoryIndexingChunker._chunk_with_embedding_budget(
        cast(RegulatoryIndexingChunker, writer),
        "body",
        source_path="fixture",
        source_file="fixture.md",
        intake_version=version,
    )
    assert captured["chunking_version"] == version
