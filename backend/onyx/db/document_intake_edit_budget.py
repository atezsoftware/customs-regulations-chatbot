"""Validate prospective intake edits against their frozen preparation budget."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk, SearchSettings, UserFile
from onyx.db.search_settings import get_current_search_settings
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.indexing.contextual_settings import effective_contextual_rag_enabled
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.natural_language_processing.utils import BaseTokenizer, count_tokens
from onyx.regulatory.amendments.annexes.context_dependencies import (
    rebuild_context_aggregates,
)
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
from onyx.regulatory.document_intake.chunking_profile import (
    SUPPORTED_INTAKE_CHUNKING_VERSIONS,
)
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
)
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE


def prospective_intake_edit_rows(
    rows: Sequence[RegulatoryChunk],
    changes: Mapping[str, Mapping[str, Any]],
) -> list[RegulatoryChunk]:
    """Rebuild the same derived text using detached copies, before any live mutation."""
    copies = canonical_snapshot_rows([_snapshot(row) for row in rows])
    for row in copies:
        for field, value in changes.get(row.id, {}).items():
            setattr(row, field, deepcopy(value))
    return rebuild_context_aggregates(copies, changed_ids=list(changes))


def _embedding_tokenizer(settings: SearchSettings) -> BaseTokenizer:
    # Construction resolves the indexing tokenizer without calling an embedding provider.
    return DefaultIndexingEmbedder.from_db_search_settings(
        search_settings=settings
    ).embedding_model.tokenizer


def validate_intake_edit_budgets(
    session: Session,
    files: Mapping[UUID, UserFile],
    prospective: Mapping[UUID, Sequence[RegulatoryChunk]],
) -> None:
    """Preserve the intake content/context reserve contract; never truncate or split."""
    from onyx.db.document_intake import IntakeConflictError

    settings = get_current_search_settings(session)
    session.refresh(settings)
    contextual = effective_contextual_rag_enabled(settings)
    # Intake indexing snapshots require context independently of the workspace flag.
    generations = {
        compute_regulatory_chunk_generation_hash(
            embedding_provider=settings.provider_type,
            embedding_model_name=settings.model_name,
            enable_contextual_rag=contextual if version is None else True,
            intake_chunking_version=version,
        ): version
        for version in (None, *sorted(SUPPORTED_INTAKE_CHUNKING_VERSIONS))
    }
    limits: dict[UUID, int] = {}
    for file_id, rows in prospective.items():
        file = files[file_id]
        generation = file.regulatory_chunk_generation_hash
        if generation is None or generation not in generations:
            raise IntakeConflictError(
                "The imported file generation no longer matches current indexing settings; reload or prepare it again"
            )
        version = generations[generation]
        if any(
            row.user_file_id != file_id
            or row.chunk_metadata.get("intake_chunking_version") not in (None, version)
            for row in rows
        ):
            raise IntakeConflictError(
                "Canonical intake chunking versions are inconsistent"
            )
        # Legacy adaptive generations did not promise the intake-specific reserve.
        reserve = DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS if version is not None else 0
        limits[file_id] = max(1, DOC_EMBEDDING_CONTEXT_SIZE - reserve)
    tokenizer = _embedding_tokenizer(settings)
    for file_id, rows in prospective.items():
        limit = limits[file_id]
        for row in rows:
            tokens = count_tokens(row.text, tokenizer)
            if tokens > limit:
                raise ValueError(
                    f"Chunk {row.id} has {tokens} tokens; intake content limit is {limit} tokens. "
                    "Shorten formatting without removing legal content, or ask an administrator to prepare the source again."
                )
