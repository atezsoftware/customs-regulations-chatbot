"""Reuse immutable native evidence within one labeling review operation only."""

import hashlib
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from uuid import UUID

from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.labeling.chapter_caption import (
    ChapterCaptionSourceProof,
    build_chapter_caption_source_proof,
    chapter_caption_qualification_sha256,
)
from onyx.regulatory.labeling.intake_abstention_repair import CaptionRepairSource

_MAX_RAW_BYTES = 32 * 1024 * 1024
_MAX_RAW_ENTRIES = 16
_MAX_PROOF_ENTRIES = 64


@dataclass(frozen=True)
class NativeCaptionIdentity:
    source_asset_id: UUID
    user_file_id: UUID
    file_id: str
    source_sha256: str
    byte_count: int
    tenant_id: str

    def key(self) -> str:
        return publication_digest(
            {
                **asdict(self),
                "source_asset_id": str(self.source_asset_id),
                "user_file_id": str(self.user_file_id),
            }
        )


@dataclass
class _OperationEvidence:
    raw: dict[str, bytes] = field(default_factory=dict)
    proofs: dict[str, ChapterCaptionSourceProof] = field(default_factory=dict)
    raw_bytes: int = 0


_EVIDENCE: ContextVar[_OperationEvidence | None] = ContextVar(
    "chapter_caption_operation_evidence", default=None
)


@contextmanager
def native_caption_proof_operation() -> Iterator[None]:
    token = _EVIDENCE.set(_OperationEvidence())
    try:
        yield
    finally:
        _EVIDENCE.reset(token)


def chapter_caption_native_proof(
    *,
    identity: NativeCaptionIdentity,
    chunk_id: str,
    sources: Sequence[CaptionRepairSource],
    read_raw: Callable[[], bytes],
) -> ChapterCaptionSourceProof:
    """Call only after reloading and validating the current owned inputs."""
    target = next(source for source in sources if source.chunk_id == chunk_id)
    raw_key = identity.key()
    proof_key = publication_digest(
        {
            "raw_identity": raw_key,
            "target": chunk_id,
            "qualification": chapter_caption_qualification_sha256(target.text),
            "sources": [
                asdict(source)
                for source in sorted(sources, key=lambda row: row.chunk_id)
            ],
        }
    )
    evidence = _EVIDENCE.get()
    if evidence is not None and proof_key in evidence.proofs:
        return evidence.proofs[proof_key]
    raw = evidence.raw.get(raw_key) if evidence is not None else None
    if raw is None:
        raw = read_raw()
    if (
        len(raw) != identity.byte_count
        or hashlib.sha256(raw).hexdigest() != identity.source_sha256
    ):
        raise ValueError("The retained native caption PDF bytes changed")
    proof = build_chapter_caption_source_proof(
        raw_pdf=raw,
        source_sha256=identity.source_sha256,
        source_asset_id=identity.source_asset_id,
        user_file_id=identity.user_file_id,
        chunk_id=chunk_id,
        sources=sources,
    )
    # Failed reads/proofs never populate either cache. Full caches keep validating.
    if evidence is not None:
        if (
            raw_key not in evidence.raw
            and len(evidence.raw) < _MAX_RAW_ENTRIES
            and evidence.raw_bytes + len(raw) <= _MAX_RAW_BYTES
        ):
            evidence.raw[raw_key] = raw
            evidence.raw_bytes += len(raw)
        if len(evidence.proofs) < _MAX_PROOF_ENTRIES:
            evidence.proofs[proof_key] = proof
    return proof
