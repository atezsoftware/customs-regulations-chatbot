"""Scoped publication observations carried from retrieval to public delivery."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import TypeVar
from uuid import UUID

from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import (
    PublicationReadEvidence as PublicationReadEvidence,
)
from onyx.document_index.publication_models import PublicationScope, ReadObservation
from onyx.regulatory.amendments.annexes.config import (
    ANNEX_DATABASE_IDENTITY,
    REGULATORY_ANNEX_ENVIRONMENT,
)
from shared_configs.contextvars import get_current_tenant_id

T = TypeVar("T")


def public_read_store() -> PublicationStore:
    # Protection survives turning off the review/creation feature.
    return PublicationStore(
        PublicationScope(
            tenant_id=get_current_tenant_id(),
            environment=REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=ANNEX_DATABASE_IDENTITY,
        )
    )


def evidence_unavailable(evidence: PublicationReadEvidence) -> frozenset[UUID]:
    if not evidence.user_file_ids:
        return frozenset()
    return public_read_store().unavailable(evidence.observation, evidence.user_file_ids)


class PublicationReadChanged(RuntimeError):
    def __init__(self) -> None:
        super().__init__("A source changed during this response. Please try again.")


class PublicationReadTracker:
    """One turn's sources; copied thread contexts share this synchronized tracker."""

    def __init__(self, observation: ReadObservation | None = None) -> None:
        self._observation = observation
        self._files: set[UUID] = set()
        self._lock = Lock()

    def observe(self) -> ReadObservation:
        with self._lock:
            if self._observation is None:
                self._observation = public_read_store().observe()
            return self._observation

    def include(self, files: Sequence[UUID]) -> None:
        with self._lock:
            self._files.update(files)

    def include_evidence(self, evidence: PublicationReadEvidence) -> None:
        with self._lock:
            if (
                self._observation is None
                or evidence.observation.committed_epoch
                < self._observation.committed_epoch
            ):
                self._observation = evidence.observation
            self._files.update(evidence.user_file_ids)

    def evidence(self) -> PublicationReadEvidence | None:
        with self._lock:
            if self._observation is None or not self._files:
                return None
            return PublicationReadEvidence(
                observation=self._observation, user_file_ids=tuple(sorted(self._files))
            )

    def validate(self) -> None:
        evidence = self.evidence()
        if evidence is not None and evidence_unavailable(evidence):
            raise PublicationReadChanged()


_CURRENT_READ: ContextVar[PublicationReadTracker | None] = ContextVar(
    "publication_read", default=None
)


@contextmanager
def track_publication_reads(tracker: PublicationReadTracker) -> Iterator[None]:
    token = _CURRENT_READ.set(tracker)
    try:
        yield
    finally:
        _CURRENT_READ.reset(token)


def observe_publication_read() -> ReadObservation:
    tracker = _CURRENT_READ.get()
    return tracker.observe() if tracker is not None else public_read_store().observe()


def file_uuid(document_id: str) -> UUID | None:
    try:
        return UUID(document_id)
    except ValueError:
        return None


def filter_publication_read(
    observation: ReadObservation,
    items: Sequence[T],
    document_id: Callable[[T], str],
    *,
    record_evidence: bool = True,
) -> list[T]:
    files = {
        file for item in items if (file := file_uuid(document_id(item))) is not None
    }
    if not files:
        return list(items)
    unavailable = public_read_store().unavailable(observation, tuple(files))
    retained = [
        item for item in items if file_uuid(document_id(item)) not in unavailable
    ]
    tracker = _CURRENT_READ.get()
    if tracker is not None and record_evidence:
        tracker.include(tuple(files - unavailable))
    return retained


def require_publication_files(
    observation: ReadObservation, files: tuple[UUID, ...]
) -> None:
    from onyx.error_handling.error_codes import OnyxErrorCode
    from onyx.error_handling.exceptions import OnyxError

    if files and public_read_store().unavailable(observation, files):
        raise OnyxError(
            OnyxErrorCode.SERVICE_UNAVAILABLE,
            "A source changed or is being published. Please try again.",
        )
    tracker = _CURRENT_READ.get()
    if tracker is not None:
        tracker.include_evidence(
            PublicationReadEvidence(observation=observation, user_file_ids=files)
        )
