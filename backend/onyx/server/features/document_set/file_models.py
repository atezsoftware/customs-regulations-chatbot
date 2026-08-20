from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from onyx.db.models import UserFile
from onyx.db.regulatory_indexing_jobs import RegulatoryIndexingProgress
from onyx.server.features.projects.models import UserFileSnapshot


class RegulatoryIndexingProgressSnapshot(BaseModel):
    job_id: UUID
    status: str
    stage: str
    total_items: int
    completed_items: int
    context_ready_items: int
    embedded_items: int
    failed_items: int
    attempt_count: int
    next_retry_at: datetime | None
    error_summary: str | None
    provider_batch_state: str | None

    @classmethod
    def from_progress(
        cls, progress: RegulatoryIndexingProgress
    ) -> "RegulatoryIndexingProgressSnapshot":
        return cls(
            job_id=progress.job_id,
            status=progress.status.value,
            stage=progress.stage.value,
            total_items=progress.total_items,
            completed_items=progress.completed_items,
            context_ready_items=progress.context_ready_items,
            embedded_items=progress.embedded_items,
            failed_items=progress.failed_items,
            attempt_count=progress.attempt_count,
            next_retry_at=progress.next_retry_at,
            error_summary=progress.error_summary,
            provider_batch_state=progress.provider_batch_state,
        )


class DocumentSetUserFileSnapshot(UserFileSnapshot):
    regulatory_indexing_progress: RegulatoryIndexingProgressSnapshot | None = None

    @classmethod
    def from_user_file(
        cls,
        model: UserFile,
        *,
        progress: RegulatoryIndexingProgress | None,
    ) -> "DocumentSetUserFileSnapshot":
        base_snapshot = UserFileSnapshot.from_model(model)
        return cls(
            **base_snapshot.model_dump(),
            regulatory_indexing_progress=(
                RegulatoryIndexingProgressSnapshot.from_progress(progress)
                if progress is not None
                else None
            ),
        )
