"""Small resource snapshots independent of amendment text and proposal payloads."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from onyx.regulatory.amendments.model_choice import AmendmentAnalysisModel


class AmendmentRuntime(BaseModel):
    analysis_model: AmendmentAnalysisModel = AmendmentAnalysisModel.FLASH
    batch_id: int
    status: str
    stage: str
    lease_generation: int
    scope: Literal["worker_container"] = "worker_container"
    raw_text_chars: int = Field(ge=0)
    current_bytes: int | None = None
    limit_bytes: int | None = None
    peak_bytes: int | None = None
    reserve_bytes: int | None = None
    active: int | None = None
    peak_active: int | None = None
    max_parallel: int | None = None
    admission_limited: bool | None = None
    dependency_limited: bool | None = None
    calibrating: bool | None = None
    memory_checked_at: datetime | None = None
    activity_checked_at: datetime | None = None
