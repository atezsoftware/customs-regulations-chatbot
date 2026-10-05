"""Request attribution copied into research threads and buffered usage records."""

from contextvars import ContextVar
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UsageMeasurementOrigin(BaseModel):
    model_config = ConfigDict(frozen=True)

    epoch: str | None
    started_at: datetime


class UsageMeasurementContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    epoch: str
    request_id: UUID
    started_at: datetime
    user_id: str
    session_id: UUID
    question_id: int
    workflow: str
    benchmark: bool = False


CURRENT_USAGE_MEASUREMENT: ContextVar[UsageMeasurementContext | None] = ContextVar(
    "usage_measurement", default=None
)
