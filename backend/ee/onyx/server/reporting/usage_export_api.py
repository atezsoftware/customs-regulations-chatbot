from collections.abc import Generator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ee.onyx.db.usage_export import (
    UsageReportMetadata,
    get_all_usage_reports,
    get_usage_report_data,
    get_usage_summary,
)
from ee.onyx.server.reporting.usage_export_generation import resolve_report_period
from ee.onyx.server.reporting.usage_export_models import UsageSummary
from onyx.auth.permissions import require_permission
from onyx.background.celery.versioned_apps.client import app as client_app
from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryQueues, OnyxCeleryTask
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.file_store.constants import STANDARD_CHUNK_SIZE
from shared_configs.contextvars import get_current_tenant_id

router = APIRouter()


class GenerateUsageReportParams(BaseModel):
    period_from: str | None = None
    period_to: str | None = None


@router.post("/admin/usage-report", status_code=204)
def generate_report(
    params: GenerateUsageReportParams,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
) -> None:
    if bool(params.period_from) != bool(params.period_to):
        raise HTTPException(
            status_code=400,
            detail="period_from and period_to must be provided together",
        )
    if params.period_from and params.period_to:
        try:
            period_from = datetime.fromisoformat(params.period_from)
            period_to = datetime.fromisoformat(params.period_to)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if period_from > period_to:
            raise HTTPException(
                status_code=400, detail="period_from must not exceed period_to"
            )

    tenant_id = get_current_tenant_id()
    client_app.send_task(
        OnyxCeleryTask.GENERATE_USAGE_REPORT_TASK,
        priority=OnyxCeleryPriority.MEDIUM,
        queue=OnyxCeleryQueues.CSV_GENERATION,
        expires=JOB_TIMEOUT,
        kwargs={
            "tenant_id": tenant_id,
            "user_id": str(user.id) if user else None,
            "period_from": params.period_from,
            "period_to": params.period_to,
        },
    )

    return None


@router.get("/admin/usage-report/summary")
def fetch_usage_summary(
    period_from: datetime | None = None,
    period_to: datetime | None = None,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> UsageSummary:
    if (period_from is None) != (period_to is None):
        raise HTTPException(
            status_code=400,
            detail="period_from and period_to must be provided together",
        )
    if period_from is not None and period_to is not None and period_from > period_to:
        raise HTTPException(
            status_code=400, detail="period_from must not exceed period_to"
        )

    return get_usage_summary(
        db_session,
        resolve_report_period(
            (period_from, period_to)
            if period_from is not None and period_to is not None
            else None
        ),
    )


@router.get("/admin/usage-report/{report_name}")
def read_usage_report(
    report_name: str,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),  # noqa: ARG001
) -> Response:
    try:
        file = get_usage_report_data(report_name)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    def iterfile() -> Generator[bytes, None, None]:
        while True:
            chunk = file.read(STANDARD_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(
        content=iterfile(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={report_name}"},
    )


@router.get("/admin/usage-report")
def fetch_usage_reports(
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[UsageReportMetadata]:
    try:
        return get_all_usage_reports(db_session)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
