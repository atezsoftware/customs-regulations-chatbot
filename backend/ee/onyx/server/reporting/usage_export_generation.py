import csv
import io
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta, timezone

from fastapi_users_db_sqlalchemy import UUID_ID
from sqlalchemy import cast
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session

from ee.onyx.db.usage_export import (
    get_all_empty_chat_message_entries,
    get_usage_summary,
    write_usage_report,
)
from ee.onyx.server.reporting.usage_export_models import (
    UsageReportMetadata,
    UsageSummary,
    UserSkeleton,
)
from onyx.configs.constants import FileOrigin
from onyx.db.models import User
from onyx.db.user_usage import (
    PerUserUsageSummary,
    get_usage_totals_by_email,
    get_user_activity_counts_by_email,
    summarize_usage_by_email,
)
from onyx.db.users import get_all_users
from onyx.file_store.constants import MAX_IN_MEMORY_SIZE
from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.utils.csv_utils import sanitize_csv_cell_or_none


def resolve_report_period(
    period: tuple[datetime, datetime] | None,
) -> tuple[datetime, datetime]:
    if period is None:
        return (
            datetime.fromtimestamp(0, tz=timezone.utc),
            datetime.now(tz=timezone.utc),
        )
    return period


def render_usage_summary_csv(
    summary: UsageSummary,
    period: tuple[datetime, datetime] | None,
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "period_from",
            "period_to",
            "total_user_queries",
            "total_user_sessions",
            "total_tokens",
            "total_cost_cents",
            "average_tokens_per_query",
            "average_tokens_per_session",
            "average_cost_cents_per_query",
            "average_cost_cents_per_session",
            "average_queries_per_session",
        ]
    )
    writer.writerow(
        [
            period[0].isoformat() if period else "all_time",
            period[1].isoformat() if period else "all_time",
            summary.total_user_queries,
            summary.total_user_sessions,
            summary.total_tokens,
            summary.total_cost_cents,
            summary.average_tokens_per_query,
            summary.average_tokens_per_session,
            summary.average_cost_cents_per_query,
            summary.average_cost_cents_per_session,
            summary.average_queries_per_session,
        ]
    )
    return stream.getvalue()


def render_per_user_usage_csv(summaries: list[PerUserUsageSummary]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "email",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cost_cents",
            "total_tokens",
            "total_user_queries",
            "total_user_sessions",
            "average_tokens_per_query",
            "average_tokens_per_session",
            "average_cost_cents_per_query",
            "average_cost_cents_per_session",
            "average_queries_per_session",
        ]
    )
    for summary in summaries:
        writer.writerow(
            [
                sanitize_csv_cell_or_none(summary.email),
                summary.input_tokens,
                summary.output_tokens,
                summary.cache_read_tokens,
                summary.cost_cents,
                summary.total_tokens,
                summary.total_user_queries,
                summary.total_user_sessions,
                summary.average_tokens_per_query,
                summary.average_tokens_per_session,
                summary.average_cost_cents_per_query,
                summary.average_cost_cents_per_session,
                summary.average_queries_per_session,
            ]
        )
    return stream.getvalue()


def generate_chat_messages_report(
    db_session: Session,
    file_store: FileStore,
    report_id: str,
    period: tuple[datetime, datetime],
) -> str:
    file_name = f"{report_id}_chat_sessions"

    with tempfile.SpooledTemporaryFile(
        max_size=MAX_IN_MEMORY_SIZE, mode="w+"
    ) as temp_file:
        csvwriter = csv.writer(temp_file, delimiter=",")
        csvwriter.writerow(
            [
                "session_id",
                "user_id",
                "flow_type",
                "time_sent",
                "assistant_name",
                "user_email",
                "number_of_tokens",
                "llm_model",
            ]
        )
        for chat_message_skeleton_batch in get_all_empty_chat_message_entries(
            db_session, period
        ):
            for chat_message_skeleton in chat_message_skeleton_batch:
                # assistant_name and user_email are user-supplied — sanitize
                # to prevent CSV/formula injection against whoever opens the
                # report in a spreadsheet. The remaining fields are
                # system-generated (UUIDs, enums, timestamps, ints).
                csvwriter.writerow(
                    [
                        chat_message_skeleton.chat_session_id,
                        chat_message_skeleton.user_id,
                        chat_message_skeleton.flow_type,
                        chat_message_skeleton.time_sent.isoformat(),
                        sanitize_csv_cell_or_none(chat_message_skeleton.assistant_name),
                        sanitize_csv_cell_or_none(chat_message_skeleton.user_email),
                        chat_message_skeleton.number_of_tokens,
                        chat_message_skeleton.llm_model,
                    ]
                )

        # after writing seek to beginning of buffer
        temp_file.seek(0)
        file_id = file_store.save_file(
            content=temp_file,
            display_name=file_name,
            file_origin=FileOrigin.GENERATED_REPORT,
            file_type="text/csv",
        )

    return file_id


def generate_user_report(
    db_session: Session,
    file_store: FileStore,
    report_id: str,
) -> str:
    file_name = f"{report_id}_users"

    with tempfile.SpooledTemporaryFile(
        max_size=MAX_IN_MEMORY_SIZE, mode="w+"
    ) as temp_file:
        csvwriter = csv.writer(temp_file, delimiter=",")
        csvwriter.writerow(["user_id", "is_active"])

        users = get_all_users(db_session)
        for user in users:
            user_skeleton = UserSkeleton(
                user_id=str(user.id),
                is_active=user.is_active,
            )
            csvwriter.writerow([user_skeleton.user_id, user_skeleton.is_active])

        temp_file.seek(0)
        file_id = file_store.save_file(
            content=temp_file,
            display_name=file_name,
            file_origin=FileOrigin.GENERATED_REPORT,
            file_type="text/csv",
        )

    return file_id


def create_new_usage_report(
    db_session: Session,
    user_id: UUID_ID | None,  # None = auto-generated
    period: tuple[datetime, datetime] | None,
) -> UsageReportMetadata:
    report_id = str(uuid.uuid4())
    file_store = get_default_file_store()
    resolved_period = resolve_report_period(period)

    messages_file_id = generate_chat_messages_report(
        db_session, file_store, report_id, resolved_period
    )
    users_file_id = generate_user_report(db_session, file_store, report_id)
    summary = get_usage_summary(db_session, resolved_period)
    summary_csv = render_usage_summary_csv(summary, period)
    # The UI sends an inclusive end-of-day timestamp. The ledger/activity
    # helpers use half-open ranges, so advance by one microsecond for parity.
    end_exclusive = resolved_period[1] + timedelta(microseconds=1)
    usage_rows = get_usage_totals_by_email(
        db_session,
        start=resolved_period[0],
        end=end_exclusive,
    )
    activity_rows = get_user_activity_counts_by_email(
        db_session,
        start=resolved_period[0],
        end=end_exclusive,
    )
    per_user_usage_csv = render_per_user_usage_csv(
        summarize_usage_by_email(
            usage_rows,
            activity_rows,
            include_activity_only=True,
        )
    )

    with tempfile.SpooledTemporaryFile(max_size=MAX_IN_MEMORY_SIZE) as zip_buffer:
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            # write messages
            chat_messages_tmpfile = file_store.read_file(
                messages_file_id, mode="b", use_tempfile=True
            )
            zip_file.writestr(
                "chat_messages.csv",
                chat_messages_tmpfile.read(),
            )

            # write users
            users_tmpfile = file_store.read_file(
                users_file_id, mode="b", use_tempfile=True
            )
            zip_file.writestr("users.csv", users_tmpfile.read())
            zip_file.writestr("usage_summary.csv", summary_csv)
            zip_file.writestr("per_user_usage.csv", per_user_usage_csv)

        zip_buffer.seek(0)

        # store zip blob to file_store
        report_name = f"{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}_{report_id}_usage_report.zip"
        file_store.save_file(
            content=zip_buffer,
            display_name=report_name,
            file_origin=FileOrigin.GENERATED_REPORT,
            file_type="application/zip",
            file_id=report_name,
        )

    # add report after zip file is written
    new_report = write_usage_report(db_session, report_name, user_id, period)

    # get user email
    requestor_user = (
        db_session.query(User)
        .filter(cast(User.id, UUID) == new_report.requestor_user_id)
        .one_or_none()
        if new_report.requestor_user_id
        else None
    )
    requestor_email = requestor_user.email if requestor_user else None

    return UsageReportMetadata(
        report_name=new_report.report_name,
        requestor=requestor_email,
        time_created=new_report.time_created,
        period_from=new_report.period_from,
        period_to=new_report.period_to,
    )
