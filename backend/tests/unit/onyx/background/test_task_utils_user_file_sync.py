from unittest.mock import MagicMock

from sqlalchemy.dialects import postgresql

from onyx.background.task_utils import _claim_next_sync_file


def test_sync_drain_claims_failed_files_only_for_document_set_sync() -> None:
    db_session = MagicMock()
    db_session.execute.return_value.scalar_one_or_none.return_value = None

    _claim_next_sync_file(db_session)

    statement = db_session.execute.call_args.args[0]
    compiled_sql = " ".join(
        str(
            statement.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        ).split()
    )

    assert (
        "user_file.status = 'FAILED' AND user_file.needs_document_set_sync IS true"
    ) in compiled_sql
