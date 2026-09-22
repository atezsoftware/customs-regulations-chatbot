"""Inspect where selected sources are used, without applying an amendment.

Run with the application's DB environment and PYTHONPATH=backend. Context input
membership is reported as a candidate, never as a promise to re-embed that chunk.
"""

import argparse
from datetime import date
from pathlib import Path
from uuid import UUID

from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from onyx.db.regulatory_amendment_impact import inspect_amendment_source_usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", required=True, type=UUID)
    parser.add_argument("--source-id", required=True, action="append")
    parser.add_argument("--as-of-date", required=True, type=date.fromisoformat)
    parser.add_argument("--tenant", default="public")
    parser.add_argument(
        "--index-uuid", help="Required when multiple physical indexes exist"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    SqlEngine.init_engine(pool_size=1, max_overflow=0)
    with get_session_with_tenant(tenant_id=args.tenant) as session:
        report = inspect_amendment_source_usage(
            session,
            user_file_id=args.file_id,
            source_ids=set(args.source_id),
            as_of_date=args.as_of_date,
            index_uuid=args.index_uuid,
        )
        session.rollback()
    rendered = report.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
