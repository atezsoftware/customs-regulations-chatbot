"""Bounded HTTP child, separate from the thread-pool Celery worker."""

import json
import sys
from pathlib import Path

from onyx.regulatory.amendments.annexes.source_parser import apply_source_process_limits


def main() -> None:
    apply_source_process_limits()
    from onyx.regulatory.amendments.annexes.sources import (
        SourceAcquisitionError,
        download_source,
    )

    try:
        result = download_source(sys.argv[1])
        Path(sys.argv[2]).write_bytes(result.content)
        output = {"mime_type": result.mime_type, "final_url": result.final_url}
    except SourceAcquisitionError as exc:
        output = {"error": exc.code}
    Path(sys.argv[3]).write_text(json.dumps(output))


if __name__ == "__main__":
    main()
