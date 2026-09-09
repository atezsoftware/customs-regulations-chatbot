"""Child-process parser boundary for untrusted source files (including PDF)."""

import json
import resource
import sys
from pathlib import Path


def apply_source_process_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    resource.setrlimit(resource.RLIMIT_FSIZE, (150 * 1024 * 1024, 150 * 1024 * 1024))
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))


def main() -> None:
    apply_source_process_limits()
    from onyx.regulatory.amendments.annexes.sources import (
        SourceAcquisitionError,
        inspect_source,
    )

    try:
        result = inspect_source(Path(sys.argv[1]).read_bytes(), sys.argv[3] or None)
        output = result.model_dump_json()
    except SourceAcquisitionError as exc:
        output = json.dumps({"error": exc.code})
    except MemoryError:
        output = json.dumps({"error": "parse_resource_limit"})
    except Exception:
        output = json.dumps({"error": "parse_failed"})
    Path(sys.argv[2]).write_text(output)


if __name__ == "__main__":
    main()
