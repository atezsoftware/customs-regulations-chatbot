"""Start the bounded annex consumer using the producer's exact scoped queues."""

import os
import sys

from onyx.regulatory.amendments.annexes.config import (
    analysis_queue_name,
    publication_queue_name,
    source_queue_name,
)


def worker_command(arguments: list[str]) -> list[str]:
    if any(not argument.startswith("--hostname=") for argument in arguments):
        raise ValueError(
            "annex worker accepts only its hostname; queues and concurrency are fixed"
        )
    return [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.regulatory_annex",
        "worker",
        *arguments,
        "-Q",
        ",".join(
            (source_queue_name(), analysis_queue_name(), publication_queue_name())
        ),
        "--concurrency=1",
    ]


def main() -> None:
    command = worker_command(sys.argv[1:])
    os.execvp(command[0], command)  # noqa: S606 - preserve Supervisor signal handling


if __name__ == "__main__":
    main()
