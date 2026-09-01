import os
import sys

from onyx.background.celery.queue_names import REGULATORY_AMENDMENT_QUEUE


def main() -> None:
    command = [
        "celery",
        "-A",
        "onyx.background.celery.versioned_apps.regulatory_benchmark",
        "worker",
        *sys.argv[1:],
        "-Q",
        f"regulatory_benchmark,{REGULATORY_AMENDMENT_QUEUE}",
    ]
    os.execvp(command[0], command)  # noqa: S606 - preserve supervisor signal handling


if __name__ == "__main__":
    main()
