"""Bounded failure receipts without source content or credentials."""

import json
import re


def safe_failure_detail(stage: str, error: BaseException) -> str:
    """Retain bounded code locations, never messages, source lines or frame locals."""
    from typing import get_args

    from pydantic import ValidationError
    from pydantic_core import ErrorType

    stages = {
        "scope",
        "startup",
        "configuration",
        "native",
        "calibration",
        "pdf_vision",
        "canary",
        "token",
        "capabilities",
        "baseline",
        "source_review",
        "source_package",
        "approval",
        "historical_chat",
        "current_chat",
        "markdown",
        "cleanup",
        "chat_cleanup",
        "token_cleanup",
    }
    types = {
        "UnicodeDecodeError",
        "ValueError",
        "ValidationError",
        "StructuredOutputValidationError",
        "RuntimeError",
        "TypeError",
        "TimeoutError",
        "OperationalError",
        "ImportError",
        "ModuleNotFoundError",
        "IsolatedProcessTimeout",
        "IsolatedProcessCrashed",
        "AssertionError",
        "KeyError",
        "BadRequestError",
        "AuthenticationError",
        "RateLimitError",
        "APIError",
        "APIConnectionError",
        "HTTPStatusError",
        "ConnectError",
        "ReadTimeout",
        "PermissionError",
        "IntegrityError",
        "NotFoundError",
    }
    exceptions: list[dict[str, object]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(exceptions) < 3:
        seen.add(id(current))
        frames: list[dict[str, str | int]] = []
        trace = current.__traceback__
        while trace is not None:
            module = trace.tb_frame.f_globals.get("__name__")
            function = trace.tb_frame.f_code.co_name
            if (
                isinstance(module, str)
                and re.fullmatch(
                    r"(?:onyx|ee\.onyx|shared_configs)\.[A-Za-z0-9_.]{1,100}", module
                )
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", function)
            ):
                frames.append(
                    {"module": module, "function": function, "line": trace.tb_lineno}
                )
                frames = frames[:2] + frames[-3:] if len(frames) > 5 else frames
            trace = trace.tb_next
        name = type(current).__name__
        exceptions.append(
            {"type": name if name in types else "Exception", "frames": frames}
        )
        if isinstance(current, ValidationError):
            # Schema names only: extra-key locations and custom error codes can be inputs.
            fields = {
                "supported",
                "rationale",
                "input_sha256",
                "model_snapshot",
                "model_provider",
                "model_name",
                "elements",
                "kind",
                "text",
                "box",
                "status",
                "issues",
                "table_role",
                "new_chunk",
                "dates",
                "effective_start_date",
                "effective_end_date",
                "reference_date",
                "heading_path",
                "metadata_changes",
                "chunk_type",
            }
            validation_errors = [
                {
                    "loc": [
                        part
                        if isinstance(part, str) and part in fields
                        else part
                        if type(part) is int and 0 <= part <= 1000
                        else "unknown"
                        for part in item["loc"][:3]
                    ],
                    "type": item["type"]
                    if item["type"] in get_args(ErrorType)
                    else "unknown",
                }
                for item in current.errors(
                    include_url=False, include_context=False, include_input=False
                )[:3]
            ]
            exceptions[-1]["validation_errors"] = validation_errors
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    # Keep optional schema diagnostics inside the established transport bound.
    for item in reversed(exceptions):
        if len(json.dumps(exceptions)) <= 3900:
            break
        item.pop("validation_errors", None)
    return json.dumps(
        {"stage": stage if stage in stages else "unknown", "exceptions": exceptions},
        separators=(",", ":"),
        sort_keys=True,
    )
