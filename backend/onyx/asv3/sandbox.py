"""Bounded isolated code, deterministic calculation and tool-composition capabilities."""

import json
import time
from calendar import monthrange
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from datetime import date, timedelta
from decimal import Decimal, localcontext
from hashlib import sha256
from typing import cast

import requests
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, guarded, json_value, schema
from onyx.asv3.models import (
    Artifact,
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.progress import ProgressReporter, action_narration, public_action_id
from onyx.asv3.registry import CapabilityRegistry
from onyx.configs.app_configs import CODE_INTERPRETER_BASE_URL
from onyx.db.asv3_corpus import research_code_enabled
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.tools.tool_implementations.python.code_interpreter_client import (
    CodeInterpreterClient,
    CodeInterpreterVersionError,
    ExecuteResponse,
    FileInput,
)

MAX_CODE_CHARS = 64_000
MAX_STAGE_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_CHARS = 32_000


def decimal_value(value: JsonValue) -> Decimal:
    if isinstance(value, (bool, dict, list)) or value is None:
        raise ValueError("Numbers must be decimal strings or finite numbers.")
    result = Decimal(str(value))
    if not result.is_finite() or len(str(value)) > 100:
        raise ValueError("Numeric input is nonfinite or too large.")
    return result


def calculate(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
    context.check_active()
    operation = str(args["operation"])
    output: JsonValue
    if operation in {"add", "multiply", "divide", "percent", "allocate"}:
        numbers = args.get("values")
        if not isinstance(numbers, list) or not 1 <= len(numbers) <= 100:
            raise ValueError("Supply 1..100 explicit decimal values.")
        values = [decimal_value(value) for value in numbers]
        with localcontext() as precision:
            precision.prec = 50
            if operation == "add":
                result = sum(values, Decimal(0))
            elif operation == "multiply":
                result = Decimal(1)
                for value in values:
                    result *= value
            else:
                if operation != "allocate" and len(values) != 2:
                    raise ValueError("This operation requires two values.")
                if operation == "divide":
                    result = values[0] / values[1]
                elif operation == "percent":
                    result = values[0] * values[1] / Decimal(100)
                else:
                    weights = args.get("weights")
                    if not isinstance(weights, list) or not weights:
                        raise ValueError("Allocation requires explicit weights.")
                    parsed = [decimal_value(value) for value in weights]
                    if any(value < 0 for value in parsed) or sum(parsed) == 0:
                        raise ValueError(
                            "Allocation weights must be nonnegative and nonzero in total."
                        )
                    output = [str(values[0] * value / sum(parsed)) for value in parsed]
                    return ToolOutcome(
                        status=OutcomeStatus.FOUND,
                        summary="Decimal allocation computed from supplied inputs; no legal rule was inferred.",
                        data={"result": output, "inputs": args, "precision": 50},
                    )
            output = str(result)
    elif operation in {"calendar_days", "business_days", "following_month_day"}:
        current = date.fromisoformat(str(args["date"]))
        count = int(cast(int, args.get("days", 0)))
        if not 0 <= count <= 3660:
            raise ValueError("Day count must be between zero and 3660.")
        if operation == "calendar_days":
            current += timedelta(days=count)
        elif operation == "following_month_day":
            target = int(cast(int, args["day"]))
            year, month = (
                (current.year + 1, 1)
                if current.month == 12
                else (current.year, current.month + 1)
            )
            if not 1 <= target <= monthrange(year, month)[1]:
                raise ValueError("Target day does not exist in the following month.")
            current = date(year, month, target)
        else:
            holidays = args.get("holidays")
            if not isinstance(holidays, list) or not args.get("calendar_name"):
                return ToolOutcome(
                    status=OutcomeStatus.INVALID,
                    summary="Business-day computation requires an explicit calendar name and holiday dates (an empty list explicitly means weekends only).",
                )
            calendar = {date.fromisoformat(str(value)) for value in holidays}
            progressed = 0
            while progressed < count:
                current += timedelta(days=1)
                if current.weekday() < 5 and current not in calendar:
                    progressed += 1
            context.check_active()
        output = current.isoformat()
    else:
        raise ValueError("Unsupported calculation operation.")
    return ToolOutcome(
        status=OutcomeStatus.FOUND,
        summary="Reproducible calculation from explicit inputs; legal triggers and calendar authority must be supported separately.",
        data={"result": output, "inputs": args, "precision": 50},
    )


def resolve_program_value(
    value: JsonValue, results: dict[str, ToolOutcome]
) -> JsonValue:
    if isinstance(value, list):
        return [resolve_program_value(item, results) for item in value]
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            path = str(value["$ref"]).split(".")
            if path[0] not in results:
                raise ValueError("Program reference is not an available dependency.")
            resolved: JsonValue = json_value(results[path[0]].model_dump(mode="json"))
            for component in path[1:]:
                if isinstance(resolved, dict) and component in resolved:
                    resolved = resolved[component]
                elif (
                    isinstance(resolved, list)
                    and component.isdigit()
                    and int(component) < len(resolved)
                ):
                    resolved = resolved[int(component)]
                else:
                    raise ValueError("Program reference path is unavailable.")
            return resolved
        return {
            key: resolve_program_value(item, results) for key, item in value.items()
        }
    return value


def compose(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
    registry = context.services.get("registry")
    if not isinstance(registry, CapabilityRegistry):
        return ToolOutcome(
            status=OutcomeStatus.UNAVAILABLE,
            summary="Programmatic capability registry is not attached to this run.",
        )
    raw = args.get("steps")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
        raise ValueError("Composition requires 1..20 bounded program steps.")
    steps: dict[str, dict[str, JsonValue]] = {}
    for item in raw:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not isinstance(item.get("tool"), str)
        ):
            raise ValueError(
                "Each step requires id, tool, arguments and optional dependency IDs."
            )
        key = str(item["id"])
        if key in steps or item["tool"] == "compose_tool_calls":
            raise ValueError("Duplicate step or recursive composition is not allowed.")
        if not isinstance(item.get("arguments", {}), dict) or not isinstance(
            item.get("depends_on", []), list
        ):
            raise ValueError("Arguments must be an object and dependencies a list.")
        steps[key] = item
    dependencies = {
        key: {str(value) for value in cast(list[JsonValue], item.get("depends_on", []))}
        for key, item in steps.items()
    }
    if any(
        not deps <= steps.keys() or key in deps for key, deps in dependencies.items()
    ):
        raise ValueError("Program dependencies are unknown or self-referential.")
    remaining = set(steps)
    settled: set[str] = set()
    while remaining:
        ready = {key for key in remaining if dependencies[key] <= settled}
        if not ready:
            raise ValueError("Program dependency graph contains a cycle.")
        settled.update(ready)
        remaining -= ready
    results: dict[str, ToolOutcome] = {}
    remaining = set(steps)
    reporter = context.services.get("progress")
    narrated_capabilities: set[str] = set()
    if isinstance(reporter, ProgressReporter):
        for definition in registry.definitions(context):
            function = definition.get("function")
            if not isinstance(function, dict):
                continue
            parameters = function.get("parameters")
            properties = (
                parameters.get("properties") if isinstance(parameters, dict) else None
            )
            name = function.get("name")
            if (
                isinstance(name, str)
                and isinstance(properties, dict)
                and "_public_update" in properties
            ):
                narrated_capabilities.add(name)

    def dispatch_step(call: CapabilityCall) -> ToolOutcome:
        narration = (
            action_narration(call.arguments, context)
            if call.name in narrated_capabilities
            else None
        )
        parent = context.services.get("task_id")
        parent_task_id = parent if isinstance(parent, str) else None
        action_id = public_action_id(call.call_id)
        phase = "final" if call.name == "verify_claim" else "tools"
        if (
            isinstance(reporter, ProgressReporter)
            and narration is not None
            and not context.is_cancelled()
        ):
            title, message = narration
            if context.depth == 0:
                reporter.report(phase, title=title, message=message)
            reporter.report(
                phase,
                task_id=action_id,
                parent_task_id=parent_task_id,
                title=title,
                message=message,
            )
        outcome = registry.dispatch(call, context)
        if (
            isinstance(reporter, ProgressReporter)
            and narration is not None
            and not context.is_cancelled()
        ):
            status = (
                "cancelled"
                if outcome.status == OutcomeStatus.CANCELLED
                else "failed"
                if outcome.status
                in {
                    OutcomeStatus.TRUNCATED,
                    OutcomeStatus.INVALID,
                    OutcomeStatus.ERROR,
                    OutcomeStatus.DENIED,
                    OutcomeStatus.UNAVAILABLE,
                }
                else "completed"
            )
            reporter.report(
                phase,
                status=status,
                task_id=action_id,
                parent_task_id=parent_task_id,
                title=narration[0],
                message=narration[1],
            )
        return outcome

    pool = ThreadPoolExecutor(
        max_workers=min(4, int(cast(int, args.get("max_parallel", 4))))
    )
    try:
        while remaining:
            context.check_active()
            ready = [
                key
                for key in steps
                if key in remaining and dependencies[key] <= results.keys()
            ]
            active = {}
            for key in ready:
                item = steps[key]
                if any(
                    results[dependency].status
                    not in {
                        OutcomeStatus.FOUND,
                        OutcomeStatus.PARTIAL,
                        OutcomeStatus.VERSION_UNKNOWN,
                        OutcomeStatus.AMBIGUOUS,
                    }
                    for dependency in dependencies[key]
                ):
                    results[key] = ToolOutcome(
                        status=OutcomeStatus.CANCELLED,
                        summary="A required dependency failed.",
                    )
                    remaining.remove(key)
                    continue
                when = item.get("when")
                if isinstance(when, dict) and resolve_program_value(
                    when.get("value"), results
                ) != when.get("equals"):
                    results[key] = ToolOutcome(
                        status=OutcomeStatus.CANCELLED,
                        summary="Program condition did not match.",
                    )
                    remaining.remove(key)
                    continue
                arguments = resolve_program_value(item.get("arguments", {}), results)
                if not isinstance(arguments, dict):
                    raise ValueError("Resolved arguments are not an object.")
                active[
                    pool.submit(
                        copy_context().run,
                        dispatch_step,
                        CapabilityCall(name=str(item["tool"]), arguments=arguments),
                    )
                ] = key
            while active:
                context.check_active()
                done, _ = wait(active, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    key = active.pop(future)
                    results[key] = future.result()
                    remaining.remove(key)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    evidence = [item for result in results.values() for item in result.evidence]
    artifacts = [item for result in results.values() for item in result.artifacts]
    bad = any(
        result.status
        in {
            OutcomeStatus.ERROR,
            OutcomeStatus.UNAVAILABLE,
            OutcomeStatus.DENIED,
            OutcomeStatus.INVALID,
        }
        for result in results.values()
    )
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL if bad else OutcomeStatus.FOUND,
        summary="Executed dependency-aware capability program; every nested call used this run's registry, budget and authorization.",
        data={
            "steps": {
                key: cast(
                    JsonValue,
                    result.model_dump(mode="json", exclude={"evidence", "artifacts"}),
                )
                for key, result in results.items()
            }
        },
        evidence=evidence,
        original_reads=[
            read for result in results.values() for read in result.original_reads
        ],
        artifacts=artifacts,
    )


def run_code(
    broker: CorpusBroker, args: dict[str, JsonValue], context: RunContext
) -> ToolOutcome:
    context.check_active()
    code = str(args["code"])
    if not 1 <= len(code) <= MAX_CODE_CHARS:
        raise ValueError("Code exceeds the research program size limit.")
    language = str(args.get("language", "python"))
    if language not in {"python", "bash"}:
        raise ValueError("Research language must be python or bash.")
    client_factory = context.services.get("code_interpreter_factory")
    if client_factory is None:
        if not CODE_INTERPRETER_BASE_URL:
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="The isolated Code Interpreter service is not configured.",
            )
        with get_session_with_current_tenant() as session:
            if not research_code_enabled(session):
                return ToolOutcome(
                    status=OutcomeStatus.UNAVAILABLE,
                    summary="Isolated code execution is disabled by deployment policy.",
                )
        client_factory = CodeInterpreterClient
    if not callable(client_factory):
        return ToolOutcome(
            status=OutcomeStatus.UNAVAILABLE,
            summary="Configured code service adapter is unavailable.",
        )
    payloads: list[tuple[str, bytes]] = []
    source_ids = args.get("source_ids", [])
    if not isinstance(source_ids, list) or len(source_ids) > 5:
        raise ValueError("Stage at most five authorized sources.")
    evidence = []
    for source_id in source_ids:
        source, chunks, more = broker.page(str(source_id), context, limit=100)
        payload = json.dumps(
            {
                "source_id": str(source.id),
                "name": source.name,
                "scope_date": broker.filters.as_of_date.isoformat()
                if broker.filters.as_of_date
                else None,
                "partial": more,
                "chunks": [
                    {
                        "chunk_id": item.id,
                        "text": item.text,
                        "sha256": sha256(item.text.encode()).hexdigest(),
                        "position": item.position,
                    }
                    for item in chunks
                ],
            },
            ensure_ascii=False,
        ).encode()
        payloads.append((f"source-{source.id}.json", payload))
        from onyx.asv3.corpus_tools import bounded_evidence

        items, _ = bounded_evidence(source, chunks)
        evidence.extend(items)
    if sum(len(payload) for _, payload in payloads) > MAX_STAGE_BYTES:
        raise ValueError(
            "Selected sandbox inputs exceed the byte budget; choose narrower sources."
        )
    uploaded = []
    try:
        with cast(Callable[[], CodeInterpreterClient], client_factory)() as client:
            if not client.health(use_cache=True).healthy:
                return ToolOutcome(
                    status=OutcomeStatus.UNAVAILABLE,
                    summary="Isolated code service is unavailable.",
                )
            staged: list[FileInput] = []
            for filename, payload in payloads:
                context.check_active()
                identifier = client.upload_file(payload, filename)
                uploaded.append(identifier)
                staged.append({"path": filename, "file_id": identifier})
            timeout = max(
                1,
                min(
                    30_000,
                    int((context.deadline - time.monotonic()) * 1000),
                    int(cast(int, args.get("timeout_ms", 30000))),
                ),
            )
            try:
                if language == "bash":
                    if not client.supports(
                        client.create_session,
                        client.execute_bash_in_session,
                        client.delete_session,
                    ):
                        return ToolOutcome(
                            status=OutcomeStatus.UNAVAILABLE,
                            summary="Isolated Bash requires a session-capable code service.",
                        )
                    execution_session = client.create_session(
                        ttl_seconds=60, files=staged
                    )
                    try:
                        bash = client.execute_bash_in_session(
                            execution_session.session_id, code, timeout_ms=timeout
                        )
                        result = ExecuteResponse(**bash.model_dump(), files=[])
                    finally:
                        client.delete_session(execution_session.session_id)
                else:
                    result = client.execute(code, timeout_ms=timeout, files=staged)
                context.check_active()
                artifacts = []
                for item in result.files[:10]:
                    if item.file_id:
                        artifacts.append(
                            Artifact(
                                artifact_id=item.file_id,
                                name=item.path,
                                source_ids=[str(value) for value in source_ids],
                                metadata={
                                    "isolated_service_file_id": item.file_id,
                                    "derived": True,
                                },
                            )
                        )
                receipt = {
                    "code_sha256": sha256(code.encode()).hexdigest(),
                    "language": language,
                    "inputs": [
                        {
                            "path": name,
                            "sha256": sha256(payload).hexdigest(),
                            "byte_count": len(payload),
                        }
                        for name, payload in payloads
                    ],
                    "duration_ms": result.duration_ms,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "stdout": result.stdout[:MAX_OUTPUT_CHARS],
                    "stderr": result.stderr[:MAX_OUTPUT_CHARS],
                    "output_truncated": len(result.stdout) > MAX_OUTPUT_CHARS
                    or len(result.stderr) > MAX_OUTPUT_CHARS,
                }
                return ToolOutcome(
                    status=OutcomeStatus.ERROR
                    if result.timed_out or result.exit_code not in (0, None)
                    else OutcomeStatus.PARTIAL
                    if receipt["output_truncated"]
                    else OutcomeStatus.FOUND,
                    summary="Isolated research program executed; derived output does not establish a legal rule.",
                    data=cast(dict[str, JsonValue], json_value(receipt)),
                    evidence=evidence,
                    artifacts=artifacts,
                )
            finally:
                for identifier in uploaded:
                    try:
                        client.delete_file(identifier)
                    except requests.RequestException:
                        pass
    except (requests.RequestException, CodeInterpreterVersionError):
        return ToolOutcome(
            status=OutcomeStatus.UNAVAILABLE,
            summary="Isolated code service could not complete the request.",
        )


def build_sandbox_specs(broker: CorpusBroker) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="calculate",
            description="Compute explicit Decimal amounts/allocations or dated deadlines. Business days require an explicit holiday calendar; does not infer legal rules.",
            parameters=schema(
                {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "multiply",
                            "divide",
                            "percent",
                            "allocate",
                            "calendar_days",
                            "business_days",
                            "following_month_day",
                        ],
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": ["string", "number"]},
                        "maxItems": 100,
                    },
                    "weights": {
                        "type": "array",
                        "items": {"type": ["string", "number"]},
                        "maxItems": 100,
                    },
                    "date": {"type": "string", "format": "date"},
                    "days": {"type": "integer", "minimum": 0, "maximum": 3660},
                    "day": {"type": "integer", "minimum": 1, "maximum": 31},
                    "calendar_name": {"type": "string"},
                    "holidays": {
                        "type": "array",
                        "items": {"type": "string", "format": "date"},
                        "maxItems": 1000,
                    },
                },
                ["operation"],
            ),
            handler=guarded(calculate),
        ),
        ToolSpec(
            name="compose_tool_calls",
            orchestrates=True,
            description="Run a bounded dependency-aware tool program, with parallel independent reads, $ref input paths and optional conditions. Every nested call uses the same registry and scope.",
            parameters=schema(
                {
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "tool": {"type": "string"},
                                "arguments": {"type": "object"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "when": {"type": "object"},
                            },
                            "required": ["id", "tool"],
                        },
                    },
                    "max_parallel": {"type": "integer", "minimum": 1, "maximum": 4},
                },
                ["steps"],
            ),
            handler=guarded(compose),
        ),
        ToolSpec(
            name="run_research_code",
            description="Execute Python or Bash in the configured isolated research service; stage authorized canonical source manifests only. Bash uses a disposable network-disabled service session. No host commands, production DB or S3 credentials are provided.",
            parameters=schema(
                {
                    "code": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_CODE_CHARS,
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "maxItems": 5,
                    },
                    "language": {"type": "string", "enum": ["python", "bash"]},
                    "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 30000},
                },
                ["code"],
            ),
            handler=guarded(lambda args, context: run_code(broker, args, context)),
            parallel_safe=False,
        ),
    ]
