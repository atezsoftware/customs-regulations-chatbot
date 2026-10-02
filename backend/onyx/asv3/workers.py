from __future__ import annotations

import builtins
import contextvars
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, cast
from uuid import uuid4

from pydantic import JsonValue

from onyx.asv3.artifacts import artifact_reference, compact_json
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    RunStopped,
    TaskSnapshot,
    TaskStatus,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.progress import ProgressReporter

ResearchRunner = Callable[
    [str, RunContext, Callable[[], builtins.list[str]]], ToolOutcome
]


class WorkerPool:
    def __init__(
        self,
        context: RunContext,
        runner: ResearchRunner,
        *,
        max_workers: int = 3,
        max_nested_workers: int = 2,
        max_tasks: int = 12,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.context = context
        self.runner = runner
        self.max_tasks = max_tasks
        self.progress = progress
        self._executors = {
            depth: ThreadPoolExecutor(
                max_workers=max_workers if depth == 1 else max_nested_workers,
                thread_name_prefix=f"asv3-depth-{depth}",
            )
            for depth in range(1, context.max_depth + 1)
        }
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._tasks: dict[str, TaskSnapshot] = {}
        self._contexts: dict[str, RunContext] = {}
        self._futures: dict[str, Future[None]] = {}
        self._closed = False

    def spawn(
        self,
        task: str,
        *,
        parent_task_id: str | None = None,
        request_context: RunContext | None = None,
        public_title: str | None = None,
        public_message: str | None = None,
    ) -> str:
        delegation = request_context or self.context
        delegation.check_active()
        if not task.strip() or len(task) > 4000:
            raise ValueError("Research task must contain 1–4000 characters")
        if (public_title is None) != (public_message is None):
            raise ValueError("Public task narration needs both title and message")
        if public_title is not None and public_message is not None:
            from onyx.asv3.supplemental_tools import public_narration_valid

            if not public_narration_valid(public_title, public_message, delegation):
                raise ValueError(
                    "Task narration must use natural question-language labels without internal details"
                )
        with self._lock:
            if self._closed or len(self._tasks) >= self.max_tasks:
                raise RunStopped("Research task capacity exhausted")
            if delegation.depth >= delegation.max_depth:
                raise RunStopped("Research delegation depth exhausted")
            task_id = str(uuid4())
            child = delegation.child()
            child.services["task_id"] = task_id
            child.services["parent_task_id"] = parent_task_id
            self._contexts[task_id] = child
            self._tasks[task_id] = TaskSnapshot(
                task_id=task_id,
                task=task,
                parent_task_id=parent_task_id,
                status=TaskStatus.QUEUED,
                public_title=public_title,
                public_message=public_message,
            )
            captured = contextvars.copy_context()
            self._futures[task_id] = cast(
                Future[None],
                self._executors[child.depth].submit(captured.run, self._run, task_id),
            )
            return task_id

    def _run(self, task_id: str) -> None:
        with self._lock:
            snapshot = self._tasks[task_id]
            child = self._contexts[task_id]
            if self._closed or child.is_cancelled():
                snapshot.status = TaskStatus.CANCELLED
                self._changed.notify_all()
                return
            snapshot.status = TaskStatus.RUNNING
            self._changed.notify_all()
        self._report_task(task_id)
        try:
            outcome = self.runner(snapshot.task, child, lambda: self.messages(task_id))
            child.check_active()
        except RunStopped as error:
            outcome = ToolOutcome(status=OutcomeStatus.CANCELLED, summary=str(error))
        except Exception:
            import logging

            logging.getLogger(__name__).exception("ASv3 researcher failed")
            outcome = ToolOutcome(
                status=OutcomeStatus.ERROR, summary="Researcher execution failed"
            )
        with self._lock:
            if (
                self._closed
                or child.is_cancelled()
                or snapshot.status == TaskStatus.CANCELLED
            ):
                snapshot.status = TaskStatus.CANCELLED
                snapshot.outcome = None
            else:
                ledger = self.context.services.get("evidence")
                if isinstance(ledger, EvidenceLedger) and outcome.evidence:
                    try:
                        numbers = ledger.add(outcome.evidence, child)
                        outcome = outcome.model_copy(
                            update={
                                "evidence": [],
                                "data": {**outcome.data, "evidence_numbers": numbers},
                            }
                        )
                    except (RunStopped, ValueError):
                        outcome = ToolOutcome(
                            status=OutcomeStatus.TRUNCATED,
                            summary="Research evidence budget reached",
                        )
                snapshot.outcome = outcome
                snapshot.status = (
                    TaskStatus.CANCELLED
                    if outcome.status == OutcomeStatus.CANCELLED
                    else TaskStatus.FAILED
                    if outcome.status
                    in (OutcomeStatus.ERROR, OutcomeStatus.UNAVAILABLE)
                    else TaskStatus.COMPLETED
                )
            self._changed.notify_all()
        self._report_task(task_id)

    def _report_task(self, task_id: str) -> None:
        if self.progress is None:
            return
        with self._lock:
            snapshot = self._tasks[task_id]
            status = snapshot.status.value
            active = sum(
                task.status in (TaskStatus.RUNNING, TaskStatus.QUEUED)
                for task in self._tasks.values()
            )
            completed = sum(
                task.status == TaskStatus.COMPLETED for task in self._tasks.values()
            )
        self.progress.report(
            "worker",
            status=status,
            task_id=task_id,
            parent_task_id=snapshot.parent_task_id,
            title=snapshot.public_title,
            message=snapshot.public_message,
            active_workers=active,
            completed_workers=completed,
        )

    def messages(self, task_id: str) -> builtins.list[str]:
        with self._lock:
            return list(self._tasks[task_id].updates)

    def send_update(self, task_id: str, message: str) -> None:
        self.context.check_active()
        if not message.strip() or len(message) > 4000:
            raise ValueError("Update must contain 1–4000 characters")
        with self._lock:
            self._tasks[task_id].updates.append(message)
            self._changed.notify_all()

    def followup(
        self, task_id: str, message: str, *, request_context: RunContext | None = None
    ) -> str:
        self.send_update(task_id, message)
        with self._lock:
            snapshot = self._tasks[task_id]
            if snapshot.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                return task_id
            return self.spawn(
                snapshot.task + "\nFollow-up: " + message,
                parent_task_id=task_id,
                request_context=request_context,
                public_title=snapshot.public_title,
                public_message=snapshot.public_message,
            )

    def cancel(self, task_id: str) -> None:
        with self._lock:
            self._contexts[task_id].cancel()
            self._tasks[task_id].status = TaskStatus.CANCELLED
            self._tasks[task_id].outcome = None
            future = self._futures.get(task_id)
            if future is not None:
                future.cancel()
            self._changed.notify_all()
        self._report_task(task_id)

    @staticmethod
    def _task_view(task: TaskSnapshot) -> TaskSnapshot:
        outcome = task.outcome
        if outcome is not None:
            data = compact_json(outcome.data, max_chars=6000)
            assert isinstance(data, dict)
            outcome = outcome.model_copy(
                deep=False,
                update={
                    "summary": outcome.summary[:3000],
                    "data": {**data, "summary_truncated": len(outcome.summary) > 3000},
                    "artifacts": [
                        artifact_reference(item) for item in outcome.artifacts
                    ],
                },
            )
        return task.model_copy(
            deep=False, update={"updates": list(task.updates[-16:]), "outcome": outcome}
        )

    def list(self) -> builtins.list[TaskSnapshot]:
        with self._lock:
            return [self._task_view(item) for item in self._tasks.values()]

    def wait(
        self,
        task_id: str | None = None,
        timeout_seconds: float = 5,
        *,
        request_context: RunContext | None = None,
    ) -> builtins.list[TaskSnapshot]:
        timeout_seconds = min(30, max(0, timeout_seconds))
        deadline = min(time.monotonic() + timeout_seconds, self.context.deadline)
        with self._changed:
            selected = [self._tasks[task_id]] if task_id else list(self._tasks.values())
            pending_at_start = {
                task.task_id
                for task in selected
                if task.status in (TaskStatus.RUNNING, TaskStatus.QUEUED)
            }
            while True:
                (request_context or self.context).check_active()
                tasks = (
                    [self._tasks[task_id]] if task_id else list(self._tasks.values())
                )
                if not pending_at_start or any(
                    task.task_id in pending_at_start
                    and task.status not in (TaskStatus.RUNNING, TaskStatus.QUEUED)
                    for task in tasks
                ):
                    return [self._task_view(task) for task in tasks]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [self._task_view(task) for task in tasks]
                self._changed.wait(min(remaining, 0.05))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for task_id, snapshot in self._tasks.items():
                if snapshot.status in (TaskStatus.RUNNING, TaskStatus.QUEUED):
                    self.cancel(task_id)
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)

    def tool_specs(self) -> builtins.list[ToolSpec]:
        def outcome(items: builtins.list[TaskSnapshot]) -> ToolOutcome:
            evidence = [
                item for task in items if task.outcome for item in task.outcome.evidence
            ]
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="Research task state",
                data={"tasks": [task.model_dump(mode="json") for task in items]},
                evidence=evidence,
            )

        def spawn(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
            parent = context.services.get("task_id")
            task_id = self.spawn(
                str(args["task"]),
                request_context=context,
                parent_task_id=parent if isinstance(parent, str) else None,
                public_title=str(args["public_title"])
                if args.get("public_title")
                else None,
                public_message=str(args["public_message"])
                if args.get("public_message")
                else None,
            )
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="Researcher queued",
                data={"task_id": task_id},
            )

        def send(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
            self.send_update(str(args["task_id"]), str(args["message"]))
            return ToolOutcome(status=OutcomeStatus.FOUND, summary="Update delivered")

        def followup(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
            task_id = self.followup(
                str(args["task_id"]), str(args["message"]), request_context=context
            )
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="Follow-up delivered",
                data={"task_id": task_id},
            )

        def wait(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
            task_id = str(args["task_id"]) if args.get("task_id") else None
            timeout = args.get("timeout_seconds", 5)
            return outcome(
                self.wait(task_id, float(str(timeout)), request_context=context)
            )

        def cancel(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
            self.cancel(str(args["task_id"]))
            return ToolOutcome(
                status=OutcomeStatus.FOUND, summary="Researcher cancelled"
            )

        def schema(
            properties: dict[str, JsonValue], required: builtins.list[str]
        ) -> dict[str, JsonValue]:
            return {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

        identifier: dict[str, JsonValue] = {"type": "string", "minLength": 1}
        message: dict[str, JsonValue] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 4000,
        }
        return [
            ToolSpec(
                name="spawn_researcher",
                description="Delegate one independent information need. The researcher chooses its own tools.",
                parameters=schema(
                    {
                        "task": message,
                        "public_title": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 240,
                        },
                        "public_message": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1600,
                        },
                    },
                    ["task"],
                ),
                orchestrates=True,
                handler=spawn,
            ),
            ToolSpec(
                name="send_update",
                description="Deliver additional facts to an existing researcher without starting a new task.",
                parameters=schema(
                    {"task_id": identifier, "message": message}, ["task_id", "message"]
                ),
                orchestrates=True,
                handler=send,
            ),
            ToolSpec(
                name="followup_researcher",
                description="Follow up with a running researcher or start a linked continuation for a finished researcher.",
                parameters=schema(
                    {"task_id": identifier, "message": message}, ["task_id", "message"]
                ),
                orchestrates=True,
                handler=followup,
            ),
            ToolSpec(
                name="list_researchers",
                description="Inspect all researcher states and completed evidence.",
                parameters=schema({}, []),
                orchestrates=True,
                handler=lambda _args, _context: outcome(self.list()),
            ),
            ToolSpec(
                name="wait_researcher",
                description="Wait briefly for a researcher result; running status is not a completed answer.",
                parameters=schema(
                    {
                        "task_id": identifier,
                        "timeout_seconds": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 30,
                        },
                    },
                    [],
                ),
                orchestrates=True,
                handler=wait,
            ),
            ToolSpec(
                name="cancel_researcher",
                description="Cancel one researcher and reject its late results.",
                parameters=schema({"task_id": identifier}, ["task_id"]),
                orchestrates=True,
                handler=cancel,
            ),
        ]

    def export(self) -> dict[str, JsonValue]:
        return {
            "version": 1,
            "run_id": self.context.run_id,
            "tasks": [
                task.model_dump(mode="json", exclude={"outcome": {"evidence"}})
                for task in self.list()
            ],
        }

    def restore(self, payload: dict[str, JsonValue]) -> None:
        if payload.get("version") != 1 or payload.get("run_id") != self.context.run_id:
            raise ValueError("Worker checkpoint identity mismatch")
        raw = payload.get("tasks")
        if not isinstance(raw, list) or len(raw) > self.max_tasks:
            raise ValueError("Invalid worker checkpoint")
        snapshots = [TaskSnapshot.model_validate(item) for item in raw]
        if len({item.task_id for item in snapshots}) != len(snapshots):
            raise ValueError("Duplicate restored task")
        with self._lock:
            if self._tasks:
                raise ValueError("Cannot restore into an active pool")
            for snapshot in snapshots:
                if snapshot.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                    snapshot.status = TaskStatus.INTERRUPTED
                    snapshot.outcome = None
                self._tasks[snapshot.task_id] = snapshot
                self._contexts[snapshot.task_id] = self.context.child()
