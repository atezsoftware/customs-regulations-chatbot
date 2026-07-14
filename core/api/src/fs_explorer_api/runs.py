"""In-memory registry for resumable `/ws/explore` runs.

A run is kept here only while it is *interrupted* (the WebSocket connection
was lost or the client explicitly stopped it) but not yet finished — a run
that completes normally (an `answer_done`/`complete` event was sent) is never
registered. A later `/ws/explore` connection can send
`{"type": "resume", "run_id": ...}` instead of a fresh `{"task": ...}`
payload to look a record up and continue the same `FsExplorerAgent` (with
its accumulated `_chat_history`/`_step_count` intact) via
`workflow.resume_agent_run()`, instead of starting over from scratch.

Registry is process-local, plain-dict state (mirrors `new_workflow()`'s
per-request `ResourceManager` pattern elsewhere in this service — no
cross-process store). If `core-api` ever runs multiple replicas behind a
load balancer, a resume request only succeeds if it happens to land back on
the same pod that held the interrupted run; that is an accepted limitation,
not a bug, given nothing else in this service persists run state externally
either.
"""

import time
import uuid
from dataclasses import dataclass, field

from .agent import FsExplorerAgent
from .exploration_trace import ExplorationTrace

# How long an interrupted run stays resumable before being swept. Generous
# on purpose — the point is to survive "I stepped away and came back" and
# "my wifi dropped for a minute," not to be a tight cache.
RUN_TTL_SECONDS = 1800


@dataclass
class RunRecord:
    run_id: str
    agent: FsExplorerAgent
    trace: ExplorationTrace
    step_number: int
    folder: str
    use_index: bool
    enable_semantic: bool
    enable_metadata: bool
    index_folders: list[str]
    database_url: str | None
    original_task: str
    updated_at: float = field(default_factory=time.monotonic)


_REGISTRY: dict[str, RunRecord] = {}


def new_run_id() -> str:
    return uuid.uuid4().hex


def _sweep_expired() -> None:
    now = time.monotonic()
    expired = [
        run_id
        for run_id, record in _REGISTRY.items()
        if now - record.updated_at > RUN_TTL_SECONDS
    ]
    for run_id in expired:
        _REGISTRY.pop(run_id, None)


def register_run(record: RunRecord) -> None:
    _sweep_expired()
    record.updated_at = time.monotonic()
    _REGISTRY[record.run_id] = record


def get_run(run_id: str) -> RunRecord | None:
    _sweep_expired()
    return _REGISTRY.get(run_id)


def remove_run(run_id: str) -> None:
    _REGISTRY.pop(run_id, None)
