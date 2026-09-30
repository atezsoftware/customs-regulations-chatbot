"""Durable, tenant-scoped capture for answer execution graphs.

Capture failures never interrupt chat. They remain visible as PARTIAL runs.
Payloads are redacted before encryption and stored only as PostgreSQL
ciphertext, separately from graph metadata.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import threading
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import BaseModel

from onyx.db.answer_graph import (
    add_answer_graph_edge,
    create_answer_graph_run,
    finish_answer_graph_message,
    finish_answer_graph_node,
    finish_answer_graph_run,
    get_answer_graph_run_by_message,
    latest_answer_graph_node_id,
    mark_answer_graph_capture_partial,
    mark_answer_graph_node_partial,
    start_answer_graph_node,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import ChatMessage
from onyx.tracing.framework.create import get_current_span, get_current_trace
from onyx.tracing.framework.processor_interface import TracingProcessor
from onyx.tracing.framework.span_data import (
    FunctionSpanData,
    GenerationSpanData,
)
from onyx.tracing.framework.spans import Span
from onyx.tracing.framework.traces import Trace
from onyx.utils.encryption import EncryptionError
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

logger = setup_logger(__name__)

_CIPHERTEXT_VERSION = b"AG01"
_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_MAX_NESTING = 16
_MAX_COLLECTION_ITEMS = 100_000
_SECRET_KEY = re.compile(
    r"(authorization|cookie|password|secret|api.?key|access.?token|"
    r"refresh.?token|private.?key|credential|signature|signed.?url|"
    r"mcp.?header)",
    re.IGNORECASE,
)
_SECRET_TEXT = (
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+"),
    re.compile(r"(?i)(basic\s+)[A-Za-z0-9+/=]+"),
    re.compile(
        r"(?i)((?:api[_-]?key|password|secret|token|cookie|set-cookie)"
        r"['\"]?\s*[:=]\s*['\"]?)[^\s,'\";}]+"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
_SIGNED_QUERY = re.compile(
    r"(?i)([?&](?:X-Amz-Signature|X-Amz-Credential|X-Amz-Security-Token|"
    r"X-Goog-Signature|X-Goog-Credential|sig|signature|token|key)=)[^&#\s]+"
)
_URL_USERINFO = re.compile(r"(://)[^/\s@:]+:[^/\s@]+@")
_ACTIVE_RUNS: dict[str, UUID] = {}
_ROOT_NODES: dict[str, str] = {}
_ACTIVE_RUNS_LOCK = threading.Lock()
_ACTIVE_STEP: ContextVar[str | None] = ContextVar("answer_graph_step", default=None)


def _active_run() -> UUID | None:
    trace = get_current_trace()
    if trace is None:
        return None
    with _ACTIVE_RUNS_LOCK:
        return _ACTIVE_RUNS.get(trace.trace_id)


def _root_node(trace_id: str) -> str | None:
    with _ACTIVE_RUNS_LOCK:
        return _ROOT_NODES.get(trace_id)


def redact_graph_value(value: Any, *, depth: int = 0) -> Any:
    """Return only serializable, redacted values; never stringify unknown objects."""
    if depth > _MAX_NESTING:
        return "[DEPTH_LIMIT]"
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        cleaned = value
        for pattern in _SECRET_TEXT:
            cleaned = pattern.sub(
                lambda match: (
                    match.group(1) + "[REDACTED]" if match.lastindex else "[REDACTED]"
                ),
                cleaned,
            )
        cleaned = _SIGNED_QUERY.sub(r"\1[REDACTED]", cleaned)
        return _URL_USERINFO.sub(r"\1[REDACTED]@", cleaned)
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_COLLECTION_ITEMS:
            return "[COLLECTION_LIMIT]"
        return [redact_graph_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > _MAX_COLLECTION_ITEMS:
            return "[COLLECTION_LIMIT]"
        return {
            str(key): (
                "[REDACTED]"
                if _SECRET_KEY.search(str(key))
                else redact_graph_value(item, depth=depth + 1)
            )
            for key, item in value.items()
        }
    return "[UNSUPPORTED_VALUE]"


def _datetime(value: str | None) -> datetime.datetime:
    return (
        datetime.datetime.fromisoformat(value)
        if value
        else datetime.datetime.now(datetime.timezone.utc)
    )


def _graph_key() -> bytes:
    secret = os.environ.get("ANSWER_GRAPH_ENCRYPTION_KEY") or os.environ.get(
        "ENCRYPTION_KEY_SECRET"
    )
    if secret is None or len(secret.encode("utf-8")) < 32:
        raise EncryptionError(
            "ANSWER_GRAPH_ENCRYPTION_KEY or ENCRYPTION_KEY_SECRET must contain at least 32 bytes"
        )
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"onyx-answer-graph-v1",
        info=b"node-payload",
    ).derive(secret.encode("utf-8"))


def _associated_data(run_id: UUID, node_id: str, part: str) -> bytes:
    return f"{run_id.hex}:{node_id}:{part}".encode("utf-8")


def _serialize_and_encrypt(
    value: Any, *, run_id: UUID, node_id: str, part: str
) -> bytes | None:
    if value is None:
        return None
    sanitized = redact_graph_value(value)
    raw = json.dumps(
        sanitized, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    if len(raw.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise ValueError("Graph payload exceeds the capture limit")
    nonce = os.urandom(12)
    ciphertext = AESGCM(_graph_key()).encrypt(
        nonce, raw.encode("utf-8"), _associated_data(run_id, node_id, part)
    )
    return _CIPHERTEXT_VERSION + nonce + ciphertext


def load_graph_part(
    ciphertext: bytes | None,
    *,
    run_id: UUID,
    node_id: str,
    part: str,
) -> Any | None:
    if ciphertext is None:
        return None
    if not ciphertext.startswith(_CIPHERTEXT_VERSION):
        raise ValueError("Unsupported answer graph ciphertext version")
    nonce = ciphertext[4:16]
    encrypted = ciphertext[16:]
    raw = AESGCM(_graph_key()).decrypt(
        nonce, encrypted, _associated_data(run_id, node_id, part)
    )
    return json.loads(raw)


def _span_contents(span: Span[Any]) -> tuple[Any, Any, Any, dict[str, Any], str]:
    data = span.span_data
    if isinstance(data, GenerationSpanData):
        attributes: dict[str, Any] = {
            "model": data.model,
            "usage": redact_graph_value(data.usage),
            "image_count": data.image_count,
            "time_to_first_action_seconds": data.time_to_first_action_seconds,
        }
        operation = str(
            redact_graph_value(
                str((data.model_config or {}).get("flow") or "generation")
            )
        )[:128]
        return (
            {
                "messages": data.input,
                "model_config": data.model_config,
                "request_params": data.request_params,
                "tools": data.tools,
            },
            data.output,
            data.reasoning,
            attributes,
            operation,
        )
    if isinstance(data, FunctionSpanData):
        recorded_agent = (data.mcp_data or {}).get("answer_graph_agent")
        return (
            data.input,
            data.output,
            None,
            {"agent": recorded_agent} if isinstance(recorded_agent, str) else {},
            str(redact_graph_value(data.name))[:128],
        )
    exported = data.export()
    return None, None, None, redact_graph_value(exported), data.type


class AnswerGraphTracingProcessor(TracingProcessor):
    """Persist only chat traces whose metadata has a reserved assistant message."""

    def _run_id(self, trace_id: str) -> UUID | None:
        with _ACTIVE_RUNS_LOCK:
            return _ACTIVE_RUNS.get(trace_id)

    def _mark_partial(self, run_id: UUID, reason: str) -> None:
        try:
            with get_session_with_current_tenant() as db_session:
                mark_answer_graph_capture_partial(db_session, run_id, reason)
        except Exception:
            logger.exception("Could not mark answer graph capture incomplete")

    def on_trace_start(self, trace: Trace) -> None:
        exported = trace.export() or {}
        metadata = exported.get("metadata") or {}
        assistant_id = metadata.get("assistant_message_id")
        user_id = metadata.get("user_message_id")
        session_id = metadata.get("chat_session_id")
        if not (assistant_id and user_id and session_id):
            return
        try:
            if metadata.get("tenant_id") != get_current_tenant_id():
                raise ValueError("Trace tenant context mismatch")
            with get_session_with_current_tenant() as db_session:
                run_id = create_answer_graph_run(
                    db_session,
                    trace_id=trace.trace_id,
                    assistant_message_id=int(assistant_id),
                    user_message_id=int(user_id),
                    chat_session_id=UUID(str(session_id)),
                    model_name=metadata.get("model_name"),
                )
            with _ACTIVE_RUNS_LOCK:
                _ACTIVE_RUNS[trace.trace_id] = run_id
        except Exception:
            logger.exception("Could not start answer graph capture")

    def on_trace_end(self, trace: Trace) -> None:
        with _ACTIVE_RUNS_LOCK:
            run_id = _ACTIVE_RUNS.pop(trace.trace_id, None)
            _ROOT_NODES.pop(trace.trace_id, None)
            another_attempt_active = run_id in _ACTIVE_RUNS.values()
        if run_id is None:
            return
        if another_attempt_active:
            return
        try:
            with get_session_with_current_tenant() as db_session:
                finish_answer_graph_run(db_session, run_id, failed=False)
        except Exception:
            logger.exception("Could not finish answer graph capture")

    def on_span_start(self, span: Span[Any]) -> None:  # noqa: ARG002
        return None

    def on_span_end(self, span: Span[Any]) -> None:
        run_id = self._run_id(span.trace_id)
        if run_id is None:
            return
        try:
            input_value, output_value, reasoning, attributes, operation = (
                _span_contents(span)
            )
            input_ciphertext = _serialize_and_encrypt(
                input_value, run_id=run_id, node_id=span.span_id, part="input"
            )
            output_ciphertext = _serialize_and_encrypt(
                output_value, run_id=run_id, node_id=span.span_id, part="output"
            )
            reasoning_ciphertext = _serialize_and_encrypt(
                reasoning, run_id=run_id, node_id=span.span_id, part="reasoning"
            )
            error = "span_error" if span.error is not None else None
            with get_session_with_current_tenant() as db_session:
                finish_answer_graph_node(
                    db_session,
                    run_id=run_id,
                    node_id=span.span_id,
                    parent_node_id=_ACTIVE_STEP.get()
                    or span.parent_id
                    or _root_node(span.trace_id),
                    kind=span.span_data.type,
                    operation=operation,
                    started_at=_datetime(span.started_at),
                    ended_at=_datetime(span.ended_at),
                    status="FAILED" if span.error else "COMPLETE",
                    attributes=attributes,
                    input_ciphertext=input_ciphertext,
                    output_ciphertext=output_ciphertext,
                    reasoning_ciphertext=reasoning_ciphertext,
                    error=error,
                    capture_status="COMPLETE",
                )
        except Exception:
            logger.exception("Could not record answer graph node")
            try:
                with get_session_with_current_tenant() as db_session:
                    _, _, _, _, operation = _span_contents(span)
                    start_answer_graph_node(
                        db_session,
                        run_id=run_id,
                        node_id=span.span_id,
                        parent_node_id=_ACTIVE_STEP.get()
                        or span.parent_id
                        or _root_node(span.trace_id),
                        kind=span.span_data.type,
                        operation=operation,
                        started_at=_datetime(span.started_at),
                    )
                    mark_answer_graph_node_partial(
                        db_session, run_id, span.span_id, "node_payload_failed"
                    )
            except Exception:
                self._mark_partial(run_id, "node_payload_failed")

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None


class AnswerGraphStep:
    """Record a decision or physical call without changing the existing trace tree."""

    def __init__(self, operation: str, input_value: Any = None) -> None:
        self.operation = operation
        self.input_value = input_value
        self.output_value: Any = None
        self.node_id: str | None = None
        self._run_id: UUID | None = None
        self._started_at: datetime.datetime | None = None
        self._parent_node_id: str | None = None
        self._step_token: Token[str | None] | None = None
        self._agent_name: str | None = None

    def __enter__(self) -> AnswerGraphStep:
        self._run_id = _active_run()
        if self._run_id is None:
            return self
        self.node_id = f"step_{uuid4().hex}"
        self._started_at = datetime.datetime.now(datetime.timezone.utc)
        parent = get_current_span()
        trace = get_current_trace()
        if trace is not None and self.operation == "chat.input":
            self._agent_name = {
                "run_llm_loop": "Chat agent",
                "run_deep_research_llm_loop": "Deep Research agent",
            }.get(trace.name, trace.name)
        self._parent_node_id = (
            _ACTIVE_STEP.get()
            or (parent.span_id if parent else None)
            or (
                _root_node(trace.trace_id)
                if trace is not None and self.operation != "chat.input"
                else None
            )
        )
        self._step_token = _ACTIVE_STEP.set(self.node_id)
        if trace is not None and self.operation == "chat.input":
            with _ACTIVE_RUNS_LOCK:
                _ROOT_NODES[trace.trace_id] = self.node_id
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._run_id is None or self.node_id is None or self._started_at is None:
            return
        try:
            input_ciphertext = _serialize_and_encrypt(
                self.input_value,
                run_id=self._run_id,
                node_id=self.node_id,
                part="input",
            )
            output_ciphertext = _serialize_and_encrypt(
                self.output_value,
                run_id=self._run_id,
                node_id=self.node_id,
                part="output",
            )
            with get_session_with_current_tenant() as db_session:
                finish_answer_graph_node(
                    db_session,
                    run_id=self._run_id,
                    node_id=self.node_id,
                    parent_node_id=self._parent_node_id,
                    kind="step",
                    operation=self.operation,
                    started_at=self._started_at,
                    ended_at=datetime.datetime.now(datetime.timezone.utc),
                    status="FAILED" if exc_type else "COMPLETE",
                    attributes={"agent": self._agent_name} if self._agent_name else {},
                    input_ciphertext=input_ciphertext,
                    output_ciphertext=output_ciphertext,
                    reasoning_ciphertext=None,
                    error=exc_type.__name__ if exc_type else None,
                    capture_status="COMPLETE",
                )
        except Exception:
            logger.exception("Could not finish answer graph step")
            try:
                with get_session_with_current_tenant() as db_session:
                    start_answer_graph_node(
                        db_session,
                        run_id=self._run_id,
                        node_id=self.node_id,
                        parent_node_id=self._parent_node_id,
                        kind="step",
                        operation=self.operation,
                        started_at=self._started_at,
                    )
                    mark_answer_graph_node_partial(
                        db_session, self._run_id, self.node_id, "step_payload_failed"
                    )
            except Exception:
                logger.exception("Could not mark answer graph step incomplete")
        finally:
            if self._step_token is not None:
                _ACTIVE_STEP.reset(self._step_token)


def graph_step(operation: str, input_value: Any = None) -> AnswerGraphStep:
    return AnswerGraphStep(operation, input_value)


def link_graph_nodes(from_node_id: str | None, to_node_id: str | None) -> None:
    run_id = _active_run()
    if run_id is None or from_node_id is None or to_node_id is None:
        return
    try:
        with get_session_with_current_tenant() as db_session:
            add_answer_graph_edge(
                db_session,
                run_id=run_id,
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                kind="data",
            )
    except Exception:
        logger.exception("Could not record answer graph data link")
        try:
            with get_session_with_current_tenant() as db_session:
                mark_answer_graph_capture_partial(
                    db_session, run_id, "data_link_failed"
                )
        except Exception:
            logger.exception("Could not mark answer graph data link incomplete")


def record_final_answer_message(assistant_message_id: int) -> None:
    """Bind the delivered, persisted answer after the LLM trace has ended."""
    run_id: UUID | None = None
    node_id = f"answer_{assistant_message_id}"
    try:
        with get_session_with_current_tenant() as db_session:
            run = get_answer_graph_run_by_message(db_session, assistant_message_id)
            message = db_session.get(ChatMessage, assistant_message_id)
            if run is None or message is None:
                return
            run_id = run.id
            answer = message.message
            reasoning = message.reasoning_tokens
            error = message.error
            citations = message.citations
        now = datetime.datetime.now(datetime.timezone.utc)
        ciphertext = _serialize_and_encrypt(
            {"answer": answer, "citations": citations},
            run_id=run_id,
            node_id=node_id,
            part="output",
        )
        reasoning_ciphertext = _serialize_and_encrypt(
            reasoning, run_id=run_id, node_id=node_id, part="reasoning"
        )
        with get_session_with_current_tenant() as db_session:
            finish_answer_graph_node(
                db_session,
                run_id=run_id,
                node_id=node_id,
                parent_node_id=None,
                kind="answer",
                operation="answer.delivered",
                started_at=now,
                ended_at=now,
                status="FAILED" if error else "COMPLETE",
                attributes={"source": "chat_message"},
                input_ciphertext=None,
                output_ciphertext=ciphertext,
                reasoning_ciphertext=reasoning_ciphertext,
                error="assistant_message_error" if error else None,
                capture_status="COMPLETE",
            )
            previous = latest_answer_graph_node_id(db_session, run_id)
            if previous is not None:
                add_answer_graph_edge(
                    db_session,
                    run_id=run_id,
                    from_node_id=previous,
                    to_node_id=node_id,
                    kind="data",
                )
            finish_answer_graph_message(db_session, assistant_message_id)
    except Exception:
        logger.exception("Could not record final answer graph node")
        if run_id is not None:
            try:
                with get_session_with_current_tenant() as db_session:
                    mark_answer_graph_node_partial(
                        db_session, run_id, node_id, "final_answer_node_failed"
                    )
                    finish_answer_graph_message(db_session, assistant_message_id)
            except Exception:
                logger.exception("Could not mark answer graph finalization incomplete")
