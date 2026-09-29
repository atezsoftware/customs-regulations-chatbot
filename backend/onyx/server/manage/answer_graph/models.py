"""Typed responses for the read-only answer graph API."""

from __future__ import annotations

import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class AnswerGraphRunView(BaseModel):
    run_id: UUID | None
    assistant_message_id: int
    status: str
    capture_status: str | None = None
    capture_error: str | None = None
    model_name: str | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    final_answer_sha256: str | None = None


class AnswerGraphNodeView(BaseModel):
    node_id: str
    parent_node_id: str | None
    kind: str
    operation: str
    status: str
    capture_status: str
    started_at: datetime.datetime
    ended_at: datetime.datetime | None
    attributes: dict[str, Any]
    has_input: bool
    has_output: bool
    has_reasoning: bool
    error: str | None


class AnswerGraphEdgeView(BaseModel):
    from_node_id: str
    to_node_id: str
    kind: str


class AnswerGraphNodePage(BaseModel):
    nodes: list[AnswerGraphNodeView]
    next_offset: int | None


class AnswerGraphEdgePage(BaseModel):
    edges: list[AnswerGraphEdgeView]
    next_offset: int | None


class AnswerGraphNodeDetail(BaseModel):
    node: AnswerGraphNodeView
    input: Any | None
    output: Any | None
    reasoning: Any | None
    input_state: str
    output_state: str
    reasoning_state: str
