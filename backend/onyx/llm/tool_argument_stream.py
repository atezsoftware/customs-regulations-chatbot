"""Recognize stalled JSON formatting without restricting substantive tool output."""

import json
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, JsonValue

from onyx.llm.model_response import ModelResponseStream

# A formatting run this large has no semantic content. This is not an output,
# research or answer length limit; whitespace inside JSON strings is untouched.
TOOL_ARGUMENT_WHITESPACE_RUN = 4096
TOOL_ARGUMENT_STREAM_ABORT = "tool_argument_stream_aborted"


class ToolArgumentStreamAbort(BaseModel):
    call_index: int
    reason: str = "nonprogress_json_whitespace"
    whitespace_run: int
    received_argument_characters: int


class _ArgumentProgress(BaseModel):
    in_string: bool = False
    escaped: bool = False
    whitespace_run: int = 0
    longest_run: int = 0
    fragments: list[str] = Field(default_factory=list)
    received_characters: int = 0

    def accept(self, fragment: str) -> bool:
        self.fragments.append(fragment)
        self.received_characters += len(fragment)
        for character in fragment:
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif character == "\\":
                    self.escaped = True
                elif character == '"':
                    self.in_string = False
                continue
            if character in " \t\r\n":
                self.whitespace_run += 1
                self.longest_run = max(self.longest_run, self.whitespace_run)
            else:
                self.whitespace_run = 0
                if character == '"':
                    self.in_string = True
        if self.longest_run < TOOL_ARGUMENT_WHITESPACE_RUN:
            return False
        # If the provider already supplied the complete object in this delta,
        # retain it exactly, including unusual but legal formatting.
        try:
            return not isinstance(json.loads("".join(self.fragments)), dict)
        except (ValueError, RecursionError):
            return True


class ToolArgumentStreamGuard:
    def __init__(self) -> None:
        self._arguments: dict[int, _ArgumentProgress] = {}

    def accept(self, chunk: ModelResponseStream) -> ToolArgumentStreamAbort | None:
        for call in chunk.choice.delta.tool_calls:
            if call.function is None or call.function.arguments is None:
                continue
            progress = self._arguments.setdefault(call.index, _ArgumentProgress())
            if progress.accept(call.function.arguments):
                return ToolArgumentStreamAbort(
                    call_index=call.index,
                    whitespace_run=progress.longest_run,
                    received_argument_characters=progress.received_characters,
                )
        return None


def complete_object_prefix_fields(
    arguments: str, *, decoder: json.JSONDecoder
) -> dict[str, JsonValue]:
    """Retain fully decoded fields preceding an interrupted object member."""
    fields: dict[str, JsonValue] = {}
    position = len(arguments) - len(arguments.lstrip())
    if arguments[position : position + 1] != "{":
        return fields
    position += 1
    try:
        while True:
            while position < len(arguments) and arguments[position] in " \t\r\n":
                position += 1
            key, position = decoder.raw_decode(arguments, position)
            if not isinstance(key, str) or key in fields:
                return {}
            while position < len(arguments) and arguments[position] in " \t\r\n":
                position += 1
            if arguments[position : position + 1] != ":":
                break
            position += 1
            while position < len(arguments) and arguments[position] in " \t\r\n":
                position += 1
            value, position = decoder.raw_decode(arguments, position)
            while position < len(arguments) and arguments[position] in " \t\r\n":
                position += 1
            if position < len(arguments) and arguments[position] not in ",}":
                break
            fields[key] = value
            if arguments[position : position + 1] != ",":
                break
            position += 1
    except (ValueError, RecursionError):
        pass
    return fields


def close_sync_stream(stream: Any) -> None:
    """Close public LiteLLM wrappers down to their owned HTTP response."""
    pending = [stream]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        for attribute in ("completion_stream", "streaming_response", "response"):
            child = getattr(current, attribute, None)
            if child is not None:
                pending.append(child)
        close = getattr(current, "close", None)
        if callable(close):
            close()


def stream_with_first_chunk(
    response: Any, first_chunk: Any, chunks: Iterator[Any]
) -> Iterator[Any]:
    """Retain the provider stream's ownership after peeking its first delta."""
    try:
        yield first_chunk
        yield from chunks
    finally:
        close_sync_stream(response)
