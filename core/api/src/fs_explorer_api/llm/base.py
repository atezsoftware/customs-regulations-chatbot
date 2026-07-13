"""
Provider-agnostic LLM client interface.

`FsExplorerAgent` talks to this interface, never to a specific provider SDK
directly. Today only Gemini is implemented (`gemini.py`); swapping providers
later means adding a new `LLMClient` implementation, not touching the agent's
decision logic.
"""

from typing import AsyncIterator, Literal, Protocol, TypeVar

from pydantic import BaseModel

Role = Literal["user", "model"]


class ChatTurn(BaseModel):
    """One turn of conversation history, independent of any provider's wire format."""

    role: Role
    text: str


class LLMUsage(BaseModel):
    """Token accounting for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    duration_ms: float = 0


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient(Protocol):
    """Minimal surface `FsExplorerAgent` needs from any LLM provider."""

    async def generate_structured(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, LLMUsage]:
        """Request a structured (schema-validated) response."""
        ...

    def stream_text(
        self,
        history: list[ChatTurn],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream a plain-text response chunk by chunk."""
        ...

    def last_stream_usage(self) -> LLMUsage | None:
        """Usage for the most recently completed `stream_text` call, if known."""
        ...
