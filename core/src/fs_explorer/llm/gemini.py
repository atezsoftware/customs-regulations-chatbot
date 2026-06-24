"""Gemini implementation of the provider-agnostic `LLMClient` interface."""

from typing import AsyncIterator

from google.genai import Client as GenAIClient
from google.genai.types import Content, HttpOptions, Part

from .base import ChatTurn, LLMUsage, SchemaT

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


def _to_contents(history: list[ChatTurn]) -> list[Content]:
    return [Content(role=turn.role, parts=[Part.from_text(text=turn.text)]) for turn in history]


class GeminiLLMClient:
    """Wraps `google.genai.Client`. Same calls/config the agent always made."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        client: GenAIClient | None = None,
    ) -> None:
        self.model = model or DEFAULT_GEMINI_MODEL
        self.temperature = temperature
        if client is not None:
            self.raw_client = client
        else:
            if api_key is None:
                raise ValueError(
                    "GeminiLLMClient requires an api_key (or a pre-built client)."
                )
            self.raw_client = GenAIClient(
                api_key=api_key,
                http_options=HttpOptions(api_version="v1beta"),
            )
        self._last_stream_usage: LLMUsage | None = None

    def _generation_config(self) -> dict:
        config: dict = {}
        if self.temperature is not None:
            config["temperature"] = self.temperature
        return config

    async def generate_structured(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        schema: type[SchemaT],
    ) -> tuple[SchemaT, LLMUsage]:
        response = await self.raw_client.aio.models.generate_content(
            model=self.model,
            contents=_to_contents(history),  # type: ignore[arg-type]
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_schema": schema,
                **self._generation_config(),
            },
        )

        usage = LLMUsage()
        if response.usage_metadata:
            usage = LLMUsage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
                thinking_tokens=getattr(
                    response.usage_metadata, "thoughts_token_count", None
                )
                or 0,
            )

        if response.text is None:
            raise ValueError("Gemini returned no text for a structured request.")

        return schema.model_validate_json(response.text), usage

    async def stream_text(
        self,
        history: list[ChatTurn],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        self._last_stream_usage = None
        stream_fn = getattr(self.raw_client.aio.models, "generate_content_stream", None)
        if stream_fn is None:
            return

        input_tokens = 0
        output_tokens = 0
        thinking_tokens = 0
        saw_usage = False

        async for chunk in stream_fn(
            model=self.model,
            contents=_to_contents(history),  # type: ignore[arg-type]
            config={
                "system_instruction": system_prompt,
                **self._generation_config(),
            },
        ):
            if getattr(chunk, "usage_metadata", None):
                saw_usage = True
                usage = chunk.usage_metadata
                input_tokens = usage.prompt_token_count or input_tokens
                output_tokens = usage.candidates_token_count or output_tokens
                thinking_tokens = (
                    getattr(usage, "thoughts_token_count", None) or thinking_tokens
                )

            text = getattr(chunk, "text", None)
            if text:
                yield text

        if saw_usage:
            self._last_stream_usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
            )

    def last_stream_usage(self) -> LLMUsage | None:
        return self._last_stream_usage
