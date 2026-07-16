import pytest

from fs_explorer_api.llm.gemini import GeminiLLMClient


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None


class _AsyncStream:
    def __init__(self) -> None:
        self._chunks = iter([_Chunk("hello"), _Chunk(" world")])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class _AsyncModels:
    async def generate_content_stream(self, **_kwargs):
        return _AsyncStream()


class _Aio:
    models = _AsyncModels()


class _Client:
    aio = _Aio()


@pytest.mark.asyncio
async def test_stream_text_accepts_awaitable_sdk_stream() -> None:
    client = GeminiLLMClient(client=_Client())

    chunks = [
        chunk async for chunk in client.stream_text([], "system", thinking_level="high")
    ]

    assert chunks == ["hello", " world"]
