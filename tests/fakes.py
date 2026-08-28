from __future__ import annotations

from collections.abc import AsyncIterator

from app.llm.base import LLMProvider
from app.schemas.chat import ChatOptions, LLMResponse, Message, StreamChunk


class FakeProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        *,
        responses: list[LLMResponse | Exception] | None = None,
        chunks: list[StreamChunk | Exception] | None = None,
    ) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.api_key = "test-key"
        self.responses = responses or []
        self.chunks = chunks or []
        self.chat_calls = 0
        self.stream_calls = 0
        self.last_messages: list[Message] = []
        self.last_options: ChatOptions | None = None

    async def chat(self, messages: list[Message], options: ChatOptions) -> LLMResponse:
        self.chat_calls += 1
        self.last_messages = messages
        self.last_options = options
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def stream(
        self, messages: list[Message], options: ChatOptions
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        self.last_messages = messages
        self.last_options = options
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def close(self) -> None:
        return None


def response(provider: str, content: str = "hello") -> LLMResponse:
    return LLMResponse(
        content=content,
        provider=provider,
        model=f"{provider}-model",
        input_tokens=12,
        output_tokens=5,
        latency_ms=20,
    )

