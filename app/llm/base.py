from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from typing import Any

from app.schemas.chat import ChatOptions, LLMResponse, Message, StreamChunk


# Hợp đồng chung mà mọi nhà cung cấp LLM phải tuân theo.
class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        options: ChatOptions,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate one complete response."""

    @abstractmethod
    def stream(
        self, messages: list[Message], options: ChatOptions
    ) -> AsyncIterator[StreamChunk]:
        """Yield incremental content and a final usage chunk."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider-owned network resources."""
