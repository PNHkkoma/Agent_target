from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.errors import (
    AllProvidersFailedError,
    LLMError,
    MalformedResponseError,
    ProviderConfigurationError,
)
from app.schemas.chat import ChatOptions, LLMResponse, Message, StreamChunk, TaskType

T = TypeVar("T")
ResponseValidator = Callable[[str], T]
logger = logging.getLogger("agent_lab.llm")


# Chọn provider, retry có giới hạn và fallback khi provider lỗi.
class ModelRouter:
    """Config-first V1 router with bounded retries and ordered fallback."""

    TASK_ROUTES = {
        TaskType.CHEAP_CHAT: "deepseek",
        TaskType.COMPLEX_REASONING: "kimi",
        TaskType.FALLBACK: "qwen",
    }

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        settings: Settings,
        *,
        sleep: Callable[[float], object] = asyncio.sleep,
    ) -> None:
        self.providers = providers
        self.settings = settings
        self._sleep = sleep

    def provider_order(self, task: TaskType) -> list[str]:
        primary = self.TASK_ROUTES.get(task, self.settings.llm_provider)
        ordered = [primary, *self.settings.llm_fallback_providers]
        return list(dict.fromkeys(name.lower() for name in ordered))

    async def chat(
        self,
        messages: list[Message],
        options: ChatOptions,
        *,
        task: TaskType,
        request_id: str,
        validate: ResponseValidator[T] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[LLMResponse, T | None]:
        errors: list[LLMError] = []
        for provider_name in self.provider_order(task):
            provider = self.providers.get(provider_name)
            if provider is None or not getattr(provider, "api_key", "test-key"):
                error = ProviderConfigurationError(
                    "Provider has no API key or is unknown", provider=provider_name
                )
                errors.append(error)
                self._log_failure(request_id, provider_name, "unknown", 0, 0, error)
                continue
            for attempt in range(self.settings.llm_max_retries + 1):
                started = time.perf_counter()
                try:
                    response = await provider.chat(
                        messages,
                        options,
                        tools=tools,
                        tool_choice=tool_choice,
                    )
                    parsed: T | None = None
                    if validate is not None:
                        try:
                            parsed = validate(response.content)
                        except Exception as exc:
                            raise MalformedResponseError(
                                "Response failed application schema validation",
                                provider=provider_name,
                            ) from exc
                    self._log_success(request_id, response, attempt + 1)
                    return response, parsed
                except LLMError as error:
                    errors.append(error)
                    latency_ms = round((time.perf_counter() - started) * 1000)
                    self._log_failure(
                        request_id,
                        provider_name,
                        provider.model,
                        latency_ms,
                        attempt + 1,
                        error,
                    )
                    if not error.retryable or attempt >= self.settings.llm_max_retries:
                        break
                    await self._sleep(
                        self.settings.llm_retry_base_delay_seconds * (2**attempt)
                    )
        raise AllProvidersFailedError(errors)

    async def stream(
        self,
        messages: list[Message],
        options: ChatOptions,
        *,
        task: TaskType,
        request_id: str,
    ) -> AsyncIterator[StreamChunk]:
        errors: list[LLMError] = []
        for provider_name in self.provider_order(task):
            provider = self.providers.get(provider_name)
            if provider is None or not getattr(provider, "api_key", "test-key"):
                errors.append(
                    ProviderConfigurationError(
                        "Provider has no API key or is unknown", provider=provider_name
                    )
                )
                continue
            for attempt in range(self.settings.llm_max_retries + 1):
                started = time.perf_counter()
                emitted = False
                input_tokens = output_tokens = 0
                try:
                    async for chunk in provider.stream(messages, options):
                        emitted = emitted or bool(chunk.content)
                        input_tokens = chunk.input_tokens or input_tokens
                        output_tokens = chunk.output_tokens or output_tokens
                        yield chunk
                    response = LLMResponse(
                        content="streamed",
                        provider=provider.name,
                        model=provider.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=round((time.perf_counter() - started) * 1000),
                    )
                    self._log_success(request_id, response, attempt + 1)
                    return
                except LLMError as error:
                    errors.append(error)
                    self._log_failure(
                        request_id,
                        provider_name,
                        provider.model,
                        round((time.perf_counter() - started) * 1000),
                        attempt + 1,
                        error,
                    )
                    # Once bytes reach the client, switching providers would duplicate/corrupt text.
                    if emitted:
                        raise
                    if not error.retryable or attempt >= self.settings.llm_max_retries:
                        break
                    await self._sleep(
                        self.settings.llm_retry_base_delay_seconds * (2**attempt)
                    )
        raise AllProvidersFailedError(errors)

    async def close(self) -> None:
        await asyncio.gather(*(provider.close() for provider in self.providers.values()))

    @staticmethod
    def _log_success(request_id: str, response: LLMResponse, attempt: int) -> None:
        logger.info(
            "LLM request completed",
            extra={
                "requestId": request_id,
                "provider": response.provider,
                "model": response.model,
                "latencyMs": response.latency_ms,
                "inputTokens": response.input_tokens,
                "outputTokens": response.output_tokens,
                "status": "SUCCESS",
                "attempt": attempt,
            },
        )

    @staticmethod
    def _log_failure(
        request_id: str,
        provider: str,
        model: str,
        latency_ms: int,
        attempt: int,
        error: LLMError,
    ) -> None:
        logger.warning(
            "LLM request failed",
            extra={
                "requestId": request_id,
                "provider": provider,
                "model": model,
                "latencyMs": latency_ms,
                "inputTokens": 0,
                "outputTokens": 0,
                "status": "FAILED",
                "attempt": attempt,
                "errorCode": error.code,
            },
        )
