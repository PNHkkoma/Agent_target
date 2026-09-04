from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.llm.base import LLMProvider
from app.llm.errors import (
    ContextLengthError,
    EmptyResponseError,
    LLMError,
    MalformedResponseError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.schemas.chat import (
    ChatOptions,
    FunctionCall,
    LLMResponse,
    Message,
    ResponseFormat,
    StreamChunk,
    ToolCall,
)


# Dùng chung logic HTTP cho các API có định dạng OpenAI Chat Completions.
class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        default_temperature: float,
        default_max_tokens: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def _payload(
        self,
        messages: list[Message],
        options: ChatOptions,
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump(exclude_none=True) for message in messages],
            "temperature": (
                options.temperature
                if options.temperature is not None
                else self.default_temperature
            ),
            "max_tokens": options.max_tokens or self.default_max_tokens,
            "stream": stream,
        }
        if options.response_format is ResponseFormat.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def chat(
        self,
        messages: list[Message],
        options: ChatOptions,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self.client.post(
                "/chat/completions",
                json=self._payload(
                    messages,
                    options,
                    stream=False,
                    tools=tools,
                    tool_choice=tool_choice,
                ),
            )
            self._raise_for_status(response)
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            raw_tool_calls = message.get("tool_calls") or []
            tool_calls = [
                ToolCall(
                    id=item["id"],
                    type=item.get("type", "function"),
                    function=FunctionCall(
                        name=item["function"]["name"],
                        arguments=item["function"]["arguments"],
                    ),
                )
                for item in raw_tool_calls
            ]
            if not content.strip() and not tool_calls:
                raise EmptyResponseError("Provider returned empty content", provider=self.name)
            usage = body.get("usage") or {}
            return LLMResponse(
                content=content,
                provider=self.name,
                model=body.get("model") or self.model,
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                latency_ms=round((time.perf_counter() - started) * 1000),
                finish_reason=choice.get("finish_reason"),
                tool_calls=tool_calls,
            )
        except LLMError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(str(exc), provider=self.name) from exc
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MalformedResponseError(
                "Provider returned malformed JSON", provider=self.name
            ) from exc

    async def stream(
        self, messages: list[Message], options: ChatOptions
    ) -> AsyncIterator[StreamChunk]:
        try:
            async with self.client.stream(
                "POST",
                "/chat/completions",
                json=self._payload(messages, options, stream=True),
            ) as response:
                self._raise_for_status(response)
                saw_content = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        body = json.loads(data)
                        choices = body.get("choices") or []
                        usage = body.get("usage") or {}
                        if choices:
                            choice = choices[0]
                            content = (choice.get("delta") or {}).get("content") or ""
                            saw_content = saw_content or bool(content)
                            yield StreamChunk(
                                content=content,
                                provider=self.name,
                                model=body.get("model") or self.model,
                                finish_reason=choice.get("finish_reason"),
                            )
                        if usage:
                            yield StreamChunk(
                                provider=self.name,
                                model=body.get("model") or self.model,
                                input_tokens=int(usage.get("prompt_tokens") or 0),
                                output_tokens=int(usage.get("completion_tokens") or 0),
                            )
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise MalformedResponseError(
                            "Provider returned malformed SSE data", provider=self.name
                        ) from exc
                if not saw_content:
                    raise EmptyResponseError(
                        "Provider stream returned no content", provider=self.name
                    )
        except LLMError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc), provider=self.name) from exc
        except httpx.TransportError as exc:
            raise ProviderConnectionError(str(exc), provider=self.name) from exc

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        status = response.status_code
        text = response.text[:500].lower()
        if status in (401, 403):
            error: LLMError = ProviderAuthenticationError(
                "Provider rejected API credentials", provider=self.name
            )
        elif status == 429:
            error = ProviderRateLimitError("Provider rate limit exceeded", provider=self.name)
        elif status in (408, 504):
            error = ProviderTimeoutError("Provider request timed out", provider=self.name)
        elif "context" in text and any(word in text for word in ("length", "long", "token")):
            error = ContextLengthError("Context window exceeded", provider=self.name)
        elif status >= 500 or status in (404, 409):
            error = ProviderUnavailableError(
                f"Provider unavailable (HTTP {status})", provider=self.name
            )
        else:
            error = LLMError(f"Provider request failed (HTTP {status})", provider=self.name)
        raise error

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
