from __future__ import annotations

import json

import httpx
import pytest

from app.llm.errors import (
    ContextLengthError,
    EmptyResponseError,
    MalformedResponseError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.schemas.chat import ChatOptions, Message, ResponseFormat


def make_provider(handler) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(
        base_url="https://provider.test/v1",
        transport=httpx.MockTransport(handler),
    )
    return OpenAICompatibleProvider(
        name="test",
        api_key="secret",
        base_url="https://provider.test/v1",
        model="test-model",
        timeout_seconds=1,
        default_temperature=0.7,
        default_max_tokens=100,
        client=client,
    )


@pytest.mark.asyncio
async def test_chat_maps_openai_response_and_usage() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "model": "actual-model",
                "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )

    provider = make_provider(handler)
    result = await provider.chat(
        [Message(role="user", content="hi")],
        ChatOptions(response_format=ResponseFormat.JSON_OBJECT),
    )
    assert result.content == '{"ok":true}'
    assert (result.input_tokens, result.output_tokens) == (10, 3)
    await provider.client.aclose()


@pytest.mark.asyncio
async def test_chat_sends_tools_and_parses_tool_calls() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "calculate"
        return httpx.Response(
            200,
            json={
                "model": "actual-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expression":"2+2"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8},
            },
        )

    provider = make_provider(handler)
    result = await provider.chat(
        [Message(role="user", content="2+2")],
        ChatOptions(),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="auto",
    )
    assert result.content == ""
    assert result.tool_calls[0].function.name == "calculate"
    await provider.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "error_type"),
    [
        (401, "bad key", ProviderAuthenticationError),
        (429, "slow down", ProviderRateLimitError),
        (500, "server error", ProviderUnavailableError),
        (404, "model unavailable", ProviderUnavailableError),
        (400, "maximum context length exceeded", ContextLengthError),
    ],
)
async def test_http_errors_are_classified(status, body, error_type) -> None:
    provider = make_provider(lambda _: httpx.Response(status, text=body))
    with pytest.raises(error_type):
        await provider.chat([Message(role="user", content="hi")], ChatOptions())
    await provider.client.aclose()


@pytest.mark.asyncio
async def test_malformed_json_is_safe() -> None:
    provider = make_provider(lambda _: httpx.Response(200, text="not json"))
    with pytest.raises(MalformedResponseError):
        await provider.chat([Message(role="user", content="hi")], ChatOptions())
    await provider.client.aclose()


@pytest.mark.asyncio
async def test_empty_response_is_safe() -> None:
    provider = make_provider(
        lambda _: httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}], "usage": {}},
        )
    )
    with pytest.raises(EmptyResponseError):
        await provider.chat([Message(role="user", content="hi")], ChatOptions())
    await provider.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_error", "error_type"),
    [
        (httpx.ReadTimeout("timeout"), ProviderTimeoutError),
        (httpx.ReadError("reset"), ProviderConnectionError),
    ],
)
async def test_transport_errors_are_classified(transport_error, error_type) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise transport_error

    provider = make_provider(handler)
    with pytest.raises(error_type):
        await provider.chat([Message(role="user", content="hi")], ChatOptions())
    await provider.client.aclose()


@pytest.mark.asyncio
async def test_stream_parses_tokens_and_usage() -> None:
    body = "".join(
        [
            'data: {"model":"m","choices":[{"delta":{"content":"Xin"},"finish_reason":null}]}\n\n',
            'data: {"model":"m","choices":[{"delta":{"content":" chào"},"finish_reason":"stop"}]}\n\n',
            'data: {"model":"m","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    provider = make_provider(lambda _: httpx.Response(200, text=body))
    chunks = [
        chunk
        async for chunk in provider.stream(
            [Message(role="user", content="hi")], ChatOptions()
        )
    ]
    assert "".join(chunk.content for chunk in chunks) == "Xin chào"
    assert chunks[-1].input_tokens == 7
    await provider.client.aclose()
