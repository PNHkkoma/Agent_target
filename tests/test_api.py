from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.router import ModelRouter
from app.main import create_app
from app.schemas.chat import StreamChunk
from tests.fakes import FakeProvider, response


def make_client(provider: FakeProvider) -> TestClient:
    settings = Settings(
        _env_file=None,
        llm_provider=provider.name,
        llm_fallback_providers=[],
        llm_max_retries=0,
    )
    router = ModelRouter({provider.name: provider}, settings)
    return TestClient(create_app(settings=settings, model_router=router))


def test_chat_endpoint_returns_unified_camel_case_response() -> None:
    provider = FakeProvider("deepseek", responses=[response("deepseek", "Vì ngưng tụ.")])
    with make_client(provider) as client:
        result = client.post("/api/chat", json={"message": "Tại sao trời mưa?"})
    assert result.status_code == 200
    assert result.json() == {
        "content": "Vì ngưng tụ.",
        "provider": "deepseek",
        "model": "deepseek-model",
        "inputTokens": 12,
        "outputTokens": 5,
        "latencyMs": 20,
    }


def test_openai_is_available_as_a_provider() -> None:
    provider = FakeProvider("openai", responses=[response("openai", "Hello")])
    with make_client(provider) as client:
        result = client.post("/api/chat", json={"message": "Hello"})
    assert result.status_code == 200
    assert result.json()["provider"] == "openai"


def test_structured_endpoint_parses_and_validates_json() -> None:
    provider = FakeProvider(
        "qwen",
        responses=[
            response(
                "qwen",
                '{"category":"travel_bag","budget_max":1000000,'
                '"requirements":["lightweight","carry-on"]}',
            )
        ],
    )
    with make_client(provider) as client:
        result = client.post(
            "/api/chat/structured",
            json={"message": "Tôi cần balo đi Nhật khoảng 1 triệu, nhẹ và mang cabin."},
        )
    assert result.status_code == 200
    assert result.json()["data"]["budget_max"] == 1_000_000
    assert provider.last_options.response_format == "json_object"


def test_stream_endpoint_emits_token_usage_and_done_events() -> None:
    provider = FakeProvider(
        "kimi",
        chunks=[
            StreamChunk(content="Xin", provider="kimi", model="kimi-model"),
            StreamChunk(content=" chào", provider="kimi", model="kimi-model"),
            StreamChunk(
                provider="kimi",
                model="kimi-model",
                input_tokens=8,
                output_tokens=2,
            ),
        ],
    )
    with make_client(provider) as client:
        result = client.post("/api/chat/stream", json={"message": "hello"})
    assert result.status_code == 200
    assert result.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in result.text
    assert "event: usage" in result.text
    assert "event: done" in result.text
