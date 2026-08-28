import pytest

from app.config import Settings
from app.llm.errors import MalformedResponseError, ProviderTimeoutError
from app.llm.router import ModelRouter
from app.schemas.chat import ChatOptions, Message, ShoppingIntent, TaskType
from tests.fakes import FakeProvider, response


def settings(**overrides) -> Settings:
    values = {
        "llm_provider": "deepseek",
        "llm_fallback_providers": ["qwen"],
        "llm_max_retries": 1,
        "llm_retry_base_delay_seconds": 0,
        **overrides,
    }
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_timeout_retries_once_then_falls_back() -> None:
    deepseek = FakeProvider(
        "deepseek",
        responses=[
            ProviderTimeoutError("timeout", provider="deepseek"),
            ProviderTimeoutError("timeout", provider="deepseek"),
        ],
    )
    qwen = FakeProvider("qwen", responses=[response("qwen")])
    router = ModelRouter({"deepseek": deepseek, "qwen": qwen}, settings())
    result, _ = await router.chat(
        [Message(role="user", content="hi")],
        ChatOptions(),
        task=TaskType.DEFAULT,
        request_id="request-1",
    )
    assert result.provider == "qwen"
    assert deepseek.chat_calls == 2
    assert qwen.chat_calls == 1


@pytest.mark.asyncio
async def test_task_routing_policy_overrides_default() -> None:
    kimi = FakeProvider("kimi", responses=[response("kimi")])
    router = ModelRouter({"kimi": kimi}, settings(llm_fallback_providers=[]))
    result, _ = await router.chat(
        [Message(role="user", content="hard")],
        ChatOptions(),
        task=TaskType.COMPLEX_REASONING,
        request_id="request-2",
    )
    assert result.provider == "kimi"


@pytest.mark.asyncio
async def test_schema_validation_failure_uses_fallback() -> None:
    deepseek = FakeProvider("deepseek", responses=[response("deepseek", "not-json")])
    qwen = FakeProvider(
        "qwen",
        responses=[
            response(
                "qwen",
                '{"category":"travel_bag","budget_max":1000000,"requirements":["lightweight"]}',
            )
        ],
    )
    router = ModelRouter({"deepseek": deepseek, "qwen": qwen}, settings(llm_max_retries=0))

    def validate(value: str) -> ShoppingIntent:
        try:
            return ShoppingIntent.model_validate_json(value)
        except ValueError as exc:
            raise MalformedResponseError("bad json") from exc

    result, parsed = await router.chat(
        [Message(role="user", content="bag")],
        ChatOptions(),
        task=TaskType.DEFAULT,
        request_id="request-3",
        validate=validate,
    )
    assert result.provider == "qwen"
    assert parsed and parsed.budget_max == 1_000_000
