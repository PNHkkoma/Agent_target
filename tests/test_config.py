from app.config import Settings


# Không nhận đầu vào; xác nhận cấu hình mặc định production dùng semantic OpenAI thay vì local hash.
def test_production_embedding_default_is_openai() -> None:
    configured = Settings(_env_file=None)
    assert configured.embedding_provider == "openai"
    assert configured.embedding_model == "text-embedding-3-small"


def test_provider_is_switched_only_by_config() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="QWEN",
        llm_fallback_providers="kimi, deepseek",
    )
    assert settings.llm_provider == "qwen"
    assert settings.llm_fallback_providers == ["kimi", "deepseek"]


def test_openai_model_can_be_pinned_by_config() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4.1-mini-2025-04-14",
    )
    assert settings.llm_provider == "openai"
    assert settings.openai_model == "gpt-4.1-mini-2025-04-14"
