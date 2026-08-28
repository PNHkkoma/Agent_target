from app.config import Settings


def test_provider_is_switched_only_by_config() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="QWEN",
        llm_fallback_providers="kimi, deepseek",
    )
    assert settings.llm_provider == "qwen"
    assert settings.llm_fallback_providers == ["kimi", "deepseek"]

