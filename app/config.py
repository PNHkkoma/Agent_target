from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Đọc và kiểm tra cấu hình ứng dụng từ biến môi trường hoặc .env.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        enable_decoding=False,
    )

    app_name: str = "Agent Lab - Phase 1"
    log_level: str = "INFO"

    llm_provider: str = "deepseek"
    llm_fallback_providers: list[str] = Field(default_factory=lambda: ["qwen", "kimi"])
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0, le=5)
    llm_retry_base_delay_seconds: float = Field(default=0.25, ge=0, le=10)
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    llm_max_tokens: int = Field(default=1024, gt=0)

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    kimi_api_key: str = ""
    kimi_base_url: str = "https://api.moonshot.ai/v1"
    kimi_model: str = "kimi-k2.5"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini-2025-04-14"

    @field_validator("llm_fallback_providers", mode="before")
    @classmethod
    def parse_provider_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return value

    @field_validator("llm_provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.strip().lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()
