from __future__ import annotations

from functools import lru_cache
from typing import Literal

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

    app_name: str = "Agent Lab - Phase 3"
    log_level: str = "INFO"

    llm_provider: str = "deepseek"
    llm_fallback_providers: list[str] = Field(default_factory=lambda: ["qwen", "kimi"])
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0, le=5)
    llm_retry_base_delay_seconds: float = Field(default=0.25, ge=0, le=10)
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    llm_max_tokens: int = Field(default=1024, gt=0)

    agent_max_steps: int = Field(default=8, ge=1, le=20)
    agent_max_tool_calls: int = Field(default=12, ge=1, le=50)
    agent_max_duplicate_calls: int = Field(default=1, ge=0, le=5)
    agent_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    tool_timeout_seconds: float = Field(default=2.0, gt=0, le=30)

    rag_store: str = "memory"
    rag_database_url: str = "postgresql://agent:agent@localhost:5433/agent_lab"
    rag_chunk_size_chars: int = Field(default=800, ge=200, le=4_000)
    rag_chunk_overlap_chars: int = Field(default=120, ge=0, le=1_000)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_dense_candidate_k: int = Field(default=20, ge=1, le=50)
    rag_lexical_candidate_k: int = Field(default=20, ge=1, le=50)
    rag_hybrid_candidate_k: int = Field(default=20, ge=1, le=50)
    rag_rrf_k: int = Field(default=60, ge=1, le=200)
    rag_retrieval_mode: str = "hybrid"
    rag_hnsw_iterative_scan: Literal["off", "strict_order", "relaxed_order"] = "strict_order"
    reranker_enabled: bool = True
    reranker_candidate_k: int = Field(default=20, ge=1, le=50)
    reranker_top_k: int = Field(default=5, ge=1, le=20)
    rag_high_relevance_threshold: float = Field(default=1.0, ge=0, le=1)
    rag_medium_relevance_threshold: float = Field(default=0.5, ge=0, le=1)
    rag_min_reranker_confidence: float = Field(default=0.7, ge=0, le=1)
    rag_min_score: float = Field(default=0.15, ge=-1, le=1)
    embedding_provider: str = "openai"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=8, le=2_000)
    embedding_version: str = "2"
    embedding_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

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

    # Nhận tên backend RAG/embedding; trả tên chữ thường để factory so khớp ổn định.
    @field_validator(
        "rag_store", "embedding_provider", "rag_retrieval_mode", "rag_hnsw_iterative_scan"
    )
    @classmethod
    def normalize_rag_backend(cls, value: str) -> str:
        return value.strip().lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()
