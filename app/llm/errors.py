from __future__ import annotations


# Lỗi cơ sở cho mọi lỗi phát sinh từ tầng LLM.
class LLMError(Exception):
    code = "llm_error"
    retryable = False
    http_status = 502

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


# Provider thiếu API key hoặc chưa được khai báo.
class ProviderConfigurationError(LLMError):
    code = "provider_not_configured"
    http_status = 503


# Provider từ chối API key hoặc quyền truy cập.
class ProviderAuthenticationError(LLMError):
    code = "authentication_failed"


# Provider giới hạn tần suất gọi API (HTTP 429).
class ProviderRateLimitError(LLMError):
    code = "rate_limited"
    retryable = True
    http_status = 429


# Provider không phản hồi trong thời gian chờ.
class ProviderTimeoutError(LLMError):
    code = "provider_timeout"
    retryable = True
    http_status = 504


# Không thể kết nối ổn định tới provider.
class ProviderConnectionError(LLMError):
    code = "provider_connection_error"
    retryable = True
    http_status = 503


# Provider hoặc model đang tạm không khả dụng.
class ProviderUnavailableError(LLMError):
    code = "provider_unavailable"
    retryable = True
    http_status = 503


# Messages vượt giới hạn context của model.
class ContextLengthError(LLMError):
    code = "context_too_long"
    http_status = 400


# Provider trả dữ liệu không đúng JSON hoặc schema mong đợi.
class MalformedResponseError(LLMError):
    code = "malformed_provider_response"
    retryable = True


# Provider trả phản hồi rỗng.
class EmptyResponseError(LLMError):
    code = "empty_provider_response"
    retryable = True


# Tất cả provider trong chuỗi retry/fallback đều thất bại.
class AllProvidersFailedError(LLMError):
    code = "all_providers_failed"
    http_status = 503

    def __init__(self, errors: list[LLMError]) -> None:
        self.errors = errors
        providers = ", ".join(error.provider or "unknown" for error in errors)
        super().__init__(f"All configured providers failed: {providers}")
