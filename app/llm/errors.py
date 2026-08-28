from __future__ import annotations


class LLMError(Exception):
    code = "llm_error"
    retryable = False
    http_status = 502

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class ProviderConfigurationError(LLMError):
    code = "provider_not_configured"
    http_status = 503


class ProviderAuthenticationError(LLMError):
    code = "authentication_failed"


class ProviderRateLimitError(LLMError):
    code = "rate_limited"
    retryable = True
    http_status = 429


class ProviderTimeoutError(LLMError):
    code = "provider_timeout"
    retryable = True
    http_status = 504


class ProviderConnectionError(LLMError):
    code = "provider_connection_error"
    retryable = True
    http_status = 503


class ProviderUnavailableError(LLMError):
    code = "provider_unavailable"
    retryable = True
    http_status = 503


class ContextLengthError(LLMError):
    code = "context_too_long"
    http_status = 400


class MalformedResponseError(LLMError):
    code = "malformed_provider_response"
    retryable = True


class EmptyResponseError(LLMError):
    code = "empty_provider_response"
    retryable = True


class AllProvidersFailedError(LLMError):
    code = "all_providers_failed"
    http_status = 503

    def __init__(self, errors: list[LLMError]) -> None:
        self.errors = errors
        providers = ", ".join(error.provider or "unknown" for error in errors)
        super().__init__(f"All configured providers failed: {providers}")

