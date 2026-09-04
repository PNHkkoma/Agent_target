from app.config import Settings
from app.llm.openai_compatible import OpenAICompatibleProvider


# Cấu hình adapter gọi trực tiếp OpenAI Chat Completions API.
class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI Chat Completions provider, pinned to a configured model snapshot."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            name="openai",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
        )
