from app.config import Settings
from app.llm.openai_compatible import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            name="qwen",
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model=settings.qwen_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
        )

