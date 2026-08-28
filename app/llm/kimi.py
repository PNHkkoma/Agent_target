from app.config import Settings
from app.llm.openai_compatible import OpenAICompatibleProvider


class KimiProvider(OpenAICompatibleProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            name="kimi",
            api_key=settings.kimi_api_key,
            base_url=settings.kimi_base_url,
            model=settings.kimi_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
        )

