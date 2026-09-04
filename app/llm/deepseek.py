from app.config import Settings
from app.llm.openai_compatible import OpenAICompatibleProvider


# Cấu hình adapter gọi API tương thích OpenAI của DeepSeek.
class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(
            name="deepseek",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            timeout_seconds=settings.llm_timeout_seconds,
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
        )
