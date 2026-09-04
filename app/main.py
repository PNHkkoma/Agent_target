from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.config import Settings, get_settings
from app.llm.deepseek import DeepSeekProvider
from app.llm.errors import AllProvidersFailedError, LLMError
from app.llm.kimi import KimiProvider
from app.llm.openai import OpenAIProvider
from app.llm.qwen import QwenProvider
from app.llm.router import ModelRouter
from app.logging import configure_logging


def build_router(settings: Settings) -> ModelRouter:
    providers = {
        "deepseek": DeepSeekProvider(settings),
        "qwen": QwenProvider(settings),
        "kimi": KimiProvider(settings),
        "openai": OpenAIProvider(settings),
    }
    return ModelRouter(providers, settings)


def create_app(
    *, settings: Settings | None = None, model_router: ModelRouter | None = None
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_router = model_router or build_router(resolved_settings)
        yield
        await app.state.model_router.close()

    app = FastAPI(title=resolved_settings.app_name, version="1.0.0", lifespan=lifespan)
    app.include_router(chat_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        errors = None
        if isinstance(exc, AllProvidersFailedError):
            errors = [
                {"provider": error.provider, "code": error.code}
                for error in exc.errors
            ]
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "code": exc.code,
                "message": str(exc),
                "requestId": request.headers.get("X-Request-ID", "unavailable"),
                "details": {"attempts": errors} if errors else None,
            },
        )

    return app


app = create_app()
