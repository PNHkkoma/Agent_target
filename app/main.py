from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.runner import AgentRunner
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.config import Settings, get_settings
from app.llm.deepseek import DeepSeekProvider
from app.llm.errors import AllProvidersFailedError, LLMError
from app.llm.kimi import KimiProvider
from app.llm.openai import OpenAIProvider
from app.llm.qwen import QwenProvider
from app.llm.router import ModelRouter
from app.logging import configure_logging
from app.tools import build_default_registry


# Nhận cấu hình provider; trả ModelRouter đã gắn DeepSeek, Qwen, Kimi và OpenAI.
def build_router(settings: Settings) -> ModelRouter:
    providers = {
        "deepseek": DeepSeekProvider(settings),
        "qwen": QwenProvider(settings),
        "kimi": KimiProvider(settings),
        "openai": OpenAIProvider(settings),
    }
    return ModelRouter(providers, settings)


# Nhận settings/router/runner tùy chọn; trả ứng dụng FastAPI hoàn chỉnh để chạy hoặc test.
def create_app(
    *,
    settings: Settings | None = None,
    model_router: ModelRouter | None = None,
    agent_runner: AgentRunner | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    # Nhận ứng dụng FastAPI; khởi tạo dependency khi startup và đóng router khi shutdown.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_router = model_router or build_router(resolved_settings)
        app.state.agent_runner = agent_runner or AgentRunner(
            app.state.model_router,
            build_default_registry(resolved_settings.tool_timeout_seconds),
            resolved_settings,
        )
        yield
        await app.state.model_router.close()

    app = FastAPI(title=resolved_settings.app_name, version="2.0.0", lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(agent_router)

    # Nhận request và handler kế tiếp; trả response có cùng X-Request-ID để truy vết.
    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    # Không nhận body; trả trạng thái ok để hệ thống giám sát biết server đang hoạt động.
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Nhận request và LLMError; trả JSONResponse chuẩn hóa gồm code, message và request ID.
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
                "requestId": getattr(request.state, "request_id", "unavailable"),
                "details": {"attempts": errors} if errors else None,
            },
        )

    return app


app = create_app()
