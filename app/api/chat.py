from __future__ import annotations

import json
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.llm.errors import LLMError, MalformedResponseError
from app.llm.router import ModelRouter
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ResponseFormat,
    ShoppingIntent,
    StructuredChatResponse,
)

router = APIRouter(prefix="/api", tags=["chat"])


def get_router(request: Request) -> ModelRouter:
    return request.app.state.model_router


RouterDependency = Annotated[ModelRouter, Depends(get_router)]


@router.post("/chat", response_model=ChatResponse, response_model_by_alias=True)
async def chat(payload: ChatRequest, model_router: RouterDependency) -> ChatResponse:
    request_id = str(uuid4())
    response, _ = await model_router.chat(
        payload.to_messages(),
        payload.options,
        task=payload.task,
        request_id=request_id,
    )
    return ChatResponse.from_llm(response)


def parse_shopping_intent(content: str) -> ShoppingIntent:
    try:
        return ShoppingIntent.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        raise MalformedResponseError("Invalid ShoppingIntent JSON") from exc


@router.post(
    "/chat/structured",
    response_model=StructuredChatResponse,
    response_model_by_alias=True,
)
async def structured_chat(
    payload: ChatRequest, model_router: RouterDependency
) -> StructuredChatResponse:
    request_id = str(uuid4())
    structured_prompt = (
        "Extract the shopping intent. Return only a JSON object with exactly these fields: "
        "category (string), budget_max (positive integer in VND), requirements "
        "(non-empty array of strings). Do not wrap JSON in Markdown."
    )
    messages = payload.to_messages()
    messages[0].content = f"{structured_prompt}\n\nAdditional instructions:\n{payload.system_prompt}"
    options = payload.options.model_copy(update={"response_format": ResponseFormat.JSON_OBJECT})
    response, parsed = await model_router.chat(
        messages,
        options,
        task=payload.task,
        request_id=request_id,
        validate=parse_shopping_intent,
    )
    assert parsed is not None
    return StructuredChatResponse(
        data=parsed,
        provider=response.provider,
        model=response.model,
        inputTokens=response.input_tokens,
        outputTokens=response.output_tokens,
        latencyMs=response.latency_ms,
    )


@router.post("/chat/stream")
async def stream_chat(
    payload: ChatRequest, model_router: RouterDependency
) -> StreamingResponse:
    request_id = str(uuid4())

    async def event_source():
        try:
            async for chunk in model_router.stream(
                payload.to_messages(),
                payload.options,
                task=payload.task,
                request_id=request_id,
            ):
                event = "usage" if chunk.input_tokens is not None else "token"
                data = chunk.model_dump(by_alias=True, exclude_none=True)
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except LLMError as exc:
            error = {"code": exc.code, "message": str(exc), "requestId": request_id}
            yield f"event: error\ndata: {json.dumps(error, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

