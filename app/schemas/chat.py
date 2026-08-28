from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class TaskType(str, Enum):
    DEFAULT = "default"
    CHEAP_CHAT = "cheap_chat"
    COMPLEX_REASONING = "complex_reasoning"
    FALLBACK = "fallback"


class ResponseFormat(str, Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"


class ChatOptions(BaseModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0, le=100_000)
    response_format: ResponseFormat = ResponseFormat.TEXT


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=200)
    system_prompt: str = Field(
        default="You are a helpful assistant.", min_length=1, max_length=20_000
    )
    task: TaskType = TaskType.DEFAULT
    options: ChatOptions = Field(default_factory=ChatOptions)

    @field_validator("message", "system_prompt")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_history_order(self) -> "ChatRequest":
        for previous, current in zip(self.history, self.history[1:]):
            if previous.role == current.role:
                raise ValueError("history roles must alternate between user and assistant")
        return self

    def to_messages(self) -> list[Message]:
        return [
            Message(role="system", content=self.system_prompt),
            *(Message(role=item.role, content=item.content) for item in self.history),
            Message(role="user", content=self.message),
        ]


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: str
    provider: str
    model: str
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    latency_ms: int = Field(alias="latencyMs")

    @classmethod
    def from_llm(cls, response: LLMResponse) -> "ChatResponse":
        return cls(**response.model_dump(exclude={"finish_reason"}))


class ShoppingIntent(BaseModel):
    category: str = Field(min_length=1)
    budget_max: int = Field(gt=0)
    requirements: list[str] = Field(min_length=1)


class StructuredChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: ShoppingIntent
    provider: str
    model: str
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    latency_ms: int = Field(alias="latencyMs")


class StreamChunk(BaseModel):
    content: str = ""
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] | None = None

