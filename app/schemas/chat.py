from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Một function call do model yêu cầu ứng dụng thực thi.
class FunctionCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: str


# Một tool call có ID để ghép kết quả tool vào đúng yêu cầu của model.
class ToolCall(BaseModel):
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: FunctionCall


# Một message chuẩn gửi đến model, gồm cả assistant/tool trong agent loop.
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = Field(default=None, max_length=100_000)
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    # Nhận Message sau khi Pydantic parse; trả message hợp lệ theo role hoặc báo lỗi contract.
    @model_validator(mode="after")
    def validate_role_fields(self) -> "Message":
        if self.role in {"system", "user"} and not (self.content or "").strip():
            raise ValueError(f"{self.role} message requires non-empty content")
        if self.role == "assistant" and not (self.content or "").strip() and not self.tool_calls:
            raise ValueError("assistant message requires content or tool_calls")
        if self.role == "tool" and (not self.tool_call_id or self.content is None):
            raise ValueError("tool message requires tool_call_id and content")
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("only assistant messages may contain tool_calls")
        return self


# Một message trong lịch sử hội thoại do client gửi lên.
class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


# Các loại tác vụ dùng cho chính sách chọn model.
#mặc định
# trò truyện giá rẻ
# lý luận phức tạp
# dự phòng
class TaskType(str, Enum):
    DEFAULT = "default"
    CHEAP_CHAT = "cheap_chat"
    COMPLEX_REASONING = "complex_reasoning"
    FALLBACK = "fallback"


# Các định dạng đầu ra model mà ứng dụng hỗ trợ.
class ResponseFormat(str, Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"


# Tuỳ chọn điều khiển một lần gọi model.
# temperature: mức ngẫu nhiên/sáng tạo của câu trả lời.
# - 0: ổn định, phù hợp extract JSON, phân loại, test.
# - 0.7: mặc định cân bằng.
# - 1–2: đa dạng/sáng tạo hơn, nhưng ít ổn định hơn.
# - None: không gửi giá trị riêng; provider dùng LLM_TEMPERATURE trong config.

# max_tokens: giới hạn số token model được phép sinh ra, tức giới hạn độ dài output và chi phí.
# - Ví dụ max_tokens=100 cho câu trả lời ngắn.
# - None: provider dùng LLM_MAX_TOKENS trong config.
# - Không phải giới hạn input/context; history dài vẫn làm tăng input tokens.

# response_format:
# - text: trả lời văn bản bình thường.
# - json_object: yêu cầu model trả JSON. Endpoint /api/chat/structured tự ép giá trị này để parse thành ShoppingIntent
class ChatOptions(BaseModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0, le=100_000)
    response_format: ResponseFormat = ResponseFormat.TEXT


# Dữ liệu đầu vào cho chat đơn lượt hoặc nhiều lượt.
# message: tin nhắn chuẩn
# history: dạng list, mỗi phần tử sẽ gồm "role" và "content"
# system_prompt: là promt hệ thống, làm ngữ cảnh của AI
# task: cách AI trả lời
# options: thiết kế mặc định trong code, sau này mình có thể tùy chỉnh thêm
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


# Phản hồi nội bộ đã được chuẩn hoá từ provider.
class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


# Phản hồi chat công khai với tên trường camelCase.
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


# Cấu trúc ý định mua sắm được trích xuất từ câu người dùng.
class ShoppingIntent(BaseModel):
    category: str = Field(min_length=1)
    budget_max: int = Field(gt=0)
    requirements: list[str] = Field(min_length=1)


# Phản hồi chứa dữ liệu có cấu trúc và thông tin sử dụng model.
class StructuredChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: ShoppingIntent
    provider: str
    model: str
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    latency_ms: int = Field(alias="latencyMs")


# Một phần nội dung hoặc usage phát ra trong luồng SSE.
class StreamChunk(BaseModel):
    content: str = ""
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


# Khuôn dạng lỗi thống nhất trả về cho API client.
class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] | None = None
