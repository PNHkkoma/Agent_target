from app.llm.errors import LLMError


# Lỗi khi request yêu cầu tool không nằm trong registry.
class AgentConfigurationError(LLMError):
    code = "invalid_agent_configuration"
    http_status = 422


# Lỗi khi agent vượt số vòng model tối đa.
class MaxAgentStepsExceededError(LLMError):
    code = "max_agent_steps_exceeded"
    http_status = 508


# Lỗi khi agent yêu cầu quá nhiều tool trong một request.
class MaxToolCallsExceededError(LLMError):
    code = "max_tool_calls_exceeded"
    http_status = 508


# Lỗi khi agent lặp lại cùng tool và arguments quá nhiều lần.
class DuplicateToolLoopError(LLMError):
    code = "duplicate_tool_loop"
    http_status = 508


# Lỗi khi toàn bộ agent loop vượt thời gian cho phép.
class AgentTimeoutError(LLMError):
    code = "agent_timeout"
    http_status = 504

