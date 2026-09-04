import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest


def test_raw_input_builds_system_and_user_messages() -> None:
    request = ChatRequest(message="Tại sao trời có mưa?")
    assert [message.model_dump(exclude_none=True) for message in request.to_messages()] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tại sao trời có mưa?"},
    ]


def test_multi_turn_history_is_sent_back_to_model() -> None:
    request = ChatRequest(
        message="Tên tôi là gì?",
        history=[
            {"role": "user", "content": "Tôi tên An"},
            {"role": "assistant", "content": "Chào An"},
        ],
    )
    assert [message.role for message in request.to_messages()] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_history_must_alternate_roles() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            message="next",
            history=[
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
        )
