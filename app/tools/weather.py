from __future__ import annotations

import unicodedata
from typing import Any

from pydantic import Field

from app.tools.registry import ToolArguments, ToolDefinition, ToolPermission


FAKE_WEATHER = {
    "da lat": {
        "city": "Da Lat",
        "temperature_c": 16,
        "condition": "rain",
        "forecast_period": "this weekend",
    },
    "hanoi": {
        "city": "Hanoi",
        "temperature_c": 27,
        "condition": "rain",
        "forecast_period": "today",
    },
    "ho chi minh city": {
        "city": "Ho Chi Minh City",
        "temperature_c": 31,
        "condition": "partly cloudy",
        "forecast_period": "today",
    },
}


# Thành phố cần tra trong dịch vụ thời tiết giả lập.
class WeatherArguments(ToolArguments):
    city: str = Field(min_length=1, max_length=100)


# Nhận tên thành phố có thể có dấu; trả khóa chuẩn để tra catalog thời tiết giả lập.
def _normalize_city(city: str) -> str:
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFD", city.lower())
        if unicodedata.category(char) != "Mn"
    ).strip()
    return normalized.replace("đ", "d")


# Nhận tên thành phố; trả dữ liệu thời tiết giả lập hoặc found=false nếu chưa có dữ liệu.
def get_weather(city: str) -> dict[str, Any]:
    normalized = _normalize_city(city)
    aliases = {"dalat": "da lat", "ha noi": "hanoi", "hcmc": "ho chi minh city"}
    normalized = aliases.get(normalized, normalized)
    weather = FAKE_WEATHER.get(normalized)
    if weather is None:
        return {
            "found": False,
            "city": city,
            "source": "fake_weather_catalog",
        }
    return {"found": True, **weather, "source": "fake_weather_catalog"}


# Nhận timeout cho handler; trả định nghĩa weather tool để đăng ký vào registry.
def build_weather_tool(timeout_seconds: float) -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description=(
            "Read the deterministic fake weather catalog for a city. Use this tool "
            "when weather is needed to choose suitable products. Never claim live "
            "weather; report that the data is simulated."
        ),
        input_model=WeatherArguments,
        handler=get_weather,
        timeout_seconds=timeout_seconds,
        permission=ToolPermission.WEATHER_READ,
    )
