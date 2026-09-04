"""Các tool hẹp và có kiểu dữ liệu rõ ràng dành cho Shopping Agent V0."""

from app.tools.calculator import build_calculator_tool
from app.tools.products import build_product_tools
from app.tools.registry import ToolRegistry
from app.tools.weather import build_weather_tool


# Nhận timeout mặc định của mỗi tool; trả registry chứa toàn bộ tool Phase 2.
def build_default_registry(default_timeout_seconds: float = 2.0) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_calculator_tool(default_timeout_seconds))
    for tool in build_product_tools(default_timeout_seconds):
        registry.register(tool)
    registry.register(build_weather_tool(default_timeout_seconds))
    return registry


__all__ = ["ToolRegistry", "build_default_registry"]
