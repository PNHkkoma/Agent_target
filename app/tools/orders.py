from __future__ import annotations

import asyncio
from typing import Any

from pydantic import Field

from app.tools.registry import ToolArguments, ToolDefinition, ToolPermission


ORDER_CATALOG: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "status": "shipped",
        "updated_at": "2026-09-03T09:00:00+07:00",
        "tracking_code": "VN123456",
    },
    "ORD-1002": {
        "status": "processing",
        "updated_at": "2026-09-04T08:30:00+07:00",
        "tracking_code": None,
    },
    "ORD-1003": {
        "status": "cancelled",
        "updated_at": "2026-09-02T14:00:00+07:00",
        "tracking_code": None,
    },
}


# Mã đơn hàng dùng để truy vấn trạng thái trong kho dữ liệu giả lập.
class OrderStatusArguments(ToolArguments):
    order_id: str = Field(min_length=3, max_length=50)


# Nhận mã đơn; trả trạng thái giả lập, found=false, hoặc tạo lỗi/timeout cho bài kiểm thử.
async def get_order_status(order_id: str) -> dict[str, Any]:
    normalized = order_id.strip().upper()
    if normalized == "ORD-SLOW":
        await asyncio.sleep(10)
    if normalized == "ORD-ERROR":
        raise RuntimeError("Simulated order database failure")
    order = ORDER_CATALOG.get(normalized)
    if order is None:
        return {"found": False, "order_id": normalized}
    return {"found": True, "order_id": normalized, **order, "source": "fake_order_db"}


# Nhận timeout của handler; trả định nghĩa tool tra trạng thái đơn hàng.
def build_order_tool(timeout_seconds: float) -> ToolDefinition:
    return ToolDefinition(
        name="get_order_status",
        description=(
            "Get authoritative status for an order ID. Requires an explicit order_id; "
            "if the user did not provide one, ask for it instead of calling the tool."
        ),
        input_model=OrderStatusArguments,
        handler=get_order_status,
        timeout_seconds=timeout_seconds,
        permission=ToolPermission.ORDER_READ,
    )
