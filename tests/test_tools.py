from __future__ import annotations

import pytest

from app.tools import build_default_registry
from app.tools.registry import ToolPermission


# Không nhận dữ liệu ngoài fixture; xác nhận registry chỉ có đúng năm shopping tool.
def test_registry_contains_exactly_five_shopping_tools() -> None:
    registry = build_default_registry()
    assert registry.names == [
        "search_products",
        "get_product_detail",
        "check_inventory",
        "calculate_shipping_fee",
        "get_order_status",
    ]


# Gửi bộ lọc giá; xác nhận search trả đúng các balo phù hợp mà không lộ tồn kho.
@pytest.mark.asyncio
async def test_search_products_applies_budget() -> None:
    registry = build_default_registry()
    result = await registry.execute(
        "search_products", {"category": "backpack", "max_price": 800_000}
    )
    assert result.status == "SUCCESS"
    assert [item["id"] for item in result.data["products"]] == ["P001", "P002"]
    assert all("inventory" not in item for item in result.data["products"])


# Gửi ID và tên rút gọn; xác nhận tool detail phân giải đúng cùng một sản phẩm.
@pytest.mark.asyncio
async def test_get_product_detail_accepts_id_and_alias() -> None:
    registry = build_default_registry()
    by_id = await registry.execute("get_product_detail", {"product_ref": "P001"})
    by_alias = await registry.execute("get_product_detail", {"product_ref": "A"})
    assert by_id.data == by_alias.data
    assert by_id.data["product"]["price"] == 650_000


# Gửi sản phẩm hết hàng; xác nhận inventory trả quantity=0 và in_stock=false.
@pytest.mark.asyncio
async def test_check_inventory_reports_out_of_stock() -> None:
    registry = build_default_registry()
    result = await registry.execute("check_inventory", {"product_ref": "P003"})
    assert result.data["quantity"] == 0
    assert result.data["in_stock"] is False


# Gửi hai sản phẩm và Hà Nội; xác nhận phí ship được tính từ tổng cân nặng.
@pytest.mark.asyncio
async def test_shipping_fee_uses_products_and_destination() -> None:
    registry = build_default_registry()
    result = await registry.execute(
        "calculate_shipping_fee",
        {"product_refs": ["P001", "P002"], "destination": "Hà Nội"},
    )
    assert result.status == "SUCCESS"
    assert result.data["total_weight_grams"] == 1_600
    assert result.data["fee_vnd"] == 40_000


# Gửi mã sản phẩm không tồn tại; xác nhận tool không tự bịa dữ liệu thay thế.
@pytest.mark.asyncio
async def test_missing_product_returns_found_false() -> None:
    registry = build_default_registry()
    result = await registry.execute(
        "get_product_detail", {"product_ref": "P999"}
    )
    assert result.data == {"found": False, "product_ref": "P999"}


# Gửi mã đơn hợp lệ; xác nhận trạng thái đến từ DB giả lập.
@pytest.mark.asyncio
async def test_order_status_reads_fake_database() -> None:
    registry = build_default_registry()
    result = await registry.execute("get_order_status", {"order_id": "ORD-1001"})
    assert result.status == "SUCCESS"
    assert result.data["status"] == "shipped"


# Gửi arguments thiếu/sai kiểu; xác nhận handler không chạy và trả INVALID_ARGUMENTS.
@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected() -> None:
    registry = build_default_registry()
    result = await registry.execute(
        "calculate_shipping_fee", {"product_refs": "P001"}
    )
    assert result.status == "ERROR"
    assert result.error == "INVALID_ARGUMENTS"


# Gửi tool không có trong allowlist; xác nhận registry trả UNKNOWN_TOOL.
@pytest.mark.asyncio
async def test_unknown_tool_is_rejected() -> None:
    registry = build_default_registry()
    result = await registry.execute("delete_database", {})
    assert result.error == "UNKNOWN_TOOL"


# Gửi tool đọc đơn nhưng chỉ cấp quyền catalog; xác nhận quyền được chặn trước handler.
@pytest.mark.asyncio
async def test_permission_is_enforced() -> None:
    registry = build_default_registry()
    result = await registry.execute(
        "get_order_status",
        {"order_id": "ORD-1001"},
        allowed_permissions={ToolPermission.CATALOG_READ},
    )
    assert result.error == "PERMISSION_DENIED"


# Gửi mã tạo lỗi giả lập; xác nhận exception được chuyển thành lỗi tool có cấu trúc.
@pytest.mark.asyncio
async def test_tool_exception_is_structured() -> None:
    registry = build_default_registry()
    result = await registry.execute("get_order_status", {"order_id": "ORD-ERROR"})
    assert result.error == "TOOL_EXECUTION_ERROR"


# Gửi mã phản hồi chậm với timeout rất ngắn; xác nhận trả TOOL_TIMEOUT có thể retry.
@pytest.mark.asyncio
async def test_tool_timeout_is_structured() -> None:
    registry = build_default_registry(0.01)
    result = await registry.execute("get_order_status", {"order_id": "ORD-SLOW"})
    assert result.error == "TOOL_TIMEOUT"
    assert result.retryable is True
