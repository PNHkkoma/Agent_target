from __future__ import annotations

import unicodedata
from typing import Any, Literal

from pydantic import Field

from app.tools.registry import ToolArguments, ToolDefinition, ToolPermission


PRODUCT_CATALOG: list[dict[str, Any]] = [
    {
        "id": 1,
        "name": "Travel Bag A",
        "category": "travel_bag",
        "price": 450_000,
        "weight": 600,
        "features": ["carry-on", "water-resistant"],
        "in_stock": True,
    },
    {
        "id": 2,
        "name": "Travel Bag B",
        "category": "travel_bag",
        "price": 1_200_000,
        "weight": 450,
        "features": ["carry-on", "lightweight", "waterproof"],
        "in_stock": True,
    },
    {
        "id": 3,
        "name": "Jacket A",
        "category": "jacket",
        "price": 650_000,
        "weight": 520,
        "features": ["warm", "water-resistant"],
        "in_stock": True,
    },
    {
        "id": 4,
        "name": "Jacket B",
        "category": "jacket",
        "price": 890_000,
        "weight": 420,
        "features": ["lightweight", "warm", "waterproof"],
        "in_stock": True,
    },
    {
        "id": 5,
        "name": "Jacket C",
        "category": "jacket",
        "price": 1_150_000,
        "weight": 380,
        "features": ["lightweight", "windproof"],
        "in_stock": False,
    },
]


# Bộ lọc có kiểu dữ liệu rõ ràng cho tìm kiếm catalog.
class SearchProductsArguments(ToolArguments):
    category: Literal["travel_bag", "jacket"] = Field(
        description=(
            "Exact catalog category. Use travel_bag for túi/balo/túi du lịch; "
            "use jacket for áo/áo khoác."
        )
    )
    max_price: int | None = Field(default=None, gt=0)
    max_weight: int | None = Field(default=None, gt=0)


# ID duy nhất dùng để lấy thông tin một sản phẩm.
class GetProductArguments(ToolArguments):
    product_id: int = Field(gt=0)


# Nhận tên category tự nhiên; trả mã category chuẩn dùng trong catalog giả lập.
def _normalize(value: str) -> str:
    value = "".join(
        char
        for char in unicodedata.normalize("NFD", value.lower())
        if unicodedata.category(char) != "Mn"
    )
    value = value.replace("đ", "d")
    aliases = {
        "bag": "travel_bag",
        "travel bag": "travel_bag",
        "backpack": "travel_bag",
        "balo": "travel_bag",
        "tui": "travel_bag",
        "tui du lich": "travel_bag",
        "ao khoac": "jacket",
        "coat": "jacket",
    }
    return aliases.get(value.strip(), value.strip().replace(" ", "_"))


# Nhận category cùng giới hạn giá/cân nặng; trả danh sách tóm tắt sản phẩm phù hợp còn hàng.
def search_products(
    category: str,
    max_price: int | None = None,
    max_weight: int | None = None,
) -> dict[str, Any]:
    normalized_category = _normalize(category)
    matches = [
        product
        for product in PRODUCT_CATALOG
        if product["category"] == normalized_category
        and product["in_stock"]
        and (max_price is None or product["price"] <= max_price)
        and (max_weight is None or product["weight"] <= max_weight)
    ]
    # Search chỉ trả summary; agent phải dùng get_product trước khi kết luận chi tiết.
    summaries = [
        {
            "id": product["id"],
            "name": product["name"],
            "price": product["price"],
            "weight": product["weight"],
        }
        for product in matches
    ]
    return {"count": len(summaries), "products": summaries}


# Nhận ID sản phẩm; trả chi tiết chính xác hoặc found=false nếu ID không tồn tại.
def get_product(product_id: int) -> dict[str, Any]:
    product = next(
        (item for item in PRODUCT_CATALOG if item["id"] == product_id), None
    )
    if product is None:
        return {"found": False, "product_id": product_id}
    return {"found": True, "product": product}


# Nhận timeout cho handler; trả hai định nghĩa tool tìm kiếm và xem chi tiết sản phẩm.
def build_product_tools(timeout_seconds: float) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="search_products",
            description=(
                "Search the fake product catalog using category (only travel_bag or "
                "jacket), maximum price in VND, and maximum weight in grams. Use "
                "whenever availability, current price, or specifications are needed. "
                "Never invent products that are not returned by this tool."
            ),
            input_model=SearchProductsArguments,
            handler=search_products,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.CATALOG_READ,
        ),
        ToolDefinition(
            name="get_product",
            description=(
                "Get authoritative details for one product by numeric product_id. "
                "Use when the user names a catalog product, asks its current price, "
                "or when detailed comparison is required."
            ),
            input_model=GetProductArguments,
            handler=get_product,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.CATALOG_READ,
        ),
    ]
