from __future__ import annotations

import unicodedata
from typing import Any, Literal

from pydantic import Field, model_validator

from app.tools.registry import ToolArguments, ToolDefinition, ToolPermission


PRODUCT_CATALOG: list[dict[str, Any]] = [
    {
        "id": "P001",
        "name": "Balo City A",
        "category": "backpack",
        "price": 650_000,
        "weight_grams": 700,
        "features": ["ngăn laptop", "chống nước nhẹ"],
        "inventory": 8,
    },
    {
        "id": "P002",
        "name": "Balo Travel B",
        "category": "backpack",
        "price": 790_000,
        "weight_grams": 900,
        "features": ["mang cabin", "đai trợ lực"],
        "inventory": 3,
    },
    {
        "id": "P003",
        "name": "Balo Lite C",
        "category": "backpack",
        "price": 950_000,
        "weight_grams": 500,
        "features": ["siêu nhẹ", "chống nước"],
        "inventory": 0,
    },
    {
        "id": "P004",
        "name": "Áo khoác Rain D",
        "category": "jacket",
        "price": 890_000,
        "weight_grams": 450,
        "features": ["chống mưa", "giữ ấm"],
        "inventory": 5,
    },
    {
        "id": "P005",
        "name": "Vali Cabin E",
        "category": "suitcase",
        "price": 1_500_000,
        "weight_grams": 2_300,
        "features": ["kích thước cabin", "khóa TSA"],
        "inventory": 2,
    },
]


# Bộ lọc dùng để tìm sản phẩm trong catalog giả lập.
class SearchProductsArguments(ToolArguments):
    query: str | None = Field(default=None, min_length=1, max_length=100)
    category: Literal["backpack", "jacket", "suitcase"] | None = None
    max_price: int | None = Field(default=None, gt=0)
    max_weight_grams: int | None = Field(default=None, gt=0)

    # Nhận arguments sau khi parse; trả arguments nếu có query/category hoặc báo lỗi vì tìm kiếm quá rộng.
    @model_validator(mode="after")
    def require_search_term(self) -> "SearchProductsArguments":
        if not self.query and not self.category:
            raise ValueError("query or category is required")
        return self


# Tham chiếu sản phẩm bằng ID hoặc đúng tên hiển thị trong catalog.
class ProductReferenceArguments(ToolArguments):
    product_ref: str = Field(min_length=1, max_length=100)


# Danh sách sản phẩm và điểm đến dùng để tính phí vận chuyển giả lập.
class ShippingFeeArguments(ToolArguments):
    product_refs: list[str] = Field(min_length=1, max_length=10)
    destination: str = Field(min_length=2, max_length=100)


# Nhận chuỗi tiếng Việt/Anh; trả chuỗi chữ thường bỏ dấu để so khớp ổn định.
def _normalize(value: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFD", value.lower())
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(normalized.replace("đ", "d").split())


# Nhận ID hoặc tên sản phẩm; trả record catalog tương ứng hoặc None nếu không tồn tại.
def _find_product(product_ref: str) -> dict[str, Any] | None:
    normalized_ref = _normalize(product_ref)
    aliases = {
        "a": "p001",
        "b": "p002",
        "c": "p003",
        "d": "p004",
        "e": "p005",
    }
    normalized_ref = aliases.get(normalized_ref, normalized_ref)
    return next(
        (
            product
            for product in PRODUCT_CATALOG
            if normalized_ref in {_normalize(product["id"]), _normalize(product["name"])}
        ),
        None,
    )


# Nhận query/category và các giới hạn; trả bản tóm tắt sản phẩm phù hợp, chưa kết luận tồn kho.
def search_products(
    query: str | None = None,
    category: str | None = None,
    max_price: int | None = None,
    max_weight_grams: int | None = None,
) -> dict[str, Any]:
    category_aliases = {
        "backpack": "backpack",
        "balo": "backpack",
        "jacket": "jacket",
        "ao khoac": "jacket",
        "suitcase": "suitcase",
        "vali": "suitcase",
    }
    normalized_category = category_aliases.get(_normalize(category), category) if category else None
    normalized_query = _normalize(query) if query else None
    matches = [
        product
        for product in PRODUCT_CATALOG
        if (normalized_category is None or product["category"] == normalized_category)
        and (
            normalized_query is None
            or normalized_query in _normalize(product["name"])
            or normalized_query in _normalize(product["category"])
        )
        and (max_price is None or product["price"] <= max_price)
        and (max_weight_grams is None or product["weight_grams"] <= max_weight_grams)
    ]
    summaries = [
        {
            "id": product["id"],
            "name": product["name"],
            "price": product["price"],
            "weight_grams": product["weight_grams"],
        }
        for product in matches
    ]
    return {"count": len(summaries), "products": summaries}


# Nhận ID/tên sản phẩm; trả toàn bộ thông tin mô tả hoặc found=false nếu không tồn tại.
def get_product_detail(product_ref: str) -> dict[str, Any]:
    product = _find_product(product_ref)
    if product is None:
        return {"found": False, "product_ref": product_ref}
    detail = {key: value for key, value in product.items() if key != "inventory"}
    return {"found": True, "product": detail}


# Nhận ID/tên sản phẩm; trả số lượng và trạng thái tồn kho hoặc found=false nếu không tồn tại.
def check_inventory(product_ref: str) -> dict[str, Any]:
    product = _find_product(product_ref)
    if product is None:
        return {"found": False, "product_ref": product_ref}
    quantity = product["inventory"]
    return {
        "found": True,
        "product_id": product["id"],
        "product_name": product["name"],
        "quantity": quantity,
        "in_stock": quantity > 0,
    }


# Nhận danh sách sản phẩm và nơi nhận; trả phí ship giả lập dựa trên vùng và tổng cân nặng.
def calculate_shipping_fee(
    product_refs: list[str], destination: str
) -> dict[str, Any]:
    products = [_find_product(product_ref) for product_ref in product_refs]
    missing = [
        product_ref
        for product_ref, product in zip(product_refs, products)
        if product is None
    ]
    if missing:
        return {"calculated": False, "missing_products": missing}

    destination_key = _normalize(destination)
    base_fees = {
        "ha noi": 30_000,
        "hanoi": 30_000,
        "da nang": 40_000,
        "ho chi minh": 35_000,
        "hcm": 35_000,
    }
    base_fee = base_fees.get(destination_key, 50_000)
    total_weight = sum(product["weight_grams"] for product in products if product)
    surcharge = max(0, (total_weight - 1_000 + 999) // 1_000) * 10_000
    return {
        "calculated": True,
        "destination": destination,
        "product_ids": [product["id"] for product in products if product],
        "total_weight_grams": total_weight,
        "fee_vnd": base_fee + surcharge,
        "source": "fake_shipping_table",
    }


# Nhận timeout của handler; trả đúng bốn định nghĩa tool liên quan đến catalog và vận chuyển.
def build_product_tools(timeout_seconds: float) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="search_products",
            description=(
                "Search the fake catalog. Use for discovery requests such as finding "
                "backpacks under a budget. category must be backpack, jacket, or "
                "suitcase. Search results do not contain inventory. Never use this "
                "before a direct tool when the user already supplied an exact ID/name."
            ),
            input_model=SearchProductsArguments,
            handler=search_products,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.CATALOG_READ,
        ),
        ToolDefinition(
            name="get_product_detail",
            description=(
                "Get authoritative product price, weight, category and features by "
                "product ID or exact name. Use for detail questions and comparisons."
            ),
            input_model=ProductReferenceArguments,
            handler=get_product_detail,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.CATALOG_READ,
        ),
        ToolDefinition(
            name="check_inventory",
            description=(
                "Check current stock quantity by product ID or exact name. Use whenever "
                "the user asks whether a product is available or in stock."
            ),
            input_model=ProductReferenceArguments,
            handler=check_inventory,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.CATALOG_READ,
        ),
        ToolDefinition(
            name="calculate_shipping_fee",
            description=(
                "Calculate a deterministic shipping fee for one or more product IDs or "
                "exact names and a destination. Requires both product_refs and destination. "
                "It reads weights internally, so no product-detail call is needed for a "
                "shipping-only question."
            ),
            input_model=ShippingFeeArguments,
            handler=calculate_shipping_fee,
            timeout_seconds=timeout_seconds,
            permission=ToolPermission.CATALOG_READ,
        ),
    ]
