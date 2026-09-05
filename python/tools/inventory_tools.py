from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from services.product_store import ProductStore


class _InventoryServices:
    """库存 Tool 共享的底层服务实例。"""

    def __init__(self):
        self.product_store = ProductStore()


_services: _InventoryServices | None = None


def _get_inventory_services() -> _InventoryServices:
    global _services
    if _services is None:
        _services = _InventoryServices()
    return _services


@tool
def check_stock(product_id: str) -> str:
    """
    查询单个商品的库存信息。

    Args:
        product_id: 商品 ID，例如 P001。

    Returns:
        JSON 字符串，包含 product_id、name、stock。
        如果商品不存在，返回 {"product_id": "xxx", "error": "not_found"}。
    """
    services = _get_inventory_services()
    product = services.product_store.get_product_by_id(product_id)
    if product is None:
        return json.dumps(
            {"product_id": product_id, "error": "not_found"},
            ensure_ascii=False,
        )
    return json.dumps({
        "product_id": product.get("product_id", ""),
        "name": product.get("name", ""),
        "stock": int(product.get("stock", 0)),
    }, ensure_ascii=False)
