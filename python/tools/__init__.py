"""
工具层
提供可被 LLM 调用的 Tool 函数。
"""

from .product_tools import search_products, get_product_detail
from .inventory_tools import check_stock

__all__ = [
    "search_products",
    "get_product_detail",
    "check_stock",
]
