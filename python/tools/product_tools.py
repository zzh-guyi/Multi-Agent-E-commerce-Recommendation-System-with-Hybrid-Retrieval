from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.tools import tool

from services.product_store import ProductStore
from services.vector_store import VectorStore
from services.embedding_service import EmbeddingService
from services.product_store import ProductStore
from models.schemas import Product


class _ToolServices:
    """Tool 共享的底层服务实例。"""

    def __init__(self):
        self.product_store = ProductStore()
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()


_services: _ToolServices | None = None


def _get_services() -> _ToolServices:
    global _services
    if _services is None:
        _services = _ToolServices()
    return _services


def _format_products(products: list[Product]) -> list[dict[str, Any]]:
    result = []
    for p in products:
        result.append({
            "product_id": p.product_id,
            "name": p.name,
            "category": p.category,
            "price": p.price,
            "brand": p.brand,
            "tags": p.tags or [],
            "stock": p.stock,
            "image_url": p.image_url or "",
        })
    return result


@tool
def search_products(
    query: str,
    min_price: float = 0.0,
    max_price: float = 999999.0,
    limit: int = 20,
) -> str:
    """
    搜索商品。根据用户意图搜索商品，返回商品列表。

    Args:
        query: 搜索关键词或商品描述。
        min_price: 最低价格，默认 0。
        max_price: 最高价格，默认 999999。
        limit: 返回商品数量上限，默认 20。

    Returns:
        JSON 数组，每个元素包含 product_id、name、category、price、brand、tags、stock、image_url。
    """
    services = _get_services()

    # 1. 关键词搜索（MySQL LIKE）
    keyword_results = services.product_store.keyword_search_products(
        query=query,
        limit=limit * 2,
    )

    # 2. Embedding + 向量搜索
    vector_results: list[str] = []
    try:
        embedding = services.embedding_service.embed(query)
        vector_hits = services.vector_store.search(
            query_embedding=embedding,
            limit=limit * 2,
        )
        vector_results = [
            hit["entity"]["product_id"]
            for hit in vector_hits
            if hit.get("entity", {}).get("product_id")
        ][:limit * 2]
    except Exception:
        vector_results = []

    # 3. RRF 融合
    k = 60
    scores: dict[str, float] = {}
    for rank, pid in enumerate(vector_results, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    for rank, row in enumerate(keyword_results, start=1):
        pid = row.get("product_id", "")
        if pid:
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:limit]

    # 4. 查询完整商品信息并过滤价格
    all_products = services.product_store.get_all_products()
    id_to_row: dict[str, dict[str, Any]] = {}
    for row in all_products:
        pid = row.get("product_id", "")
        if pid:
            id_to_row[pid] = row

    results: list[dict[str, Any]] = []
    for pid in ranked_ids:
        row = id_to_row.get(pid)
        if not row:
            continue
        try:
            price = float(row.get("price", 0))
        except (TypeError, ValueError):
            price = 0.0
        if price < min_price or price > max_price:
            continue
        results.append({
            "product_id": row.get("product_id", ""),
            "name": row.get("name", ""),
            "category": row.get("category", ""),
            "price": price,
            "brand": row.get("brand", ""),
            "tags": row.get("tags", ""),
            "stock": int(row.get("stock", 0)),
            "image_url": row.get("image_url", ""),
        })
        if len(results) >= limit:
            break

    return json.dumps(results, ensure_ascii=False, default=str)


@tool
def get_product_detail(product_id: str) -> str:
    """
    根据商品 ID 查询单个商品的详细信息。

    Args:
        product_id: 商品 ID，例如 P001。

    Returns:
        JSON 字符串，包含 product_id、name、category、price、description、brand、seller_id、stock、tags、image_url。
        如果商品不存在，返回空 JSON 对象 {}。
    """
    services = _get_services()
    product = services.product_store.get_product_by_id(product_id)
    if product is None:
        return json.dumps({}, ensure_ascii=False)
    tags = product.get("tags", "")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return json.dumps({
        "product_id": product.get("product_id", ""),
        "name": product.get("name", ""),
        "category": product.get("category", ""),
        "price": float(product.get("price", 0)),
        "description": product.get("description", ""),
        "brand": product.get("brand", ""),
        "seller_id": product.get("seller_id", ""),
        "stock": int(product.get("stock", 0)),
        "tags": tags,
        "image_url": product.get("image_url", ""),
    }, ensure_ascii=False)
