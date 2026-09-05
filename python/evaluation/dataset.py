"""
Evaluation Dataset

Each entry contains:
- query: the search query string
- relevant_product_ids: list of ground-truth product IDs
- relevance: optional dict mapping product_id -> relevance score (0-3)

Run this script to populate the dataset from the real product database:
    python -m evaluation.dataset --populate
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Query templates — based on real product categories in the database
# ------------------------------------------------------------------

# 这些查询基于数据库中真实存在的商品类目设计。
# 请将 queries.json 中的 relevant_product_ids 替换为实际相关的商品 ID。

QUERIES_TEMPLATE = [
    {
        "query": "预算500元以内的跑鞋",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的跑鞋商品ID补充",
    },
    {
        "query": "适合学生党的降噪耳机",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的耳机商品ID补充",
    },
    {
        "query": "轻便便携的旅行背包",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的背包商品ID补充",
    },
    {
        "query": "性价比高的机械键盘",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的键盘商品ID补充",
    },
    {
        "query": "适合办公的无线鼠标",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的鼠标商品ID补充",
    },
    {
        "query": "夏季透气运动T恤",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的T恤商品ID补充",
    },
    {
        "query": "大容量充电宝快充",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的充电宝商品ID补充",
    },
    {
        "query": "家用智能LED灯泡",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的灯泡商品ID补充",
    },
    {
        "query": "便携式蓝牙音箱",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的音箱商品ID补充",
    },
    {
        "query": "防晒保湿防晒霜",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的护肤商品ID补充",
    },
    {
        "query": "儿童安全座椅汽车用",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的母婴商品ID补充",
    },
    {
        "query": "不粘锅炒锅家用",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的厨具商品ID补充",
    },
    {
        "query": "瑜伽垫加厚防滑",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的瑜伽用品ID补充",
    },
    {
        "query": "电动牙刷声波震动",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的个护商品ID补充",
    },
    {
        "query": "护眼台灯书桌LED",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的台灯商品ID补充",
    },
    {
        "query": "保温杯大容量不锈钢",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的保温杯商品ID补充",
    },
    {
        "query": "USB集线器多接口拓展",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的配件商品ID补充",
    },
    {
        "query": "懒人沙发小户型客厅",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的家具商品ID补充",
    },
    {
        "query": "猫咪自动喂食器智能",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的宠物用品ID补充",
    },
    {
        "query": "折叠电动车便携式",
        "relevant_product_ids": [],
        "relevance": {},
        "note": "请根据实际数据库中的代步车商品ID补充",
    },
]


def load_dataset(path: str | None = None) -> list[dict]:
    """Load evaluation dataset from JSON file."""
    if path is None:
        path = str(Path(__file__).parent / "data" / "queries.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(queries: list[dict], path: str | None = None) -> None:
    """Save evaluation dataset to JSON file."""
    if path is None:
        path = str(Path(__file__).parent / "data" / "queries.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"Dataset saved to {path}, {len(queries)} queries")


def populate_from_db(db_url: str | None = None) -> list[dict]:
    """
    从数据库加载商品列表，生成带真实 product_id 的查询模板。
    返回查询列表（relevant_product_ids 仍为空，需要人工标注）。

    运行方式：
        python -m evaluation.dataset --populate
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import get_settings
        from services.product_store import ProductStore

        settings = get_settings()
        store = ProductStore()
        products = store.get_all_products()
        print(f"Loaded {len(products)} products from database")

        # Group by category
        by_category: dict[str, list[dict]] = {}
        for p in products:
            cat = p.get("category", "unknown")
            by_category.setdefault(cat, []).append(p)

        print("\nProduct categories:")
        for cat, prods in sorted(by_category.items()):
            ids = [p["product_id"] for p in prods[:5]]
            print(f"  {cat}: {len(prods)} products, sample IDs: {ids}")

        # Return template with notes for each category
        result = []
        for q in QUERIES_TEMPLATE:
            result.append(q)

        return result
    except Exception as e:
        print(f"Warning: could not connect to database: {e}")
        print("Using template queries without ground truth.")
        return QUERIES_TEMPLATE


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation dataset manager")
    parser.add_argument("--populate", action="store_true", help="Populate dataset from database")
    args = parser.parse_args()

    if args.populate:
        queries = populate_from_db()
    else:
        queries = QUERIES_TEMPLATE

    save_dataset(queries)
