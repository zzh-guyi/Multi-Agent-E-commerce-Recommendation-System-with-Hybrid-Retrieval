from __future__ import annotations

import types
from typing import Any

from pymilvus import DataType, MilvusClient

from config import get_settings


class _AttrObj:
    """
    将 dict 商品数据转为对象属性访问兼容格式。
    使 ensure_products_indexed 同时支持：
        list[Product]  (对象，已有 .product_id 等)
        list[dict]     (字典，如 get_all_products 返回)
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, value)


def _to_attr_obj(product):
    """dict → _AttrObj，已是对象则原样返回。"""
    if isinstance(product, dict):
        return _AttrObj(product)
    return product


class VectorStore:

    """
    Milvus 商品向量数据库。

    Collection:

        product_id
        embedding

    负责：

        1. 创建 Collection
        2. 商品 Embedding 写入
        3. 商品 Embedding Upsert
        4. 删除
        5. Vector Search
        6. 检查商品是否已经建立 Embedding
    """

    def __init__(self):

        settings = get_settings()

        self.collection_name = (
            settings.milvus_collection
        )

        self.client = MilvusClient(
            uri=(
                f"http://"
                f"{settings.milvus_host}:"
                f"{settings.milvus_port}"
            )
        )

    # ========================================================
    # Collection
    # ========================================================

    def create_collection(
        self,
        dim: int = 1024,
    ) -> None:

        if self.client.has_collection(
            self.collection_name
        ):
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        schema.add_field(
            field_name="product_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )

        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )

        index_params = (
            self.client.prepare_index_params()
        )

        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

        print(
            "[VectorStore] Collection 创建成功:",
            self.collection_name,
        )

    # ========================================================
    # Collection 是否存在
    # ========================================================

    def exists(self) -> bool:

        return self.client.has_collection(
            self.collection_name
        )

    # ========================================================
    # 当前向量数量
    # ========================================================

    def count(self) -> int:

        if not self.exists():
            return 0

        result = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=[
                "product_id"
            ],
            limit=16384,
        )

        return len(result)

    # ========================================================
    # 获取已有 Product IDs
    # ========================================================

    def get_existing_product_ids(
        self,
    ) -> set[str]:

        if not self.exists():
            return set()

        rows = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=[
                "product_id"
            ],
            limit=16384,
        )

        return {
            str(row["product_id"])
            for row in rows
            if row.get("product_id")
        }

    # ========================================================
    # Insert
    # ========================================================

    def insert(
        self,
        product_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:

        if not product_ids or not embeddings:
            return

        if len(product_ids) != len(
            embeddings
        ):
            raise ValueError(
                "product_ids 和 embeddings 数量不一致"
            )

        if not self.exists():
            self.create_collection(
                dim=len(embeddings[0])
            )

        data = [
            {
                "product_id": product_id,
                "embedding": embedding,
            }
            for product_id, embedding
            in zip(
                product_ids,
                embeddings,
            )
        ]

        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )

        print(
            "[VectorStore] Embedding 插入成功:",
            len(data),
        )

    # ========================================================
    # Upsert
    # ========================================================

    def upsert(
        self,
        product_id: str,
        embedding: list[float],
    ) -> None:

        if not product_id:
            return

        if not embedding:
            raise ValueError(
                "embedding 不能为空"
            )

        if not self.exists():
            self.create_collection(
                dim=len(embedding)
            )

        self.client.upsert(
            collection_name=self.collection_name,
            data=[
                {
                    "product_id": product_id,
                    "embedding": embedding,
                }
            ],
        )

        print(
            "[VectorStore] Embedding upsert:",
            product_id,
        )

    # ========================================================
    # 删除
    # ========================================================

    def delete(
        self,
        product_id: str,
    ) -> None:

        if not product_id:
            return

        if not self.exists():
            return

        self.client.delete(
            collection_name=self.collection_name,
            filter=(
                f'product_id == "{product_id}"'
            ),
        )

        print(
            "[VectorStore] 删除:",
            product_id,
        )

    # ========================================================
    # Vector Search
    # ========================================================

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter: str = "",
    ) -> list[dict[str, Any]]:

        if not query_embedding:
            print(
                "[VectorStore] query_embedding 为空"
            )
            return []

        if not self.exists():
            print(
                "[VectorStore] Collection 不存在:",
                self.collection_name,
            )
            return []

        # ----------------------------------------------------
        # 检查 Collection 是否有数据
        # ----------------------------------------------------

        try:

            entity_count = self.count()

            print(
                "[VectorStore] Collection:",
                self.collection_name,
                "vector_count:",
                entity_count,
            )

            if entity_count == 0:

                print(
                    "[VectorStore] Collection 中没有任何商品向量"
                )

                return []

        except Exception as exc:

            print(
                "[VectorStore] 获取 vector_count 失败:",
                repr(exc),
            )

        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [query_embedding],
            "limit": limit,
            "output_fields": [
                "product_id"
            ],
            "search_params": {
                "metric_type": "COSINE",
            },
        }

        if filter:

            search_kwargs["filter"] = filter

        try:

            results = self.client.search(
                **search_kwargs
            )

        except Exception as exc:

            print(
                "[VectorStore] Vector Search 失败:",
                repr(exc),
            )

            return []

        if not results:

            return []

        return results[0]

    # ========================================================
    # Ensure Product Embeddings
    # ========================================================

    def ensure_products_indexed(
        self,
        products: list[Any],
        embedding_service: Any,
        batch_size: int = 32,
    ) -> dict[str, int]:
        """
        确保商品已经存在于 Milvus。

        这是解决：

            Vector Search 返回 []

        的关键。

        流程：

            Product
                ↓
            product_id
                ↓
            检查 Milvus
                ↓
            找到缺失商品
                ↓
            Product Text
                ↓
            Embedding
                ↓
            Upsert
        """

        # ----------------------------------------------------
        # 统一数据类型：支持 dict 和对象两种输入
        # ----------------------------------------------------

        products = [
            _to_attr_obj(p)
            for p in products
        ]

        if not products:

            return {
                "total": 0,
                "existing": 0,
                "created": 0,
            }

        # ----------------------------------------------------
        # 如果 Collection 不存在
        # ----------------------------------------------------

        if not self.exists():

            first_product = products[0]

            text = embedding_service.build_product_text(
                first_product
            )

            embedding = embedding_service.embed(
                text
            )

            self.create_collection(
                dim=len(embedding)
            )

            product_ids = [
                str(product.product_id)
                for product in products
            ]

            texts = [
                embedding_service.build_product_text(
                    product
                )
                for product in products
            ]

            embeddings = (
                embedding_service.embed_batch(
                    texts
                )
            )

            self.insert(
                product_ids=product_ids,
                embeddings=embeddings,
            )

            print(
                "[VectorStore] 首次建立商品向量:",
                len(products),
            )

            return {
                "total": len(products),
                "existing": 0,
                "created": len(products),
            }

        # ----------------------------------------------------
        # 获取已有 ID
        # ----------------------------------------------------

        existing_ids = (
            self.get_existing_product_ids()
        )

        missing_products = [
            product
            for product in products
            if str(product.product_id)
            not in existing_ids
        ]

        print(
            "[VectorStore] 商品总数:",
            len(products),
        )

        print(
            "[VectorStore] 已存在 Embedding:",
            len(
                products
            )
            - len(missing_products),
        )

        print(
            "[VectorStore] 缺失 Embedding:",
            len(missing_products),
        )

        if not missing_products:

            return {
                "total": len(products),
                "existing": len(products),
                "created": 0,
            }

        # ----------------------------------------------------
        # 批量建立缺失 Embedding
        # ----------------------------------------------------

        created = 0

        for start in range(
            0,
            len(missing_products),
            batch_size,
        ):

            batch = missing_products[
                start:
                start + batch_size
            ]

            texts = [
                embedding_service.build_product_text(
                    product
                )
                for product in batch
            ]

            embeddings = (
                embedding_service.embed_batch(
                    texts
                )
            )

            product_ids = [
                str(product.product_id)
                for product in batch
            ]

            self.insert(
                product_ids=product_ids,
                embeddings=embeddings,
            )

            created += len(batch)

            print(
                "[VectorStore] Embedding 创建进度:",
                f"{created}/{len(missing_products)}",
            )

        return {
            "total": len(products),
            "existing": len(products)
            - len(missing_products),
            "created": created,
        }