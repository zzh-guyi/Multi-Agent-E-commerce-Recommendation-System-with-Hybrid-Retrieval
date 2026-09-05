from __future__ import annotations

from typing import Any

from services.product_store import ProductStore
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


class ProductService:
    """
    商品业务服务层。

    负责协调：

        MySQL
          ↓
      商品数据
          ↓
    EmbeddingService
          ↓
       Milvus

    ProductService 不直接处理 HTTP 请求，
    由 FastAPI API 层调用。

    主要职责：

    1. 新增商品
    2. 修改商品
    3. 删除商品
    4. 查询商品
    5. 保证 MySQL 与 Milvus 基本同步
    """

    def __init__(self):
        self.product_store = ProductStore()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    # ============================================================
    # 新增商品
    # ============================================================

    def add_product(
        self,
        product: dict[str, Any],
    ) -> dict[str, Any]:
        """
        新增商品。

        流程：

            API
             ↓
        ProductService
             ↓
        MySQL
             ↓
        Embedding
             ↓
        Milvus
        """

        product_id = product["product_id"]

        # --------------------------------------------------------
        # 1. 检查商品是否已经存在
        # --------------------------------------------------------

        existing = self.product_store.get_product_by_id(
            product_id
        )

        if existing is not None:
            raise ValueError(
                f"商品已存在: {product_id}"
            )

        # --------------------------------------------------------
        # 2. 写入 MySQL
        # --------------------------------------------------------

        self.product_store.insert_product(
            product
        )

        print(
            f"[ProductService] MySQL 插入成功: "
            f"{product_id}"
        )

        # --------------------------------------------------------
        # 3. 构造 Embedding 文本
        # --------------------------------------------------------

        embedding_text = self._build_embedding_text(
            product
        )

        print(
            f"[ProductService] 开始生成 Embedding: "
            f"{product_id}"
        )

        # --------------------------------------------------------
        # 4. 生成 Embedding
        # --------------------------------------------------------

        embedding = self.embedding_service.embed(
            embedding_text
        )

        print(
            f"[ProductService] Embedding 生成成功: "
            f"{product_id}"
        )

        # --------------------------------------------------------
        # 5. 写入 Milvus
        # --------------------------------------------------------

        self.vector_store.upsert(
            product_id=product_id,
            embedding=embedding,
        )

        print(
            f"[ProductService] Milvus 写入成功: "
            f"{product_id}"
        )

        # --------------------------------------------------------
        # 6. 返回数据库中的商品
        # --------------------------------------------------------

        result = self.product_store.get_product_by_id(
            product_id
        )

        return result or product

    # ============================================================
    # 修改商品
    # ============================================================

    def update_product(
        self,
        product_id: str,
        product: dict[str, Any],
    ) -> dict[str, Any]:
        """
        修改商品。

        根据修改字段决定是否重新生成 Embedding。

        例如：

            {"stock": 80}

        只修改库存：

            MySQL
              ↓
            完成

        不重新生成 Embedding。

        如果：

            {
                "name": "小米充电宝 Pro",
                "tags": ["快充", "便携"]
            }

        则：

            MySQL
              ↓
          Embedding
              ↓
           Milvus
        """

        # --------------------------------------------------------
        # 1. 查询旧商品
        # --------------------------------------------------------

        existing = self.product_store.get_product_by_id(
            product_id
        )

        if existing is None:
            raise ValueError(
                f"商品不存在: {product_id}"
            )

        # --------------------------------------------------------
        # 2. 判断哪些字段发生变化
        # --------------------------------------------------------

        semantic_fields = {
            "name",
            "category",
            "description",
            "brand",
            "tags",
        }

        need_reembedding = any(
            field in product
            for field in semantic_fields
        )

        # --------------------------------------------------------
        # 3. 更新 MySQL
        # --------------------------------------------------------

        self.product_store.update_product(
            product_id=product_id,
            product=product,
        )

        print(
            f"[ProductService] MySQL 更新成功: "
            f"{product_id}"
        )

        # --------------------------------------------------------
        # 4. 如果修改的是商品语义字段
        #    重新生成 Embedding
        # --------------------------------------------------------

        if need_reembedding:

            # ----------------------------------------------------
            # 获取更新后的完整商品
            # ----------------------------------------------------

            updated = (
                self.product_store.get_product_by_id(
                    product_id
                )
            )

            if updated is None:
                raise ValueError(
                    f"商品更新后查询失败: {product_id}"
                )

            # ----------------------------------------------------
            # 构造新的 Embedding 文本
            # ----------------------------------------------------

            embedding_text = (
                self._build_embedding_text(
                    updated
                )
            )

            print(
                "[ProductService] 商品语义字段发生变化，"
                "开始重新生成 Embedding:",
                product_id,
            )

            # ----------------------------------------------------
            # 生成新的 Embedding
            # ----------------------------------------------------

            embedding = (
                self.embedding_service.embed(
                    embedding_text
                )
            )

            print(
                "[ProductService] 新 Embedding 生成成功:",
                product_id,
            )

            # ----------------------------------------------------
            # Upsert 到 Milvus
            # ----------------------------------------------------

            self.vector_store.upsert(
                product_id=product_id,
                embedding=embedding,
            )

            print(
                "[ProductService] Milvus Embedding 更新成功:",
                product_id,
            )

        else:

            print(
                "[ProductService] 本次修改不涉及商品语义字段，"
                "无需更新 Embedding:",
                product_id,
            )

        # --------------------------------------------------------
        # 5. 返回最新商品
        # --------------------------------------------------------

        updated = (
            self.product_store.get_product_by_id(
                product_id
            )
        )

        if updated is None:
            raise ValueError(
                f"商品更新后查询失败: {product_id}"
            )

        return updated

    # ============================================================
    # 删除商品
    # ============================================================

    def delete_product(
        self,
        product_id: str,
    ) -> None:
        """
        删除商品。

        流程：

            MySQL 删除
                +
            Milvus 删除
        """

        # --------------------------------------------------------
        # 1. 检查商品是否存在
        # --------------------------------------------------------

        existing = self.product_store.get_product_by_id(
            product_id
        )

        if existing is None:
            raise ValueError(
                f"商品不存在: {product_id}"
            )

        # --------------------------------------------------------
        # 2. 删除 MySQL 商品
        # --------------------------------------------------------

        self.product_store.delete_product(
            product_id
        )

        print(
            f"[ProductService] MySQL 删除成功: "
            f"{product_id}"
        )

        # --------------------------------------------------------
        # 3. 删除 Milvus Embedding
        # --------------------------------------------------------

        self.vector_store.delete(
            product_id
        )

        print(
            f"[ProductService] Milvus 删除成功: "
            f"{product_id}"
        )

    # ============================================================
    # 查询单个商品
    # ============================================================

    def get_product(
        self,
        product_id: str,
    ) -> dict[str, Any] | None:
        """Get one product."""

        return self.product_store.get_product_by_id(
            product_id
        )

    # ============================================================
    # 查询全部商品
    # ============================================================

    def get_all_products(
        self,
    ) -> list[dict[str, Any]]:
        """Get all products."""

        return self.product_store.get_all_products()

    # ============================================================
    # 构造 Embedding 文本
    # ============================================================

    @staticmethod
    def _build_embedding_text(
        product: dict[str, Any],
    ) -> str:
        """
        构造商品 Embedding 文本。

        注意：

        price 和 stock 不放进 Embedding。

        因为：

        price / stock
            ↓
        实时业务属性

        name / category / description / brand / tags
            ↓
        商品语义属性

        推荐系统中：

            MySQL
            ├── 价格
            ├── 库存
            └── 商品基础信息

            Milvus
            └── 商品语义向量
        """

        tags = product.get(
            "tags",
            "",
        )

        if isinstance(tags, list):
            tags = ", ".join(
                str(tag)
                for tag in tags
            )

        return (
            f"商品名称：{product.get('name') or ''}；"
            f"商品类别：{product.get('category') or ''}；"
            f"商品描述：{product.get('description') or ''}；"
            f"品牌：{product.get('brand') or ''}；"
            f"商品标签：{tags}"
        )
