from __future__ import annotations

from typing import Any

from openai import OpenAI

from config import get_settings


class EmbeddingService:
    """
    Embedding Service

    使用 SiliconFlow BGE-M3。

    负责：

    1. Query Embedding
    2. Product Embedding
    3. Batch Embedding

    Query 和 Product 必须使用同一个 embedding model。
    """

    def __init__(self):

        settings = get_settings()

        self.client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )

        self.model = settings.embedding_model

    # ========================================================
    # 单文本 Embedding
    # ========================================================

    def embed(
        self,
        text: str,
    ) -> list[float]:

        text = (text or "").strip()

        if not text:
            raise ValueError(
                "Embedding 输入文本不能为空"
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        embedding = response.data[0].embedding

        if not embedding:
            raise ValueError(
                "Embedding API 返回空向量"
            )

        return embedding

    # ========================================================
    # Batch Embedding
    # ========================================================

    def embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        clean_texts = [
            (text or "").strip()
            for text in texts
        ]

        if any(
            not text
            for text in clean_texts
        ):
            raise ValueError(
                "Embedding batch 中存在空文本"
            )

        response = self.client.embeddings.create(
            model=self.model,
            input=clean_texts,
        )

        embeddings = [
            item.embedding
            for item in response.data
        ]

        if len(embeddings) != len(
            clean_texts
        ):
            raise ValueError(
                "Embedding 数量与输入文本数量不一致"
            )

        return embeddings

    # ========================================================
    # Product → Embedding Text
    # ========================================================

    @staticmethod
    def build_product_text(
        product: Any,
    ) -> str:
        """
        把商品结构转换成语义 Embedding 文本。

        注意：

        商品文本和 Query 文本必须在语义空间中具有
        尽可能一致的表达方式。
        """

        tags = getattr(
            product,
            "tags",
            [],
        ) or []

        if isinstance(tags, list):
            tags_text = " ".join(
                str(tag)
                for tag in tags
            )
        else:
            tags_text = str(tags)

        return (
            f"商品名称："
            f"{getattr(product, 'name', '')}\n"

            f"商品类别："
            f"{getattr(product, 'category', '')}\n"

            f"品牌："
            f"{getattr(product, 'brand', '')}\n"

            f"商品标签："
            f"{tags_text}\n"

            f"商品描述："
            f"{getattr(product, 'description', '')}"
        ).strip()