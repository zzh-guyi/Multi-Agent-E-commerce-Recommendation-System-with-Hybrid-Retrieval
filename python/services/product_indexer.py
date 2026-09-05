from __future__ import annotations

from services.product_store import ProductStore
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


class ProductIndexer:
    """Build Milvus vector index for products."""

    def __init__(self):
        self.product_store = ProductStore()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def index_all_products(self) -> None:
        """Index all MySQL products into Milvus."""

        # ============================================================
        # 1. 从 MySQL 获取全部商品
        # ============================================================

        products = self.product_store.get_all_products()

        if not products:
            print("MySQL 中没有商品，无法建立 Milvus 索引")
            return

        print(f"MySQL 商品数量：{len(products)}")

        # ============================================================
        # 2. 构造 Embedding 文本
        # ============================================================

        texts = []

        for product in products:

            tags = product.get("tags") or ""

            if isinstance(tags, list):
                tags = ", ".join(
                    str(tag)
                    for tag in tags
                )

            product_text = (
                f"商品名称：{product.get('name') or ''}；"
                f"商品类别：{product.get('category') or ''}；"
                f"商品描述：{product.get('description') or ''}；"
                f"品牌：{product.get('brand') or ''}；"
                f"商品标签：{tags}"
            )

            texts.append(product_text)

        # ============================================================
        # 3. 生成 Embedding
        # ============================================================

        print(
            f"准备为 {len(texts)} 个商品生成 Embedding..."
        )

        embeddings = self.embedding_service.embed_batch(
            texts
        )

        if not embeddings:
            print("Embedding 生成失败，没有得到向量")
            return

        print(
            f"Embedding 数量：{len(embeddings)}"
        )

        print(
            f"Embedding Dim：{len(embeddings[0])}"
        )

        # ============================================================
        # 4. 商品 ID
        # ============================================================

        product_ids = [
            str(product["product_id"])
            for product in products
        ]

        # ============================================================
        # 5. 创建 Milvus Collection
        #
        # VectorStore 内部负责：
        #
        # product_embeddings
        #
        # collection schema
        # ============================================================

        self.vector_store.create_collection(
            dim=len(embeddings[0])
        )

        # ============================================================
        # 6. 插入向量
        # ============================================================

        self.vector_store.insert(
            product_ids=product_ids,
            embeddings=embeddings,
        )

        # ============================================================
        # 7. 完成
        # ============================================================

        print("=" * 60)
        print("Milvus 索引建立完成")
        print(f"商品数量：{len(product_ids)}")
        print(f"Embedding Dim：{len(embeddings[0])}")
        print("=" * 60)


def main():
    indexer = ProductIndexer()
    indexer.index_all_products()


if __name__ == "__main__":
    main()