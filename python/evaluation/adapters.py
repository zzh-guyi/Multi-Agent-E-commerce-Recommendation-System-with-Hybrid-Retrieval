""" Retrieval and Rerank adapters for Evaluation.
Adapters wrap the existing production code so that Evaluation can call the same retrieval logic without duplicating it.

Design:
RetrievalAdapter — abstract interface for retrieval strategies
KeywordAdapter — MySQL keyword search only
VectorAdapter — Milvus vector search only
HybridAdapter — existing Hybrid Retrieval + RRF (production code)
RerankAdapter — LLM Rerank (production code)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from models.schemas import Product, UserProfile


# ============================================================
# Base Adapter
# ============================================================

class RetrievalAdapter(ABC):
    """Interface for retrieval strategies."""

    @abstractmethod
    async def search(
            self,
            query: str,
            limit: int,
    ) -> list[str]:
        """Return ranked product IDs."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class RerankAdapter(ABC):
    """Interface for reranking strategies."""

    @abstractmethod
    async def rerank(
            self,
            query: str,
            candidates: list[Product],
            num_items: int,
            profile: UserProfile | None = None,
    ) -> list[str]:
        """Return reranked product IDs."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ============================================================
# Keyword Adapter
# ============================================================

class KeywordAdapter(RetrievalAdapter):
    """
    MySQL keyword search only.
    Wraps ProductStore.keyword_search_products.
    """

    def __init__(self):
        from services.product_store import ProductStore
        self._store = ProductStore()

    @property
    def name(self) -> str:
        return "keyword"

    async def search(
            self,
            query: str,
            limit: int,
    ) -> list[str]:
        rows = self._store.keyword_search_products(
            query=query,
            limit=limit,
        )
        return [
            row["product_id"]
            for row in rows
            if row.get("product_id")
               and row["product_id"] != "TEST001"
        ]


# ============================================================
# Vector Adapter
# ============================================================

class VectorAdapter(RetrievalAdapter):
    """
    Milvus vector search only.
    Wraps EmbeddingService + VectorStore.
    """

    def __init__(self):
        from services.embedding_service import EmbeddingService
        from services.vector_store import VectorStore
        self._embedding = EmbeddingService()
        self._vector_store = VectorStore()

    @property
    def name(self) -> str:
        return "vector"

    async def search(
            self,
            query: str,
            limit: int,
    ) -> list[str]:
        try:
            embedding = self._embedding.embed(query)
        except Exception:
            return []

        hits = self._vector_store.search(
            query_embedding=embedding,
            limit=limit,
        )

        results: list[str] = []
        for hit in hits:
            entity = hit.get("entity", {})
            pid = entity.get("product_id")
            if pid and pid != "TEST001":
                results.append(pid)

        return results


# ============================================================
# Hybrid Adapter
# ============================================================

class HybridAdapter(RetrievalAdapter):
    """
    Evaluation 使用的 Hybrid Retrieval Adapter。

    真实调用链：
        Evaluation Query
            ↓
        ProductRecAgent._recall(
            query_text=query
        )
            ↓
        MySQL 硬过滤
            ↓
        Embedding
            ↓
        Milvus Vector Search + Keyword Search
            ↓
        RRF Fusion
            ↓
        Product IDs

    注意：
        - 不修改正式业务调用逻辑
        - query_text 只作为 Evaluation 的显式 Query 注入
        - TEST001 为测试商品，不参与正式 Evaluation
    """

    EXCLUDED_PRODUCT_IDS = {"TEST001"}

    def __init__(self):
        from agents.product_rec_agent import ProductRecAgent
        self._agent = ProductRecAgent(
            enable_tool_calling=False
        )

    @property
    def name(self) -> str:
        return "hybrid"

    async def search(
            self,
            query: str,
            limit: int,
    ) -> list[str]:
        if not query or not query.strip():
            return []

        query = query.strip()
        print(
            "[Evaluation][HybridAdapter]"
            " query=",
            query,
        )

        profile = UserProfile(
            user_id="evaluation",
            preferred_categories=[],
            price_range=(0.0, 99999.0),
        )

        result = await self._agent._recall(
            user_profile=profile,
            limit=limit,
            query_text=query,
            excluded_product_ids=self.EXCLUDED_PRODUCT_IDS,
        )

        products = result.get(
            "products",
            [],
        )

        product_ids = []
        for product in products:
            product_id = product.product_id
            if product_id in self.EXCLUDED_PRODUCT_IDS:
                continue
            if product_id in product_ids:
                continue
            product_ids.append(
                product_id
            )
            if len(product_ids) >= limit:
                break

        print(
            "[Evaluation][HybridAdapter]"
            " result=",
            product_ids,
        )
        print(
            "[Evaluation][HybridAdapter]"
            " query_used=",
            result.get("query", ""),
        )
        print(
            "[Evaluation][HybridAdapter]"
            " vector_results=",
            result.get("vector_results", []),
        )
        print(
            "[Evaluation][HybridAdapter]"
            " keyword_results=",
            result.get("keyword_results", []),
        )
        print(
            "[Evaluation][HybridAdapter]"
            " rrf_results=",
            result.get("rrf_results", []),
        )

        return product_ids


# ============================================================
# Rerank Adapter
# ============================================================

class LlmRerankAdapter(RerankAdapter):
    """
    LLM Rerank adapter.
    Reuses ProductRecAgent._rerank() directly.
    """

    def __init__(self):
        from agents.product_rec_agent import ProductRecAgent
        self._agent = ProductRecAgent(
            enable_tool_calling=False
        )

    @property
    def name(self) -> str:
        return "llm_rerank"

    async def rerank(
            self,
            query: str,
            candidates: list[Product],
            num_items: int,
            profile: UserProfile | None = None,
    ) -> list[str]:
        return await self._agent._rerank(
            profile=profile,
            candidates=candidates,
            num_items=num_items,
        )


# ============================================================
# No-Rerank Baseline
# ============================================================

class NoRerankAdapter(RerankAdapter):
    """
    Baseline: return candidates in original retrieval order.
    """

    @property
    def name(self) -> str:
        return "no_rerank"

    async def rerank(
            self,
            query: str,
            candidates: list[Product],
            num_items: int,
            profile: UserProfile | None = None,
    ) -> list[str]:
        return [
            product.product_id
            for product in candidates[:num_items]
            if product.product_id
               and product.product_id != "TEST001"
        ]