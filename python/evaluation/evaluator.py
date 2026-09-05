
"""
Evaluation Engine

RetrievalEvaluator  — runs retrieval strategies against dataset
RerankEvaluator     — runs rerank strategies against dataset

Evaluation flow:

Retrieval:
    Query
      ↓
    RetrievalAdapter
      ↓
    Product IDs
      ↓
    Recall@K / Precision@K / MRR@K / NDCG@K / Latency

Rerank:
    Query
      ↓
    Hybrid Retrieval
      ↓
    Candidate Products
      ↓
    LLM Rerank
      ↓
    Top-K Products
      ↓
    Recall@K / Precision@K / MRR@K / NDCG@K / Latency
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from models.schemas import Product, UserProfile

from .adapters import RetrievalAdapter, RerankAdapter
from .metrics import aggregate_metrics


# ============================================================
# Result Data Classes
# ============================================================


@dataclass
class QueryResult:
    """Result for a single evaluation query."""

    query: str
    strategy: str

    retrieved: list[str] = field(default_factory=list)
    relevant: list[str] = field(default_factory=list)
    relevance: dict[str, int] = field(default_factory=dict)

    latency_ms: float = 0.0

    # Separate latency components.
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0

    error: str = ""


@dataclass
class EvaluationResult:
    """Aggregated result for one strategy across all queries."""

    strategy: str
    k: int
    use_graded: bool = False

    query_results: list[QueryResult] = field(default_factory=list)

    # Computed aggregates.
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr_at_k: float = 0.0
    ndcg_at_k: float = 0.0

    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0

    sample_count: int = 0

    def compute(self) -> "EvaluationResult":
        """Compute aggregate metrics from query results."""

        agg = aggregate_metrics(
            per_query=[
                {
                    "query": qr.query,
                    "retrieved": qr.retrieved,
                    "relevant": (
                        qr.relevance
                        if self.use_graded
                        else qr.relevant
                    ),
                    "latency_ms": qr.latency_ms,
                }
                for qr in self.query_results
            ],
            k=self.k,
            use_graded=self.use_graded,
        )

        self.recall_at_k = agg["recall_at_k"]
        self.precision_at_k = agg["precision_at_k"]
        self.mrr_at_k = agg["mrr_at_k"]
        self.ndcg_at_k = agg["ndcg_at_k"]

        self.sample_count = agg["sample_count"]

        self.latency_avg = agg["latency"]["avg"]
        self.latency_p50 = agg["latency"]["p50"]
        self.latency_p95 = agg["latency"]["p95"]

        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation result to a JSON-serializable dict."""

        return {
            "strategy": self.strategy,
            "k": self.k,
            "sample_count": self.sample_count,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr_at_k": self.mrr_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "latency_ms": {
                "avg": self.latency_avg,
                "p50": self.latency_p50,
                "p95": self.latency_p95,
            },
            "query_results": [
                {
                    "query": qr.query,
                    "retrieved": qr.retrieved,
                    "relevant": qr.relevant,
                    "relevance": qr.relevance,
                    "latency_ms": qr.latency_ms,
                    "retrieval_latency_ms": qr.retrieval_latency_ms,
                    "rerank_latency_ms": qr.rerank_latency_ms,
                    "error": qr.error,
                }
                for qr in self.query_results
            ],
        }


# ============================================================
# Helper Functions
# ============================================================


def _has_graded_relevance(dataset: list[dict[str, Any]]) -> bool:
    """
    Determine whether the evaluation dataset contains graded relevance.

    This checks the entire dataset instead of accidentally relying
    on the final query only.
    """

    for item in dataset:
        relevance = item.get("relevance", {})

        if isinstance(relevance, dict):
            for value in relevance.values():
                try:
                    if int(value) > 0:
                        return True
                except (TypeError, ValueError):
                    continue

    return False


def _build_evaluation_profile(query: str) -> UserProfile:
    """
    Build a deterministic UserProfile for offline rerank evaluation.

    The production Rerank path may use a real user profile. Offline
    evaluation has no real user, so we provide a stable synthetic
    profile instead of passing profile=None.

    The query is stored in real_time_tags so the evaluation context
    remains explicit and reproducible.
    """

    return UserProfile(
        user_id="evaluation_user",
        segments=[],
        preferred_categories=[],
        price_range=(0.0, 10000.0),
        recent_views=[],
        recent_purchases=[],
        rfm_score={},
        real_time_tags={
            "evaluation": True,
            "query": query,
        },
    )


# ============================================================
# Retrieval Evaluator
# ============================================================


class RetrievalEvaluator:
    """
    Evaluate retrieval strategies against a dataset.

    Usage:
        evaluator = RetrievalEvaluator(dataset, k=10)

        result = await evaluator.evaluate(keyword_adapter)
        result = await evaluator.evaluate(vector_adapter)
        result = await evaluator.evaluate(hybrid_adapter)
    """

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        k: int = 10,
    ):
        self._dataset = dataset
        self.k = k

    async def evaluate(
        self,
        adapter: RetrievalAdapter,
    ) -> EvaluationResult:
        """
        Run one retrieval strategy across the full dataset.
        """

        query_results: list[QueryResult] = []

        use_graded = _has_graded_relevance(self._dataset)

        for item in self._dataset:
            query = item["query"]

            relevant_ids = item.get(
                "relevant_product_ids",
                [],
            )

            relevance = item.get(
                "relevance",
                {},
            )

            start = time.perf_counter()

            try:
                retrieved = await adapter.search(
                    query,
                    limit=self.k,
                )

                error = ""

            except Exception as exc:
                retrieved = []
                error = f"{type(exc).__name__}: {exc}"

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            query_results.append(
                QueryResult(
                    query=query,
                    strategy=adapter.name,
                    retrieved=retrieved,
                    relevant=relevant_ids,
                    relevance=relevance,
                    latency_ms=round(latency_ms, 2),
                    retrieval_latency_ms=round(
                        latency_ms,
                        2,
                    ),
                    error=error,
                )
            )

        result = EvaluationResult(
            strategy=adapter.name,
            k=self.k,
            use_graded=use_graded,
            query_results=query_results,
        )

        result.compute()

        return result


# ============================================================
# Rerank Evaluator
# ============================================================


class RerankEvaluator:
    """
    Evaluate reranking strategies against a dataset.

    Flow:

        Query
          ↓
        Hybrid Retrieval
          ↓
        Top-N Candidates
          ↓
        Rerank Adapter
          ↓
        Top-K Results
          ↓
        Evaluation Metrics
    """

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        recall_k: int = 10,
        rerank_k: int = 5,
    ):
        self._dataset = dataset
        self.recall_k = recall_k
        self.rerank_k = rerank_k

    async def evaluate(
        self,
        rerank_adapter: RerankAdapter,
        retrieval_adapter_name: str = "hybrid",
    ) -> EvaluationResult:
        """
        Run one rerank strategy across the full dataset.

        retrieval_adapter_name is used to look up the appropriate
        RetrievalAdapter from the ADAPTERS registry.
        """

        from .run_evaluation import ADAPTERS

        retrieval_adapter_cls = ADAPTERS.get(
            retrieval_adapter_name
        )

        if retrieval_adapter_cls is None:
            raise ValueError(
                f"Unknown retrieval adapter: "
                f"{retrieval_adapter_name}. "
                f"Available: {list(ADAPTERS.keys())}"
            )

        retrieval_adapter = retrieval_adapter_cls()

        if retrieval_adapter is None:
            raise ValueError(
                f"Unknown retrieval adapter: "
                f"{retrieval_adapter_name}. "
                f"Available: {list(ADAPTERS.keys())}"
            )

        query_results: list[QueryResult] = []

        use_graded = _has_graded_relevance(self._dataset)

        # Create one deterministic offline profile.
        #
        # We still update real_time_tags with the current query
        # inside the loop so each query has an explicit context.
        base_profile = _build_evaluation_profile("")

        for item in self._dataset:
            query = item["query"]

            relevant_ids = item.get(
                "relevant_product_ids",
                [],
            )

            relevance = item.get(
                "relevance",
                {},
            )

            # ----------------------------------------------------
            # Step 1: Retrieve candidates
            # ----------------------------------------------------

            recall_start = time.perf_counter()
            retrieval_error = ""

            try:
                candidate_ids = await retrieval_adapter.search(
                    query,
                    limit=self.recall_k,
                )

            except Exception as exc:
                candidate_ids = []
                retrieval_error = (
                    f"retrieval "
                    f"{type(exc).__name__}: {exc}"
                )

            recall_latency = (
                time.perf_counter() - recall_start
            ) * 1000

            # ----------------------------------------------------
            # Step 2: Load Product objects
            # ----------------------------------------------------

            candidates = self._load_products(
                candidate_ids
            )

            # ----------------------------------------------------
            # Step 3: Rerank
            # ----------------------------------------------------

            rerank_start = time.perf_counter()
            rerank_error = ""

            # Build a fresh profile for this query.
            #
            # IMPORTANT:
            # Do not pass profile=None.
            # ProductRecAgent._rerank() may treat a missing profile
            # as a condition for bypassing the LLM rerank path.
            profile = UserProfile(
                user_id=base_profile.user_id,
                segments=list(base_profile.segments),
                preferred_categories=list(
                    base_profile.preferred_categories
                ),
                price_range=base_profile.price_range,
                recent_views=list(
                    base_profile.recent_views
                ),
                recent_purchases=list(
                    base_profile.recent_purchases
                ),
                rfm_score=dict(
                    base_profile.rfm_score
                ),
                real_time_tags={
                    "evaluation": True,
                    "query": query,
                },
            )

            try:
                reranked_ids = await rerank_adapter.rerank(
                    query=query,
                    candidates=candidates,
                    num_items=self.rerank_k,
                    profile=profile,
                )

            except Exception as exc:
                # If rerank fails, fall back to the retrieval order.
                reranked_ids = candidate_ids[
                    : self.rerank_k
                ]

                rerank_error = (
                    f"rerank "
                    f"{type(exc).__name__}: {exc}"
                )

            rerank_latency = (
                time.perf_counter() - rerank_start
            ) * 1000

            total_latency = (
                recall_latency + rerank_latency
            )

            errors = [
                error
                for error in (
                    retrieval_error,
                    rerank_error,
                )
                if error
            ]

            query_results.append(
                QueryResult(
                    query=query,
                    strategy=rerank_adapter.name,
                    retrieved=reranked_ids,
                    relevant=relevant_ids,
                    relevance=relevance,
                    latency_ms=round(
                        total_latency,
                        2,
                    ),
                    retrieval_latency_ms=round(
                        recall_latency,
                        2,
                    ),
                    rerank_latency_ms=round(
                        rerank_latency,
                        2,
                    ),
                    error="; ".join(errors),
                )
            )

        result = EvaluationResult(
            strategy=rerank_adapter.name,
            k=self.rerank_k,
            use_graded=use_graded,
            query_results=query_results,
        )

        result.compute()

        return result

    # ------------------------------------------------------------
    # Product Loading
    # ------------------------------------------------------------

    def _load_products(
            self,
            product_ids: list[str],
    ) -> list[Product]:
        """Load Product objects for candidate IDs."""

        from services.product_store import ProductStore

        store = ProductStore()
        rows = store.get_products_by_ids(product_ids)

        id_to_product: dict[str, Product] = {}

        for row in rows:
            pid = row.get("product_id", "")

            if not pid:
                continue

            tags = row.get("tags", "")

            if isinstance(tags, str):
                tags = [
                    tag.strip()
                    for tag in tags.split(",")
                    if tag.strip()
                ]
            elif not isinstance(tags, list):
                tags = []

            id_to_product[pid] = Product(
                product_id=pid,
                name=row.get("name", ""),
                category=row.get("category", ""),
                price=float(row.get("price", 0) or 0),
                description=row.get("description", ""),
                brand=row.get("brand", ""),
                seller_id=row.get("seller_id", ""),
                stock=int(row.get("stock", 0) or 0),
                tags=tags,
                image_url=row.get("image_url", ""),
            )

        return [
            id_to_product[pid]
            for pid in product_ids
            if pid in id_to_product
        ]