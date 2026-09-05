"""
Pure evaluation metrics — no external dependencies.

All functions are deterministic and testable without LLM / DB / Milvus.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# Retrieval Metrics
# ============================================================


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Recall@K = |retrieved[:K] ∩ relevant| / |relevant|

    Args:
        retrieved: ranked list of retrieved product IDs
        relevant: ground-truth relevant product IDs
        k: cutoff
    """
    if not relevant:
        return 1.0
    hit = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hit / len(relevant)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@K = |retrieved[:K] ∩ relevant| / K

    Args:
        retrieved: ranked list of retrieved product IDs
        relevant: ground-truth relevant product IDs
        k: cutoff
    """
    if k <= 0:
        return 0.0
    hit = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hit / k


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Mean Reciprocal Rank @K

    1 / rank_of_first_relevant_hit (0 if no hit within K)

    Args:
        retrieved: ranked list of retrieved product IDs
        relevant: ground-truth relevant product IDs
        k: cutoff
    """
    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevant: dict[str, int],
    k: int,
) -> float:
    """
    NDCG@K with graded relevance (0=irrelevant, 1=weak, 2=relevant, 3=high)

    NDCG = DCG / IDCG

    DCG = sum_{i=1}^{k} rel_i / log2(i + 1)
    IDCG = DCG of ideal ranking

    Args:
        retrieved: ranked list of retrieved product IDs
        relevant: dict mapping product_id -> relevance score (0-3)
        k: cutoff
    """
    if k <= 0:
        return 0.0

    # Convert list relevance to dict (binary relevance)
    if isinstance(relevant, list):
        relevant = {pid: 1 for pid in relevant}

    k = min(k, len(retrieved))

    # DCG
    dcg = 0.0
    for i, pid in enumerate(retrieved[:k], start=1):
        rel = relevant.get(pid, 0)
        dcg += rel / _log2(i + 1)

    # IDCG — ideal: sorted by relevance descending
    all_rels = sorted(relevant.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(all_rels[:k], start=1):
        idcg += rel / _log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _log2(x: float) -> float:
    import math
    return math.log2(x)


# ============================================================
# Latency Statistics
# ============================================================


def latency_stats(latencies: list[float]) -> dict[str, float]:
    """
    Compute latency statistics.

    Returns:
        {
            "avg": float,
            "min": float,
            "max": float,
            "p50": float,
            "p95": float,
            "count": int,
        }
    """
    if not latencies:
        return {
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "count": 0,
        }

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        k = (len(data) - 1) * p
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[f]
        return data[f] + (k - f) * (data[c] - data[f])

    return {
        "avg": round(sum(sorted_lat) / n, 2),
        "min": round(float(sorted_lat[0]), 2),
        "max": round(float(sorted_lat[-1]), 2),
        "p50": round(percentile(sorted_lat, 0.50), 2),
        "p95": round(percentile(sorted_lat, 0.95), 2),
        "count": n,
    }


# ============================================================
# Aggregate Metrics
# ============================================================


def aggregate_metrics(
    per_query: list[dict[str, Any]],
    k: int,
    use_graded: bool = False,
) -> dict[str, Any]:
    """
    Aggregate per-query metrics into summary stats.

    Args:
        per_query: list of {"query": str, "retrieved": [...], "relevant": [... or dict], ...}
        k: cutoff for @K metrics
        use_graded: if True, use relevance dict for NDCG; else binary

    Returns:
        Summary dict with recall@k, precision@k, mrr@k, ndcg@k, latency stats
    """
    recalls = []
    precisions = []
    mrrs = []
    ndcgs = []
    latencies = []

    for q in per_query:
        retrieved = q.get("retrieved", [])
        relevant = q.get("relevant", [])

        recalls.append(recall_at_k(retrieved, relevant, k))
        precisions.append(precision_at_k(retrieved, relevant, k))
        mrrs.append(mrr_at_k(retrieved, relevant, k))

        if use_graded and isinstance(relevant, dict):
            ndcgs.append(ndcg_at_k(retrieved, relevant, k))
        else:
            rel_list = relevant if isinstance(relevant, list) else []
            ndcgs.append(ndcg_at_k(retrieved, rel_list, k))

        lat = q.get("latency_ms", 0.0)
        if lat > 0:
            latencies.append(lat)

    n = len(per_query) or 1
    return {
        "sample_count": n,
        "recall_at_k": round(sum(recalls) / n, 4),
        "precision_at_k": round(sum(precisions) / n, 4),
        "mrr_at_k": round(sum(mrrs) / n, 4),
        "ndcg_at_k": round(sum(ndcgs) / n, 4),
        "latency": latency_stats(latencies),
    }
