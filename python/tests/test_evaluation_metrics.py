"""
Unit tests for evaluation metrics.

These tests are pure functions — no LLM, no database, no Milvus.
Run with: pytest python/tests/test_evaluation_metrics.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    latency_stats,
)


# ============================================================
# Recall@K
# ============================================================


def test_recall_at_kperfect_hit():
    retrieved = ["P001", "P002", "P003"]
    relevant = ["P001", "P002"]
    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_recall_at_k_partial():
    retrieved = ["P001", "P004", "P005"]
    relevant = ["P001", "P002", "P003"]
    assert recall_at_k(retrieved, relevant, 3) == 1 / 3


def test_recall_at_k_no_hit():
    retrieved = ["P004", "P005", "P006"]
    relevant = ["P001", "P002"]
    assert recall_at_k(retrieved, relevant, 3) == 0.0


def test_recall_at_k_empty_relevant():
    retrieved = ["P001"]
    relevant = []
    assert recall_at_k(retrieved, relevant, 1) == 1.0


def test_recall_at_k_k_greater_than_retrieved():
    retrieved = ["P001"]
    relevant = ["P001", "P002"]
    assert recall_at_k(retrieved, relevant, 5) == 0.5


# ============================================================
# Precision@K
# ============================================================


def test_precision_at_kperfect():
    retrieved = ["P001", "P002"]
    relevant = ["P001", "P002"]
    assert precision_at_k(retrieved, relevant, 2) == 1.0


def test_precision_at_k_partial():
    retrieved = ["P001", "P004"]
    relevant = ["P001", "P002"]
    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_precision_at_k_no_hit():
    retrieved = ["P004", "P005"]
    relevant = ["P001"]
    assert precision_at_k(retrieved, relevant, 2) == 0.0


# ============================================================
# MRR@K
# ============================================================


def test_mrr_first_hit():
    retrieved = ["P001", "P002", "P003"]
    relevant = ["P001"]
    assert mrr_at_k(retrieved, relevant, 3) == 1.0


def test_mrr_second_hit():
    retrieved = ["P004", "P001", "P002"]
    relevant = ["P001"]
    assert mrr_at_k(retrieved, relevant, 3) == 0.5


def test_mrr_no_hit():
    retrieved = ["P004", "P005", "P006"]
    relevant = ["P001"]
    assert mrr_at_k(retrieved, relevant, 3) == 0.0


def test_mrr_hit_after_k():
    retrieved = ["P004", "P005", "P006"]
    relevant = ["P001"]
    assert mrr_at_k(retrieved, relevant, 2) == 0.0


# ============================================================
# NDCG@K (graded relevance)
# ============================================================


def test_ndcg_perfect():
    retrieved = ["P001", "P002", "P003"]
    relevant = {"P001": 3, "P002": 2, "P003": 1}
    result = ndcg_at_k(retrieved, relevant, 3)
    assert abs(result - 1.0) < 1e-9


def test_ndcg_worst():
    retrieved = ["P003", "P002", "P001"]
    relevant = {"P001": 3, "P002": 2, "P003": 1}
    result = ndcg_at_k(retrieved, relevant, 3)
    assert result < 0.5


def test_ndcg_no_relevance():
    retrieved = ["P001", "P002"]
    relevant = {}
    assert ndcg_at_k(retrieved, relevant, 2) == 0.0


def test_ndcg_partial_hit():
    retrieved = ["P001", "P004", "P005"]
    relevant = {"P001": 3, "P002": 2}
    result = ndcg_at_k(retrieved, relevant, 3)
    assert 0 < result < 1.0


# ============================================================
# Latency Stats
# ============================================================


def test_latency_stats_empty():
    result = latency_stats([])
    assert result["count"] == 0
    assert result["avg"] == 0.0


def test_latency_stats_single():
    result = latency_stats([100.0])
    assert result["avg"] == 100.0
    assert result["min"] == 100.0
    assert result["max"] == 100.0
    assert result["p50"] == 100.0
    assert result["p95"] == 100.0


def test_latency_stats_multiple():
    latencies = [10, 20, 30, 40, 50]
    result = latency_stats(latencies)
    assert result["avg"] == 30.0
    assert result["min"] == 10.0
    assert result["max"] == 50.0
    assert result["p50"] == 30.0


# ============================================================
# Integration: full pipeline with mock data
# ============================================================


def test_full_pipeline():
    """Test the full metric pipeline with known data."""
    retrieved = ["P001", "P003", "P002", "P004", "P005"]
    relevant = ["P001", "P002", "P003"]
    relevance_graded = {"P001": 3, "P002": 2, "P003": 1}

    assert recall_at_k(retrieved, relevant, 5) == 1.0
    assert precision_at_k(retrieved, relevant, 5) == 3 / 5
    assert mrr_at_k(retrieved, relevant, 5) == 1.0  # P001 is rank 1
    assert ndcg_at_k(retrieved, relevance_graded, 5) > 0.8


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
