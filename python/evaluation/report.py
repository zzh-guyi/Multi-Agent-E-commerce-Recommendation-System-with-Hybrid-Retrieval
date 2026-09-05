"""
Evaluation Report Generator

Supports:
1. Terminal table printing
2. JSON report saving
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from .evaluator import EvaluationResult


# ============================================================
# Terminal Table Printer
# ============================================================


def print_retrieval_table(results: list[EvaluationResult]) -> None:
    """Print a comparison table for retrieval strategies."""
    if not results:
        print("No results to display.")
        return

    # Header
    fmt = "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>12}"
    sep = "  " + "-" * 78
    print()
    print(sep)
    print(fmt.format("Strategy", "Recall@K", "Prec@K", "MRR@K", "NDCG@K", "Latency(ms)"))
    print(sep)

    for r in results:
        print(fmt.format(
            r.strategy,
            f"{r.recall_at_k:.4f}",
            f"{r.precision_at_k:.4f}",
            f"{r.mrr_at_k:.4f}",
            f"{r.ndcg_at_k:.4f}",
            f"{r.latency_avg:.1f}",
        ))

    print(sep)
    print(f"  Samples: {results[0].sample_count}, K: {results[0].k}")
    print()


def print_rerank_table(results: list[EvaluationResult]) -> None:
    """Print a comparison table for rerank strategies."""
    if not results:
        print("No results to display.")
        return

    fmt = "  {:<25} {:>10} {:>10} {:>10} {:>12}"
    sep = "  " + "-" * 70
    print()
    print(sep)
    print(fmt.format("Strategy", "Recall@K", "MRR@K", "NDCG@K", "Latency(ms)"))
    print(sep)

    for r in results:
        print(fmt.format(
            r.strategy,
            f"{r.recall_at_k:.4f}",
            f"{r.mrr_at_k:.4f}",
            f"{r.ndcg_at_k:.4f}",
            f"{r.latency_avg:.1f}",
        ))

    print(sep)
    print(f"  Samples: {results[0].sample_count}, K: {results[0].k}")
    print()


# ============================================================
# JSON Report Writer
# ============================================================


def save_report(
    results: list[EvaluationResult],
    report_type: str,
    output_dir: str = "evaluation_results",
) -> str:
    """
    Save evaluation results to JSON report.

    Returns the path to the saved report.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report_type}_report_{timestamp}.json"
    path = os.path.join(output_dir, filename)

    report: dict[str, Any] = {
        "report_type": report_type,
        "timestamp": datetime.now().isoformat(),
        "strategies": [],
    }

    for r in results:
        report["strategies"].append(r.to_dict())

    # Also save a summary
    summary: dict[str, Any] = {
        "report_type": report_type,
        "timestamp": datetime.now().isoformat(),
        "strategy_count": len(results),
        "strategies": [],
    }
    for r in results:
        summary["strategies"].append({
            "strategy": r.strategy,
            "k": r.k,
            "sample_count": r.sample_count,
            "recall_at_k": r.recall_at_k,
            "precision_at_k": r.precision_at_k,
            "mrr_at_k": r.mrr_at_k,
            "ndcg_at_k": r.ndcg_at_k,
            "latency_avg_ms": r.latency_avg,
            "latency_p50_ms": r.latency_p50,
            "latency_p95_ms": r.latency_p95,
        })

    summary_path = os.path.join(output_dir, f"{report_type}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved: {path}")
    print(f"Summary saved: {summary_path}")
    return path
