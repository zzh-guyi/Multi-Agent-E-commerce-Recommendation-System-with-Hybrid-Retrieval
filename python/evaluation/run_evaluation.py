"""
CLI entry point for Evaluation.

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --k 10
    python -m evaluation.run_evaluation --dataset evaluation/data/queries.json
    python -m evaluation.run_evaluation --retrieval-only
    python -m evaluation.run_evaluation --rerank-only
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.dataset import load_dataset
from evaluation.adapters import (
    KeywordAdapter,
    VectorAdapter,
    HybridAdapter,
    LlmRerankAdapter,
    NoRerankAdapter,
)
from evaluation.evaluator import RetrievalEvaluator, RerankEvaluator
from evaluation.report import (
    print_retrieval_table,
    print_rerank_table,
    save_report,
)


# ============================================================
# Adapter Registry
# ============================================================

ADAPTERS = {
    "keyword": KeywordAdapter,
    "vector": VectorAdapter,
    "hybrid": HybridAdapter,
}


def build_retrieval_adapters(selected: list[str] | None) -> list:
    """Build adapter instances for selected strategies."""
    if selected is None:
        selected = list(ADAPTERS.keys())
    adapters = []
    for name in selected:
        if name in ADAPTERS:
            adapters.append(ADAPTERS[name]())
        else:
            print(f"Warning: unknown retrieval strategy '{name}', skipping")
    return adapters


def build_rerank_adapters() -> list:
    """Build rerank adapter instances."""
    return [LlmRerankAdapter(), NoRerankAdapter()]


# ============================================================
# Main Evaluation
# ============================================================


async def run_retrieval_eval(
    dataset: list[dict],
    k: int,
    strategies: list[str] | None,
) -> list:
    """Run retrieval evaluation and return results."""
    adapters = build_retrieval_adapters(strategies)
    if not adapters:
        print("No retrieval adapters configured.")
        return []

    evaluator = RetrievalEvaluator(dataset, k=k)
    results = []

    print(f"\n{'='*60}")
    print(f"Retrieval Evaluation  (K={k})")
    print(f"{'='*60}")

    for adapter in adapters:
        print(f"\n  Running: {adapter.name} ...")
        t0 = time.perf_counter()
        result = await evaluator.evaluate(adapter)
        elapsed = time.perf_counter() - t0
        print(f"  {adapter.name}: Recall@{k}={result.recall_at_k:.4f}, "
              f"Precision@{k}={result.precision_at_k:.4f}, "
              f"MRR@{k}={result.mrr_at_k:.4f}, "
              f"NDCG@{k}={result.ndcg_at_k:.4f}, "
              f"AvgLatency={result.latency_avg:.1f}ms, "
              f"Time={elapsed:.1f}s")
        results.append(result)

    return results


async def run_rerank_eval(
    dataset: list[dict],
    recall_k: int,
    rerank_k: int,
) -> list:
    """Run rerank evaluation and return results."""
    rerank_adapters = build_rerank_adapters()
    evaluator = RerankEvaluator(
        dataset,
        recall_k=recall_k,
        rerank_k=rerank_k,
    )
    results = []

    print(f"\n{'='*60}")
    print(f"Rerank Evaluation  (Recall@{recall_k} -> Rerank@{rerank_k})")
    print(f"{'='*60}")

    for adapter in rerank_adapters:
        print(f"\n  Running: {adapter.name} ...")
        t0 = time.perf_counter()
        try:
            result = await evaluator.evaluate(adapter)
        except Exception as e:
            print(f"  {adapter.name}: ERROR -- {e}")
            continue
        elapsed = time.perf_counter() - t0
        print(f"  {adapter.name}: Recall@{rerank_k}={result.recall_at_k:.4f}, "
              f"MRR@{rerank_k}={result.mrr_at_k:.4f}, "
              f"NDCG@{rerank_k}={result.ndcg_at_k:.4f}, "
              f"AvgLatency={result.latency_avg:.1f}ms, "
              f"Time={elapsed:.1f}s")
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent E-Commerce Retrieval & Rerank Evaluator"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="K for retrieval @K metrics (default: 10)",
    )
    parser.add_argument(
        "--rerank-k", type=int, default=5,
        help="K for rerank output (default: 5)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to queries.json (default: evaluation/data/queries.json)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only run retrieval evaluation",
    )
    parser.add_argument(
        "--rerank-only", action="store_true",
        help="Only run rerank evaluation",
    )
    parser.add_argument(
        "--strategies", type=str, nargs="+", default=None,
        help="Retrieval strategies to evaluate (default: keyword vector hybrid)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    # Load dataset
    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = str(Path(__file__).parent / "data" / "queries.json")

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} queries")

    # Check ground truth
    has_ground_truth = sum(
        1 for q in dataset if q.get("relevant_product_ids")
    )
    print(f"Queries with ground truth: {has_ground_truth}/{len(dataset)}")

    if has_ground_truth == 0:
        print("\nWARNING: No ground truth available!")
        print("Please populate relevant_product_ids in queries.json")
        print("Run: python -m evaluation.dataset --populate")

    # Run evaluations
    results = []

    if not args.rerank_only:
        retrieval_results = asyncio.run(run_retrieval_eval(
            dataset, args.k, args.strategies
        ))
        results.extend(retrieval_results)
        print_retrieval_table(retrieval_results)
        save_report(retrieval_results, "retrieval", args.output_dir)

    if not args.retrieval_only:
        rerank_results = asyncio.run(run_rerank_eval(
            dataset, args.k, args.rerank_k
        ))
        results.extend(rerank_results)
        print_rerank_table(rerank_results)
        save_report(rerank_results, "rerank", args.output_dir)

    if not results:
        print("\nNo evaluation results generated.")
        sys.exit(1)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
