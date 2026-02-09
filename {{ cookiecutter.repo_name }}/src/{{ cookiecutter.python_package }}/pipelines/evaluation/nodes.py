"""Evaluation nodes with metrics from HBS assignment (MAP@K, NDCG, MRR)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def map_at_k(true_ids: list, predicted_ids: list, k: int = 10) -> float:
    """Calculate Mean Average Precision at K.

    This is the core metric from the HBS assignment.

    Args:
        true_ids: List of relevant product IDs
        predicted_ids: List of predicted product IDs (ranked)
        k: Cutoff for evaluation

    Returns:
        MAP@K score (0.0 to 1.0)
    """
    if not true_ids:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, p_id in enumerate(predicted_ids[:k]):
        if p_id in true_ids and p_id not in predicted_ids[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(true_ids), k)


def ndcg_at_k(true_ids: list, predicted_ids: list, k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        true_ids: List of relevant product IDs
        predicted_ids: List of predicted product IDs (ranked)
        k: Cutoff for evaluation

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if not true_ids:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, p_id in enumerate(predicted_ids[:k]):
        if p_id in true_ids:
            # Using binary relevance: rel = 1 for relevant, 0 otherwise
            dcg += 1.0 / np.log2(i + 2)  # +2 because position starts at 1

    # Calculate ideal DCG
    n_relevant = min(len(true_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr(true_ids: list, predicted_ids: list) -> float:
    """Calculate Mean Reciprocal Rank.

    Args:
        true_ids: List of relevant product IDs
        predicted_ids: List of predicted product IDs (ranked)

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not true_ids:
        return 0.0

    for i, p_id in enumerate(predicted_ids):
        if p_id in true_ids:
            return 1.0 / (i + 1)

    return 0.0


def recall_at_k(true_ids: list, predicted_ids: list, k: int = 10) -> float:
    """Calculate Recall at K.

    Args:
        true_ids: List of relevant product IDs
        predicted_ids: List of predicted product IDs (ranked)
        k: Cutoff for evaluation

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not true_ids:
        return 0.0

    hits = len(set(true_ids) & set(predicted_ids[:k]))
    return hits / len(true_ids)


def evaluate_predictions(
    predictions: dict,
    ground_truth: dict,
    k: int = 10,
) -> dict:
    """Evaluate predictions against ground truth.

    Args:
        predictions: Dict mapping query_id to list of predicted product_ids
        ground_truth: Dict mapping query_id to list of relevant product_ids
        k: Cutoff for evaluation

    Returns:
        Dictionary with aggregate metrics
    """
    map_scores = []
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for query_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(query_id, [])

        map_scores.append(map_at_k(true_ids, pred_ids, k))
        ndcg_scores.append(ndcg_at_k(true_ids, pred_ids, k))
        mrr_scores.append(mrr(true_ids, pred_ids))
        recall_scores.append(recall_at_k(true_ids, pred_ids, k))

    metrics = {
        f"map@{k}": float(np.mean(map_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        "mrr": float(np.mean(mrr_scores)),
        f"recall@{k}": float(np.mean(recall_scores)),
        "num_queries": len(map_scores),
    }

    return metrics


def compare_methods(
    tfidf_metrics: dict,
    embedding_metrics: dict,
    hybrid_metrics: dict,
) -> pd.DataFrame:
    """Compare all retrieval methods.

    Args:
        tfidf_metrics: Metrics from TF-IDF method
        embedding_metrics: Metrics from embedding method
        hybrid_metrics: Metrics from hybrid method

    Returns:
        Comparison DataFrame
    """
    comparison = pd.DataFrame(
        {
            "TF-IDF": tfidf_metrics,
            "Embeddings": embedding_metrics,
            "Hybrid": hybrid_metrics,
        }
    ).T

    # Calculate improvement over baseline (TF-IDF)
    baseline_map = tfidf_metrics.get("map@10", 0)
    if baseline_map > 0:
        comparison["improvement_vs_baseline"] = (
            (comparison["map@10"] - baseline_map) / baseline_map * 100
        ).round(2).astype(str) + "%"

    return comparison


def generate_report(
    comparison: pd.DataFrame,
    tfidf_metrics: dict,
    embedding_metrics: dict,
    hybrid_metrics: dict,
) -> dict[str, Any]:
    """Generate evaluation report.

    Args:
        comparison: Comparison DataFrame
        tfidf_metrics: Metrics from TF-IDF
        embedding_metrics: Metrics from embedding
        hybrid_metrics: Metrics from hybrid

    Returns:
        Report dictionary with summary and details
    """
    best_method = comparison["map@10"].idxmax()
    best_map = comparison.loc[best_method, "map@10"]

    report = {
        "summary": {
            "best_method": best_method,
            "best_map_10": best_map,
            "baseline_map_10": tfidf_metrics.get("map@10", 0),
            "improvement_achieved": (
                (best_map - tfidf_metrics.get("map@10", 0))
                / tfidf_metrics.get("map@10", 1)
                * 100
            ),
        },
        "detailed_metrics": {
            "tfidf": tfidf_metrics,
            "embeddings": embedding_metrics,
            "hybrid": hybrid_metrics,
        },
        "comparison_table": comparison.to_dict(),
    }

    logger.info("=" * 50)
    logger.info("RETRIEVAL EVALUATION REPORT")
    logger.info("=" * 50)
    logger.info(f"Best Method: {best_method}")
    logger.info(f"Best MAP@10: {best_map:.4f}")
    logger.info(f"Baseline MAP@10: {tfidf_metrics.get('map@10', 0):.4f}")
    logger.info(f"Improvement: {report['summary']['improvement_achieved']:.2f}%")
    logger.info("=" * 50)

    return report
