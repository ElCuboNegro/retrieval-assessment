"""Retrieval nodes for searching products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


def search_tfidf(
    query_vectors: "csr_matrix",
    product_matrix: "csr_matrix",
    products: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Search products using TF-IDF cosine similarity.

    Args:
        query_vectors: TF-IDF vectors for queries
        product_matrix: TF-IDF matrix for products
        products: Products DataFrame with product_id
        k: Number of results to return per query

    Returns:
        Dictionary mapping query index to list of (product_id, score) tuples
    """
    # Compute cosine similarity
    similarities = cosine_similarity(query_vectors, product_matrix)

    results = {}
    product_ids = products["product_id"].values

    for query_idx in range(similarities.shape[0]):
        scores = similarities[query_idx]

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]

        # Map to product IDs
        results[query_idx] = [
            (int(product_ids[idx]), float(score))
            for idx, score in zip(top_k_indices, top_k_scores)
        ]

    return results


def search_embeddings(
    query_embeddings: np.ndarray,
    faiss_index: Any,
    products: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Search products using FAISS semantic search.

    Args:
        query_embeddings: Query embeddings
        faiss_index: FAISS index
        products: Products DataFrame with product_id
        k: Number of results to return per query

    Returns:
        Dictionary mapping query index to list of (product_id, score) tuples
    """
    try:
        import faiss
    except ImportError as e:
        msg = "faiss-cpu not installed. Run: pip install faiss-cpu"
        raise ImportError(msg) from e

    # Normalize query embeddings
    query_embeddings = query_embeddings.astype(np.float32)
    faiss.normalize_L2(query_embeddings)

    # Search
    scores, indices = faiss_index.search(query_embeddings, k)

    results = {}
    product_ids = products["product_id"].values

    for query_idx in range(query_embeddings.shape[0]):
        results[query_idx] = [
            (int(product_ids[idx]), float(score))
            for idx, score in zip(indices[query_idx], scores[query_idx])
            if idx >= 0  # FAISS returns -1 for empty results
        ]

    return results


def hybrid_search(
    tfidf_results: dict,
    embedding_results: dict,
    alpha: float = 0.5,
    k: int = 10,
) -> dict:
    """Combine TF-IDF and embedding results with fusion.

    Uses Reciprocal Rank Fusion (RRF) to combine results.

    Args:
        tfidf_results: Results from TF-IDF search
        embedding_results: Results from embedding search
        alpha: Weight for TF-IDF (1-alpha for embeddings)
        k: Number of results to return per query

    Returns:
        Dictionary mapping query index to list of (product_id, score) tuples
    """
    # RRF constant
    rrf_k = 60

    results = {}

    query_indices = set(tfidf_results.keys()) | set(embedding_results.keys())

    for query_idx in query_indices:
        # Get rankings from both methods
        tfidf_ranking = {
            pid: rank + 1
            for rank, (pid, _) in enumerate(tfidf_results.get(query_idx, []))
        }
        embed_ranking = {
            pid: rank + 1
            for rank, (pid, _) in enumerate(embedding_results.get(query_idx, []))
        }

        # Combine all product IDs
        all_pids = set(tfidf_ranking.keys()) | set(embed_ranking.keys())

        # Calculate RRF scores
        rrf_scores = {}
        for pid in all_pids:
            tfidf_rank = tfidf_ranking.get(pid, 1000)  # Default high rank
            embed_rank = embed_ranking.get(pid, 1000)

            # Weighted RRF
            rrf_scores[pid] = alpha * (1 / (rrf_k + tfidf_rank)) + (1 - alpha) * (
                1 / (rrf_k + embed_rank)
            )

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        results[query_idx] = [(pid, score) for pid, score in sorted_results]

    return results


def format_results_for_evaluation(
    results: dict,
    queries: pd.DataFrame,
) -> dict:
    """Format results for evaluation, mapping to query_ids.

    Args:
        results: Results dict with query index as key
        queries: Queries DataFrame

    Returns:
        Dictionary mapping query_id to list of product_ids
    """
    formatted = {}
    query_ids = queries["query_id"].values

    for query_idx, result_list in results.items():
        query_id = int(query_ids[query_idx])
        formatted[query_id] = [pid for pid, _ in result_list]

    return formatted
