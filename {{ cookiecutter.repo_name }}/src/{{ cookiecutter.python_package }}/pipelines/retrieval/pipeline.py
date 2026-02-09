"""Retrieval pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    format_results_for_evaluation,
    hybrid_search,
    search_embeddings,
    search_tfidf,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the retrieval pipeline.

    This pipeline:
    1. Performs TF-IDF based search (baseline)
    2. Performs embedding-based search (advanced)
    3. Combines results with hybrid search

    Returns:
        Pipeline object
    """
    return pipeline(
        [
            # TF-IDF search
            node(
                func=search_tfidf,
                inputs=[
                    "query_tfidf_vectors",
                    "product_tfidf_matrix",
                    "processed_products",
                    "params:top_k",
                ],
                outputs="tfidf_results",
                name="search_tfidf_node",
            ),
            # Embedding search
            node(
                func=search_embeddings,
                inputs=[
                    "query_embeddings",
                    "faiss_index",
                    "processed_products",
                    "params:top_k",
                ],
                outputs="embedding_results",
                name="search_embeddings_node",
            ),
            # Hybrid search
            node(
                func=hybrid_search,
                inputs=[
                    "tfidf_results",
                    "embedding_results",
                    "params:hybrid_alpha",
                    "params:top_k",
                ],
                outputs="hybrid_results",
                name="hybrid_search_node",
            ),
            # Format results
            node(
                func=format_results_for_evaluation,
                inputs=["tfidf_results", "raw_queries"],
                outputs="tfidf_predictions",
                name="format_tfidf_results_node",
            ),
            node(
                func=format_results_for_evaluation,
                inputs=["embedding_results", "raw_queries"],
                outputs="embedding_predictions",
                name="format_embedding_results_node",
            ),
            node(
                func=format_results_for_evaluation,
                inputs=["hybrid_results", "raw_queries"],
                outputs="hybrid_predictions",
                name="format_hybrid_results_node",
            ),
        ],
        namespace="retrieval",
        inputs=[
            "query_tfidf_vectors",
            "product_tfidf_matrix",
            "query_embeddings",
            "faiss_index",
            "processed_products",
            "raw_queries",
        ],
        outputs=[
            "tfidf_predictions",
            "embedding_predictions",
            "hybrid_predictions",
        ],
    )
