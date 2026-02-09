"""Vectorization pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_embeddings,
    create_faiss_index,
    create_tfidf_vectorizer,
    embed_queries,
    vectorize_queries,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the vectorization pipeline.

    This pipeline:
    1. Creates TF-IDF representations (baseline)
    2. Creates semantic embeddings (advanced)
    3. Builds FAISS index for fast search

    Returns:
        Pipeline object
    """
    return pipeline(
        [
            # TF-IDF baseline
            node(
                func=create_tfidf_vectorizer,
                inputs=[
                    "processed_products",
                    "params:tfidf_max_features",
                    "params:tfidf_ngram_range",
                ],
                outputs=["tfidf_vectorizer", "product_tfidf_matrix"],
                name="create_tfidf_vectorizer_node",
            ),
            node(
                func=vectorize_queries,
                inputs=["raw_queries", "tfidf_vectorizer"],
                outputs="query_tfidf_vectors",
                name="vectorize_queries_tfidf_node",
            ),
            # Semantic embeddings (optional)
            node(
                func=create_embeddings,
                inputs=[
                    "processed_products",
                    "params:embedding_model",
                    "params:embedding_batch_size",
                ],
                outputs="product_embeddings",
                name="create_product_embeddings_node",
            ),
            node(
                func=embed_queries,
                inputs=["raw_queries", "params:embedding_model"],
                outputs="query_embeddings",
                name="embed_queries_node",
            ),
            node(
                func=create_faiss_index,
                inputs="product_embeddings",
                outputs="faiss_index",
                name="create_faiss_index_node",
            ),
        ],
        namespace="vectorization",
        inputs=["processed_products", "raw_queries"],
        outputs=[
            "tfidf_vectorizer",
            "product_tfidf_matrix",
            "query_tfidf_vectors",
            "product_embeddings",
            "query_embeddings",
            "faiss_index",
        ],
    )
