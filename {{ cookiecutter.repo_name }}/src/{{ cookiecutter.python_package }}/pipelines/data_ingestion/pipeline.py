"""Data ingestion pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_relevance_labels,
    group_labels_by_query,
    load_labels,
    load_products,
    load_queries,
    preprocess_products,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data ingestion pipeline.

    This pipeline:
    1. Loads raw WANDS data (products, queries, labels)
    2. Preprocesses products for vectorization
    3. Creates relevance labels for evaluation

    Returns:
        Pipeline object
    """
    return pipeline(
        [
            node(
                func=load_products,
                inputs="params:products_path",
                outputs="raw_products",
                name="load_products_node",
            ),
            node(
                func=load_queries,
                inputs="params:queries_path",
                outputs="raw_queries",
                name="load_queries_node",
            ),
            node(
                func=load_labels,
                inputs="params:labels_path",
                outputs="raw_labels",
                name="load_labels_node",
            ),
            node(
                func=preprocess_products,
                inputs="raw_products",
                outputs="processed_products",
                name="preprocess_products_node",
            ),
            node(
                func=create_relevance_labels,
                inputs="raw_labels",
                outputs="relevance_labels",
                name="create_relevance_labels_node",
            ),
            node(
                func=group_labels_by_query,
                inputs="relevance_labels",
                outputs="grouped_labels",
                name="group_labels_node",
            ),
        ],
        namespace="data_ingestion",
        inputs=None,
        outputs=["processed_products", "raw_queries", "grouped_labels"],
    )
