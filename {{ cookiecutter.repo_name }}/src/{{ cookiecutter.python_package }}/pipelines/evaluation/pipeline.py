"""Evaluation pipeline definition."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compare_methods,
    evaluate_predictions,
    generate_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the evaluation pipeline.

    This pipeline:
    1. Evaluates each retrieval method (TF-IDF, embeddings, hybrid)
    2. Compares methods and calculates improvements
    3. Generates final evaluation report

    Returns:
        Pipeline object
    """
    return pipeline(
        [
            # Evaluate TF-IDF
            node(
                func=evaluate_predictions,
                inputs=[
                    "tfidf_predictions",
                    "grouped_labels",
                    "params:eval_k",
                ],
                outputs="tfidf_metrics",
                name="evaluate_tfidf_node",
            ),
            # Evaluate embeddings
            node(
                func=evaluate_predictions,
                inputs=[
                    "embedding_predictions",
                    "grouped_labels",
                    "params:eval_k",
                ],
                outputs="embedding_metrics",
                name="evaluate_embeddings_node",
            ),
            # Evaluate hybrid
            node(
                func=evaluate_predictions,
                inputs=[
                    "hybrid_predictions",
                    "grouped_labels",
                    "params:eval_k",
                ],
                outputs="hybrid_metrics",
                name="evaluate_hybrid_node",
            ),
            # Compare methods
            node(
                func=compare_methods,
                inputs=[
                    "tfidf_metrics",
                    "embedding_metrics",
                    "hybrid_metrics",
                ],
                outputs="comparison_table",
                name="compare_methods_node",
            ),
            # Generate report
            node(
                func=generate_report,
                inputs=[
                    "comparison_table",
                    "tfidf_metrics",
                    "embedding_metrics",
                    "hybrid_metrics",
                ],
                outputs="evaluation_report",
                name="generate_report_node",
            ),
        ],
        namespace="evaluation",
        inputs=[
            "tfidf_predictions",
            "embedding_predictions",
            "hybrid_predictions",
            "grouped_labels",
        ],
        outputs=["evaluation_report", "comparison_table"],
    )
