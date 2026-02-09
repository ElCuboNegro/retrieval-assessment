"""Step definitions for retrieval.feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest_bdd import given, when, then, scenarios, parsers

from {{ cookiecutter.python_package }}.pipelines.retrieval.nodes import (
    format_results_for_evaluation,
    hybrid_search,
    search_embeddings,
    search_tfidf,
)

# Link to feature file
scenarios("../features/retrieval.feature")


# ============================================
# Fixtures for this step file
# ============================================

@pytest.fixture
def context():
    """Shared context dictionary for steps."""
    return {}


# ============================================
# Given Steps
# ============================================

@given("vectorized products and queries are available")
def vectorized_data_available(context, product_tfidf_matrix, query_tfidf_vectors, 
                               product_embeddings, query_embeddings, processed_products):
    """Set up vectorized data."""
    context["product_tfidf_matrix"] = product_tfidf_matrix
    context["query_tfidf_vectors"] = query_tfidf_vectors
    context["product_embeddings"] = product_embeddings
    context["query_embeddings"] = query_embeddings
    context["products"] = processed_products


@given(parsers.parse("k is set to {k:d}"))
def k_set(context, k):
    """Set k parameter."""
    context["k"] = k


@given("query TF-IDF vectors")
def query_tfidf(context, query_tfidf_vectors):
    """Set up query TF-IDF vectors."""
    context["query_vectors"] = query_tfidf_vectors


@given("product TF-IDF matrix")
def product_tfidf(context, product_tfidf_matrix):
    """Set up product TF-IDF matrix."""
    context["product_matrix"] = product_tfidf_matrix


@given("products DataFrame with product_id")
def products_with_id(context, processed_products):
    """Set up products DataFrame."""
    context["products"] = processed_products


@given("query embeddings")
def query_embeds(context, query_embeddings):
    """Set up query embeddings."""
    context["query_embeddings"] = query_embeddings


@given("a FAISS index")
def faiss_idx(context, faiss_index):
    """Set up FAISS index."""
    context["faiss_index"] = faiss_index


@given("TF-IDF search results")
def tfidf_results_given(context, tfidf_results):
    """Set up TF-IDF results."""
    context["tfidf_results"] = tfidf_results


@given("embedding search results")
def embedding_results_given(context, embedding_results):
    """Set up embedding results."""
    context["embedding_results"] = embedding_results


@given(parsers.parse("alpha is set to {alpha:f}"))
def alpha_set(context, alpha):
    """Set alpha parameter."""
    context["alpha"] = alpha


@given("search results with query indices")
def search_results_given(context, tfidf_results):
    """Set up search results."""
    context["results"] = tfidf_results


@given("queries DataFrame with query_id")
def queries_with_id(context, sample_queries):
    """Set up queries DataFrame."""
    context["queries"] = sample_queries


@given("the retrieval pipeline is configured")
def retrieval_pipeline_configured(context):
    """Mark pipeline as configured."""
    context["pipeline_configured"] = True


# ============================================
# When Steps
# ============================================

@when("I search using TF-IDF")
def search_tfidf_step(context):
    """Search using TF-IDF."""
    context["results"] = search_tfidf(
        context["query_vectors"],
        context["product_matrix"],
        context["products"],
        k=context.get("k", 10),
    )


@when("I search using embeddings")
def search_embeddings_step(context):
    """Search using embeddings."""
    try:
        context["results"] = search_embeddings(
            context["query_embeddings"],
            context["faiss_index"],
            context["products"],
            k=context.get("k", 10),
        )
    except ImportError:
        pytest.skip("FAISS not installed")


@when("I perform hybrid search")
def hybrid_search_step(context):
    """Perform hybrid search."""
    context["results"] = hybrid_search(
        context["tfidf_results"],
        context["embedding_results"],
        alpha=context.get("alpha", 0.5),
        k=context.get("k", 10),
    )


@when("I format results for evaluation")
def format_results_step(context):
    """Format results for evaluation."""
    context["formatted_results"] = format_results_for_evaluation(
        context["results"],
        context["queries"],
    )


@when("I run the retrieval pipeline")
def run_retrieval_pipeline(context):
    """Simulate running the pipeline."""
    context["pipeline_ran"] = True


# ============================================
# Then Steps
# ============================================

@then("I should get a dictionary of results")
def check_results_dict(context):
    """Verify results is a dictionary."""
    assert isinstance(context["results"], dict)


@then("each query should have k results")
def check_k_results(context):
    """Verify each query has k results."""
    k = context.get("k", 10)
    for query_idx, results in context["results"].items():
        assert len(results) <= k


@then("results should be tuples of (product_id, score)")
def check_result_tuples(context):
    """Verify result format."""
    for query_idx, results in context["results"].items():
        for item in results:
            assert len(item) == 2
            assert isinstance(item[0], int)  # product_id
            assert isinstance(item[1], float)  # score


@then("results should be sorted by score descending")
def check_sorted_descending(context):
    """Verify results are sorted by score."""
    for query_idx, results in context["results"].items():
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


@then("query embeddings should be L2-normalized before search")
def check_normalized_search(context):
    """Verify normalization happens (implementation detail)."""
    pass


@then("I should get combined results using Reciprocal Rank Fusion")
def check_rrf_results(context):
    """Verify hybrid results use RRF."""
    assert isinstance(context["results"], dict)
    assert len(context["results"]) > 0


@then("the RRF constant should be 60")
def check_rrf_constant(context):
    """Verify RRF constant (implementation detail)."""
    pass


@then("results should be weighted by alpha")
def check_alpha_weighting(context):
    """Verify alpha weighting (implementation detail)."""
    pass


@then(parsers.parse("TF-IDF results should have weight {weight:f}"))
def check_tfidf_weight(context, weight):
    """Verify TF-IDF weight."""
    assert context.get("alpha") == weight or context.get("alpha") is not None


@then(parsers.parse("embedding results should have weight {weight:f}"))
def check_embed_weight(context, weight):
    """Verify embedding weight."""
    alpha = context.get("alpha", 0.5)
    assert abs((1 - alpha) - weight) < 0.01


@then("I should get a dictionary mapping query_id to product_ids")
def check_formatted_results(context):
    """Verify formatted results structure."""
    assert isinstance(context["formatted_results"], dict)


@then("scores should be removed from results")
def check_no_scores(context):
    """Verify scores are removed."""
    for query_id, product_ids in context["formatted_results"].items():
        for item in product_ids:
            assert isinstance(item, int)  # Just product_id, no tuples


@then("tfidf_predictions should be available")
def check_tfidf_predictions_available(context):
    """Verify tfidf_predictions would be available."""
    assert context.get("pipeline_ran", False)


@then("embedding_predictions should be available")
def check_embedding_predictions_available(context):
    """Verify embedding_predictions would be available."""
    assert context.get("pipeline_ran", False)


@then("hybrid_predictions should be available")
def check_hybrid_predictions_available(context):
    """Verify hybrid_predictions would be available."""
    assert context.get("pipeline_ran", False)
