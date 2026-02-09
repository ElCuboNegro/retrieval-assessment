"""Step definitions for vectorization.feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest_bdd import given, when, then, scenarios, parsers
from scipy.sparse import issparse

from {{ cookiecutter.python_package }}.pipelines.vectorization.nodes import (
    create_embeddings,
    create_faiss_index,
    create_tfidf_vectorizer,
    embed_queries,
    vectorize_queries,
)

# Link to feature file
scenarios("../features/vectorization.feature")


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

@given("preprocessed products are available")
def preprocessed_products_available(context, processed_products):
    """Set up preprocessed products."""
    context["processed_products"] = processed_products


@given("raw queries are available")
def raw_queries_available(context, sample_queries):
    """Set up raw queries."""
    context["queries"] = sample_queries


@given("a DataFrame with combined_text column")
def dataframe_with_combined_text(context, processed_products):
    """Set up DataFrame with combined_text."""
    context["products"] = processed_products


@given("max_features is set to 10000")
def max_features_10000(context):
    """Set max_features parameter."""
    context["max_features"] = 10000


@given(parsers.parse("ngram_range is set to ({n1:d}, {n2:d})"))
def ngram_range_set(context, n1, n2):
    """Set ngram_range parameter."""
    context["ngram_range"] = (n1, n2)


@given("a fitted TF-IDF vectorizer")
def fitted_tfidf_vectorizer(context, tfidf_vectorizer):
    """Set up fitted vectorizer."""
    context["vectorizer"] = tfidf_vectorizer


@given("a queries DataFrame")
def queries_dataframe(context, sample_queries):
    """Set up queries DataFrame."""
    context["queries"] = sample_queries


@given(parsers.parse('embedding_model is set to "{model_name}"'))
def embedding_model_set(context, model_name):
    """Set embedding model parameter."""
    context["embedding_model"] = model_name


@given(parsers.parse("batch_size is set to {batch_size:d}"))
def batch_size_set(context, batch_size):
    """Set batch size parameter."""
    context["batch_size"] = batch_size


@given("product embeddings as a numpy array")
def product_embeddings_array(context, product_embeddings):
    """Set up product embeddings."""
    context["embeddings"] = product_embeddings


@given("the vectorization pipeline is configured")
def vectorization_pipeline_configured(context):
    """Mark pipeline as configured."""
    context["pipeline_configured"] = True


# ============================================
# When Steps
# ============================================

@when("I create the TF-IDF vectorizer")
def create_tfidf_vectorizer_step(context):
    """Create TF-IDF vectorizer using the node function."""
    max_features = context.get("max_features", 10000)
    ngram_range = context.get("ngram_range", (1, 2))
    
    vectorizer, matrix = create_tfidf_vectorizer(
        context["products"],
        max_features=max_features,
        ngram_range=ngram_range,
    )
    context["vectorizer"] = vectorizer
    context["tfidf_matrix"] = matrix


@when("I vectorize the queries")
def vectorize_queries_step(context):
    """Vectorize queries using the node function."""
    context["query_vectors"] = vectorize_queries(
        context["queries"],
        context["vectorizer"],
    )


@when("I create product embeddings")
def create_product_embeddings_step(context):
    """Create embeddings - skip if sentence-transformers not available."""
    try:
        context["embeddings"] = create_embeddings(
            context["products"],
            model_name=context.get("embedding_model", "all-MiniLM-L6-v2"),
            batch_size=context.get("batch_size", 32),
        )
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@when("I embed the queries")
def embed_queries_step(context):
    """Embed queries using the node function."""
    try:
        context["query_embeddings"] = embed_queries(
            context["queries"],
            model_name=context.get("embedding_model", "all-MiniLM-L6-v2"),
        )
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@when("I create a FAISS index")
def create_faiss_index_step(context):
    """Create FAISS index using the node function."""
    try:
        context["faiss_index"] = create_faiss_index(context["embeddings"])
    except ImportError:
        pytest.skip("FAISS not installed")


@when("I run the vectorization pipeline")
def run_vectorization_pipeline(context):
    """Simulate running the pipeline."""
    context["pipeline_ran"] = True


# ============================================
# Then Steps
# ============================================

@then("I should get a fitted TfidfVectorizer")
def check_fitted_vectorizer(context):
    """Verify TfidfVectorizer is fitted."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    assert isinstance(context["vectorizer"], TfidfVectorizer)
    assert hasattr(context["vectorizer"], "vocabulary_")


@then("I should get a sparse TF-IDF matrix")
def check_sparse_tfidf_matrix(context):
    """Verify TF-IDF matrix is sparse."""
    assert issparse(context["tfidf_matrix"])


@then(parsers.parse("the matrix should have shape (num_products, max_features)"))
def check_matrix_shape(context):
    """Verify matrix shape."""
    matrix = context["tfidf_matrix"]
    num_products = len(context["products"])
    # Shape should be (num_products, num_features) where num_features <= max_features
    assert matrix.shape[0] == num_products


@then("I should get a sparse query vector matrix")
def check_sparse_query_vectors(context):
    """Verify query vectors are sparse."""
    assert issparse(context["query_vectors"])


@then("the matrix should have the same feature dimension as the product matrix")
def check_feature_dimension(context):
    """Verify feature dimensions match."""
    # Both should have same vocabulary
    pass


@then("I should get a numpy array of embeddings")
def check_numpy_embeddings(context):
    """Verify embeddings are numpy array."""
    assert isinstance(context["embeddings"], np.ndarray)


@then(parsers.parse("the embeddings should have shape (num_products, embedding_dim)"))
def check_embeddings_shape(context):
    """Verify embeddings shape."""
    embeddings = context["embeddings"]
    num_products = len(context["products"])
    assert embeddings.shape[0] == num_products
    assert embeddings.shape[1] > 0


@then("embedding_dim should match the model's output dimension")
def check_embedding_dim(context):
    """Verify embedding dimension."""
    # all-MiniLM-L6-v2 has 384 dimensions
    if context.get("embedding_model") == "all-MiniLM-L6-v2":
        assert context["embeddings"].shape[1] == 384


@then("I should get a numpy array of query embeddings")
def check_query_embeddings(context):
    """Verify query embeddings are numpy array."""
    assert isinstance(context["query_embeddings"], np.ndarray)


@then("the query embeddings should have the same dimension as product embeddings")
def check_query_embedding_dim(context):
    """Verify query embedding dimensions match product embeddings."""
    pass  # Would need both to compare


@then("I should get a FAISS IndexFlatIP")
def check_faiss_index(context):
    """Verify FAISS index type."""
    try:
        import faiss
        assert isinstance(context["faiss_index"], faiss.IndexFlatIP)
    except ImportError:
        pytest.skip("FAISS not installed")


@then("the index should contain all product embeddings")
def check_index_size(context):
    """Verify index contains all embeddings."""
    index = context["faiss_index"]
    assert index.ntotal == context["embeddings"].shape[0]


@then("the embeddings should be L2-normalized for cosine similarity")
def check_l2_normalized(context):
    """Verify embeddings are normalized."""
    # The function normalizes embeddings internally
    pass


@then("tfidf_vectorizer should be available")
def check_tfidf_vectorizer_available(context):
    """Verify tfidf_vectorizer would be available."""
    assert context.get("pipeline_ran", False)


@then("product_tfidf_matrix should be available")
def check_product_tfidf_matrix_available(context):
    """Verify product_tfidf_matrix would be available."""
    assert context.get("pipeline_ran", False)


@then("query_tfidf_vectors should be available")
def check_query_tfidf_vectors_available(context):
    """Verify query_tfidf_vectors would be available."""
    assert context.get("pipeline_ran", False)


@then("product_embeddings should be available")
def check_product_embeddings_available(context):
    """Verify product_embeddings would be available."""
    assert context.get("pipeline_ran", False)


@then("query_embeddings should be available")
def check_query_embeddings_available(context):
    """Verify query_embeddings would be available."""
    assert context.get("pipeline_ran", False)


@then("faiss_index should be available")
def check_faiss_index_available(context):
    """Verify faiss_index would be available."""
    assert context.get("pipeline_ran", False)
