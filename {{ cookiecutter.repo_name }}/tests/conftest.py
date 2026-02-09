"""Pytest-BDD configuration and shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================
# Sample Data Fixtures
# ============================================


@pytest.fixture
def sample_products() -> pd.DataFrame:
    """Create sample products DataFrame."""
    return pd.DataFrame(
        {
            "product_id": [1, 2, 3, 4, 5],
            "product_name": [
                "Blue Cotton T-Shirt",
                "Red Wool Sweater",
                "Black Leather Jacket",
                "White Linen Pants",
                "Green Silk Dress",
            ],
            "product_description": [
                "Comfortable cotton t-shirt in blue color",
                "Warm woolen sweater perfect for winter",
                "Stylish leather jacket for casual wear",
                "Breathable linen pants for summer",
                "Elegant silk dress for special occasions",
            ],
            "product_features": [
                "cotton, blue, casual",
                "wool, red, warm",
                "leather, black, stylish",
                "linen, white, breathable",
                "silk, green, elegant",
            ],
        }
    )


@pytest.fixture
def sample_queries() -> pd.DataFrame:
    """Create sample queries DataFrame."""
    return pd.DataFrame(
        {
            "query_id": [101, 102, 103],
            "query": [
                "blue shirt",
                "winter sweater",
                "formal dress",
            ],
        }
    )


@pytest.fixture
def sample_labels() -> pd.DataFrame:
    """Create sample labels DataFrame."""
    return pd.DataFrame(
        {
            "query_id": [101, 101, 101, 102, 102, 103, 103],
            "product_id": [1, 2, 3, 2, 4, 5, 1],
            "label": [
                "Exact",
                "Irrelevant",
                "Partial",
                "Exact",
                "Irrelevant",
                "Exact",
                "Partial",
            ],
        }
    )


@pytest.fixture
def processed_products(sample_products: pd.DataFrame) -> pd.DataFrame:
    """Create preprocessed products with combined_text."""
    df = sample_products.copy()
    df["combined_text"] = (
        df["product_name"].str.lower() + " " + df["product_description"].str.lower()
    )
    return df


@pytest.fixture
def grouped_labels() -> dict:
    """Create grouped labels for evaluation."""
    return {
        101: [1],  # query 101 has exact match with product 1
        102: [2],  # query 102 has exact match with product 2
        103: [5],  # query 103 has exact match with product 5
    }


# ============================================
# TF-IDF Fixtures
# ============================================


@pytest.fixture
def tfidf_vectorizer(processed_products: pd.DataFrame) -> TfidfVectorizer:
    """Create fitted TF-IDF vectorizer."""
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    vectorizer.fit(processed_products["combined_text"])
    return vectorizer


@pytest.fixture
def product_tfidf_matrix(
    processed_products: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
) -> csr_matrix:
    """Create TF-IDF matrix for products."""
    return tfidf_vectorizer.transform(processed_products["combined_text"])


@pytest.fixture
def query_tfidf_vectors(
    sample_queries: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
) -> csr_matrix:
    """Create TF-IDF vectors for queries."""
    return tfidf_vectorizer.transform(sample_queries["query"])


# ============================================
# Embedding Fixtures
# ============================================


@pytest.fixture
def product_embeddings() -> np.ndarray:
    """Create mock product embeddings."""
    np.random.seed(42)
    return np.random.randn(5, 384).astype(np.float32)


@pytest.fixture
def query_embeddings() -> np.ndarray:
    """Create mock query embeddings."""
    np.random.seed(43)
    return np.random.randn(3, 384).astype(np.float32)


@pytest.fixture
def faiss_index(product_embeddings: np.ndarray):
    """Create FAISS index from product embeddings."""
    try:
        import faiss

        embeddings = product_embeddings.copy()
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index
    except ImportError:
        pytest.skip("FAISS not installed")


# ============================================
# Search Results Fixtures
# ============================================


@pytest.fixture
def tfidf_results() -> dict:
    """Sample TF-IDF search results."""
    return {
        0: [(1, 0.95), (3, 0.75), (2, 0.50), (4, 0.30), (5, 0.10)],
        1: [(2, 0.90), (1, 0.60), (4, 0.40), (3, 0.20), (5, 0.05)],
        2: [(5, 0.85), (1, 0.55), (3, 0.45), (2, 0.25), (4, 0.15)],
    }


@pytest.fixture
def embedding_results() -> dict:
    """Sample embedding search results."""
    return {
        0: [(1, 0.90), (2, 0.70), (3, 0.65), (5, 0.40), (4, 0.20)],
        1: [(2, 0.88), (4, 0.55), (1, 0.50), (5, 0.30), (3, 0.15)],
        2: [(5, 0.92), (3, 0.60), (1, 0.45), (4, 0.35), (2, 0.20)],
    }


# ============================================
# Metrics Fixtures
# ============================================


@pytest.fixture
def tfidf_metrics() -> dict:
    """Sample TF-IDF metrics."""
    return {
        "map@10": 0.75,
        "ndcg@10": 0.80,
        "mrr": 0.85,
        "recall@10": 0.90,
        "num_queries": 3,
    }


@pytest.fixture
def embedding_metrics() -> dict:
    """Sample embedding metrics."""
    return {
        "map@10": 0.85,
        "ndcg@10": 0.88,
        "mrr": 0.90,
        "recall@10": 0.95,
        "num_queries": 3,
    }


@pytest.fixture
def hybrid_metrics() -> dict:
    """Sample hybrid metrics."""
    return {
        "map@10": 0.90,
        "ndcg@10": 0.92,
        "mrr": 0.95,
        "recall@10": 0.98,
        "num_queries": 3,
    }
