"""Vectorization nodes for creating TF-IDF and embedding representations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


def create_tfidf_vectorizer(
    products: pd.DataFrame,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
) -> tuple[TfidfVectorizer, "csr_matrix"]:
    """Create TF-IDF vectorizer and transform products.

    Args:
        products: DataFrame with combined_text column
        max_features: Maximum number of features
        ngram_range: Range of ngrams to use

    Returns:
        Tuple of (fitted vectorizer, TF-IDF matrix)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        lowercase=True,
        dtype=np.float32,
    )

    tfidf_matrix = vectorizer.fit_transform(
        products["combined_text"].values.astype(str)
    )

    return vectorizer, tfidf_matrix


def vectorize_queries(
    queries: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> "csr_matrix":
    """Transform queries using fitted TF-IDF vectorizer.

    Args:
        queries: DataFrame with query column
        vectorizer: Fitted TF-IDF vectorizer

    Returns:
        TF-IDF matrix for queries
    """
    query_texts = queries["query"].values.astype(str)
    query_vectors = vectorizer.transform(query_texts)

    return query_vectors


def create_embeddings(
    products: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """Create semantic embeddings for products.

    Args:
        products: DataFrame with combined_text column
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding

    Returns:
        Numpy array of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        msg = "sentence-transformers not installed. Run: pip install sentence-transformers"
        raise ImportError(msg) from e

    model = SentenceTransformer(model_name)

    texts = products["combined_text"].values.astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings


def embed_queries(
    queries: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Create embeddings for queries.

    Args:
        queries: DataFrame with query column
        model_name: Sentence transformer model name

    Returns:
        Numpy array of query embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        msg = "sentence-transformers not installed. Run: pip install sentence-transformers"
        raise ImportError(msg) from e

    model = SentenceTransformer(model_name)

    query_texts = queries["query"].values.astype(str).tolist()
    query_embeddings = model.encode(
        query_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return query_embeddings


def create_faiss_index(
    embeddings: np.ndarray,
) -> Any:
    """Create FAISS index from embeddings.

    Args:
        embeddings: Numpy array of embeddings

    Returns:
        FAISS index
    """
    try:
        import faiss
    except ImportError as e:
        msg = "faiss-cpu not installed. Run: pip install faiss-cpu"
        raise ImportError(msg) from e

    # Normalize embeddings for cosine similarity
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(
        dimension
    )  # Inner product = cosine with normalized vectors
    index.add(embeddings)

    return index
