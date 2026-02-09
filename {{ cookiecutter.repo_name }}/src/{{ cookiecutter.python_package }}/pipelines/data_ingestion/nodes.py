"""Data ingestion nodes for loading and preprocessing WANDS dataset."""

from __future__ import annotations

import pandas as pd


def load_products(products_path: str) -> pd.DataFrame:
    """Load products dataset from WANDS.

    Args:
        products_path: Path to products.csv file

    Returns:
        DataFrame with product information
    """
    df = pd.read_csv(products_path, sep="\t")

    # Fill missing values in text columns
    text_cols = ["product_name", "product_description", "product_features"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def load_queries(queries_path: str) -> pd.DataFrame:
    """Load queries dataset from WANDS.

    Args:
        queries_path: Path to query.csv file

    Returns:
        DataFrame with query information
    """
    df = pd.read_csv(queries_path, sep="\t")
    return df


def load_labels(labels_path: str) -> pd.DataFrame:
    """Load labels (ground truth) dataset from WANDS.

    Args:
        labels_path: Path to label.csv file

    Returns:
        DataFrame with relevance labels
    """
    df = pd.read_csv(labels_path, sep="\t")
    return df


def preprocess_products(products: pd.DataFrame) -> pd.DataFrame:
    """Preprocess products for vectorization.

    Combines product name and description into a single text field.

    Args:
        products: Raw products DataFrame

    Returns:
        Processed products DataFrame with combined text
    """
    products = products.copy()

    # Combine text fields for vectorization
    products["combined_text"] = (
        products["product_name"].astype(str)
        + " "
        + products["product_description"].astype(str)
    )

    # Clean text
    products["combined_text"] = (
        products["combined_text"]
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return products


def create_relevance_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Create binary relevance labels from WANDS labels.

    WANDS uses: Exact, Partial, Irrelevant
    We convert to: 1 (Exact), 0.5 (Partial), 0 (Irrelevant)

    Args:
        labels: Raw labels DataFrame

    Returns:
        DataFrame with numeric relevance scores
    """
    labels = labels.copy()

    label_mapping = {"Exact": 1.0, "Partial": 0.5, "Irrelevant": 0.0}

    labels["relevance"] = labels["label"].map(label_mapping)

    return labels


def group_labels_by_query(labels: pd.DataFrame) -> dict:
    """Group labels by query_id for evaluation.

    Args:
        labels: DataFrame with relevance labels

    Returns:
        Dictionary mapping query_id to list of relevant product_ids
    """
    # Get exact matches only for MAP@K calculation
    exact_matches = labels[labels["label"] == "Exact"]

    grouped = exact_matches.groupby("query_id")["product_id"].apply(list).to_dict()

    return grouped
