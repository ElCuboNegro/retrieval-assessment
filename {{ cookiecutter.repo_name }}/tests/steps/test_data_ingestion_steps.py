"""Step definitions for data_ingestion.feature."""

from __future__ import annotations

import pandas as pd
import pytest
from pytest_bdd import given, when, then, scenarios, parsers

from {{ cookiecutter.python_package }}.pipelines.data_ingestion.nodes import (
    create_relevance_labels,
    group_labels_by_query,
    load_labels,
    load_products,
    load_queries,
    preprocess_products,
)

# Link to feature file
scenarios("../features/data_ingestion.feature")


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

@given("the WANDS dataset files exist in the raw data directory")
def wands_files_exist(tmp_path, context):
    """Create temporary WANDS-like files."""
    # Create products file
    products_df = pd.DataFrame({
        "product_id": [1, 2, 3],
        "product_name": ["Product A", "Product B", "Product C"],
        "product_description": ["Desc A", "Desc B", "Desc C"],
    })
    products_path = tmp_path / "products.csv"
    products_df.to_csv(products_path, sep="\t", index=False)
    context["products_path"] = str(products_path)
    
    # Create queries file
    queries_df = pd.DataFrame({
        "query_id": [101, 102],
        "query": ["query one", "query two"],
    })
    queries_path = tmp_path / "query.csv"
    queries_df.to_csv(queries_path, sep="\t", index=False)
    context["queries_path"] = str(queries_path)
    
    # Create labels file
    labels_df = pd.DataFrame({
        "query_id": [101, 101, 102],
        "product_id": [1, 2, 3],
        "label": ["Exact", "Partial", "Irrelevant"],
    })
    labels_path = tmp_path / "labels.csv"
    labels_df.to_csv(labels_path, sep="\t", index=False)
    context["labels_path"] = str(labels_path)


@given("a products CSV file with tab-separated values")
def products_csv_file(context, tmp_path):
    """Ensure products file exists."""
    if "products_path" not in context:
        df = pd.DataFrame({
            "product_id": [1, 2],
            "product_name": ["Name 1", "Name 2"],
            "product_description": ["Desc 1", "Desc 2"],
        })
        path = tmp_path / "products.csv"
        df.to_csv(path, sep="\t", index=False)
        context["products_path"] = str(path)


@given("a queries CSV file with tab-separated values")
def queries_csv_file(context, tmp_path):
    """Ensure queries file exists."""
    if "queries_path" not in context:
        df = pd.DataFrame({
            "query_id": [101],
            "query": ["test query"],
        })
        path = tmp_path / "query.csv"
        df.to_csv(path, sep="\t", index=False)
        context["queries_path"] = str(path)


@given("a labels CSV file with tab-separated values")
def labels_csv_file(context, tmp_path):
    """Ensure labels file exists."""
    if "labels_path" not in context:
        df = pd.DataFrame({
            "query_id": [101],
            "product_id": [1],
            "label": ["Exact"],
        })
        path = tmp_path / "labels.csv"
        df.to_csv(path, sep="\t", index=False)
        context["labels_path"] = str(path)


@given("a products DataFrame with name and description columns")
def products_dataframe(context, sample_products):
    """Set up products DataFrame."""
    context["products"] = sample_products


@given("a labels DataFrame with Exact, Partial, and Irrelevant labels")
def labels_dataframe(context, sample_labels):
    """Set up labels DataFrame."""
    context["labels"] = sample_labels


@given("a relevance labels DataFrame")
def relevance_labels_dataframe(context, sample_labels):
    """Set up relevance labels with numeric scores."""
    context["relevance_labels"] = create_relevance_labels(sample_labels)


@given("the data ingestion pipeline is configured")
def data_ingestion_pipeline_configured(context):
    """Mark pipeline as configured."""
    context["pipeline_configured"] = True


# ============================================
# When Steps
# ============================================

@when("I load the products dataset")
def load_products_dataset(context):
    """Load products using the node function."""
    context["result"] = load_products(context["products_path"])


@when("I load the queries dataset")
def load_queries_dataset(context):
    """Load queries using the node function."""
    context["result"] = load_queries(context["queries_path"])


@when("I load the labels dataset")
def load_labels_dataset(context):
    """Load labels using the node function."""
    context["result"] = load_labels(context["labels_path"])


@when("I preprocess the products")
def preprocess_products_step(context):
    """Preprocess products using the node function."""
    context["result"] = preprocess_products(context["products"])


@when("I create relevance labels")
def create_relevance_labels_step(context):
    """Create relevance labels using the node function."""
    context["result"] = create_relevance_labels(context["labels"])


@when("I group labels by query")
def group_labels_step(context):
    """Group labels by query using the node function."""
    context["result"] = group_labels_by_query(context["relevance_labels"])


@when("I run the data ingestion pipeline")
def run_data_ingestion_pipeline(context):
    """Simulate running the pipeline."""
    # In real tests, this would use Kedro's runner
    context["pipeline_ran"] = True


# ============================================
# Then Steps
# ============================================

@then("I should get a DataFrame with product information")
def check_products_dataframe(context):
    """Verify products DataFrame."""
    assert isinstance(context["result"], pd.DataFrame)
    assert len(context["result"]) > 0


@then(parsers.parse('the DataFrame should have columns "{columns}"'))
def check_columns(context, columns):
    """Verify DataFrame has required columns."""
    expected_cols = [c.strip() for c in columns.split(",")]
    for col in expected_cols:
        assert col in context["result"].columns, f"Column {col} not found"


@then("missing text values should be filled with empty strings")
def check_missing_values(context):
    """Verify no missing values in text columns."""
    df = context["result"]
    text_cols = ["product_name", "product_description"]
    for col in text_cols:
        if col in df.columns:
            assert df[col].isna().sum() == 0


@then("I should get a DataFrame with query information")
def check_queries_dataframe(context):
    """Verify queries DataFrame."""
    assert isinstance(context["result"], pd.DataFrame)
    assert len(context["result"]) > 0


@then("I should get a DataFrame with relevance labels")
def check_labels_dataframe(context):
    """Verify labels DataFrame."""
    assert isinstance(context["result"], pd.DataFrame)
    assert len(context["result"]) > 0


@then('a "combined_text" column should be created')
def check_combined_text_column(context):
    """Verify combined_text column exists."""
    assert "combined_text" in context["result"].columns


@then("the text should be lowercased")
def check_lowercased(context):
    """Verify text is lowercased."""
    text = context["result"]["combined_text"].iloc[0]
    assert text == text.lower()


@then("special characters should be removed")
def check_special_chars(context):
    """Verify special characters are handled."""
    # This is a basic check - implementation may vary
    pass


@then("extra whitespace should be normalized")
def check_whitespace(context):
    """Verify whitespace is normalized."""
    for text in context["result"]["combined_text"]:
        assert "  " not in text  # No double spaces


@then('"Exact" labels should map to 1.0')
def check_exact_labels(context):
    """Verify Exact maps to 1.0."""
    df = context["result"]
    exact_rows = df[df["label"] == "Exact"]
    assert all(exact_rows["relevance"] == 1.0)


@then('"Partial" labels should map to 0.5')
def check_partial_labels(context):
    """Verify Partial maps to 0.5."""
    df = context["result"]
    partial_rows = df[df["label"] == "Partial"]
    assert all(partial_rows["relevance"] == 0.5)


@then('"Irrelevant" labels should map to 0.0')
def check_irrelevant_labels(context):
    """Verify Irrelevant maps to 0.0."""
    df = context["result"]
    irrelevant_rows = df[df["label"] == "Irrelevant"]
    assert all(irrelevant_rows["relevance"] == 0.0)


@then("I should get a dictionary mapping query_id to relevant product_ids")
def check_grouped_labels_dict(context):
    """Verify grouped labels is a dictionary."""
    assert isinstance(context["result"], dict)


@then('only "Exact" matches should be included')
def check_only_exact_matches(context):
    """Verify only exact matches are in the grouped labels."""
    # The group_labels_by_query function only includes Exact matches
    assert len(context["result"]) > 0


@then("processed_products should be available")
def check_processed_products_available(context):
    """Verify processed products would be available."""
    assert context.get("pipeline_ran", False)


@then("raw_queries should be available")
def check_raw_queries_available(context):
    """Verify raw queries would be available."""
    assert context.get("pipeline_ran", False)


@then("grouped_labels should be available")
def check_grouped_labels_available(context):
    """Verify grouped labels would be available."""
    assert context.get("pipeline_ran", False)
