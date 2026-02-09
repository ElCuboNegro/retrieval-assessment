@retrieval
Feature: Retrieval Pipeline
  As a data scientist
  I want to search for products using different methods
  So that I can compare retrieval performance

  Background:
    Given vectorized products and queries are available
    And k is set to 10

  @search_tfidf
  Scenario: Search products using TF-IDF cosine similarity
    Given query TF-IDF vectors
    And product TF-IDF matrix
    And products DataFrame with product_id
    When I search using TF-IDF
    Then I should get a dictionary of results
    And each query should have k results
    And results should be tuples of (product_id, score)
    And results should be sorted by score descending

  @search_embeddings
  Scenario: Search products using FAISS semantic search
    Given query embeddings
    And a FAISS index
    And products DataFrame with product_id
    When I search using embeddings
    Then I should get a dictionary of results
    And each query should have k results
    And results should be tuples of (product_id, score)
    And query embeddings should be L2-normalized before search

  @hybrid_search
  Scenario: Combine TF-IDF and embedding results with RRF
    Given TF-IDF search results
    And embedding search results
    And alpha is set to 0.5
    When I perform hybrid search
    Then I should get combined results using Reciprocal Rank Fusion
    And the RRF constant should be 60
    And results should be weighted by alpha

  @hybrid_search_weights
  Scenario Outline: Hybrid search with different alpha values
    Given TF-IDF search results
    And embedding search results
    And alpha is set to <alpha>
    When I perform hybrid search
    Then TF-IDF results should have weight <tfidf_weight>
    And embedding results should have weight <embed_weight>

    Examples:
      | alpha | tfidf_weight | embed_weight |
      | 0.0   | 0.0          | 1.0          |
      | 0.5   | 0.5          | 0.5          |
      | 1.0   | 1.0          | 0.0          |
      | 0.7   | 0.7          | 0.3          |

  @format_results
  Scenario: Format results for evaluation
    Given search results with query indices
    And queries DataFrame with query_id
    When I format results for evaluation
    Then I should get a dictionary mapping query_id to product_ids
    And scores should be removed from results

  @pipeline_integration
  Scenario: Run complete retrieval pipeline
    Given the retrieval pipeline is configured
    When I run the retrieval pipeline
    Then tfidf_predictions should be available
    And embedding_predictions should be available
    And hybrid_predictions should be available
