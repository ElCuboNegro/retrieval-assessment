@evaluation
Feature: Evaluation Pipeline
  As a data scientist
  I want to evaluate retrieval performance using standard metrics
  So that I can compare different methods and track improvements

  Background:
    Given predictions and ground truth labels are available
    And k is set to 10

  @map_at_k
  Scenario: Calculate Mean Average Precision at K
    Given true relevant product IDs for a query
    And predicted product IDs ranked by score
    When I calculate MAP@K
    Then the score should be between 0.0 and 1.0
    And it should reward results appearing earlier in the ranking
    And it should only count each relevant item once

  @map_at_k_examples
  Scenario Outline: MAP@K calculation examples
    Given true_ids are <true_ids>
    And predicted_ids are <predicted_ids>
    And k is <k>
    When I calculate MAP@K
    Then the score should be <expected_score>

    Examples:
      | true_ids | predicted_ids       | k  | expected_score |
      | [1, 2]   | [1, 2, 3, 4, 5]     | 5  | 1.0            |
      | [1, 2]   | [3, 1, 4, 2, 5]     | 5  | 0.5833         |
      | [1]      | [2, 3, 4, 5, 1]     | 5  | 0.2            |
      | [1, 2]   | [3, 4, 5, 6, 7]     | 5  | 0.0            |
      | []       | [1, 2, 3, 4, 5]     | 5  | 0.0            |

  @ndcg_at_k
  Scenario: Calculate Normalized Discounted Cumulative Gain at K
    Given true relevant product IDs for a query
    And predicted product IDs ranked by score
    When I calculate NDCG@K
    Then the score should be between 0.0 and 1.0
    And it should use logarithmic discounting for position
    And it should be normalized by ideal DCG

  @mrr
  Scenario: Calculate Mean Reciprocal Rank
    Given true relevant product IDs for a query
    And predicted product IDs ranked by score
    When I calculate MRR
    Then the score should be between 0.0 and 1.0
    And it should return 1/rank of the first relevant result
    And it should return 0.0 if no relevant results are found

  @recall_at_k
  Scenario: Calculate Recall at K
    Given true relevant product IDs for a query
    And predicted product IDs ranked by score
    When I calculate Recall@K
    Then the score should be between 0.0 and 1.0
    And it should measure the fraction of relevant items retrieved

  @evaluate_predictions
  Scenario: Evaluate predictions against ground truth
    Given predictions mapping query_id to product_ids
    And ground_truth mapping query_id to relevant product_ids
    When I evaluate predictions
    Then I should get aggregate metrics including:
      | metric     |
      | map@10     |
      | ndcg@10    |
      | mrr        |
      | recall@10  |
      | num_queries|

  @compare_methods
  Scenario: Compare multiple retrieval methods
    Given metrics for TF-IDF method
    And metrics for embedding method
    And metrics for hybrid method
    When I compare methods
    Then I should get a comparison DataFrame
    And it should include improvement percentages over baseline

  @generate_report
  Scenario: Generate evaluation report
    Given comparison table
    And metrics for all methods
    When I generate the report
    Then the report should identify the best method
    And it should show best MAP@10 score
    And it should calculate improvement over baseline

  @pipeline_integration
  Scenario: Run complete evaluation pipeline
    Given the evaluation pipeline is configured
    When I run the evaluation pipeline
    Then evaluation_report should be available
    And comparison_table should be available
