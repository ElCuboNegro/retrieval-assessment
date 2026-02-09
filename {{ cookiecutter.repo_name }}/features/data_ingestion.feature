@data_ingestion
Feature: Data Ingestion Pipeline
  As a data scientist
  I want to load and preprocess the WANDS dataset
  So that I can use it for retrieval evaluation

  Background:
    Given the WANDS dataset files exist in the raw data directory

  @load_products
  Scenario: Load products from CSV file
    Given a products CSV file with tab-separated values
    When I load the products dataset
    Then I should get a DataFrame with product information
    And the DataFrame should have columns "product_id", "product_name", "product_description"
    And missing text values should be filled with empty strings

  @load_queries
  Scenario: Load queries from CSV file
    Given a queries CSV file with tab-separated values
    When I load the queries dataset
    Then I should get a DataFrame with query information
    And the DataFrame should have columns "query_id", "query"

  @load_labels
  Scenario: Load labels from CSV file
    Given a labels CSV file with tab-separated values
    When I load the labels dataset
    Then I should get a DataFrame with relevance labels
    And the DataFrame should have columns "query_id", "product_id", "label"

  @preprocess_products
  Scenario: Preprocess products for vectorization
    Given a products DataFrame with name and description columns
    When I preprocess the products
    Then a "combined_text" column should be created
    And the text should be lowercased
    And special characters should be removed
    And extra whitespace should be normalized

  @create_relevance_labels
  Scenario: Create numeric relevance labels
    Given a labels DataFrame with Exact, Partial, and Irrelevant labels
    When I create relevance labels
    Then "Exact" labels should map to 1.0
    And "Partial" labels should map to 0.5
    And "Irrelevant" labels should map to 0.0

  @group_labels
  Scenario: Group labels by query for evaluation
    Given a relevance labels DataFrame
    When I group labels by query
    Then I should get a dictionary mapping query_id to relevant product_ids
    And only "Exact" matches should be included

  @pipeline_integration
  Scenario: Run complete data ingestion pipeline
    Given the data ingestion pipeline is configured
    When I run the data ingestion pipeline
    Then processed_products should be available
    And raw_queries should be available
    And grouped_labels should be available
