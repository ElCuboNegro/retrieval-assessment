@utils
Feature: Utility Functions
  As a developer
  I want utility functions to work correctly
  So that the project runs smoothly

  @log_metrics
  Scenario: Log evaluation metrics
    Given a dictionary of metrics
    And a method name
    When I log the metrics
    Then the metrics should be formatted and logged
    And float values should be displayed with 4 decimal places

  @calculate_improvement
  Scenario: Calculate improvement over baseline
    Given a baseline value of 0.20
    And a current value of 0.25
    When I calculate the improvement
    Then the improvement should be 25.0 percent

  @calculate_improvement_zero_baseline
  Scenario: Handle zero baseline in improvement calculation
    Given a baseline value of 0.0
    And a current value of 0.25
    When I calculate the improvement
    Then the improvement should be 0.0 percent


@download_data
Feature: Data Download Script
  As a user
  I want to download the WANDS dataset
  So that I can run the retrieval assessment

  @download_wands
  Scenario: Download WANDS dataset files
    Given the data directory does not contain WANDS files
    When I run the download script
    Then products.csv should be downloaded
    And query.csv should be downloaded
    And labels.csv should be downloaded

  @skip_existing
  Scenario: Skip already downloaded files
    Given products.csv already exists in the data directory
    When I run the download script
    Then products.csv should not be re-downloaded
    And a skip message should be shown
