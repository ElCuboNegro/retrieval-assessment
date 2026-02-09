@vectorization
Feature: Vectorization Pipeline
  As a data scientist
  I want to create vector representations of products and queries
  So that I can perform similarity-based retrieval

  Background:
    Given preprocessed products are available
    And raw queries are available

  @tfidf_vectorizer
  Scenario: Create TF-IDF vectorizer and transform products
    Given a DataFrame with combined_text column
    And max_features is set to 10000
    And ngram_range is set to (1, 2)
    When I create the TF-IDF vectorizer
    Then I should get a fitted TfidfVectorizer
    And I should get a sparse TF-IDF matrix
    And the matrix should have shape (num_products, max_features)

  @vectorize_queries_tfidf
  Scenario: Vectorize queries using TF-IDF
    Given a fitted TF-IDF vectorizer
    And a queries DataFrame
    When I vectorize the queries
    Then I should get a sparse query vector matrix
    And the matrix should have the same feature dimension as the product matrix

  @create_embeddings
  Scenario: Create semantic embeddings for products
    Given a DataFrame with combined_text column
    And embedding_model is set to "all-MiniLM-L6-v2"
    And batch_size is set to 32
    When I create product embeddings
    Then I should get a numpy array of embeddings
    And the embeddings should have shape (num_products, embedding_dim)
    And embedding_dim should match the model's output dimension

  @embed_queries
  Scenario: Create embeddings for queries
    Given a queries DataFrame
    And embedding_model is set to "all-MiniLM-L6-v2"
    When I embed the queries
    Then I should get a numpy array of query embeddings
    And the query embeddings should have the same dimension as product embeddings

  @create_faiss_index
  Scenario: Create FAISS index from embeddings
    Given product embeddings as a numpy array
    When I create a FAISS index
    Then I should get a FAISS IndexFlatIP
    And the index should contain all product embeddings
    And the embeddings should be L2-normalized for cosine similarity

  @pipeline_integration
  Scenario: Run complete vectorization pipeline
    Given the vectorization pipeline is configured
    When I run the vectorization pipeline
    Then tfidf_vectorizer should be available
    And product_tfidf_matrix should be available
    And query_tfidf_vectors should be available
    And product_embeddings should be available
    And query_embeddings should be available
    And faiss_index should be available
