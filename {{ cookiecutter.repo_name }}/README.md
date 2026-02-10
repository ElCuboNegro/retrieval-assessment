# Retrieval Assessment - Kedro + LangGraph

## Overview

This is a Kedro project for evaluating retrieval systems, inspired by the HBS WANDS retrieval assignment.

The project demonstrates how to build a production-ready retrieval pipeline using:

- **Kedro**: For pipeline orchestration and data management
- **LangGraph**: For agentic workflows
- **scikit-learn**: For TF-IDF baseline
- **sentence-transformers**: For semantic embeddings
- **FAISS/ChromaDB**: For vector storage

## Pipelines

| Pipeline | Description |
|----------|-------------|
| `data_ingestion` | Load and preprocess the WANDS dataset |
| `vectorization` | Create TF-IDF and/or embedding representations |
| `retrieval` | Implement search functionality with ranking |
| `evaluation` | Calculate MAP@K, NDCG, and other metrics |

## Quick Start

```bash
# Install dependencies
pip install -e .

# Download WANDS dataset
python -m {{ cookiecutter.python_package }}.download_data

# Run all pipelines
kedro run

# Run a specific pipeline
kedro run --pipeline=data_ingestion
kedro run --pipeline=vectorization
kedro run --pipeline=retrieval
kedro run --pipeline=evaluation

# Visualize pipelines
kedro viz run
```

## Docker Usage

```bash
# Build and run with Docker Compose
docker compose build
docker compose up kedro

# Run tests in Docker
docker compose --profile test up test

# Start Kedro Viz (port 4141)
docker compose --profile viz up viz

# Start Jupyter Lab (port 8888)
docker compose --profile jupyter up jupyter

# Start ChromaDB (port 8000)
docker compose --profile chromadb up chromadb

# Or use the Makefile shortcuts
make docker-run
make docker-test
make docker-viz
```

## Project Structure

```
{{ cookiecutter.repo_name }}/
├── conf/                      # Configuration files
│   ├── base/                  # Base configuration
│   │   ├── catalog.yml        # Data catalog
│   │   └── parameters.yml     # Parameters
│   └── local/                 # Local overrides
├── data/                      # Data directory (gitignored)
│   ├── 01_raw/               # Raw WANDS data
│   ├── 02_intermediate/      # Processed data
│   ├── 03_primary/           # Primary datasets
│   └── 07_model_output/      # Results and metrics
├── src/                       # Source code
│   └── {{ cookiecutter.python_package }}/
│       ├── pipelines/        # Kedro pipelines
│       │   ├── data_ingestion/
│       │   ├── vectorization/
│       │   ├── retrieval/
│       │   └── evaluation/
│       └── utils.py          # Utility functions
├── notebooks/                 # Jupyter notebooks
└── tests/                     # Unit tests
```

## Assessment Challenge

Your task is to **improve the baseline retrieval system**:

1. **Baseline**: TF-IDF with cosine similarity (MAP@10 ≈ 0.23)
2. **Goal**: Achieve >10% improvement in MAP@10
3. **Options**:
   - Semantic embeddings (sentence-transformers)
   - Query expansion
   - Hybrid retrieval (sparse + dense)
   - Re-ranking

## Evaluation Metrics

- **MAP@K**: Mean Average Precision at K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Recall@K**: Recall at K

## License

MIT License
