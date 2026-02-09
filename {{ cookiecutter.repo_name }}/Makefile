# ============================================
# Makefile for {{ cookiecutter.project_name }}
# ============================================

.PHONY: help install run test lint viz clean docker-build docker-run docker-test docker-viz docker-jupyter docker-down

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "  Local Development:"
	@echo "    install      - Install the project with dev dependencies"
	@echo "    run          - Run all Kedro pipelines"
	@echo "    test         - Run pytest with BDD tests"
	@echo "    lint         - Run ruff linter"
	@echo "    viz          - Start Kedro Viz server"
	@echo "    clean        - Remove build artifacts and cache"
	@echo ""
	@echo "  Docker Commands:"
	@echo "    docker-build - Build Docker image"
	@echo "    docker-run   - Run Kedro pipelines in Docker"
	@echo "    docker-test  - Run tests in Docker"
	@echo "    docker-viz   - Start Kedro Viz in Docker"
	@echo "    docker-jupyter - Start Jupyter Lab in Docker"
	@echo "    docker-down  - Stop all Docker containers"
	@echo ""
	@echo "  Data:"
	@echo "    download     - Download WANDS dataset"

# ============================================
# Local Development
# ============================================

install:
	pip install -e ".[dev]"

run:
	kedro run

run-pipeline:
	kedro run --pipeline=$(PIPELINE)

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src/{{ cookiecutter.python_package }} --cov-report=html

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

viz:
	kedro viz run

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

download:
	python -m {{ cookiecutter.python_package }}.download_data

# ============================================
# Docker Commands
# ============================================

docker-build:
	docker compose build

docker-run:
	docker compose up kedro

docker-test:
	docker compose --profile test up test

docker-viz:
	docker compose --profile viz up viz

docker-jupyter:
	docker compose --profile jupyter up jupyter

docker-chromadb:
	docker compose --profile chromadb up chromadb

docker-down:
	docker compose down

docker-clean:
	docker compose down -v --rmi local

# ============================================
# Pipeline shortcuts
# ============================================

ingest:
	kedro run --pipeline=data_ingestion

vectorize:
	kedro run --pipeline=vectorization

retrieve:
	kedro run --pipeline=retrieval

evaluate:
	kedro run --pipeline=evaluation
