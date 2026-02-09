# Multi-stage Dockerfile for Kedro Retrieval Assessment
# Optimized for development and production use

# ============================================
# Build stage - Install dependencies
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Development stage
# ============================================
FROM python:3.11-slim as development

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dev dependencies
RUN pip install --no-cache-dir pytest pytest-bdd pytest-cov ruff

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV KEDRO_LOGGING_CONFIG=/app/conf/logging.yml

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Expose Kedro Viz port
EXPOSE 4141

# Default command
CMD ["kedro", "run"]

# ============================================
# Production stage
# ============================================
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r kedro && useradd -r -g kedro kedro

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV KEDRO_LOGGING_CONFIG=/app/conf/logging.yml

# Copy only necessary files
COPY conf/ ./conf/
COPY data/ ./data/
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package
RUN pip install --no-cache-dir .

# Change ownership to non-root user
RUN chown -R kedro:kedro /app

USER kedro

# Default command
CMD ["kedro", "run"]
