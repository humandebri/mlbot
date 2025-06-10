# Multi-stage build for cloud deployment
FROM python:3.13-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.3

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Development stage
FROM base as development

# Install all dependencies including dev
RUN poetry install && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

# Create non-root user
RUN addgroup --gid 1001 --system appgroup && \
    adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group appuser

# Change ownership of app directory
RUN chown -R appuser:appgroup /app

USER appuser

# Production stage
FROM base as production

# Install only production dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY src ./src
COPY config ./config

# Create non-root user
RUN addgroup --gid 1001 --system appgroup && \
    adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group appuser

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/models /app/artifacts && \
    chown -R appuser:appgroup /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["poetry", "run", "python", "-m", "src.main"]

# Service-specific stages
FROM production as ingestor
ENV SERVICE_NAME=ingestor
CMD ["poetry", "run", "python", "-m", "src.ingestor.main"]

FROM production as feature-hub
ENV SERVICE_NAME=feature_hub
CMD ["poetry", "run", "python", "-m", "src.feature_hub.main"]

FROM production as model-server
ENV SERVICE_NAME=model_server
CMD ["poetry", "run", "python", "-m", "src.model_server.main"]

FROM production as order-router
ENV SERVICE_NAME=order_router
CMD ["poetry", "run", "python", "-m", "src.order_router.main"]