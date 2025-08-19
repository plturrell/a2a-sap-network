# Multi-stage Docker build for A2A Agents
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Build the application
RUN python -m compileall . && \
    find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -delete

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    A2A_ENVIRONMENT=production

# Create non-root user for security
RUN groupadd -r a2auser && useradd -r -g a2auser a2auser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directory and set ownership
WORKDIR /app
RUN chown -R a2auser:a2auser /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --from=builder --chown=a2auser:a2auser /app .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R a2auser:a2auser /app/data /app/logs

# Switch to non-root user
USER a2auser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python scripts/health_check.py || exit 1

# Default ports
EXPOSE 8000 8001 8002 8003 8004 8005

# Default command
CMD ["python", "-m", "app.a2a.main"]