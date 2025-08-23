# Catalog Manager Agent Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy shared library
COPY shared/a2a_common /app/a2a_common
RUN pip install -e /app/a2a_common

# Copy requirements and install
COPY catalog_manager/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source code
COPY catalog_manager/src ./src
COPY catalog_manager/config ./config

# Create catalog directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AGENT_PORT=8009
ENV CATALOG_DB_PATH=/data/a2a_catalog.db

# Expose port
EXPOSE 8009

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8009/health || exit 1

# Run the agent
CMD ["python", "-m", "src.main"]