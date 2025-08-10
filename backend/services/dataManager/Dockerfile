# Data Manager Agent Dockerfile
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
COPY data_manager/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source code
COPY data_manager/src ./src
COPY data_manager/config ./config

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AGENT_PORT=8008
ENV STORAGE_BACKEND=sqlite
ENV DATA_MANAGER_DB_PATH=/data/a2a_data.db

# Expose port
EXPOSE 8008

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

# Run the agent
CMD ["python", "-m", "src.main"]