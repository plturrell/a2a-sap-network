FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy shared dependencies first (better layer caching)
COPY services/shared /app/shared

# Copy agent-specific files
COPY services/agent0_data_product/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source code
COPY services/agent0_data_product/src /app/src
COPY services/agent0_data_product/config /app/config

# Set Python path to include shared modules
ENV PYTHONPATH=/app/shared:/app:$PYTHONPATH

# Expose agent port
EXPOSE 8001

# Run the agent
CMD ["python", "-m", "src.main"]