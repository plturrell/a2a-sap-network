FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install shared A2A common library
COPY services/shared/a2a_common /tmp/a2a_common
RUN cd /tmp/a2a_common && pip install .

# Copy agent-specific files
COPY services/agent1_standardization/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source code
COPY services/agent1_standardization/src /app/src
COPY services/agent1_standardization/config /app/config

# Expose A2A agent port
EXPOSE 8002

# Run the A2A agent
CMD ["python", "-m", "src.main"]