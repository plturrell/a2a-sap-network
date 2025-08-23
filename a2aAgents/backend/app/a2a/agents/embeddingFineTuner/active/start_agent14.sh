#!/bin/bash
# Start Agent 14 (Embedding Fine-Tuner) REST API Server

echo "Starting Agent 14 - Embedding Fine-Tuner API Server..."

# Navigate to the agent directory
cd "$(dirname "$0")"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(cd ../../../../../.. && pwd)"
export AGENT14_PORT=8014
export A2A_AGENT_ID="embedding-fine-tuner-agent-14"
export A2A_BLOCKCHAIN_URL="${A2A_BLOCKCHAIN_URL:-http://localhost:8545}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the server
echo "Agent 14 will be available at http://localhost:8014"
echo "Press Ctrl+C to stop the server"

python agent14_server.py 2>&1 | tee logs/agent14_$(date +%Y%m%d_%H%M%S).log