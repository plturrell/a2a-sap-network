#!/bin/bash
# Start Agent 9 (Reasoning Agent) REST API Server

echo "Starting Agent 9 - Reasoning Agent API Server..."

# Navigate to the agent directory
cd "$(dirname "$0")"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(cd ../../../../../.. && pwd)"
export AGENT9_PORT=8086
export A2A_AGENT_ID="reasoning-agent-9"
export A2A_BLOCKCHAIN_URL="${A2A_BLOCKCHAIN_URL:-http://localhost:8545}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the server
echo "Agent 9 will be available at http://localhost:8086"
echo "Press Ctrl+C to stop the server"

python agent9_server.py 2>&1 | tee logs/agent9_$(date +%Y%m%d_%H%M%S).log