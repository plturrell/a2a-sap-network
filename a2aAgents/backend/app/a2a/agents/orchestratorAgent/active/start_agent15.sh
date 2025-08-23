#!/bin/bash
# Start Agent 15 (Orchestrator) REST API Server

echo "Starting Agent 15 - Orchestrator API Server..."

# Navigate to the agent directory
cd "$(dirname "$0")"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(cd ../../../../../.. && pwd)"
export AGENT15_PORT=8015
export A2A_AGENT_ID="orchestrator-agent-15"
export A2A_BLOCKCHAIN_URL="${A2A_BLOCKCHAIN_URL:-http://localhost:8545}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the server
echo "Agent 15 will be available at http://localhost:8015"
echo "Press Ctrl+C to stop the server"

python agent15_server.py 2>&1 | tee logs/agent15_$(date +%Y%m%d_%H%M%S).log