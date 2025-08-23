#!/bin/bash
# Setup environment variables required for A2A SDK

export A2A_SERVICE_URL="http://localhost:3000"
export A2A_SERVICE_HOST="localhost"
export A2A_BASE_URL="http://localhost:3000"
export GLEAN_AGENT_URL="http://localhost:8016"
export GLEAN_SERVICE_URL="http://localhost:4000/api/glean"
export A2A_NETWORK_URL="http://localhost:3001"
export A2A_ROUTER_URL="http://localhost:3002"

# Disable blockchain for testing
export BLOCKCHAIN_ENABLED="false"

# Add blockchain URL to prevent connection errors
export WEB3_PROVIDER_URL="http://localhost:8545"

# Run the test
echo "Environment variables set. Running test..."
cd /Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/gleanAgent
python3 test_real_project.py