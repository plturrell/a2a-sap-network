#!/bin/bash

echo "Starting A2A Financial Data Standardization Agent..."
echo "=============================================="
echo ""
echo "This is an A2A-compliant agent that implements:"
echo "- JSON-RPC 2.0 endpoint at /a2a/v1/rpc"
echo "- REST endpoint at /a2a/v1/messages"
echo "- Agent Card at /a2a/v1/.well-known/agent.json"
echo ""

# Navigate to backend directory
cd /Users/apple/projects/finsight_cib/backend

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Export environment variables
export PYTHONPATH=/Users/apple/projects/finsight_cib/backend:$PYTHONPATH

# Start the A2A agent server
echo ""
echo "Starting A2A agent server on http://localhost:8000"
echo "=============================================="
echo "Agent Card: http://localhost:8000/a2a/v1/.well-known/agent.json"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "The agent is ready to receive standardization requests via:"
echo "- JSON-RPC 2.0: POST http://localhost:8000/a2a/v1/rpc"
echo "- REST API: POST http://localhost:8000/a2a/v1/messages"
echo ""
echo "Press Ctrl+C to stop the agent"
echo ""

# Run the A2A agent server
python main.py