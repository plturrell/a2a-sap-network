#!/bin/bash
# Start all A2A agents as separate processes on their designated ports

set -euo pipefail

# Colors for output
readonly GREEN='\033[0;32m'
readonly RED='\033[0;31m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# Configuration
readonly ROOT_DIR="/app"
readonly LOG_DIR="$ROOT_DIR/logs"
readonly PID_DIR="$ROOT_DIR/pids"

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# Function to start an agent
start_agent() {
    local agent_num=$1
    local port=$2
    local name=$3
    
    echo -e "${YELLOW}Starting Agent $agent_num ($name) on port $port...${NC}"
    
    cd "$ROOT_DIR/a2aAgents/backend"
    
    # Set environment for this specific agent
    export AGENT_ID="agent$agent_num"
    export AGENT_PORT=$port
    export AGENT_NAME="$name"
    
    # Start the agent
    nohup python3 -m uvicorn main:app \
        --host 0.0.0.0 \
        --port $port \
        --app-dir . \
        --reload-dir app \
        > "$LOG_DIR/agent${agent_num}.log" 2>&1 &
    
    local pid=$!
    echo $pid > "$PID_DIR/agent${agent_num}.pid"
    
    # Wait a moment for startup
    sleep 2
    
    # Check if running
    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}✓ Agent $agent_num started successfully (PID: $pid)${NC}"
        return 0
    else
        echo -e "${RED}✗ Agent $agent_num failed to start${NC}"
        return 1
    fi
}

# Function to start MCP server
start_mcp_server() {
    local name=$1
    local port=$2
    local module=$3
    
    echo -e "${YELLOW}Starting MCP Server: $name on port $port...${NC}"
    
    cd "$ROOT_DIR/a2aAgents/backend"
    
    nohup python3 -m $module \
        --port $port \
        > "$LOG_DIR/mcp-$name.log" 2>&1 &
    
    local pid=$!
    echo $pid > "$PID_DIR/mcp-$name.pid"
    
    sleep 1
    
    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}✓ MCP $name started successfully (PID: $pid)${NC}"
        return 0
    else
        echo -e "${RED}✗ MCP $name failed to start${NC}"
        return 1
    fi
}

echo "═══════════════════════════════════════════════════════"
echo "     Starting Complete A2A Platform - All Services     "
echo "═══════════════════════════════════════════════════════"

# Start all 18 agents
echo -e "\n${YELLOW}Starting 18 A2A Agents...${NC}"

start_agent 0 8000 "Data Product Registration"
start_agent 1 8001 "Financial Data Standardization"
start_agent 2 8002 "AI Data Preparation"
start_agent 3 8003 "Vector Processing"
start_agent 4 8004 "Calculation & Validation"
start_agent 5 8005 "Quality Assurance"
start_agent 6 8006 "Quality Control"
start_agent 7 8007 "Agent Builder"
start_agent 8 8008 "Data Manager"
start_agent 9 8009 "Reasoning Agent"
start_agent 10 8010 "Calculation Agent"
start_agent 11 8011 "SQL Agent"
start_agent 12 8012 "Catalog Manager"
start_agent 13 8013 "Orchestrator Agent"
start_agent 14 8014 "Security Agent"
start_agent 15 8015 "Performance Agent"
start_agent 16 8016 "Agent Manager"
start_agent 17 8017 "System Monitor"

# Start MCP servers
echo -e "\n${YELLOW}Starting MCP Servers...${NC}"

start_mcp_server "enhanced-test" 8100 "tests.a2a_mcp.server.enhanced_mcp_server"
start_mcp_server "data-standardization" 8101 "app.a2a.mcp.servers.data_standardization_server"
start_mcp_server "vector-similarity" 8102 "app.a2a.mcp.servers.vector_similarity_server"
start_mcp_server "vector-ranking" 8103 "app.a2a.mcp.servers.vector_ranking_server"
start_mcp_server "transport-layer" 8104 "app.a2a.mcp.servers.transport_layer_server"
start_mcp_server "reasoning" 8105 "app.a2a.mcp.servers.reasoning_server"
start_mcp_server "session-management" 8106 "app.a2a.mcp.servers.session_management_server"
start_mcp_server "resource-streaming" 8107 "app.a2a.mcp.servers.resource_streaming_server"
start_mcp_server "confidence-calculator" 8108 "app.a2a.mcp.servers.confidence_calculator_server"
start_mcp_server "semantic-similarity" 8109 "app.a2a.mcp.servers.semantic_similarity_server"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}     All A2A Services Started Successfully!            ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"

# Keep the script running
echo -e "\n${YELLOW}Services are running. Press Ctrl+C to stop all services.${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping all services...${NC}"
    
    # Stop all agents
    for i in {0..17}; do
        if [ -f "$PID_DIR/agent${i}.pid" ]; then
            kill $(cat "$PID_DIR/agent${i}.pid") 2>/dev/null || true
            rm -f "$PID_DIR/agent${i}.pid"
        fi
    done
    
    # Stop all MCP servers
    for pid_file in "$PID_DIR"/mcp-*.pid; do
        if [ -f "$pid_file" ]; then
            kill $(cat "$pid_file") 2>/dev/null || true
            rm -f "$pid_file"
        fi
    done
    
    echo -e "${GREEN}All services stopped.${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Wait indefinitely
while true; do
    sleep 60
    
    # Basic health check every minute
    running_agents=$(ls "$PID_DIR"/agent*.pid 2>/dev/null | wc -l)
    running_mcp=$(ls "$PID_DIR"/mcp-*.pid 2>/dev/null | wc -l)
    
    echo -e "${GREEN}Status: $running_agents agents, $running_mcp MCP servers running${NC}"
done