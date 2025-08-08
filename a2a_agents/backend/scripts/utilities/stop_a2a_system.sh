#!/bin/bash

# A2A Business Data Cloud Stop Script
# Gracefully stops all A2A services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LOG_DIR="./deployment_logs"

echo -e "${BLUE}🛑 Stopping A2A Business Data Cloud${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Function to stop service by PID file
stop_service() {
    local name=$1
    local pid_file="$LOG_DIR/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping $name (PID: $pid)...${NC}"
            kill $pid
            sleep 1
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                kill -9 $pid
            fi
            
            rm "$pid_file"
            echo -e "${GREEN}✅ $name stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $name not running (stale PID file)${NC}"
            rm "$pid_file"
        fi
    else
        echo -e "${YELLOW}⚠️  No PID file for $name${NC}"
    fi
}

# Function to stop service by port
stop_by_port() {
    local name=$1
    local port=$2
    
    local pid=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}Stopping $name on port $port (PID: $pid)...${NC}"
        kill $pid 2>/dev/null || true
        sleep 1
        
        # Force kill if still running
        if lsof -ti:$port > /dev/null 2>&1; then
            kill -9 $(lsof -ti:$port) 2>/dev/null || true
        fi
        
        echo -e "${GREEN}✅ $name stopped${NC}"
    else
        echo -e "${YELLOW}⚠️  $name not running on port $port${NC}"
    fi
}

# Stop all A2A agents
echo -e "${BLUE}Stopping A2A Agents...${NC}"
stop_service "agent5_qa_validation"
stop_service "agent4_calc_validation"
stop_service "agent3_vector_processing"
stop_service "agent2_ai_preparation"
stop_service "agent1_standardization"
stop_service "agent0_data_product"

# Stop supporting services
echo -e "\n${BLUE}Stopping Supporting Services...${NC}"
stop_service "catalog_manager"
stop_service "data_manager"

# Stop by port as fallback
echo -e "\n${BLUE}Checking for services by port...${NC}"
stop_by_port "agent5" 8007
stop_by_port "agent4" 8006
stop_by_port "agent3" 8008
stop_by_port "agent2" 8005
stop_by_port "agent1" 8004
stop_by_port "agent0" 8003
stop_by_port "catalog_manager" 8002
stop_by_port "data_manager" 8001

# Stop Anvil blockchain
echo -e "\n${BLUE}Stopping Blockchain...${NC}"
stop_service "anvil"
stop_by_port "anvil" 8545

# Clean up any orphaned processes
echo -e "\n${BLUE}Cleaning up...${NC}"

# Kill any remaining Python processes related to A2A
for proc in launch_data_manager launch_catalog_manager launch_agent; do
    pkill -f "$proc" 2>/dev/null || true
done

echo -e "${GREEN}✅ All services stopped${NC}"

# Show final status
echo -e "\n${BLUE}Final Status Check:${NC}"
all_stopped=true

for port in 8001 8002 8003 8004 8005 8006 8007 8008 8545; do
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${RED}❌ Port $port still in use${NC}"
        all_stopped=false
    else
        echo -e "${GREEN}✅ Port $port is free${NC}"
    fi
done

if [ "$all_stopped" = true ]; then
    echo -e "\n${GREEN}✅ A2A system successfully stopped${NC}"
else
    echo -e "\n${YELLOW}⚠️  Some services may still be running${NC}"
    echo "Check with: lsof -i:8001-8008"
fi

exit 0