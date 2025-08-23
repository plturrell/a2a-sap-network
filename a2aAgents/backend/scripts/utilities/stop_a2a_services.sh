#!/bin/bash
# Stop all A2A services

echo "🛑 Stopping A2A Microservices"
echo "============================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Kill processes
echo -e "${RED}Stopping Agent 0...${NC}"
pkill -f "launch_agent0.py"

echo -e "${RED}Stopping Agent 1...${NC}"
pkill -f "launch_agent1.py"

echo -e "${RED}Stopping Registry...${NC}"
pkill -f "uvicorn main:app.*8000"

# Wait for processes to stop
sleep 2

# Check if any are still running
if pgrep -f "launch_agent0.py" > /dev/null; then
    echo -e "${RED}⚠️  Agent 0 still running, force killing...${NC}"
    pkill -9 -f "launch_agent0.py"
fi

if pgrep -f "launch_agent1.py" > /dev/null; then
    echo -e "${RED}⚠️  Agent 1 still running, force killing...${NC}"
    pkill -9 -f "launch_agent1.py"
fi

if pgrep -f "uvicorn main:app.*8000" > /dev/null; then
    echo -e "${RED}⚠️  Registry still running, force killing...${NC}"
    pkill -9 -f "uvicorn main:app.*8000"
fi

echo -e "${GREEN}✅ All A2A services stopped${NC}"