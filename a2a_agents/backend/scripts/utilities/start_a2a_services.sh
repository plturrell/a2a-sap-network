#!/bin/bash
# Start all A2A services as true microservices

echo "🚀 Starting True A2A Microservices Architecture"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Kill any existing services
echo -e "${BLUE}Stopping any existing services...${NC}"
pkill -f "launch_agent0.py" 2>/dev/null
pkill -f "launch_agent1.py" 2>/dev/null
pkill -f "uvicorn main:app.*8000" 2>/dev/null
sleep 2

# Start Registry Service (Main App)
echo -e "${GREEN}Starting A2A Registry Service on port 8000...${NC}"
cd /Users/apple/projects/finsight_cib/backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
REGISTRY_PID=$!
echo "Registry PID: $REGISTRY_PID"
sleep 5

# Wait for registry to be ready
echo -e "${BLUE}Waiting for Registry to be ready...${NC}"
until curl -s http://localhost:8000/health > /dev/null; do
    echo -n "."
    sleep 1
done
echo -e "${GREEN}✓ Registry is ready${NC}"

# Start Agent 0
echo -e "${GREEN}Starting Agent 0 (Data Product Registration) on port 8002...${NC}"
AGENT0_PORT=8002 A2A_REGISTRY_URL=http://localhost:8000/api/v1/a2a python3 launch_agent0.py &
AGENT0_PID=$!
echo "Agent 0 PID: $AGENT0_PID"
sleep 3

# Start Agent 1
echo -e "${GREEN}Starting Agent 1 (Financial Standardization) on port 8001...${NC}"
AGENT1_PORT=8001 A2A_REGISTRY_URL=http://localhost:8000/api/v1/a2a python3 launch_agent1.py &
AGENT1_PID=$!
echo "Agent 1 PID: $AGENT1_PID"
sleep 3

# Wait for agents to register
echo -e "${BLUE}Waiting for agents to register with registry...${NC}"
sleep 5

# Check agent health
echo -e "${GREEN}Checking agent health...${NC}"
curl -s http://localhost:8000/health | jq '.app' || echo "Registry health check failed"
curl -s http://localhost:8002/health | jq '.' || echo "Agent 0 health check failed"
curl -s http://localhost:8001/health | jq '.' || echo "Agent 1 health check failed"

# Check registered agents
echo -e "${GREEN}Checking registered agents...${NC}"
curl -s http://localhost:8000/api/v1/a2a/agents/search | jq '.agents[] | {name: .name, url: .url, status: .status}'

echo -e "${GREEN}=============================================="
echo "✅ A2A Microservices Started Successfully!"
echo "=============================================="
echo ""
echo "Services running:"
echo "  - Registry:    http://localhost:8000"
echo "  - Agent 0:     http://localhost:8002" 
echo "  - Agent 1:     http://localhost:8001"
echo ""
echo "PIDs:"
echo "  - Registry:    $REGISTRY_PID"
echo "  - Agent 0:     $AGENT0_PID"
echo "  - Agent 1:     $AGENT1_PID"
echo ""
echo "To stop all services, run: ./stop_a2a_services.sh"
echo ""
echo "To test true A2A communication:"
echo "  python3 test_true_a2a.py"
echo "=============================================="

# Keep script running
wait