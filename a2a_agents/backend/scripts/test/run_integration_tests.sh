#!/bin/bash

# A2A Integration Test Runner
# Ensures all services are running and executes comprehensive tests

set -e

echo "🚀 A2A Integration Test Runner"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a service is running
check_service() {
    local name=$1
    local port=$2
    local url="http://localhost:${port}/health"
    
    if curl -s -f -o /dev/null "${url}"; then
        echo -e "${GREEN}✅ ${name} is running on port ${port}${NC}"
        return 0
    else
        echo -e "${RED}❌ ${name} is not running on port ${port}${NC}"
        return 1
    fi
}

# Function to start a service if not running
start_service_if_needed() {
    local name=$1
    local port=$2
    local script=$3
    
    if ! check_service "${name}" "${port}"; then
        echo -e "${YELLOW}Starting ${name}...${NC}"
        python3 "${script}" &
        sleep 5  # Give service time to start
        
        if check_service "${name}" "${port}"; then
            echo -e "${GREEN}✅ ${name} started successfully${NC}"
        else
            echo -e "${RED}❌ Failed to start ${name}${NC}"
            return 1
        fi
    fi
}

echo ""
echo "1️⃣ Checking prerequisite services..."
echo "------------------------------------"

# Check if Anvil is running
if lsof -i:8545 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Anvil blockchain is running on port 8545${NC}"
else
    echo -e "${YELLOW}⚠️ Anvil not detected. Starting Anvil...${NC}"
    anvil &
    ANVIL_PID=$!
    sleep 3
fi

echo ""
echo "2️⃣ Checking A2A services..."
echo "----------------------------"

# Define all services
declare -A services=(
    ["Data Manager"]=8001
    ["Catalog Manager"]=8002
    ["Agent 0 (Data Product)"]=8003
    ["Agent 1 (Standardization)"]=8004
    ["Agent 2 (AI Preparation)"]=8005
    ["Agent 3 (Vector Processing)"]=8008
    ["Agent 4 (Calc Validation)"]=8006
    ["Agent 5 (QA Validation)"]=8007
)

# Check each service
all_running=true
for service in "${!services[@]}"; do
    port=${services[$service]}
    if ! check_service "$service" "$port"; then
        all_running=false
    fi
done

if [ "$all_running" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠️ Some services are not running.${NC}"
    echo "Please start all services using:"
    echo "  ./start_all_agents.sh"
    echo ""
    echo "Or start them individually:"
    echo "  python launch_data_manager.py"
    echo "  python launch_catalog_manager.py"
    echo "  python launch_agent0.py"
    echo "  python launch_agent1.py"
    echo "  python launch_agent2.py"
    echo "  python launch_agent3.py"
    echo "  python launch_agent4.py"
    echo "  python launch_agent5.py"
    exit 1
fi

echo ""
echo "3️⃣ Checking smart contracts..."
echo "------------------------------"

# Check if BusinessDataCloudA2A contract is deployed
BDC_ADDRESS="0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"
if cast code $BDC_ADDRESS --rpc-url http://localhost:8545 2>/dev/null | grep -q "0x"; then
    echo -e "${GREEN}✅ BusinessDataCloudA2A contract deployed at ${BDC_ADDRESS}${NC}"
else
    echo -e "${RED}❌ BusinessDataCloudA2A contract not found${NC}"
    echo "Please deploy contracts first using:"
    echo "  python integrate_bdc_smart_contract.py"
    exit 1
fi

echo ""
echo "4️⃣ Running integration tests..."
echo "-------------------------------"

# Set test environment
export A2A_TEST_MODE=integration
export A2A_BLOCKCHAIN_URL=http://localhost:8545
export A2A_BDC_CONTRACT=$BDC_ADDRESS

# Run the integration tests
echo ""
python3 test_full_a2a_integration.py

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ All integration tests passed!${NC}"
    echo ""
    echo "📊 Test Report:"
    echo "---------------"
    if [ -f integration_test_report.json ]; then
        # Extract summary from JSON report
        python3 -c "
import json
with open('integration_test_report.json', 'r') as f:
    report = json.load(f)
    summary = report['summary']
    print(f'  Total Tests: {summary[\"total_tests\"]}')
    print(f'  Passed: {summary[\"passed_tests\"]}')
    print(f'  Failed: {summary[\"failed_tests\"]}')
    print(f'  Success Rate: {summary[\"success_rate\"]}%')
    print(f'  Overall Health: {summary[\"overall_health\"]}')
"
    fi
else
    echo ""
    echo -e "${RED}❌ Integration tests failed${NC}"
    echo "Check integration_test_report.json for details"
    exit 1
fi

echo ""
echo "5️⃣ Additional validation tests..."
echo "---------------------------------"

# Test cross-agent workflow
echo "Testing cross-agent workflow..."
curl -s -X POST http://localhost:8003/api/register \
    -H "Content-Type: application/json" \
    -d '{
        "title": "Quick Test Product",
        "description": "Testing cross-agent communication",
        "creator": "Test Runner",
        "type": "Dataset"
    }' > /dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Cross-agent workflow test passed${NC}"
else
    echo -e "${RED}❌ Cross-agent workflow test failed${NC}"
fi

# Test trust verification
echo "Testing trust verification..."
TRUST_RESULT=$(curl -s http://localhost:8003/trust/public-key | grep -c "public_key")
if [ $TRUST_RESULT -gt 0 ]; then
    echo -e "${GREEN}✅ Trust system operational${NC}"
else
    echo -e "${RED}❌ Trust system not responding${NC}"
fi

echo ""
echo "✨ Integration testing complete!"
echo ""
echo "📄 Reports generated:"
echo "  - integration_test_report.json"
echo "  - Test logs in each agent's log file"
echo ""

# Cleanup if we started Anvil
if [ ! -z "$ANVIL_PID" ]; then
    echo "Stopping Anvil (PID: $ANVIL_PID)..."
    kill $ANVIL_PID 2>/dev/null || true
fi

exit 0