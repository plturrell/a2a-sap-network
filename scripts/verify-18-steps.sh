#!/bin/bash

# A2A System 18-Step Verification Script
# This script validates that all components required for the 18-step startup are present
# Can be run standalone or within Docker container

# Removed set -e to allow individual tests to fail gracefully

# Support running from different locations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 A2A System 18-Step Verification"
echo "=================================="
echo ""

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_STEPS=18

verify_step() {
    local step_num=$1
    local step_name=$2
    local check_command=$3
    
    echo -n "Step $step_num/$TOTAL_STEPS: $step_name... "
    
    if eval "$check_command" > /dev/null 2>&1; then
        echo "✅ PASS"
        ((PASS_COUNT++))
    else
        echo "❌ FAIL"
        ((FAIL_COUNT++))
    fi
}

# Step 1: Pre-flight Checks
verify_step 1 "Pre-flight Checks" "test -f start.sh && test -f Dockerfile"

# Step 2: Environment Setup
verify_step 2 "Environment Setup" "test -d a2aAgents && test -d a2aNetwork"

# Step 3: Infrastructure Services
verify_step 3 "Infrastructure Services (Redis config)" "test -f docker-compose.yml || test -f docker-compose.production.yml"

# Step 4: Blockchain Services
verify_step 4 "Blockchain Services (Smart contracts)" "test -d a2aNetwork/src && ls a2aNetwork/src/*.sol > /dev/null 2>&1"

# Step 5: Core Services
verify_step 5 "Core Services (Backend)" "test -d a2aAgents/backend && test -f a2aAgents/backend/main.py"

# Step 6: Trust Systems
verify_step 6 "Trust Systems" "test -d a2aAgents/backend/trustSystem || test -f a2aAgents/backend/app/a2a/core/trustManager.py"

# Step 7: MCP Servers
verify_step 7 "MCP Servers" "test -d a2aAgents/backend/app/a2a/mcp/servers && test -f a2aAgents/backend/app/a2a/sdk/mcpServer.py"

# Step 8: Network Services
verify_step 8 "Network Services (CDS/CAP)" "test -f a2aNetwork/package.json && test -f a2aNetwork/dev-services/server.js"

# Step 9: Agent Services
verify_step 9 "Agent Services (16 agents)" "ls a2aAgents/backend/app/a2a/agents/agent*/active/*.py 2>/dev/null | wc -l | grep -E '1[0-9]|[2-9][0-9]'"

# Step 10: Frontend Service
verify_step 10 "Frontend Service" "test -d a2aAgents/frontend || test -d a2aNetwork/app"

# Step 11: Notification System
verify_step 11 "Notification System" "grep -r 'notification' a2aAgents/backend/ > /dev/null 2>&1 || grep -r 'websocket' a2aNetwork/ > /dev/null 2>&1"

# Step 12: Telemetry Services
verify_step 12 "Telemetry Services" "grep -r 'prometheus\|telemetry\|monitoring' a2aAgents/backend a2aNetwork --include='*.py' --include='*.js' --include='*.yml' --include='*.yaml' > /dev/null 2>&1"

# Step 13: Agent Communication Testing
verify_step 13 "Agent Communication" "test -f a2aAgents/backend/app/a2a/sdk/client.py || test -f a2aAgents/backend/app/a2a/core/a2aClient.py"

# Step 14: Blockchain Integration Testing
verify_step 14 "Blockchain Integration" "test -f a2aAgents/backend/app/a2a/sdk/blockchain/web3Client.py"

# Step 15: Database Connectivity
verify_step 15 "Database Connectivity" "grep -r 'DATABASE_URL\|postgres\|sqlite' a2aAgents/backend/ > /dev/null 2>&1"

# Step 16: Security Configuration
verify_step 16 "Security Configuration" "test -f a2aAgents/backend/app/a2a/core/securityContext.py || grep -r 'security' a2aAgents/backend/ > /dev/null 2>&1"

# Step 17: Performance Optimization
verify_step 17 "Performance Optimization" "test -f a2aAgents/backend/app/a2a/core/performanceOptimizer.py"

# Step 18: Final Validation
verify_step 18 "Final Validation (Makefile)" "test -f Makefile && grep 'start-all' Makefile > /dev/null 2>&1"

echo ""
echo "=================================="
echo "Verification Summary:"
echo "✅ Passed: $PASS_COUNT/$TOTAL_STEPS"
echo "❌ Failed: $FAIL_COUNT/$TOTAL_STEPS"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 All 18 steps verified successfully!"
    exit 0
else
    echo "⚠️  Some verification steps failed. Please check the components."
    exit 1
fi