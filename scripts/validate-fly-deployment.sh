#!/bin/bash
# Validate Fly.io deployment

set -e

APP_NAME="${1:-a2a-platform}"
BASE_URL="https://${APP_NAME}.fly.dev"

echo "🔍 Validating A2A Platform deployment on Fly.io"
echo "=============================================="
echo "App: $APP_NAME"
echo "URL: $BASE_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local description=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $description... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL$endpoint" || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✅ OK${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC} (HTTP $response)"
        return 1
    fi
}

# Function to check JSON endpoint
check_json_endpoint() {
    local endpoint=$1
    local description=$2
    
    echo -n "Checking $description... "
    
    if response=$(curl -s "$BASE_URL$endpoint" | jq . 2>/dev/null); then
        echo -e "${GREEN}✅ OK${NC}"
        echo "$response" | jq -r '. | "\(.status // "N/A") - \(.app // .service // "Service") v\(.version // "?")"' 2>/dev/null || true
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

# Start validation
TOTAL_CHECKS=0
PASSED_CHECKS=0

echo "1️⃣ Core Endpoints"
echo "=================="

# Health check
((TOTAL_CHECKS++))
if check_endpoint "/health" "Health endpoint"; then
    ((PASSED_CHECKS++))
fi

# Root endpoint
((TOTAL_CHECKS++))
if check_json_endpoint "/" "Root endpoint"; then
    ((PASSED_CHECKS++))
fi

# API documentation
((TOTAL_CHECKS++))
if check_endpoint "/docs" "API documentation" "200"; then
    ((PASSED_CHECKS++))
fi

echo ""
echo "2️⃣ Monitoring Endpoints"
echo "======================="

# Monitoring dashboard
((TOTAL_CHECKS++))
if check_json_endpoint "/api/v1/monitoring/dashboard" "Monitoring dashboard"; then
    ((PASSED_CHECKS++))
    
    # Extract and display summary
    curl -s "$BASE_URL/api/v1/monitoring/dashboard" | jq -r '
        "   Status: \(.status)
   Agents: \(.summary.agents.healthy)/\(.summary.agents.total) healthy
   Services: \(.summary.services.healthy)/\(.summary.services.total) healthy
   CPU: \(.system.cpu.percent)%
   Memory: \(.system.memory.percent)%"
    ' 2>/dev/null || true
fi

# Metrics endpoint
((TOTAL_CHECKS++))
if check_endpoint "/api/v1/monitoring/metrics" "Prometheus metrics"; then
    ((PASSED_CHECKS++))
fi

# Alerts endpoint
((TOTAL_CHECKS++))
if check_json_endpoint "/api/v1/monitoring/alerts" "System alerts"; then
    ((PASSED_CHECKS++))
    
    # Show any active alerts
    alerts=$(curl -s "$BASE_URL/api/v1/monitoring/alerts" | jq -r '.alert_count // 0' 2>/dev/null || echo "0")
    if [ "$alerts" -gt 0 ]; then
        echo -e "   ${YELLOW}⚠️ Active alerts: $alerts${NC}"
        curl -s "$BASE_URL/api/v1/monitoring/alerts" | jq -r '.alerts[] | "   - \(.severity): \(.message)"' 2>/dev/null || true
    fi
fi

echo ""
echo "3️⃣ Agent Status"
echo "==============="

# Check agent status
((TOTAL_CHECKS++))
if agent_data=$(curl -s "$BASE_URL/api/v1/monitoring/agents/status" 2>/dev/null); then
    ((PASSED_CHECKS++))
    
    total_agents=$(echo "$agent_data" | jq -r '.total_agents // 0')
    healthy_agents=$(echo "$agent_data" | jq -r '[.agents[] | select(.health.status == "healthy")] | length' || echo "0")
    
    echo -e "${GREEN}✅ Agent Status Retrieved${NC}"
    echo "   Total agents: $total_agents"
    echo "   Healthy agents: $healthy_agents"
    
    # Show unhealthy agents if any
    unhealthy=$(echo "$agent_data" | jq -r '.agents[] | select(.health.status != "healthy") | .name' 2>/dev/null || true)
    if [ -n "$unhealthy" ]; then
        echo -e "   ${YELLOW}Unhealthy agents:${NC}"
        echo "$unhealthy" | while read agent; do
            echo "   - $agent"
        done
    fi
else
    echo -e "${RED}❌ Failed to retrieve agent status${NC}"
fi

echo ""
echo "4️⃣ Performance Check"
echo "===================="

# Measure response time
echo -n "Checking response time... "
response_time=$(curl -s -o /dev/null -w "%{time_total}" "$BASE_URL/health" || echo "N/A")

if [ "$response_time" != "N/A" ]; then
    time_ms=$(echo "$response_time * 1000" | bc 2>/dev/null || echo "N/A")
    if [ "$time_ms" != "N/A" ]; then
        echo -e "${GREEN}✅ ${time_ms}ms${NC}"
    else
        echo -e "${GREEN}✅ ${response_time}s${NC}"
    fi
else
    echo -e "${RED}❌ Failed${NC}"
fi

echo ""
echo "📊 Summary"
echo "=========="
echo "Passed: $PASSED_CHECKS/$TOTAL_CHECKS checks"

if [ $PASSED_CHECKS -eq $TOTAL_CHECKS ]; then
    echo -e "${GREEN}✅ All checks passed! Deployment is healthy.${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Some checks failed. Review the output above.${NC}"
    exit 1
fi