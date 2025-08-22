#!/bin/bash

# ==============================================================================
# A2A System Status Script  
# Quick status check for all A2A services
# ==============================================================================

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Service port definitions
AGENT_PORTS="8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012 8013 8014 8015 8888"
CORE_PORTS="4004 8080 8090 8091 8889 3001 8020"
INFRASTRUCTURE_PORTS="6379 9090 8545"
MCP_PORTS="8100 8101 8102 8103 8104 8105 8106 8107 8108 8109"

# Service name lookup function
get_service_name() {
    local port=$1
    case $port in
        8001) echo "Agent 0 - Data Product" ;;
        8002) echo "Agent 1 - Standardization" ;;
        8003) echo "Agent 2 - AI Preparation" ;;
        8004) echo "Agent 3 - Vector Processing" ;;
        8005) echo "Agent 4 - Calc Validation" ;;
        8006) echo "Agent 5 - QA Validation" ;;
        8007) echo "Agent 6 - Quality Control" ;;
        8008) echo "Reasoning Agent" ;;
        8009) echo "SQL Agent" ;;
        8010) echo "Agent Manager" ;;
        8011) echo "Data Manager" ;;
        8012) echo "Catalog Manager" ;;
        8013) echo "Calculation Agent" ;;
        8014) echo "Agent Builder" ;;
        8015) echo "Embedding Fine-tuner" ;;
        8888) echo "Agents Service (Unified)" ;;
        4004) echo "CAP/CDS Network" ;;
        8080) echo "API Gateway" ;;
        8090) echo "A2A Registry" ;;
        8091) echo "ORD Registry" ;;
        8889) echo "Health Dashboard" ;;
        3001) echo "Developer Portal" ;;
        8020) echo "Trust System" ;;
        6379) echo "Redis Cache" ;;
        9090) echo "Prometheus Monitoring" ;;
        8545) echo "Blockchain (Anvil)" ;;
        8100) echo "Enhanced Test Suite" ;;
        8101) echo "Data Standardization" ;;
        8102) echo "Vector Similarity" ;;
        8103) echo "Vector Ranking" ;;
        8104) echo "Transport Layer" ;;
        8105) echo "Reasoning Agent MCP" ;;
        8106) echo "Session Management" ;;
        8107) echo "Resource Streaming" ;;
        8108) echo "Confidence Calculator" ;;
        8109) echo "Semantic Similarity" ;;
        *) echo "Service on port $port" ;;
    esac
}

# Check if service is running on port
is_port_active() {
    local port=$1
    lsof -ti :$port >/dev/null 2>&1
}

# Check service group status
check_service_group() {
    local ports="$1"
    local group_name="$2"
    local running=0
    local total=0
    
    echo -e "\n${BOLD}$group_name:${NC}"
    for port in $ports; do
        total=$((total + 1))
        local service_name=$(get_service_name $port)
        if is_port_active $port; then
            echo -e "  ${GREEN}✓${NC} $service_name (port $port)"
            running=$((running + 1))
        else
            echo -e "  ${RED}✗${NC} $service_name (port $port)"
        fi
    done
    
    local percentage=$((running * 100 / total))
    if [ $running -eq $total ]; then
        echo -e "  ${GREEN}Status: $running/$total running ($percentage%)${NC}"
    elif [ $running -eq 0 ]; then
        echo -e "  ${RED}Status: $running/$total running ($percentage%)${NC}"
    else
        echo -e "  ${YELLOW}Status: $running/$total running ($percentage%)${NC}"
    fi
    
    return $running
}

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                              A2A SYSTEM                                     ║"
    echo "║                         STATUS OVERVIEW                                     ║"
    echo "║                                                                              ║"
    echo "║  📊 Service Status • 🔍 Health Check • ⚡ Quick Overview                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Quick summary
show_summary() {
    local agent_count=0
    local core_count=0
    local infra_count=0
    local mcp_count=0
    local total_count=0
    
    for port in $AGENT_PORTS; do
        total_count=$((total_count + 1))
        is_port_active $port && agent_count=$((agent_count + 1))
    done
    
    for port in $CORE_PORTS; do
        total_count=$((total_count + 1))
        is_port_active $port && core_count=$((core_count + 1))
    done
    
    for port in $INFRASTRUCTURE_PORTS; do
        total_count=$((total_count + 1))
        is_port_active $port && infra_count=$((infra_count + 1))
    done
    
    for port in $MCP_PORTS; do
        total_count=$((total_count + 1))
        is_port_active $port && mcp_count=$((mcp_count + 1))
    done
    
    local running_count=$((agent_count + core_count + infra_count + mcp_count))
    local overall_percentage=$((running_count * 100 / total_count))
    
    echo -e "${BOLD}System Overview:${NC}"
    echo -e "  Agents: ${agent_count}/16 running"
    echo -e "  Core Services: ${core_count}/7 running"
    echo -e "  Infrastructure: ${infra_count}/3 running" 
    echo -e "  MCP Servers: ${mcp_count}/10 running"
    echo -e "  ${BOLD}Total: ${running_count}/${total_count} services running (${overall_percentage}%)${NC}"
    
    if [ $overall_percentage -ge 80 ]; then
        echo -e "  ${GREEN}System Status: OPERATIONAL${NC}"
    elif [ $overall_percentage -ge 50 ]; then
        echo -e "  ${YELLOW}System Status: PARTIAL${NC}"
    else
        echo -e "  ${RED}System Status: DEGRADED${NC}"
    fi
}

# Health check for running services
health_check() {
    echo -e "\n${BOLD}Health Check (Running Services Only):${NC}"
    local healthy_count=0
    local unhealthy_count=0
    
    # Check agents with health endpoints
    for port in 8003; do # Only check known working agent
        if is_port_active $port; then
            local service_name=$(get_service_name $port)
            if curl -s --max-time 2 "http://localhost:$port/health" >/dev/null 2>&1; then
                echo -e "  ${GREEN}✓${NC} $service_name - HEALTHY"
                healthy_count=$((healthy_count + 1))
            else
                echo -e "  ${YELLOW}⚠${NC} $service_name - RUNNING (no health endpoint)"
            fi
        fi
    done
    
    # Check MCP servers
    for port in $MCP_PORTS; do
        if is_port_active $port; then
            local service_name=$(get_service_name $port)
            if curl -s --max-time 2 "http://localhost:$port/health" >/dev/null 2>&1; then
                echo -e "  ${GREEN}✓${NC} $service_name - HEALTHY"
                healthy_count=$((healthy_count + 1))
            else
                echo -e "  ${YELLOW}⚠${NC} $service_name - RUNNING (no health endpoint)"
            fi
        fi
    done
    
    # Check other services with health endpoints
    for port in 8889; do # Health Dashboard
        if is_port_active $port; then
            local service_name=$(get_service_name $port)
            if curl -s --max-time 2 "http://localhost:$port/health" >/dev/null 2>&1; then
                echo -e "  ${GREEN}✓${NC} $service_name - HEALTHY"
                healthy_count=$((healthy_count + 1))
            else
                echo -e "  ${RED}✗${NC} $service_name - UNHEALTHY"
                unhealthy_count=$((unhealthy_count + 1))
            fi
        fi
    done
    
    if [ $healthy_count -gt 0 ]; then
        echo -e "  ${GREEN}Health Summary: $healthy_count services responding to health checks${NC}"
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --summary   Show only summary (default)"
    echo "  --detailed  Show detailed service status"
    echo "  --health    Include health checks for running services"
    echo "  --json      Output in JSON format"
    echo "  --help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0              # Quick summary"
    echo "  $0 --detailed   # Detailed service list"
    echo "  $0 --health     # Include health checks"
    echo "  $0 --json       # JSON output for scripts"
}

# JSON output
json_output() {
    local agent_list=""
    local core_list=""
    local infra_list=""
    local mcp_list=""
    
    # Build JSON arrays
    for port in $AGENT_PORTS; do
        local status=$(is_port_active $port && echo "running" || echo "stopped")
        local name=$(get_service_name $port)
        agent_list="$agent_list{\"name\":\"$name\",\"port\":$port,\"status\":\"$status\"},"
    done
    agent_list=${agent_list%,} # Remove trailing comma
    
    for port in $CORE_PORTS; do
        local status=$(is_port_active $port && echo "running" || echo "stopped")
        local name=$(get_service_name $port)
        core_list="$core_list{\"name\":\"$name\",\"port\":$port,\"status\":\"$status\"},"
    done
    core_list=${core_list%,}
    
    for port in $INFRASTRUCTURE_PORTS; do
        local status=$(is_port_active $port && echo "running" || echo "stopped")
        local name=$(get_service_name $port)
        infra_list="$infra_list{\"name\":\"$name\",\"port\":$port,\"status\":\"$status\"},"
    done
    infra_list=${infra_list%,}
    
    for port in $MCP_PORTS; do
        local status=$(is_port_active $port && echo "running" || echo "stopped")
        local name=$(get_service_name $port)
        mcp_list="$mcp_list{\"name\":\"$name\",\"port\":$port,\"status\":\"$status\"},"
    done
    mcp_list=${mcp_list%,}
    
    echo "{"
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"agents\": [$agent_list],"
    echo "  \"core\": [$core_list],"
    echo "  \"infrastructure\": [$infra_list],"
    echo "  \"mcp\": [$mcp_list]"
    echo "}"
}

# Parse command line arguments
MODE="summary"
INCLUDE_HEALTH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --summary)
            MODE="summary"
            shift
            ;;
        --detailed)
            MODE="detailed"
            shift
            ;;
        --health)
            INCLUDE_HEALTH=true
            shift
            ;;
        --json)
            json_output
            exit 0
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
show_banner
show_summary

if [ "$MODE" = "detailed" ]; then
    check_service_group "$AGENT_PORTS" "A2A Agents"
    check_service_group "$CORE_PORTS" "Core Services"
    check_service_group "$INFRASTRUCTURE_PORTS" "Infrastructure"
    check_service_group "$MCP_PORTS" "MCP Servers"
fi

if [ "$INCLUDE_HEALTH" = true ]; then
    health_check
fi

echo ""
echo -e "${BLUE}💡 Tip: Use './start.sh complete' to start all services or './stop.sh' to stop them${NC}"