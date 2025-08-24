#!/bin/bash

# ==============================================================================
# A2A Unified System Stop Script
# Comprehensive shutdown for Network, Agents, Blockchain and all services
# ==============================================================================

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly GRAY='\033[0;37m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="$SCRIPT_DIR/logs"
readonly PID_DIR="$SCRIPT_DIR/pids"

# Create required directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# Progress tracking variables
TOTAL_STEPS=10
CURRENT_STEP=0
START_TIME=$(date +%s)

# Service port definitions (using simple arrays)
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

# Enhanced logging functions with progress tracking
log_progress() {
    local step_name="$1"
    local step_detail="${2:-}"
    
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local progress=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    local elapsed=$(($(date +%s) - START_TIME))
    
    printf "\n${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${CYAN}║${NC} ${BOLD}STEP $CURRENT_STEP/$TOTAL_STEPS [$progress%%] - $step_name${NC}${CYAN}%*s║${NC}\n" $((78 - ${#step_name} - ${#CURRENT_STEP} - ${#TOTAL_STEPS} - 10)) ""
    if [ -n "$step_detail" ]; then
        printf "${CYAN}║${NC} $step_detail%*s${CYAN}║${NC}\n" $((78 - ${#step_detail})) ""
    fi
    printf "${CYAN}║${NC} Elapsed: ${elapsed}s%*s${CYAN}║${NC}\n" $((69 - ${#elapsed})) ""
    printf "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}\n"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') [STOP-PROGRESS] Step $CURRENT_STEP/$TOTAL_STEPS ($progress%) - $step_name" >> "$LOG_DIR/stop.log"
    if [ -n "$step_detail" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [DETAIL] $step_detail" >> "$LOG_DIR/stop.log"
    fi
}

log_substep() {
    local substep_name="$1"
    local detail="${2:-}"
    
    echo -e "  ${PURPLE}▶${NC} $substep_name" | tee -a "$LOG_DIR/stop.log"
    if [ -n "$detail" ]; then
        echo -e "    ${GRAY}$detail${NC}" | tee -a "$LOG_DIR/stop.log"
    fi
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/stop.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/stop.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/stop.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/stop.log"
}

# Check if service is running on port
is_port_active() {
    local port=$1
    lsof -ti :$port >/dev/null 2>&1
}

# Get PID for port
get_port_pid() {
    local port=$1
    lsof -ti :$port 2>/dev/null || echo ""
}

# Stop service by port with graceful then forceful termination
stop_service_by_port() {
    local port=$1
    local service_name="${2:-$(get_service_name $port)}"
    local timeout=${3:-10}
    
    if ! is_port_active $port; then
        log_info "$service_name already stopped"
        return 0
    fi
    
    local pid=$(get_port_pid $port)
    if [ -z "$pid" ]; then
        log_warning "No PID found for $service_name on port $port"
        return 0
    fi
    
    log_info "Stopping $service_name (PID: $pid, Port: $port)"
    
    # Try graceful shutdown first
    if kill -TERM $pid 2>/dev/null; then
        local count=0
        while [ $count -lt $timeout ] && kill -0 $pid 2>/dev/null; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            log_warning "Graceful shutdown timeout, force killing $service_name"
            kill -KILL $pid 2>/dev/null
            sleep 1
        fi
    fi
    
    # Verify stopped
    if is_port_active $port; then
        log_error "Failed to stop $service_name on port $port"
        return 1
    else
        log_success "$service_name stopped successfully"
        return 0
    fi
}

# Stop services by process name pattern
stop_services_by_pattern() {
    local pattern="$1"
    local service_type="$2"
    
    log_substep "Stopping $service_type processes"
    
    local pids=$(pgrep -f "$pattern" 2>/dev/null || echo "")
    if [ -z "$pids" ]; then
        log_info "No $service_type processes found"
        return 0
    fi
    
    local stopped_count=0
    for pid in $pids; do
        local cmd=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
        log_info "Stopping $service_type process: $cmd (PID: $pid)"
        
        # Try graceful then force
        if kill -TERM $pid 2>/dev/null; then
            sleep 2
            if kill -0 $pid 2>/dev/null; then
                kill -KILL $pid 2>/dev/null
            fi
            stopped_count=$((stopped_count + 1))
        fi
    done
    
    log_success "Stopped $stopped_count $service_type processes"
}

# Stop port group with progress reporting
stop_port_group() {
    local ports="$1"
    local group_name="$2"
    local timeout=${3:-10}
    
    log_substep "Stopping $group_name services"
    
    local total_ports=$(echo $ports | wc -w)
    local stopped_count=0
    local failed_count=0
    
    for port in $ports; do
        local service_name=$(get_service_name $port)
        if stop_service_by_port $port "$service_name" $timeout; then
            stopped_count=$((stopped_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
    done
    
    if [ $failed_count -eq 0 ]; then
        log_success "All $group_name services stopped ($stopped_count/$total_ports)"
    else
        log_warning "$group_name services: $stopped_count stopped, $failed_count failed"
    fi
}

# Cleanup functions
cleanup_pid_files() {
    log_substep "Cleaning up PID files"
    if [ -d "$PID_DIR" ]; then
        local count=$(find "$PID_DIR" -name "*.pid" 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            rm -f "$PID_DIR"/*.pid 2>/dev/null
            log_success "Removed $count PID files"
        else
            log_info "No PID files to clean"
        fi
    fi
}

cleanup_temp_files() {
    log_substep "Cleaning up temporary files"
    local cleaned_count=0
    
    for pattern in "/tmp/a2a_*" "/tmp/anvil_*" "/tmp/redis_*" "/tmp/prometheus_*"; do
        if ls $pattern >/dev/null 2>&1; then
            rm -rf $pattern 2>/dev/null
            cleaned_count=$((cleaned_count + 1))
        fi
    done
    
    log_success "Cleaned $cleaned_count temporary file groups"
}

clear_old_logs() {
    log_substep "Archiving current logs"
    if [ "$1" = "--clear-logs" ]; then
        if [ -d "$LOG_DIR" ]; then
            local timestamp=$(date +%Y%m%d_%H%M%S)
            local archive_dir="$LOG_DIR/archive_$timestamp"
            mkdir -p "$archive_dir"
            
            find "$LOG_DIR" -maxdepth 1 -name "*.log" -exec mv {} "$archive_dir/" \; 2>/dev/null
            log_success "Archived logs to $archive_dir"
        fi
    else
        log_info "Logs preserved (use --clear-logs to archive)"
    fi
}

# System resource cleanup
cleanup_system_resources() {
    log_substep "Cleaning up system resources"
    
    # Clear shared memory segments (if available)
    if command -v ipcs >/dev/null 2>&1; then
        local shm_count=$(ipcs -m 2>/dev/null | grep $(whoami) | wc -l || echo 0)
        if [ $shm_count -gt 0 ]; then
            ipcs -m | grep $(whoami) | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null
            log_success "Cleaned $shm_count shared memory segments"
        fi
    fi
    
    log_info "System resource cleanup completed"
}

# Show usage
show_usage() {
    echo -e "${CYAN}"
    echo "A2A Unified System Stop Script"
    echo "============================="
    echo -e "${NC}"
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  all         - Stop all A2A services (default)"
    echo "  agents      - Stop agents only"
    echo "  core        - Stop core services only" 
    echo "  mcp         - Stop MCP servers only"
    echo "  infrastructure - Stop infrastructure only"
    echo "  blockchain  - Stop blockchain only"
    echo "  network     - Stop network services only"
    echo ""
    echo "Options:"
    echo "  --force     Fast forceful shutdown (SIGKILL)"
    echo "  --graceful  Extended graceful shutdown timeout"
    echo "  --clear-logs Archive current logs"
    echo "  --dry-run   Show what would be stopped without stopping"
    echo "  --help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                    # Stop all services"
    echo "  $0 agents             # Stop agents only"
    echo "  $0 all --clear-logs   # Stop all and archive logs"
    echo "  $0 --dry-run          # Preview stop operations"
}

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                              A2A SYSTEM                                     ║"
    echo "║                        SHUTDOWN SEQUENCE                                    ║"
    echo "║                                                                              ║"
    echo "║  🛑 Safe Shutdown • 🔄 Resource Cleanup • 📁 Log Management               ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Dry run mode
dry_run_report() {
    echo -e "${YELLOW}[DRY RUN]${NC} The following services would be stopped:"
    
    echo -e "\n${BOLD}A2A Agents:${NC}"
    for port in $AGENT_PORTS; do
        if is_port_active $port; then
            echo "  ✓ $(get_service_name $port) (port $port)"
        fi
    done
    
    echo -e "\n${BOLD}Core Services:${NC}"
    for port in $CORE_PORTS; do
        if is_port_active $port; then
            echo "  ✓ $(get_service_name $port) (port $port)"
        fi
    done
    
    echo -e "\n${BOLD}Infrastructure:${NC}"
    for port in $INFRASTRUCTURE_PORTS; do
        if is_port_active $port; then
            echo "  ✓ $(get_service_name $port) (port $port)"
        fi
    done
    
    echo -e "\n${BOLD}MCP Servers:${NC}"
    for port in $MCP_PORTS; do
        if is_port_active $port; then
            echo "  ✓ $(get_service_name $port) (port $port)"
        fi
    done
    
    exit 0
}

# Main execution functions
stop_all_services() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [STOP] Beginning A2A System shutdown in $MODE mode" >> "$LOG_DIR/stop.log"
    
    log_progress "Pre-shutdown Assessment" "Identifying running services and resources"
    log_info "Assessing running A2A services..."
    
    # Count active services
    local agent_count=0
    local core_count=0  
    local infra_count=0
    local mcp_count=0
    
    for port in $AGENT_PORTS; do
        is_port_active $port && agent_count=$((agent_count + 1))
    done
    for port in $CORE_PORTS; do
        is_port_active $port && core_count=$((core_count + 1))
    done
    for port in $INFRASTRUCTURE_PORTS; do
        is_port_active $port && infra_count=$((infra_count + 1))
    done
    for port in $MCP_PORTS; do
        is_port_active $port && mcp_count=$((mcp_count + 1))
    done
    
    log_info "Found: $agent_count agents, $core_count core services, $infra_count infrastructure, $mcp_count MCP servers"
    
    log_progress "Agent Services Shutdown" "Stopping all A2A agent processes"
    stop_port_group "$AGENT_PORTS" "A2A Agent" $SHUTDOWN_TIMEOUT
    
    log_progress "MCP Servers Shutdown" "Stopping Model Context Protocol servers"
    stop_port_group "$MCP_PORTS" "MCP Server" $SHUTDOWN_TIMEOUT
    
    log_progress "Core Services Shutdown" "Stopping registries, gateways, and web services"
    stop_port_group "$CORE_PORTS" "Core Service" $SHUTDOWN_TIMEOUT
    
    log_progress "Infrastructure Shutdown" "Stopping Redis, Prometheus, and blockchain"
    stop_port_group "$INFRASTRUCTURE_PORTS" "Infrastructure" $SHUTDOWN_TIMEOUT
    
    log_progress "Process Pattern Cleanup" "Stopping remaining processes by pattern"
    stop_services_by_pattern "anvil" "Blockchain"
    stop_services_by_pattern "prometheus" "Monitoring" 
    stop_services_by_pattern "redis-server" "Cache"
    stop_services_by_pattern "uvicorn.*main" "FastAPI"
    stop_services_by_pattern "node.*server" "Node.js"
    stop_services_by_pattern "python.*mcp" "MCP"
    
    log_progress "File System Cleanup" "Cleaning PID files and temporary data"
    cleanup_pid_files
    cleanup_temp_files
    clear_old_logs $CLEAR_LOGS_FLAG
    
    log_progress "System Resource Cleanup" "Freeing shared memory and message queues"
    cleanup_system_resources
    
    log_progress "Verification" "Confirming all services stopped"
    local remaining_count=0
    for port in $AGENT_PORTS $CORE_PORTS $INFRASTRUCTURE_PORTS $MCP_PORTS; do
        if is_port_active $port; then
            remaining_count=$((remaining_count + 1))
            log_warning "Port $port still active"
        fi
    done
    
    if [ $remaining_count -eq 0 ]; then
        log_success "All A2A services stopped successfully"
    else
        log_warning "$remaining_count services may still be running"
    fi
    
    log_progress "Shutdown Complete" "A2A System shutdown finished"
    local total_time=$(($(date +%s) - START_TIME))
    log_success "Total shutdown time: ${total_time}s"
}

# Selective stop functions
stop_agents_only() {
    log_progress "Agent Services Only" "Stopping A2A agents and unified service"
    stop_port_group "$AGENT_PORTS" "A2A Agent" $SHUTDOWN_TIMEOUT
    stop_services_by_pattern "uvicorn.*main" "Agent Service"
}

stop_core_only() {
    log_progress "Core Services Only" "Stopping registries, gateways, and web services"
    stop_port_group "$CORE_PORTS" "Core Service" $SHUTDOWN_TIMEOUT
}

stop_mcp_only() {
    log_progress "MCP Servers Only" "Stopping Model Context Protocol servers"
    stop_port_group "$MCP_PORTS" "MCP Server" $SHUTDOWN_TIMEOUT
    stop_services_by_pattern "python.*mcp" "MCP"
}

stop_infrastructure_only() {
    log_progress "Infrastructure Only" "Stopping Redis, Prometheus, and blockchain"
    stop_port_group "$INFRASTRUCTURE_PORTS" "Infrastructure" $SHUTDOWN_TIMEOUT
    stop_services_by_pattern "anvil" "Blockchain"
    stop_services_by_pattern "prometheus" "Monitoring"
    stop_services_by_pattern "redis-server" "Cache"
}

stop_blockchain_only() {
    log_progress "Blockchain Only" "Stopping Anvil blockchain"
    stop_service_by_port 8545 "Blockchain (Anvil)" $SHUTDOWN_TIMEOUT
    stop_services_by_pattern "anvil" "Blockchain"
}

stop_network_only() {
    log_progress "Network Services Only" "Stopping CAP/CDS network service"
    stop_service_by_port 4004 "CAP/CDS Network" $SHUTDOWN_TIMEOUT
    stop_services_by_pattern "node.*server" "Network Service"
}

# Parse command line arguments
MODE="all"
SHUTDOWN_TIMEOUT=10
CLEAR_LOGS_FLAG=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        all|agents|core|mcp|infrastructure|blockchain|network)
            MODE="$1"
            shift
            ;;
        --force)
            SHUTDOWN_TIMEOUT=2
            log_info "Fast forceful shutdown mode enabled"
            shift
            ;;
        --graceful)
            SHUTDOWN_TIMEOUT=30
            log_info "Extended graceful shutdown mode enabled"
            shift
            ;;
        --clear-logs)
            CLEAR_LOGS_FLAG="--clear-logs"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
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

if [ "$DRY_RUN" = true ]; then
    dry_run_report
fi

log_info "A2A System Stop Script - Mode: $MODE"

case $MODE in
    "all")
        stop_all_services
        ;;
    "agents")
        stop_agents_only
        ;;
    "core")
        stop_core_only
        ;;
    "mcp")
        stop_mcp_only
        ;;
    "infrastructure")
        stop_infrastructure_only
        ;;
    "blockchain")
        stop_blockchain_only
        ;;
    "network")
        stop_network_only
        ;;
    *)
        log_error "Unknown mode: $MODE"
        show_usage
        exit 1
        ;;
esac

echo ""
log_success "🎯 A2A System stop script completed successfully"
echo -e "${GREEN}✅ All requested services have been stopped${NC}"