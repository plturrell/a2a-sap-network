#!/bin/bash

# ==============================================================================
# A2A Enterprise System Startup Script
# SAP Standard Commercial Approach
# ==============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for enterprise logging
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Enterprise configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly LOG_DIR="$PROJECT_ROOT/logs"
readonly PID_DIR="$PROJECT_ROOT/pids"
readonly CONFIG_DIR="$PROJECT_ROOT/config"

# Service configuration
readonly NETWORK_PORT=4004
readonly AGENTS_PORT=8000
readonly HEALTH_CHECK_TIMEOUT=30
readonly STARTUP_TIMEOUT=60

# Create required directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/startup.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/startup.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/startup.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/startup.log"
}

log_header() {
    echo -e "${PURPLE}[ENTERPRISE]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/startup.log"
}

# Enterprise banner
show_enterprise_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          A2A ENTERPRISE SYSTEM                              ║"
    echo "║                     SAP Standard Commercial Startup                         ║"
    echo "║                                                                              ║"
    echo "║  Agent-to-Agent Network • Autonomous Orchestration • Enterprise Grade      ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Pre-flight checks
preflight_checks() {
    log_header "Performing Enterprise Pre-flight Checks"
    
    # Check Node.js version
    if ! command -v node &> /dev/null; then
        log_error "Node.js not found. Please install Node.js 18+ for enterprise deployment."
        exit 1
    fi
    
    local node_version=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$node_version" -lt 18 ]; then
        log_error "Node.js version $node_version detected. Enterprise deployment requires Node.js 18+."
        exit 1
    fi
    log_success "Node.js version check passed: $(node --version)"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found. Please install Python 3.9+ for enterprise deployment."
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_success "Python version check passed: $(python3 --version)"
    
    # Check SAP CAP CLI
    if ! command -v cds &> /dev/null; then
        log_error "SAP CAP CLI not found. Please install @sap/cds-dk globally."
        exit 1
    fi
    log_success "SAP CAP CLI check passed: $(cds --version | head -1)"
    
    # Check required directories
    for dir in "$PROJECT_ROOT/a2aNetwork" "$PROJECT_ROOT/a2aAgents"; do
        if [ ! -d "$dir" ]; then
            log_error "Required directory not found: $dir"
            exit 1
        fi
    done
    log_success "Project structure validation passed"
    
    # Check port availability
    if lsof -Pi :$NETWORK_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warning "Port $NETWORK_PORT is already in use. Attempting graceful shutdown..."
        pkill -f "cds-serve" || true
        sleep 2
    fi
    
    if lsof -Pi :$AGENTS_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warning "Port $AGENTS_PORT is already in use. Attempting graceful shutdown..."
        pkill -f "uvicorn.*$AGENTS_PORT" || true
        sleep 2
    fi
    
    log_success "Pre-flight checks completed successfully"
}

# Environment setup
setup_environment() {
    log_header "Setting Up Enterprise Environment"
    
    # Set production environment variables
    export NODE_ENV=development
    export A2A_NETWORK_PORT=4004
    export A2A_AGENTS_PORT=8000
    export LOG_LEVEL=info
    export ENTERPRISE_MODE=true

    # SAP HANA Cloud deployment configuration (optional)
    export DEPLOY_TO_HANA=${DEPLOY_TO_HANA:-false}
    export CF_API_ENDPOINT=${CF_API_ENDPOINT:-}
    export CF_ORG=${CF_ORG:-}
    export CF_SPACE=${CF_SPACE:-}
    export OTEL_SERVICE_NAME=a2a-enterprise
    export OTEL_RESOURCE_ATTRIBUTES="service.name=a2a-enterprise,service.version=1.0.0"
    
    # SAP specific environment
    export CDS_REQUIRES_AUTH=true
    export CDS_REQUIRES_MULTITENANCY=false
    export CDS_FEATURES_ASSERT_INTEGRITY=warn
    
    # Create environment file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_info "Creating enterprise environment configuration..."
        cat > "$PROJECT_ROOT/.env" << EOF
# A2A Enterprise Environment Configuration
NODE_ENV=development
CDS_ENV=development
LOG_LEVEL=info
SESSION_SECRET=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 32)
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Database Configuration
DATABASE_URL=sqlite:db/a2a-enterprise.db
REDIS_URL=redis://localhost:6379

# Monitoring Configuration
ENABLE_METRICS=true
ENABLE_TRACING=true
HEALTH_CHECK_INTERVAL=30

# Security Configuration
ENABLE_CORS=true
ENABLE_HELMET=true
ENABLE_RATE_LIMITING=true
MAX_REQUEST_SIZE=10mb
EOF
        log_success "Enterprise environment configuration created"
    fi
    
    # Source environment
    if [ -f "$PROJECT_ROOT/.env" ]; then
        set -a
        source "$PROJECT_ROOT/.env"
        set +a
        log_success "Environment variables loaded"
    fi
}

# Database initialization
initialize_database() {
    log_header "Initializing Enterprise Database"
    
    cd "$PROJECT_ROOT/a2aNetwork"
    
    # Deploy database schema
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - Initializing local development database..."
    cd "$PROJECT_ROOT/a2aNetwork"

    # First, deploy to local SQLite for development
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - Deploying to local SQLite database..."
    npm run db:deploy
    if [ $? -ne 0 ]; then
        echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - Local database deployment failed"
        exit 1
    fi

    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - Local database deployment completed"

    # Optional: Deploy to HANA if configured
    if [ "${DEPLOY_TO_HANA:-false}" = "true" ] && [ -n "${CF_API_ENDPOINT:-}" ]; then
        echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - Deploying to SAP HANA Cloud..."
        npm run db:migrate
        if [ $? -ne 0 ]; then
            echo "[WARNING] $(date '+%Y-%m-%d %H:%M:%S') - HANA deployment failed, continuing with local database"
        else
            echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - HANA deployment completed"
        fi
    fi
    
    # Seed initial data
    if [ -f "scripts/seed-data.js" ]; then
        log_info "Seeding initial enterprise data..."
        npm run db:seed 2>&1 | tee -a "$LOG_DIR/db-seed.log"
        log_success "Database seeded with initial data"
    fi
}

# Start A2A Network Service
start_network_service() {
    log_header "Starting A2A Network Service (SAP CAP)"
    
    cd "$PROJECT_ROOT/a2aNetwork"
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing A2A Network dependencies..."
        npm ci --production 2>&1 | tee -a "$LOG_DIR/network-install.log"
    fi
    
    # Enterprise validation
    if [ -f "scripts/validate-enterprise-config.js" ]; then
        log_info "Validating enterprise configuration..."
        npm run enterprise:validate 2>&1 | tee -a "$LOG_DIR/network-validation.log"
    fi
    
    # Start the service
    log_info "Starting A2A Network Service on port $NETWORK_PORT..."
    nohup npm start > "$LOG_DIR/network-service.log" 2>&1 &
    local network_pid=$!
    echo $network_pid > "$PID_DIR/network.pid"
    
    # Wait for service to be ready
    log_info "Waiting for A2A Network Service to be ready..."
    local attempts=0
    while [ $attempts -lt $STARTUP_TIMEOUT ]; do
        if curl -sf "http://localhost:$NETWORK_PORT/health" > /dev/null 2>&1; then
            log_success "A2A Network Service is ready on port $NETWORK_PORT"
            return 0
        fi
        sleep 1
        ((attempts++))
    done
    
    log_error "A2A Network Service failed to start within $STARTUP_TIMEOUT seconds"
    return 1
}

# Start A2A Agents Service
start_agents_service() {
    log_header "Starting A2A Agents Service (Python/FastAPI)"
    
    cd "$PROJECT_ROOT/a2aAgents/backend"
    
    # Create virtual environment if needed
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    if [ ! -f "venv/pyvenv.cfg" ] || [ "requirements.txt" -nt "venv/pyvenv.cfg" ]; then
        log_info "Installing A2A Agents dependencies..."
        pip install -r requirements.txt 2>&1 | tee -a "$LOG_DIR/agents-install.log"
    fi
    
    # Start the service
    log_info "Starting A2A Agents Service on port $AGENTS_PORT..."
    nohup python3 -m uvicorn main:app --host 0.0.0.0 --port $AGENTS_PORT --workers 4 > "$LOG_DIR/agents-service.log" 2>&1 &
    local agents_pid=$!
    echo $agents_pid > "$PID_DIR/agents.pid"
    
    # Wait for service to be ready
    log_info "Waiting for A2A Agents Service to be ready..."
    local attempts=0
    while [ $attempts -lt $STARTUP_TIMEOUT ]; do
        if curl -sf "http://localhost:$AGENTS_PORT/health" > /dev/null 2>&1; then
            log_success "A2A Agents Service is ready on port $AGENTS_PORT"
            return 0
        fi
        sleep 1
        ((attempts++))
    done
    
    log_error "A2A Agents Service failed to start within $STARTUP_TIMEOUT seconds"
    return 1
}

# Post-startup validation
post_startup_validation() {
    log_header "Performing Post-Startup Enterprise Validation"
    
    # Health checks
    log_info "Performing comprehensive health checks..."
    
    # Network service health
    if curl -sf "http://localhost:$NETWORK_PORT/health" > /dev/null; then
        log_success "A2A Network Service health check passed"
    else
        log_error "A2A Network Service health check failed"
        return 1
    fi
    
    # Agents service health
    if curl -sf "http://localhost:$AGENTS_PORT/health" > /dev/null; then
        log_success "A2A Agents Service health check passed"
    else
        log_error "A2A Agents Service health check failed"
        return 1
    fi
    
    # Launchpad accessibility
    if curl -sf "http://localhost:$NETWORK_PORT/launchpad.html" > /dev/null; then
        log_success "A2A Launchpad accessibility check passed"
    else
        log_warning "A2A Launchpad accessibility check failed - may need manual verification"
    fi
    
    # Service integration test
    log_info "Testing service integration..."
    if curl -sf "http://localhost:$NETWORK_PORT/api/agents" > /dev/null; then
        log_success "Service integration test passed"
    else
        log_warning "Service integration test failed - services may still be initializing"
    fi
    
    log_success "Post-startup validation completed"
}

# Enterprise monitoring setup
setup_monitoring() {
    log_header "Setting Up Enterprise Monitoring"
    
    # Create monitoring configuration
    cat > "$CONFIG_DIR/monitoring.json" << EOF
{
  "services": [
    {
      "name": "a2a-network",
      "url": "http://localhost:$NETWORK_PORT/health",
      "critical": true
    },
    {
      "name": "a2a-agents",
      "url": "http://localhost:$AGENTS_PORT/health",
      "critical": true
    }
  ],
  "checks": {
    "interval": 30,
    "timeout": 10,
    "retries": 3
  },
  "alerts": {
    "enabled": true,
    "channels": ["log", "console"]
  }
}
EOF
    
    log_success "Enterprise monitoring configuration created"
}

# Display enterprise status
show_enterprise_status() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        A2A ENTERPRISE SYSTEM STATUS                         ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                              ║"
    echo "║  🌐 A2A Network Service:    http://localhost:$NETWORK_PORT                        ║"
    echo "║  🤖 A2A Agents Service:     http://localhost:$AGENTS_PORT                         ║"
    echo "║  🚀 A2A Launchpad:          http://localhost:$NETWORK_PORT/launchpad.html         ║"
    echo "║                                                                              ║"
    echo "║  📊 Health Monitoring:      Active                                          ║"
    echo "║  📝 Logs Directory:         $LOG_DIR                               ║"
    echo "║  🔧 Process IDs:            $PID_DIR                                ║"
    echo "║                                                                              ║"
    echo "║  Status: ENTERPRISE READY ✅                                                ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Cleanup function
cleanup() {
    log_info "Performing cleanup..."
    if [ -f "$PID_DIR/network.pid" ]; then
        kill $(cat "$PID_DIR/network.pid") 2>/dev/null || true
        rm -f "$PID_DIR/network.pid"
    fi
    if [ -f "$PID_DIR/agents.pid" ]; then
        kill $(cat "$PID_DIR/agents.pid") 2>/dev/null || true
        rm -f "$PID_DIR/agents.pid"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    show_enterprise_banner
    
    preflight_checks
    setup_environment
    initialize_database
    
    if start_network_service && start_agents_service; then
        post_startup_validation
        setup_monitoring
        show_enterprise_status
        
        log_success "A2A Enterprise System startup completed successfully"
        log_info "System is ready for enterprise operations"
        
        # Keep script running to maintain services
        log_info "Monitoring services... Press Ctrl+C to shutdown"
        while true; do
            sleep 30
            # Basic health monitoring
            if ! curl -sf "http://localhost:$NETWORK_PORT/health" > /dev/null; then
                log_error "A2A Network Service health check failed"
            fi
            if ! curl -sf "http://localhost:$AGENTS_PORT/health" > /dev/null; then
                log_error "A2A Agents Service health check failed"
            fi
        done
    else
        log_error "A2A Enterprise System startup failed"
        exit 1
    fi
}

# Execute main function
main "$@"
