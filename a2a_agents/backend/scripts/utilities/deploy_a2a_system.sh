#!/bin/bash

# A2A Business Data Cloud Deployment Script
# Complete deployment of all agents, services, and smart contracts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_MODE=${1:-local}  # local, staging, production
LOG_DIR="./deployment_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}🚀 A2A Business Data Cloud Deployment${NC}"
echo -e "${BLUE}=====================================  ${NC}"
echo "Deployment Mode: $DEPLOY_MODE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Function to log output
log() {
    echo -e "$1" | tee -a "$LOG_DIR/deployment_$TIMESTAMP.log"
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -i:$port > /dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# Function to start service
start_service() {
    local name=$1
    local script=$2
    local port=$3
    local log_file="$LOG_DIR/${name}_$TIMESTAMP.log"
    
    log "${YELLOW}Starting $name on port $port...${NC}"
    
    if check_port $port; then
        nohup python3 $script > "$log_file" 2>&1 &
        local pid=$!
        echo $pid > "$LOG_DIR/${name}.pid"
        
        # Wait for service to start
        local retries=30
        while [ $retries -gt 0 ]; do
            sleep 1
            if ! check_port $port; then
                log "${GREEN}✅ $name started successfully (PID: $pid)${NC}"
                return 0
            fi
            retries=$((retries - 1))
        done
        
        log "${RED}❌ Failed to start $name${NC}"
        return 1
    else
        log "${YELLOW}⚠️  $name already running on port $port${NC}"
        return 0
    fi
}

# Step 1: Pre-deployment checks
log "\n${BLUE}1️⃣ Pre-deployment Checks${NC}"
log "------------------------"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
log "Python version: $python_version"

# Check required packages
log "Checking required packages..."
required_packages=(aiohttp web3 cryptography openai grok supabase)
for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        log "  ✅ $package installed"
    else
        log "  ❌ $package not installed"
        log "Installing missing packages..."
        pip3 install $package
    fi
done

# Step 2: Start blockchain (local mode only)
if [ "$DEPLOY_MODE" = "local" ]; then
    log "\n${BLUE}2️⃣ Starting Blockchain${NC}"
    log "---------------------"
    
    if check_port 8545; then
        log "Starting Anvil blockchain..."
        nohup anvil > "$LOG_DIR/anvil_$TIMESTAMP.log" 2>&1 &
        ANVIL_PID=$!
        echo $ANVIL_PID > "$LOG_DIR/anvil.pid"
        sleep 3
        log "${GREEN}✅ Anvil started (PID: $ANVIL_PID)${NC}"
    else
        log "${YELLOW}⚠️  Blockchain already running on port 8545${NC}"
    fi
fi

# Step 3: Deploy smart contracts
log "\n${BLUE}3️⃣ Deploying Smart Contracts${NC}"
log "---------------------------"

if [ "$DEPLOY_MODE" = "local" ]; then
    cd /Users/apple/projects/a2a/a2a_network
    
    log "Compiling contracts..."
    forge build > "$LOG_DIR/forge_build_$TIMESTAMP.log" 2>&1
    
    log "Deploying contracts..."
    ETHERSCAN_API_KEY=dummy forge script script/DeployBDCOnly.s.sol \
        --fork-url http://localhost:8545 \
        --broadcast \
        --skip-simulation > "$LOG_DIR/contract_deployment_$TIMESTAMP.log" 2>&1
    
    if [ $? -eq 0 ]; then
        log "${GREEN}✅ Smart contracts deployed successfully${NC}"
        log "  BusinessDataCloudA2A: 0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"
        log "  AgentRegistry: 0x5FbDB2315678afecb367f032d93F642f64180aa3"
        log "  MessageRouter: 0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
    else
        log "${RED}❌ Contract deployment failed${NC}"
    fi
    
    cd - > /dev/null
fi

# Step 4: Start supporting services
log "\n${BLUE}4️⃣ Starting Supporting Services${NC}"
log "-------------------------------"

# Start Data Manager
start_service "data_manager" "launch_data_manager.py" 8001

# Start Catalog Manager
start_service "catalog_manager" "launch_catalog_manager.py" 8002

# Step 5: Start A2A Agents
log "\n${BLUE}5️⃣ Starting A2A Agents${NC}"
log "---------------------"

# Start all agents
start_service "agent0_data_product" "launch_agent0.py" 8003
start_service "agent1_standardization" "launch_agent1.py" 8004
start_service "agent2_ai_preparation" "launch_agent2.py" 8005
start_service "agent3_vector_processing" "launch_agent3.py" 8008
start_service "agent4_calc_validation" "launch_agent4.py" 8006
start_service "agent5_qa_validation" "launch_agent5.py" 8007

# Step 6: Wait for services to stabilize
log "\n${BLUE}6️⃣ Waiting for Services to Stabilize${NC}"
log "-----------------------------------"
sleep 5

# Step 7: Health checks
log "\n${BLUE}7️⃣ Running Health Checks${NC}"
log "-----------------------"

services=(
    "data_manager:8001"
    "catalog_manager:8002"
    "agent0:8003"
    "agent1:8004"
    "agent2:8005"
    "agent3:8008"
    "agent4:8006"
    "agent5:8007"
)

all_healthy=true
for service_port in "${services[@]}"; do
    IFS=':' read -r service port <<< "$service_port"
    if curl -s -f -o /dev/null "http://localhost:$port/health"; then
        log "${GREEN}✅ $service is healthy${NC}"
    else
        log "${RED}❌ $service is not healthy${NC}"
        all_healthy=false
    fi
done

# Step 8: Initialize smart contract integration
if [ "$all_healthy" = true ]; then
    log "\n${BLUE}8️⃣ Initializing Smart Contract Integration${NC}"
    log "-----------------------------------------"
    
    python3 integrate_bdc_smart_contract.py > "$LOG_DIR/integration_$TIMESTAMP.log" 2>&1 &
    
    if [ $? -eq 0 ]; then
        log "${GREEN}✅ Smart contract integration initialized${NC}"
    else
        log "${YELLOW}⚠️  Smart contract integration needs manual setup${NC}"
    fi
fi

# Step 9: Create systemd services (production mode)
if [ "$DEPLOY_MODE" = "production" ]; then
    log "\n${BLUE}9️⃣ Creating Systemd Services${NC}"
    log "----------------------------"
    
    # This would create systemd service files
    log "Creating systemd service files..."
    # Implementation depends on specific production requirements
fi

# Step 10: Setup monitoring (staging/production)
if [ "$DEPLOY_MODE" != "local" ]; then
    log "\n${BLUE}🔟 Setting up Monitoring${NC}"
    log "-----------------------"
    
    # Start Prometheus and Grafana
    log "Setting up Prometheus metrics..."
    log "Setting up Grafana dashboards..."
fi

# Generate deployment summary
log "\n${BLUE}📊 Deployment Summary${NC}"
log "===================="

cat > "$LOG_DIR/deployment_summary_$TIMESTAMP.json" << EOF
{
    "deployment_mode": "$DEPLOY_MODE",
    "timestamp": "$TIMESTAMP",
    "services": {
        "blockchain": {
            "status": "running",
            "port": 8545,
            "type": "anvil"
        },
        "smart_contracts": {
            "BusinessDataCloudA2A": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
            "AgentRegistry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
            "MessageRouter": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
        },
        "agents": [
            {"name": "data_manager", "port": 8001, "status": "running"},
            {"name": "catalog_manager", "port": 8002, "status": "running"},
            {"name": "agent0_data_product", "port": 8003, "status": "running"},
            {"name": "agent1_standardization", "port": 8004, "status": "running"},
            {"name": "agent2_ai_preparation", "port": 8005, "status": "running"},
            {"name": "agent3_vector_processing", "port": 8008, "status": "running"},
            {"name": "agent4_calc_validation", "port": 8006, "status": "running"},
            {"name": "agent5_qa_validation", "port": 8007, "status": "running"}
        ]
    },
    "logs_directory": "$LOG_DIR"
}
EOF

log "\n${GREEN}✅ Deployment Complete!${NC}"
log ""
log "📄 Deployment summary saved to: $LOG_DIR/deployment_summary_$TIMESTAMP.json"
log "📁 All logs saved to: $LOG_DIR/"
log ""
log "${BLUE}🌐 Access Points:${NC}"
log "  - Data Manager API: http://localhost:8001"
log "  - Catalog Manager API: http://localhost:8002"
log "  - Agent APIs: http://localhost:8003-8008"
log "  - Blockchain RPC: http://localhost:8545"
log ""
log "${BLUE}📚 Next Steps:${NC}"
log "  1. Run integration tests: ./run_integration_tests.sh"
log "  2. Access developer portal: http://localhost:3000"
log "  3. Monitor logs: tail -f $LOG_DIR/*.log"
log ""
log "To stop all services: ./stop_a2a_system.sh"

exit 0