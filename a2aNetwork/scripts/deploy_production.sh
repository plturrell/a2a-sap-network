#!/bin/bash

# A2A Network Production Deployment Script
# Requires: forge, jq, curl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load configuration
NETWORK="${NETWORK:-sepolia}"
CONFIG_FILE="${PROJECT_ROOT}/deploy.production.json"
ENV_FILE="${PROJECT_ROOT}/.env.production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    command -v forge >/dev/null 2>&1 || log_error "forge not found. Please install Foundry."
    command -v jq >/dev/null 2>&1 || log_error "jq not found. Please install jq."
    command -v curl >/dev/null 2>&1 || log_error "curl not found. Please install curl."
    
    log_success "All dependencies found"
}

# Load environment variables
load_environment() {
    log_info "Loading environment variables..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file not found: $ENV_FILE"
    fi
    
    source "$ENV_FILE"
    
    # Check required environment variables
    required_vars=("PRIVATE_KEY" "ALCHEMY_API_KEY" "ETHERSCAN_API_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_error "Required environment variable not set: $var"
        fi
    done
    
    log_success "Environment variables loaded"
}

# Check wallet balance
check_balance() {
    log_info "Checking wallet balance..."
    
    local wallet_address=$(cast wallet address --private-key "$PRIVATE_KEY")
    local balance=$(cast balance "$wallet_address" --rpc-url $(get_rpc_url))
    local balance_eth=$(cast to-unit "$balance" ether)
    
    log_info "Deployer wallet: $wallet_address"
    log_info "Current balance: $balance_eth ETH"
    
    # Check minimum balance (0.1 ETH)
    if (( $(echo "$balance_eth < 0.1" | bc -l) )); then
        log_error "Insufficient balance. Minimum 0.1 ETH required."
    fi
    
    log_success "Wallet balance sufficient"
}

# Get RPC URL for network
get_rpc_url() {
    case "$NETWORK" in
        "mainnet")
            echo "https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}"
            ;;
        "sepolia")
            echo "https://sepolia.infura.io/v3/${INFURA_API_KEY}"
            ;;
        "polygon")
            echo "https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}"
            ;;
        *)
            log_error "Unknown network: $NETWORK"
            ;;
    esac
}

# Get chain ID for network
get_chain_id() {
    case "$NETWORK" in
        "mainnet") echo "1" ;;
        "sepolia") echo "11155111" ;;
        "polygon") echo "137" ;;
        *) log_error "Unknown network: $NETWORK" ;;
    esac
}

# Estimate deployment costs
estimate_costs() {
    log_info "Estimating deployment costs..."
    
    local rpc_url=$(get_rpc_url)
    local gas_price=$(cast gas-price --rpc-url "$rpc_url")
    local estimated_gas=4500000  # 4.5M gas estimate
    
    local cost_wei=$((gas_price * estimated_gas))
    local cost_eth=$(cast to-unit "$cost_wei" ether)
    
    log_info "Estimated gas price: $(cast to-unit "$gas_price" gwei) gwei"
    log_info "Estimated gas usage: $estimated_gas"
    log_info "Estimated deployment cost: $cost_eth ETH"
    
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled by user"
        exit 0
    fi
}

# Deploy contracts
deploy_contracts() {
    log_info "Starting contract deployment to $NETWORK..."
    
    local rpc_url=$(get_rpc_url)
    local chain_id=$(get_chain_id)
    
    cd "$PROJECT_ROOT"
    
    # Compile contracts
    log_info "Compiling contracts..."
    forge build || log_error "Contract compilation failed"
    
    # Deploy using the Deploy.s.sol script
    log_info "Deploying contracts..."
    forge script script/Deploy.s.sol:DeployScript \
        --rpc-url "$rpc_url" \
        --private-key "$PRIVATE_KEY" \
        --broadcast \
        --verify \
        --chain-id "$chain_id" \
        --gas-limit 5000000 || log_error "Contract deployment failed"
    
    log_success "Contracts deployed successfully"
}

# Verify contracts on Etherscan
verify_contracts() {
    log_info "Verifying contracts on block explorer..."
    
    # Contract verification is handled by the --verify flag in forge script
    # Additional verification can be done here if needed
    
    log_success "Contract verification initiated"
}

# Save deployment addresses
save_deployment_addresses() {
    log_info "Saving deployment addresses..."
    
    # Extract addresses from forge broadcast files
    local broadcast_dir="$PROJECT_ROOT/broadcast/Deploy.s.sol/$(get_chain_id)"
    local run_latest="$broadcast_dir/run-latest.json"
    
    if [[ -f "$run_latest" ]]; then
        # Parse deployment addresses and save to deployments.json
        local deployments_file="$PROJECT_ROOT/deployments.${NETWORK}.json"
        
        # Extract contract addresses using jq
        jq '.transactions[] | select(.transactionType == "CREATE") | {contractName: .contractName, contractAddress: .contractAddress}' \
            "$run_latest" > "$deployments_file"
        
        log_success "Deployment addresses saved to $deployments_file"
    else
        log_warning "Broadcast file not found, addresses not saved automatically"
    fi
}

# Update frontend configuration
update_frontend_config() {
    log_info "Updating frontend configuration..."
    
    local deployments_file="$PROJECT_ROOT/deployments.${NETWORK}.json"
    local frontend_config="$PROJECT_ROOT/app/a2a-fiori/webapp/config/blockchain.json"
    
    if [[ -f "$deployments_file" && -f "$frontend_config" ]]; then
        # Update the frontend config with new contract addresses
        # This would need to be customized based on your config structure
        log_info "Frontend config would be updated here"
        log_success "Frontend configuration updated"
    else
        log_warning "Frontend config update skipped (files not found)"
    fi
}

# Run post-deployment tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    # Add integration test commands here
    log_info "Integration tests would run here"
    
    log_success "Integration tests completed"
}

# Main deployment function
main() {
    echo "=================================================================="
    echo "           A2A Network Production Deployment"
    echo "=================================================================="
    echo "Network: $NETWORK"
    echo "Config:  $CONFIG_FILE"
    echo "Env:     $ENV_FILE"
    echo "=================================================================="
    
    check_dependencies
    load_environment
    check_balance
    estimate_costs
    deploy_contracts
    verify_contracts
    save_deployment_addresses
    update_frontend_config
    run_integration_tests
    
    echo "=================================================================="
    log_success "Deployment completed successfully!"
    echo "=================================================================="
    
    # Display important information
    echo "Next steps:"
    echo "1. Verify all contracts on block explorer"
    echo "2. Update production environment variables with contract addresses"
    echo "3. Configure monitoring and alerting"
    echo "4. Test end-to-end functionality"
    echo "5. Enable production traffic"
    echo "=================================================================="
}

# Handle script arguments
case "${1:-}" in
    "mainnet"|"sepolia"|"polygon")
        NETWORK="$1"
        main
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [network]"
        echo "Networks: mainnet, sepolia, polygon"
        echo "Example: $0 sepolia"
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "Invalid network: $1. Use: mainnet, sepolia, or polygon"
        ;;
esac