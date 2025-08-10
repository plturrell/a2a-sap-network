#!/bin/bash

# SAP BTP Deployment Script for A2A Platform
# Supports multiple environments with proper MTA configurations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MBT_BUILD_DIR="${PROJECT_ROOT}/mta_archives"

# Default values
ENVIRONMENT=""
BUILD_ONLY=false
SKIP_TESTS=false
VERBOSE=false
DRY_RUN=false

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy A2A Platform to SAP BTP with environment-specific configurations

OPTIONS:
    -e, --environment ENV    Target environment (dev|staging|prod)
    -b, --build-only         Only build MTA, don't deploy
    -s, --skip-tests         Skip test execution
    -v, --verbose            Enable verbose output
    -n, --dry-run            Show what would be deployed without actually deploying
    -h, --help               Show this help message

EXAMPLES:
    $0 -e dev                Deploy to development environment
    $0 -e staging -v         Deploy to staging with verbose output
    $0 -e prod --dry-run     Show production deployment plan
    $0 -e dev --build-only   Build MTA for development environment only

ENVIRONMENTS:
    dev      - Development environment (auto-deploy on merge)
    staging  - Staging environment (manual approval required)
    prod     - Production environment (multi-approval required)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -b|--build-only)
            BUILD_ONLY=true
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ -z "$ENVIRONMENT" ]]; then
    echo -e "${RED}Error: Environment must be specified${NC}"
    usage
    exit 1
fi

if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    echo -e "${RED}Error: Environment must be one of: dev, staging, prod${NC}"
    exit 1
fi

# Set verbose mode
if [[ "$VERBOSE" == true ]]; then
    set -x
fi

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check required tools
check_prerequisites() {
    log "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command -v cf &> /dev/null; then
        missing_tools+=("cf CLI")
    fi
    
    if ! command -v mbt &> /dev/null; then
        missing_tools+=("mbt (Multi-Target Application Build Tool)")
    fi
    
    if ! command -v node &> /dev/null; then
        missing_tools+=("Node.js")
    fi
    
    if ! command -v npm &> /dev/null; then
        missing_tools+=("npm")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools:"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        echo ""
        echo "Please install the missing tools and try again."
        echo "See: https://developers.sap.com/tutorials/btp-app-prepare-dev-environment-cap.html"
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Load environment configuration
load_environment_config() {
    local config_file="${PROJECT_ROOT}/deployment-config.json"
    
    if [[ ! -f "$config_file" ]]; then
        log_error "Deployment configuration file not found: $config_file"
        exit 1
    fi
    
    # Extract CF configuration using jq if available, otherwise use basic parsing
    if command -v jq &> /dev/null; then
        CF_API=$(jq -r ".environments.${ENVIRONMENT}.cf.api" "$config_file")
        CF_ORG=$(jq -r ".environments.${ENVIRONMENT}.cf.org" "$config_file")
        CF_SPACE=$(jq -r ".environments.${ENVIRONMENT}.cf.space" "$config_file")
        MTA_EXTENSION_FILE=$(jq -r ".environments.${ENVIRONMENT}.mta.extension_file" "$config_file")
    else
        log_warning "jq not found, using basic configuration parsing"
        CF_API="https://api.cf.us10.hana.ondemand.com"
        CF_ORG="company-${ENVIRONMENT}"
        CF_SPACE="a2a-${ENVIRONMENT}"
        MTA_EXTENSION_FILE="mta-extensions/mta-${ENVIRONMENT}.mtaext"
    fi
    
    log "Environment configuration loaded:"
    log "  API: $CF_API"
    log "  Org: $CF_ORG"
    log "  Space: $CF_SPACE"
    log "  MTA Extension: $MTA_EXTENSION_FILE"
}

# CF Login
cf_login() {
    log "Logging into Cloud Foundry..."
    
    # Check if already logged in to correct target
    if cf target &> /dev/null; then
        local current_api=$(cf target | grep "api endpoint" | awk '{print $3}' || echo "")
        local current_org=$(cf target | grep "org:" | awk '{print $2}' || echo "")
        local current_space=$(cf target | grep "space:" | awk '{print $2}' || echo "")
        
        if [[ "$current_api" == "$CF_API" && "$current_org" == "$CF_ORG" && "$current_space" == "$CF_SPACE" ]]; then
            log_success "Already logged in to correct target"
            return 0
        fi
    fi
    
    # Login required
    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: Would login to CF API: $CF_API"
        return 0
    fi
    
    # Use CF_USERNAME and CF_PASSWORD environment variables if available
    if [[ -n "${CF_USERNAME:-}" && -n "${CF_PASSWORD:-}" ]]; then
        echo "$CF_PASSWORD" | cf login -a "$CF_API" -u "$CF_USERNAME" -p - -o "$CF_ORG" -s "$CF_SPACE"
    else
        cf login -a "$CF_API" -o "$CF_ORG" -s "$CF_SPACE"
    fi
    
    log_success "Logged in successfully"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_warning "Skipping tests as requested"
        return 0
    fi
    
    log "Running tests for $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"
    
    # Install dependencies
    if [[ -f "package.json" ]]; then
        npm install
    fi
    
    # Backend tests
    if [[ -d "backend" ]]; then
        cd backend
        if [[ -f "requirements.txt" ]]; then
            pip install -r requirements.txt
        fi
        
        # Run Python tests
        if command -v pytest &> /dev/null; then
            log "Running backend tests..."
            pytest tests/ -v
        fi
        
        cd ..
    fi
    
    # Frontend tests
    if [[ -d "frontend" ]]; then
        cd frontend
        if [[ -f "package.json" ]]; then
            npm install
            npm run test
        fi
        cd ..
    fi
    
    log_success "Tests completed successfully"
}

# Build MTA
build_mta() {
    log "Building MTA for $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"
    
    # Clean previous builds
    if [[ -d "$MBT_BUILD_DIR" ]]; then
        rm -rf "$MBT_BUILD_DIR"
    fi
    
    mkdir -p "$MBT_BUILD_DIR"
    
    # Validate MTA descriptor and extension
    if [[ ! -f "mta.yaml" ]]; then
        log_error "MTA descriptor (mta.yaml) not found"
        exit 1
    fi
    
    if [[ ! -f "$MTA_EXTENSION_FILE" ]]; then
        log_error "MTA extension file not found: $MTA_EXTENSION_FILE"
        exit 1
    fi
    
    # Build MTA
    local build_cmd="mbt build"
    
    if [[ "$ENVIRONMENT" != "dev" ]]; then
        build_cmd="$build_cmd --mtar \"${MBT_BUILD_DIR}/a2a-agent-platform-${ENVIRONMENT}.mtar\""
    fi
    
    build_cmd="$build_cmd -e \"$MTA_EXTENSION_FILE\""
    
    if [[ "$VERBOSE" == true ]]; then
        build_cmd="$build_cmd -v"
    fi
    
    log "Executing: $build_cmd"
    eval "$build_cmd"
    
    log_success "MTA build completed"
}

# Deploy MTA
deploy_mta() {
    if [[ "$BUILD_ONLY" == true ]]; then
        log_warning "Build-only mode, skipping deployment"
        return 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: Would deploy MTA to $ENVIRONMENT environment"
        show_deployment_plan
        return 0
    fi
    
    log "Deploying to $ENVIRONMENT environment..."
    
    # Find the built MTAR file
    local mtar_file
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        mtar_file=$(find "$PROJECT_ROOT" -name "*.mtar" -type f | head -1)
    else
        mtar_file="${MBT_BUILD_DIR}/a2a-agent-platform-${ENVIRONMENT}.mtar"
    fi
    
    if [[ ! -f "$mtar_file" ]]; then
        log_error "MTAR file not found. Please run build first."
        exit 1
    fi
    
    log "Deploying MTAR: $mtar_file"
    
    # Deploy with appropriate strategy
    local deploy_cmd="cf deploy \"$mtar_file\""
    
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        # Production deployment with blue-green strategy
        deploy_cmd="$deploy_cmd --strategy blue-green --no-confirm"
    else
        deploy_cmd="$deploy_cmd --no-confirm"
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        deploy_cmd="$deploy_cmd -v"
    fi
    
    log "Executing: $deploy_cmd"
    eval "$deploy_cmd"
    
    log_success "Deployment completed successfully"
}

# Show deployment plan
show_deployment_plan() {
    log "Deployment Plan for $ENVIRONMENT:"
    echo ""
    echo "Applications to be deployed:"
    echo "  - a2a-backend"
    echo "  - a2a-frontend" 
    echo "  - a2a-approuter"
    echo "  - a2a-db-deployer"
    echo ""
    echo "Services to be created/updated:"
    echo "  - a2a-hana-hdi"
    echo "  - a2a-postgres-db"
    echo "  - a2a-redis"
    echo "  - a2a-xsuaa"
    echo "  - a2a-connectivity"
    echo "  - a2a-destination"
    echo "  - a2a-ans"
    echo "  - a2a-application-logs"
    echo "  - a2a-html5-repo-host"
    echo "  - a2a-html5-repo-runtime"
    echo ""
}

# Post-deployment validation
post_deployment_validation() {
    if [[ "$BUILD_ONLY" == true ]] || [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    
    log "Running post-deployment validation..."
    
    # Get application URLs
    local backend_url=$(cf app a2a-backend-${ENVIRONMENT} --guid 2>/dev/null | xargs -I {} cf curl /v2/apps/{}/routes | grep -o 'https://[^"]*' | head -1 || echo "")
    local frontend_url=$(cf app a2a-approuter-${ENVIRONMENT} --guid 2>/dev/null | xargs -I {} cf curl /v2/apps/{}/routes | grep -o 'https://[^"]*' | head -1 || echo "")
    
    # Health checks
    if [[ -n "$backend_url" ]]; then
        log "Checking backend health: ${backend_url}/health"
        if command -v curl &> /dev/null; then
            local health_status=$(curl -s -o /dev/null -w "%{http_code}" "${backend_url}/health" || echo "000")
            if [[ "$health_status" == "200" ]]; then
                log_success "Backend health check passed"
            else
                log_warning "Backend health check failed (HTTP $health_status)"
            fi
        fi
    fi
    
    log_success "Post-deployment validation completed"
}

# Main deployment workflow
main() {
    log "Starting A2A Platform deployment to $ENVIRONMENT"
    echo "=================================================="
    
    check_prerequisites
    load_environment_config
    
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        log_warning "Production deployment requires additional approvals"
        read -p "Are you sure you want to deploy to production? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Deployment cancelled"
            exit 0
        fi
    fi
    
    cf_login
    run_tests
    build_mta
    deploy_mta
    post_deployment_validation
    
    echo "=================================================="
    log_success "Deployment to $ENVIRONMENT completed successfully!"
    
    if [[ "$BUILD_ONLY" == false ]] && [[ "$DRY_RUN" == false ]]; then
        echo ""
        echo "Application URLs:"
        cf apps | grep "a2a-"
    fi
}

# Run main function
main "$@"