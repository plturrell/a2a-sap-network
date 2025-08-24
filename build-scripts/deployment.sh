#!/bin/bash
# Deployment Script for A2A Platform
# Supports multiple environments and deployment strategies

set -euo pipefail

# Configuration
ENVIRONMENT=${1:-staging}
DEPLOYMENT_STRATEGY=${DEPLOYMENT_STRATEGY:-rolling}
DRY_RUN=${DRY_RUN:-false}
ROLLBACK=${ROLLBACK:-false}
VERSION=${VERSION:-latest}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ $1${NC}"
}

# Validation functions
validate_environment() {
    local env=$1
    case $env in
        development|staging|production)
            log_success "Environment '$env' is valid"
            ;;
        *)
            log_error "Invalid environment: $env"
            log "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

validate_prerequisites() {
    log "Validating deployment prerequisites..."
    
    local tools=("docker" "kubectl" "helm")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Kubernetes context
    local current_context=$(kubectl config current-context 2>/dev/null || echo "none")
    if [[ "$current_context" == "none" ]]; then
        log_error "No Kubernetes context set"
        exit 1
    fi
    
    log_success "Prerequisites validated (Context: $current_context)"
}

# Deployment functions
deploy_database() {
    log "🗄️ Deploying database changes..."
    
    local namespace="a2a-${ENVIRONMENT}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN: Would run database migrations"
        return
    fi
    
    # Run database migrations
    kubectl run db-migrate-$(date +%s) \
        --namespace="$namespace" \
        --image="ghcr.io/a2a-platform/db-migrate:$VERSION" \
        --restart=Never \
        --rm -i --tty \
        --env="DATABASE_URL=$DATABASE_URL" \
        --env="ENVIRONMENT=$ENVIRONMENT" \
        -- npm run db migrate
        
    log_success "Database deployment completed"
}

deploy_smart_contracts() {
    log "🔗 Deploying smart contracts..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN: Would deploy smart contracts"
        return
    fi
    
    # Deploy contracts based on environment
    case $ENVIRONMENT in
        development)
            forge script script/Deploy.s.sol --rpc-url http://localhost:8545 --broadcast
            ;;
        staging)
            forge script script/Deploy.s.sol --rpc-url $STAGING_RPC_URL --broadcast --verify
            ;;
        production)
            log_warning "Manual approval required for production contract deployment"
            read -p "Proceed with contract deployment? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                forge script script/Deploy.s.sol --rpc-url $PRODUCTION_RPC_URL --broadcast --verify
            else
                log "Contract deployment cancelled"
                return
            fi
            ;;
    esac
    
    log_success "Smart contracts deployed"
}

deploy_services() {
    log "🚀 Deploying services with $DEPLOYMENT_STRATEGY strategy..."
    
    local namespace="a2a-${ENVIRONMENT}"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy using Helm
    local helm_args=(
        "upgrade" "--install" "a2a-platform"
        "./helm/a2a-platform"
        "--namespace" "$namespace"
        "--values" "helm/values-${ENVIRONMENT}.yaml"
        "--set" "image.tag=$VERSION"
        "--set" "deployment.strategy=$DEPLOYMENT_STRATEGY"
    )
    
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_args+=("--dry-run")
    fi
    
    helm "${helm_args[@]}"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Wait for rollout to complete
        kubectl rollout status deployment/a2a-network -n "$namespace" --timeout=600s
        kubectl rollout status deployment/a2a-agents -n "$namespace" --timeout=600s
        
        log_success "Services deployed successfully"
    else
        log_warning "DRY RUN: Services deployment validated"
    fi
}

run_health_checks() {
    log "🏥 Running health checks..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN: Would run health checks"
        return
    fi
    
    local namespace="a2a-${ENVIRONMENT}"
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Health check attempt $attempt/$max_attempts"
        
        # Check service endpoints
        if kubectl exec -n "$namespace" deployment/a2a-network -- curl -f http://localhost:4004/health &> /dev/null; then
            log_success "Network service is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Health checks failed after $max_attempts attempts"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Run smoke tests
    log "Running smoke tests..."
    kubectl run smoke-test-$(date +%s) \
        --namespace="$namespace" \
        --image="ghcr.io/a2a-platform/smoke-tests:$VERSION" \
        --restart=Never \
        --rm -i --tty \
        --env="ENVIRONMENT=$ENVIRONMENT" \
        --env="BASE_URL=http://a2a-network.$namespace.svc.cluster.local:4004"
        
    log_success "Health checks passed"
}

rollback_deployment() {
    log "🔄 Rolling back deployment..."
    
    local namespace="a2a-${ENVIRONMENT}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN: Would rollback deployment"
        return
    fi
    
    # Rollback using Helm
    helm rollback a2a-platform --namespace "$namespace"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/a2a-network -n "$namespace" --timeout=600s
    kubectl rollout status deployment/a2a-agents -n "$namespace" --timeout=600s
    
    log_success "Rollback completed"
}

backup_current_deployment() {
    log "💾 Creating deployment backup..."
    
    local namespace="a2a-${ENVIRONMENT}"
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)_${ENVIRONMENT}"
    
    mkdir -p "$backup_dir"
    
    # Backup Helm release
    helm get all a2a-platform --namespace "$namespace" > "$backup_dir/helm-release.yaml"
    
    # Backup database
    kubectl exec -n "$namespace" deployment/postgres -- pg_dump -U postgres a2a_db > "$backup_dir/database.sql"
    
    # Backup configs
    kubectl get configmaps -n "$namespace" -o yaml > "$backup_dir/configmaps.yaml"
    kubectl get secrets -n "$namespace" -o yaml > "$backup_dir/secrets.yaml"
    
    log_success "Backup created in $backup_dir"
}

# Main deployment logic
main() {
    log "🚀 Starting A2A Platform Deployment"
    log "Environment: $ENVIRONMENT"
    log "Strategy: $DEPLOYMENT_STRATEGY"
    log "Version: $VERSION"
    log "Dry Run: $DRY_RUN"
    
    # Validate inputs
    validate_environment "$ENVIRONMENT"
    validate_prerequisites
    
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        return
    fi
    
    # Create backup before deployment
    if [[ "$ENVIRONMENT" == "production" && "$DRY_RUN" != "true" ]]; then
        backup_current_deployment
    fi
    
    # Deploy components
    deploy_database
    deploy_smart_contracts  
    deploy_services
    run_health_checks
    
    log_success "🎉 Deployment completed successfully!"
}

# Script usage
show_usage() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENT:
    development, staging, production (default: staging)

OPTIONS:
    --strategy STRATEGY    Deployment strategy: rolling, blue-green, canary (default: rolling)
    --version VERSION      Version to deploy (default: latest)
    --dry-run             Validate deployment without applying changes
    --rollback            Rollback to previous deployment
    --help                Show this help message

EXAMPLES:
    $0 staging
    $0 production --version v1.2.3
    $0 staging --dry-run
    $0 production --rollback

ENVIRONMENT VARIABLES:
    DATABASE_URL          Database connection string
    STAGING_RPC_URL       Staging blockchain RPC URL
    PRODUCTION_RPC_URL    Production blockchain RPC URL
    DEPLOYMENT_STRATEGY   Default deployment strategy
    DRY_RUN              Default dry run mode
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            DEPLOYMENT_STRATEGY="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        development|staging|production)
            ENVIRONMENT="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute main function
main