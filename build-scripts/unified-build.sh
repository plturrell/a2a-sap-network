#!/bin/bash
# Unified Build Script for A2A Platform
# Handles building all components in correct dependency order

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_ENV=${BUILD_ENV:-development}
SKIP_TESTS=${SKIP_TESTS:-false}
PARALLEL_BUILDS=${PARALLEL_BUILDS:-true}
BUILD_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BUILD_LOG="build-${BUILD_TIMESTAMP}.log"

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$BUILD_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$BUILD_LOG"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$BUILD_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ $1${NC}" | tee -a "$BUILD_LOG"
}

cleanup() {
    log "Performing cleanup..."
    # Clean temporary files
    find . -name "*.tmp" -delete 2>/dev/null || true
    find . -name ".DS_Store" -delete 2>/dev/null || true
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Main build function
main() {
    log "🚀 Starting A2A Platform Build Process"
    log "Environment: $BUILD_ENV"
    log "Skip Tests: $SKIP_TESTS"
    log "Parallel Builds: $PARALLEL_BUILDS"
    
    # Pre-build checks
    check_prerequisites
    
    # Build components in dependency order
    build_smart_contracts
    build_network_service
    build_agents_backend
    build_frontend
    build_vscode_extension
    
    # Post-build tasks
    if [[ "$SKIP_TESTS" != "true" ]]; then
        run_tests
    fi
    
    generate_build_manifest
    
    log_success "🎉 Build completed successfully!"
    log "Build artifacts saved to: ./dist/"
    log "Build log saved to: $BUILD_LOG"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Node.js version
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        exit 1
    fi
    
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    if [[ "$(printf '%s\n' "18.0.0" "$NODE_VERSION" | sort -V | head -n1)" != "18.0.0" ]]; then
        log_error "Node.js version 18+ required, found: $NODE_VERSION"
        exit 1
    fi
    
    # Check required tools
    local tools=("npm" "docker" "forge")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_warning "$tool is not installed - some builds may fail"
        fi
    done
    
    log_success "Prerequisites check passed"
}

build_smart_contracts() {
    log "🔗 Building Smart Contracts..."
    
    if [[ ! -d "a2aNetwork/contracts" ]]; then
        log_warning "Smart contracts directory not found, skipping"
        return
    fi
    
    cd a2aNetwork/contracts
    
    # Install dependencies
    if [[ -f "package.json" ]]; then
        npm ci --production
    fi
    
    # Compile contracts with Forge
    if command -v forge &> /dev/null; then
        forge build --sizes
        log_success "Smart contracts compiled successfully"
    else
        log_warning "Forge not found, skipping contract compilation"
    fi
    
    cd ../..
}

build_network_service() {
    log "🌐 Building Network Service..."
    
    cd a2aNetwork
    
    # Install dependencies
    npm ci --production
    
    # Run linting and formatting
    npm run lint check || log_warning "Linting issues found"
    
    # Build the service
    npm run build
    
    # Create distribution package
    mkdir -p ../dist/network
    cp -r dist/* ../dist/network/ 2>/dev/null || true
    cp package.json ../dist/network/
    
    log_success "Network service built successfully"
    cd ..
}

build_agents_backend() {
    log "🤖 Building Agents Backend..."
    
    cd a2aAgents/backend
    
    # Install dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    elif [[ -f "package.json" ]]; then
        npm ci --production
    fi
    
    # Build based on technology stack
    if [[ -f "setup.py" ]] || [[ -f "pyproject.toml" ]]; then
        python -m build
    elif [[ -f "package.json" ]]; then
        npm run build
    fi
    
    # Create distribution package
    mkdir -p ../../dist/agents
    cp -r dist/* ../../dist/agents/ 2>/dev/null || true
    cp -r app ../../dist/agents/ 2>/dev/null || true
    
    log_success "Agents backend built successfully"
    cd ../..
}

build_frontend() {
    log "🎨 Building Frontend..."
    
    if [[ -d "a2aAgents/frontend" ]]; then
        cd a2aAgents/frontend
        
        # Install dependencies
        npm ci --production
        
        # Build frontend
        npm run build
        
        # Create distribution package
        mkdir -p ../../dist/frontend
        cp -r dist/* ../../dist/frontend/ 2>/dev/null || true
        
        log_success "Frontend built successfully"
        cd ../..
    else
        log_warning "Frontend directory not found, skipping"
    fi
}

build_vscode_extension() {
    log "🔧 Building VS Code Extension..."
    
    if [[ -d "a2aNetwork/vscode-extension" ]]; then
        cd a2aNetwork/vscode-extension
        
        # Install dependencies
        npm ci --production
        
        # Compile TypeScript
        npm run compile
        
        # Package extension
        if command -v vsce &> /dev/null; then
            vsce package --out ../../dist/
            log_success "VS Code extension packaged"
        else
            log_warning "vsce not found, skipping extension packaging"
        fi
        
        cd ../..
    else
        log_warning "VS Code extension directory not found, skipping"
    fi
}

run_tests() {
    log "🧪 Running Tests..."
    
    local test_results=()
    
    # Test Network Service
    if [[ -d "a2aNetwork" ]]; then
        cd a2aNetwork
        if npm run test 2>&1; then
            test_results+=("Network: PASSED")
        else
            test_results+=("Network: FAILED")
        fi
        cd ..
    fi
    
    # Test Agents Backend
    if [[ -d "a2aAgents/backend" ]]; then
        cd a2aAgents/backend
        if pytest tests/ 2>&1; then
            test_results+=("Agents: PASSED")
        else
            test_results+=("Agents: FAILED")
        fi
        cd ../..
    fi
    
    # Display test results
    log "Test Results:"
    for result in "${test_results[@]}"; do
        if [[ "$result" == *"PASSED"* ]]; then
            log_success "$result"
        else
            log_error "$result"
        fi
    done
}

generate_build_manifest() {
    log "📋 Generating Build Manifest..."
    
    mkdir -p dist
    
    cat > dist/build-manifest.json << EOF
{
  "buildTimestamp": "$BUILD_TIMESTAMP",
  "buildEnvironment": "$BUILD_ENV",
  "gitCommit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "gitBranch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "nodeVersion": "$(node --version)",
  "npmVersion": "$(npm --version)",
  "components": {
    "smartContracts": {
      "built": $([ -d "a2aNetwork/contracts/out" ] && echo "true" || echo "false"),
      "path": "contracts/"
    },
    "networkService": {
      "built": $([ -d "dist/network" ] && echo "true" || echo "false"),
      "path": "network/"
    },
    "agentsBackend": {
      "built": $([ -d "dist/agents" ] && echo "true" || echo "false"),
      "path": "agents/"
    },
    "frontend": {
      "built": $([ -d "dist/frontend" ] && echo "true" || echo "false"),
      "path": "frontend/"
    },
    "vscodeExtension": {
      "built": $([ -f "dist/*.vsix" ] && echo "true" || echo "false"),
      "path": "*.vsix"
    }
  },
  "buildDuration": "$SECONDS seconds",
  "buildLog": "$BUILD_LOG"
}
EOF
    
    log_success "Build manifest generated"
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            BUILD_ENV="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-parallel)
            PARALLEL_BUILDS=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env ENV          Set build environment (default: development)"
            echo "  --skip-tests       Skip running tests"
            echo "  --no-parallel      Disable parallel builds"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Start the build
main "$@"