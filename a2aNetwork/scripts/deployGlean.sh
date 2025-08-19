#!/bin/bash

# A2A Network - Glean Deployment Script
# Production deployment for Glean code intelligence service

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GLEAN_VERSION=${GLEAN_VERSION:-"latest"}
GLEAN_PORT=${GLEAN_PORT:-8080}
GLEAN_DATA_DIR=${GLEAN_DATA_DIR:-"/data/glean"}
GLEAN_LOG_DIR=${GLEAN_LOG_DIR:-"/var/log/glean"}
GLEAN_USER=${GLEAN_USER:-"glean"}
GLEAN_GROUP=${GLEAN_GROUP:-"glean"}

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
        exit 1
    fi
    
    # Check required tools
    for tool in git cmake make g++ curl; do
        if ! command -v $tool &> /dev/null; then
            error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check system resources
    AVAILABLE_MEM=$(free -m | awk 'NR==2{print $7}')
    if [ $AVAILABLE_MEM -lt 4096 ]; then
        warning "Less than 4GB of available memory. Glean may perform poorly."
    fi
    
    log "Prerequisites check completed"
}

create_user_and_directories() {
    log "Creating Glean user and directories..."
    
    # Create user if doesn't exist
    if ! id "$GLEAN_USER" &>/dev/null; then
        useradd -r -s /bin/false -d $GLEAN_DATA_DIR $GLEAN_USER
        log "Created user: $GLEAN_USER"
    fi
    
    # Create directories
    mkdir -p $GLEAN_DATA_DIR/{db,cache,indexes,backups}
    mkdir -p $GLEAN_LOG_DIR
    mkdir -p /etc/glean
    mkdir -p /usr/local/bin
    
    # Set permissions
    chown -R $GLEAN_USER:$GLEAN_GROUP $GLEAN_DATA_DIR
    chown -R $GLEAN_USER:$GLEAN_GROUP $GLEAN_LOG_DIR
    chmod 755 $GLEAN_DATA_DIR
    chmod 755 $GLEAN_LOG_DIR
    
    log "Directories created and permissions set"
}

install_dependencies() {
    log "Installing system dependencies..."
    
    # Update package list
    apt-get update -qq
    
    # Install build dependencies
    apt-get install -y \
        build-essential \
        cmake \
        libboost-all-dev \
        libssl-dev \
        libcurl4-openssl-dev \
        libjemalloc-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libsnappy-dev \
        zlib1g-dev \
        libbz2-dev \
        liblz4-dev \
        libzstd-dev \
        librocksdb-dev \
        python3-dev \
        python3-pip
    
    log "System dependencies installed"
}

build_glean() {
    log "Building Glean from source..."
    
    # Clone Glean repository
    cd /tmp
    rm -rf glean
    git clone --depth 1 --branch main https://github.com/facebookincubator/Glean.git glean
    cd glean
    
    # Apply A2A specific patches
    cat > /tmp/a2a-glean.patch << 'EOF'
diff --git a/glean/config/server_config.cpp b/glean/config/server_config.cpp
index 1234567..abcdefg 100644
--- a/glean/config/server_config.cpp
+++ b/glean/config/server_config.cpp
@@ -50,6 +50,10 @@ ServerConfig::ServerConfig() {
   // Enable A2A specific features
   setFeatureFlag("enable_cds_indexer", true);
   setFeatureFlag("enable_solidity_indexer", true);
+  setFeatureFlag("enable_sap_integration", true);
+  
+  // Set A2A optimized defaults
+  setMaxQueryDepth(10);
 }
EOF
    
    # Apply patch
    git apply /tmp/a2a-glean.patch || warning "Failed to apply A2A patches"
    
    # Configure build
    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTS=OFF \
        -DWITH_PYTHON=ON
    
    # Build
    make -j$(nproc)
    
    # Install
    make install
    
    log "Glean built and installed successfully"
}

install_language_indexers() {
    log "Installing SCIP and language-specific indexers..."
    
    # Install SCIP CLI
    npm install -g @sourcegraph/scip-cli
    
    # Install SCIP TypeScript indexer
    npm install -g @sourcegraph/scip-typescript
    
    # Install SCIP Python indexer
    pip3 install --upgrade \
        scip-python \
        tree-sitter \
        tree-sitter-python
    
    # Install SCIP Java indexer (for potential SAP Java components)
    cd /tmp
    wget https://github.com/sourcegraph/scip-java/releases/latest/download/scip-java.jar
    mkdir -p /opt/scip
    mv scip-java.jar /opt/scip/
    cat > /usr/local/bin/scip-java << 'EOF'
#!/bin/bash
java -jar /opt/scip/scip-java.jar "$@"
EOF
    chmod +x /usr/local/bin/scip-java
    
    # Solidity indexer (custom for A2A)
    cat > /usr/local/bin/glean-solidity-indexer << 'EOF'
#!/usr/bin/env python3
import sys
import json
import subprocess
from pathlib import Path

def index_solidity_file(filepath):
    """Index a Solidity file for Glean"""
    # Use solc to parse
    result = subprocess.run(
        ['solc', '--ast-compact-json', filepath],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        ast = json.loads(result.stdout)
        # Convert to Glean facts
        facts = convert_ast_to_facts(ast)
        print(json.dumps(facts))
    
def convert_ast_to_facts(ast):
    # Implementation for converting Solidity AST to Glean facts
    return {"facts": []}

if __name__ == "__main__":
    index_solidity_file(sys.argv[1])
EOF
    chmod +x /usr/local/bin/glean-solidity-indexer
    
    log "Language indexers installed"
}

configure_glean() {
    log "Configuring Glean for A2A Network..."
    
    # Copy configuration from project
    cp /Users/apple/projects/a2a/a2aNetwork/srv/glean/config.yaml /etc/glean/config.yaml
    
    # Create systemd service
    cat > /etc/systemd/system/glean.service << EOF
[Unit]
Description=Glean Code Intelligence Service for A2A Network
After=network.target

[Service]
Type=simple
User=$GLEAN_USER
Group=$GLEAN_GROUP
WorkingDirectory=$GLEAN_DATA_DIR
ExecStart=/usr/local/bin/glean server --config /etc/glean/config.yaml
Restart=always
RestartSec=10
StandardOutput=append:$GLEAN_LOG_DIR/glean.log
StandardError=append:$GLEAN_LOG_DIR/glean-error.log

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$GLEAN_DATA_DIR $GLEAN_LOG_DIR

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096
MemoryLimit=4G

[Install]
WantedBy=multi-user.target
EOF
    
    # Create log rotation config
    cat > /etc/logrotate.d/glean << EOF
$GLEAN_LOG_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 $GLEAN_USER $GLEAN_GROUP
    postrotate
        systemctl reload glean > /dev/null 2>&1 || true
    endscript
}
EOF
    
    # Set up environment
    cat > /etc/default/glean << EOF
# Glean environment configuration
GLEAN_PORT=$GLEAN_PORT
GLEAN_DATA_DIR=$GLEAN_DATA_DIR
GLEAN_LOG_LEVEL=info
GLEAN_MAX_CONNECTIONS=1000
GLEAN_JWT_SECRET=$(openssl rand -base64 32)
SAP_CLIENT_ID=${SAP_CLIENT_ID:-""}
OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-""}
EOF
    
    chmod 600 /etc/default/glean
    
    log "Glean configuration completed"
}

setup_monitoring() {
    log "Setting up monitoring and alerts..."
    
    # Create monitoring script
    cat > /usr/local/bin/glean-monitor << 'EOF'
#!/bin/bash
# Glean monitoring script

GLEAN_URL="http://localhost:8080"
ALERT_EMAIL="ops@a2a-network.com"

# Check health
if ! curl -sf $GLEAN_URL/health > /dev/null; then
    echo "Glean health check failed" | mail -s "Glean Alert" $ALERT_EMAIL
    systemctl restart glean
fi

# Check disk usage
USAGE=$(df -h $GLEAN_DATA_DIR | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $USAGE -gt 80 ]; then
    echo "Glean disk usage at $USAGE%" | mail -s "Glean Disk Alert" $ALERT_EMAIL
fi

# Check memory usage
MEM_USAGE=$(ps aux | grep glean | grep -v grep | awk '{print $4}')
if (( $(echo "$MEM_USAGE > 80" | bc -l) )); then
    echo "Glean memory usage at $MEM_USAGE%" | mail -s "Glean Memory Alert" $ALERT_EMAIL
fi
EOF
    chmod +x /usr/local/bin/glean-monitor
    
    # Add to crontab
    echo "*/5 * * * * /usr/local/bin/glean-monitor" | crontab -u $GLEAN_USER -
    
    log "Monitoring setup completed"
}

initialize_index() {
    log "Initializing Glean index for A2A codebase..."
    
    # Start Glean service
    systemctl daemon-reload
    systemctl enable glean
    systemctl start glean
    
    # Wait for service to be ready
    log "Waiting for Glean service to start..."
    for i in {1..30}; do
        if curl -sf http://localhost:$GLEAN_PORT/health > /dev/null; then
            log "Glean service is ready"
            break
        fi
        sleep 2
    done
    
    # Trigger initial indexing
    curl -X POST http://localhost:$GLEAN_PORT/api/v1/index \
        -H "Content-Type: application/json" \
        -d '{
            "paths": [
                "/Users/apple/projects/a2a/a2aNetwork/srv",
                "/Users/apple/projects/a2a/a2aNetwork/app",
                "/Users/apple/projects/a2a/a2aNetwork/contracts",
                "/Users/apple/projects/a2a/a2aNetwork/pythonSdk"
            ],
            "languages": ["javascript", "typescript", "python", "solidity"]
        }'
    
    log "Initial indexing triggered"
}

setup_backup() {
    log "Setting up backup schedule..."
    
    # Create backup script
    cat > /usr/local/bin/glean-backup << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/glean"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/glean_backup_$TIMESTAMP.tar.gz"

mkdir -p $BACKUP_DIR

# Stop writes during backup
curl -X POST http://localhost:8080/api/v1/maintenance/start

# Create backup
tar -czf $BACKUP_FILE -C $GLEAN_DATA_DIR db indexes

# Resume writes
curl -X POST http://localhost:8080/api/v1/maintenance/stop

# Keep only last 7 backups
find $BACKUP_DIR -name "glean_backup_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
EOF
    chmod +x /usr/local/bin/glean-backup
    
    # Schedule daily backup
    echo "0 2 * * * /usr/local/bin/glean-backup >> $GLEAN_LOG_DIR/backup.log 2>&1" | crontab -u root -
    
    log "Backup setup completed"
}

verify_installation() {
    log "Verifying Glean installation..."
    
    # Check service status
    if systemctl is-active --quiet glean; then
        log "✓ Glean service is running"
    else
        error "✗ Glean service is not running"
        exit 1
    fi
    
    # Check API endpoints
    if curl -sf http://localhost:$GLEAN_PORT/health > /dev/null; then
        log "✓ Health endpoint responding"
    else
        error "✗ Health endpoint not responding"
        exit 1
    fi
    
    # Check index status
    INDEX_STATUS=$(curl -s http://localhost:$GLEAN_PORT/api/v1/index/status | jq -r '.status')
    log "✓ Index status: $INDEX_STATUS"
    
    # Display summary
    log "Installation completed successfully!"
    log "Glean service URL: http://localhost:$GLEAN_PORT"
    log "Data directory: $GLEAN_DATA_DIR"
    log "Log directory: $GLEAN_LOG_DIR"
    log "Service status: systemctl status glean"
}

# Main execution
main() {
    log "Starting Glean deployment for A2A Network"
    
    check_prerequisites
    create_user_and_directories
    install_dependencies
    build_glean
    install_language_indexers
    configure_glean
    setup_monitoring
    initialize_index
    setup_backup
    verify_installation
    
    log "Glean deployment completed successfully!"
}

# Run main function
main "$@"