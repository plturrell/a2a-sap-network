# A2A Business Data Cloud Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the A2A Business Data Cloud system across different environments.

## Deployment Options

### 1. Local Development Deployment

Quick deployment for development and testing.

```bash
# Deploy everything locally
./deploy_a2a_system.sh local

# Stop all services
./stop_a2a_system.sh
```

### 2. Docker Deployment

Container-based deployment for consistency across environments.

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Kubernetes Deployment

For production-grade deployments.

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/k8s/

# Check deployment status
kubectl get pods -n a2a-system
```

### 4. Cloud Deployment (AWS)

For AWS production deployment.

```bash
# Deploy using Terraform
cd deployment/terraform
terraform init
terraform plan
terraform apply
```

## Pre-Deployment Checklist

### Environment Variables

Create a `.env` file with required variables:

```env
# Blockchain
ETHEREUM_RPC_URL=http://localhost:8545
BDC_CONTRACT_ADDRESS=0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0
REGISTRY_CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3
ROUTER_CONTRACT_ADDRESS=0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512

# Database
HANA_CONNECTION_STRING=hana://user:pass@host:port
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key

# APIs
GROK_API_KEY=your-grok-api-key
GROK_API_URL=https://api.grok.ai/v1
ORD_REGISTRY_URL=https://ord.example.com

# Monitoring
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
PROMETHEUS_URL=http://localhost:9090
```

### System Requirements

- **CPU**: Minimum 8 cores (16 recommended)
- **Memory**: Minimum 16GB RAM (32GB recommended)
- **Storage**: 100GB SSD
- **Network**: 100Mbps bandwidth
- **OS**: Ubuntu 20.04+ or RHEL 8+

### Software Dependencies

- Python 3.9+
- Docker 20.10+
- Node.js 16+ (for blockchain tools)
- Foundry (for smart contracts)

## Step-by-Step Deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/a2a-agents.git
cd a2a-agents/backend
```

### Step 2: Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Node dependencies (for blockchain)
npm install -g @foundry-rs/toolkit
```

### Step 3: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
vim .env
```

### Step 4: Deploy Smart Contracts

```bash
# Start local blockchain (development only)
anvil &

# Deploy contracts
cd ../a2a_network
forge script script/DeployBDCOnly.s.sol --broadcast
```

### Step 5: Start Services

#### Option A: Shell Script
```bash
./deploy_a2a_system.sh local
```

#### Option B: Docker Compose
```bash
docker-compose up -d
```

#### Option C: Individual Services
```bash
# Start core services
python launch_data_manager.py &
python launch_catalog_manager.py &

# Start agents
python launch_agent0.py &
python launch_agent1.py &
python launch_agent2.py &
python launch_agent3.py &
python launch_agent4.py &
python launch_agent5.py &
```

### Step 6: Verify Deployment

```bash
# Run health checks
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# Run integration tests
./run_integration_tests.sh
```

## Production Deployment

### 1. Infrastructure Setup

#### AWS Infrastructure
```bash
# Use Terraform
cd deployment/terraform
terraform init
terraform apply -var-file=production.tfvars
```

#### Required AWS Services:
- EKS for Kubernetes
- RDS for databases
- ElastiCache for caching
- S3 for storage
- CloudWatch for logging
- Route53 for DNS

### 2. Security Configuration

```yaml
# deployment/security.yaml
security:
  ssl:
    enabled: true
    certificates:
      - domain: api.a2a.example.com
        cert: /etc/ssl/certs/api.crt
        key: /etc/ssl/private/api.key
  
  firewall:
    rules:
      - allow: 443/tcp
      - allow: 8545/tcp  # Blockchain RPC
      - deny: all
  
  secrets:
    manager: aws-secrets-manager
    rotation: enabled
```

### 3. Load Balancing

```nginx
# deployment/nginx.conf
upstream data_manager {
    server data-manager-1:8001;
    server data-manager-2:8001;
    server data-manager-3:8001;
}

upstream catalog_manager {
    server catalog-manager-1:8002;
    server catalog-manager-2:8002;
}

server {
    listen 443 ssl;
    server_name api.a2a.example.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location /data/ {
        proxy_pass http://data_manager;
    }
    
    location /catalog/ {
        proxy_pass http://catalog_manager;
    }
}
```

### 4. Database Setup

```sql
-- Create production databases
CREATE DATABASE a2a_production;
CREATE USER a2a_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE a2a_production TO a2a_user;

-- Create tables and indexes
\i deployment/sql/schema.sql
\i deployment/sql/indexes.sql
```

### 5. Monitoring Setup

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'a2a-agents'
    static_configs:
      - targets:
        - data-manager:8001
        - catalog-manager:8002
        - agent0:8003
        - agent1:8004
        - agent2:8005
        - agent3:8008
        - agent4:8006
        - agent5:8007

  - job_name: 'blockchain'
    static_configs:
      - targets:
        - ethereum-node:8545
```

## Post-Deployment Tasks

### 1. Smoke Tests

```bash
# Test each agent endpoint
for port in 8001 8002 8003 8004 8005 8006 8007 8008; do
    echo "Testing port $port..."
    curl -s http://localhost:$port/health | jq .
done
```

### 2. Configure Monitoring

```bash
# Access Grafana
open http://localhost:3001

# Default credentials
Username: admin
Password: admin
```

### 3. Set Up Backups

```bash
# Schedule automated backups
crontab -e
0 2 * * * /opt/a2a/scripts/backup.sh
```

### 4. Configure Alerts

```yaml
# alerting/rules.yml
groups:
  - name: a2a_alerts
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 5m
        annotations:
          summary: "Service {{ $labels.job }} is down"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate on {{ $labels.job }}"
```

## Scaling Guidelines

### Horizontal Scaling

```yaml
# Agent scaling recommendations
services:
  data_manager:
    min_replicas: 2
    max_replicas: 10
    target_cpu: 70%
    
  catalog_manager:
    min_replicas: 2
    max_replicas: 5
    target_cpu: 70%
    
  agents:
    min_replicas: 1
    max_replicas: 5
    target_cpu: 80%
```

### Vertical Scaling

```yaml
# Resource recommendations by load
small_load:  # < 100 req/sec
  cpu: 2
  memory: 4Gi
  
medium_load:  # 100-1000 req/sec
  cpu: 4
  memory: 8Gi
  
high_load:  # > 1000 req/sec
  cpu: 8
  memory: 16Gi
```

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   docker-compose logs agent0
   
   # Check port availability
   lsof -i:8003
   ```

2. **Blockchain connection issues**
   ```bash
   # Test RPC connection
   curl -X POST http://localhost:8545 \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
   ```

3. **Database connection failures**
   ```bash
   # Test database connectivity
   python -c "import psycopg2; conn = psycopg2.connect('${DATABASE_URL}')"
   ```

### Debug Mode

```bash
# Enable debug logging
export A2A_LOG_LEVEL=DEBUG
export A2A_DEBUG=true

# Start with verbose output
./deploy_a2a_system.sh local --debug
```

## Maintenance

### Regular Tasks

1. **Daily**
   - Check service health
   - Review error logs
   - Monitor resource usage

2. **Weekly**
   - Update dependencies
   - Run security scans
   - Review performance metrics

3. **Monthly**
   - Update SSL certificates
   - Audit access logs
   - Performance optimization

### Upgrade Process

```bash
# 1. Backup current state
./scripts/backup.sh

# 2. Deploy to staging
./deploy_a2a_system.sh staging

# 3. Run tests
./run_integration_tests.sh

# 4. Deploy to production
./deploy_a2a_system.sh production

# 5. Verify deployment
./scripts/verify_deployment.sh
```

## Support

For deployment assistance:
- Documentation: https://docs.a2a.example.com
- Support: support@a2a.example.com
- Emergency: +1-xxx-xxx-xxxx

---

Last Updated: 2025-01-07