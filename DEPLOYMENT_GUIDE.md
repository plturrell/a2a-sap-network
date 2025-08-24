# A2A Platform Docker Deployment Guide

## 🚀 Quick Start

### Pull and Run the Complete Platform
```bash
# Pull the latest image
docker pull ghcr.io/plturrell/a2a-sap-network:main

# Run complete A2A platform with all services
docker run -d --name a2a-platform \
  -p 3000:3000 \
  -p 4004:4004 \
  -p 8000-8017:8000-8017 \
  -e ENABLE_ALL_AGENTS=true \
  -e A2A_NETWORK_ENABLED=true \
  -e FRONTEND_ENABLED=true \
  -e ENABLE_BLOCKCHAIN=true \
  -v a2a-data:/app/data \
  -v a2a-logs:/app/logs \
  ghcr.io/plturrell/a2a-sap-network:main start complete
```

## 📋 Container Information

**Image Details:**
- **Registry:** `ghcr.io/plturrell/a2a-sap-network:main`
- **Digest:** `sha256:5d51cda3b14cf2b02bde12a6e1fa58a4a315f4d8a73bd29bb5d8d90be9f9c2e3`
- **Architectures:** `linux/amd64`, `linux/arm64`
- **Size:** Multi-layer optimized build

**Included Features:**
- ✅ All 16 A2A Agents with AI enhancements
- ✅ Grok AI integration (Agents 0-3)
- ✅ Perplexity API integration
- ✅ PDF processing with OCR
- ✅ Vector processing with sklearn fallbacks
- ✅ SAP Fiori Launchpad UI
- ✅ Blockchain integration
- ✅ Enterprise security features

## 🌐 Access Points

Once running, access the platform at:

| Service | URL | Description |
|---------|-----|-------------|
| **Launch Pad** | http://localhost:3000 | Main SAP Fiori interface |
| **A2A Network API** | http://localhost:4004/api/v1 | Network services API |
| **Agent 0 (Data Product)** | http://localhost:8000 | Data product agent with AI |
| **Agent 1 (Standardization)** | http://localhost:8001 | Data standardization |
| **Agent 2 (AI Preparation)** | http://localhost:8002 | AI preparation services |
| **Agent 3 (Vector Processing)** | http://localhost:8003 | Vector processing |
| **Agents 4-15** | http://localhost:8004-8015 | Additional agent services |
| **API Documentation** | http://localhost:8000/docs | OpenAPI documentation |

## 🔧 Environment Variables

### Required for AI Features
```bash
# Grok AI Integration
XAI_API_KEY=your_grok_api_key_here
# OR
GROK_API_KEY=your_grok_api_key_here

# Perplexity API Integration
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

### Optional Configuration
```bash
# Service Configuration
ENABLE_ALL_AGENTS=true
A2A_NETWORK_ENABLED=true
FRONTEND_ENABLED=true
ENABLE_BLOCKCHAIN=true

# Database Configuration
DATABASE_URL=sqlite:db/a2a.db
REDIS_URL=redis://localhost:6379

# Security Configuration
SESSION_SECRET=your_session_secret
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
```

## 📦 Deployment Options

### 1. Development Mode
```bash
docker run -d --name a2a-dev \
  -p 3000:3000 -p 4004:4004 -p 8000-8017:8000-8017 \
  -e NODE_ENV=development \
  -e DEBUG=true \
  -v $(pwd)/logs:/app/logs \
  ghcr.io/plturrell/a2a-sap-network:main start complete
```

### 2. Production Mode
```bash
docker run -d --name a2a-production \
  -p 3000:3000 -p 4004:4004 -p 8000-8017:8000-8017 \
  -e NODE_ENV=production \
  -e XAI_API_KEY=${XAI_API_KEY} \
  -e PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY} \
  --restart unless-stopped \
  -v a2a-prod-data:/app/data \
  -v a2a-prod-logs:/app/logs \
  ghcr.io/plturrell/a2a-sap-network:main start complete
```

### 3. Docker Compose (Recommended)
```yaml
version: '3.8'
services:
  a2a-platform:
    image: ghcr.io/plturrell/a2a-sap-network:main
    container_name: a2a-platform
    ports:
      - "3000:3000"
      - "4004:4004"
      - "8000-8017:8000-8017"
    environment:
      - ENABLE_ALL_AGENTS=true
      - A2A_NETWORK_ENABLED=true
      - FRONTEND_ENABLED=true
      - ENABLE_BLOCKCHAIN=true
      - XAI_API_KEY=${XAI_API_KEY}
      - PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}
    volumes:
      - a2a-data:/app/data
      - a2a-logs:/app/logs
    restart: unless-stopped
    command: start complete

volumes:
  a2a-data:
  a2a-logs:
```

## 🔍 Health Checks

### Check Platform Status
```bash
# Check if services are running
curl http://localhost:4004/api/v1/health
curl http://localhost:8000/health

# View logs
docker logs a2a-platform

# Monitor resource usage
docker stats a2a-platform
```

### Verification Commands
```bash
# Run 18-step verification
docker exec a2a-platform /app/scripts/verify-18-steps.sh

# Check agent status
curl http://localhost:8000/api/agents

# Test AI features (requires API keys)
curl -X POST http://localhost:8000/api/v1/data-products \
  -H "Content-Type: application/json" \
  -d '{"query": "test AI integration"}'
```

## 🛠️ Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check logs for startup issues
docker logs a2a-platform --tail 100

# Restart with debug mode
docker run --rm -it \
  -p 3000:3000 -p 4004:4004 -p 8000-8017:8000-8017 \
  -e DEBUG=true \
  ghcr.io/plturrell/a2a-sap-network:main start complete
```

**Port conflicts:**
```bash
# Check what's using ports
lsof -i :3000
lsof -i :4004
lsof -i :8000

# Use different ports
docker run -d --name a2a-platform \
  -p 13000:3000 -p 14004:4004 -p 18000-18017:8000-8017 \
  ghcr.io/plturrell/a2a-sap-network:main start complete
```

**AI features not working:**
```bash
# Verify API keys are set
docker exec a2a-platform env | grep -E "(XAI|GROK|PERPLEXITY)"

# Test AI connectivity
docker exec a2a-platform curl -H "Authorization: Bearer $XAI_API_KEY" \
  https://api.x.ai/v1/models
```

## 🔄 Updates and Maintenance

### Update to Latest Version
```bash
# Pull latest image
docker pull ghcr.io/plturrell/a2a-sap-network:main

# Stop current container
docker stop a2a-platform
docker rm a2a-platform

# Start with new image (data persists in volumes)
docker run -d --name a2a-platform \
  -p 3000:3000 -p 4004:4004 -p 8000-8017:8000-8017 \
  -e ENABLE_ALL_AGENTS=true \
  -v a2a-data:/app/data \
  -v a2a-logs:/app/logs \
  ghcr.io/plturrell/a2a-sap-network:main start complete
```

### Backup Data
```bash
# Backup volumes
docker run --rm -v a2a-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/a2a-data-backup.tar.gz -C /data .

docker run --rm -v a2a-logs:/logs -v $(pwd):/backup \
  alpine tar czf /backup/a2a-logs-backup.tar.gz -C /logs .
```

## 📊 Monitoring

### Resource Requirements
- **Minimum:** 4GB RAM, 2 CPU cores, 10GB disk
- **Recommended:** 8GB RAM, 4 CPU cores, 50GB disk
- **Production:** 16GB RAM, 8 CPU cores, 100GB disk

### Performance Monitoring
```bash
# Monitor container resources
docker stats a2a-platform

# Check service health
curl http://localhost:4004/api/v1/health
curl http://localhost:8000/health

# View application logs
docker logs a2a-platform -f
```

## 🚀 Production Deployment

For production deployment, consider:

1. **Load Balancer:** Use nginx or similar for SSL termination
2. **Database:** External PostgreSQL or SAP HANA
3. **Redis:** External Redis cluster for caching
4. **Monitoring:** Prometheus + Grafana setup
5. **Backup:** Automated data backup strategy
6. **Security:** Network policies, secrets management
7. **Scaling:** Kubernetes deployment for high availability

## 📞 Support

- **Documentation:** Check `/app/docs/` in container
- **Logs:** Available in `/app/logs/` directory
- **Health Checks:** Built-in endpoints for monitoring
- **API Docs:** Available at http://localhost:8000/docs
