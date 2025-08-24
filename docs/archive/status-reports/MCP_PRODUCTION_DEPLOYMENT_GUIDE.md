# MCP Production Deployment Guide

## Executive Summary

This guide provides comprehensive instructions for deploying the A2A MCP (Model Context Protocol) servers in a production environment. The infrastructure has been enhanced with commercial-grade security, monitoring, and operational features.

## Critical Security Improvements Implemented

### 1. **Security Vulnerability Fixes**
- ✅ **REMOVED eval() vulnerability** in `mcpValidationTools.py` - replaced with safe AST-based validation
- ✅ **Removed hardcoded credentials** - all secrets now use environment variables
- ✅ **Added JWT-based authentication** - production-ready auth middleware for all endpoints

### 2. **Production Infrastructure**
- ✅ **Graceful shutdown** - proper signal handling (SIGINT/SIGTERM)
- ✅ **Health checks** - `/health` endpoints on all servers
- ✅ **Structured logging** - JSON logs with correlation IDs
- ✅ **Circuit breakers** - prevent cascading failures
- ✅ **CORS middleware** - configurable cross-origin support
- ✅ **Error handling** - global exception handlers with proper status codes

### 3. **Configuration Management**
- ✅ **Environment-based configuration** - no hardcoded values
- ✅ **Security headers** - X-Frame-Options, CSP, HSTS, etc.
- ✅ **Rate limiting** - protect against DoS attacks

## Pre-Deployment Checklist

### Environment Setup
1. **Copy environment template**:
   ```bash
   cp .env.template .env
   ```

2. **Set all required secrets**:
   ```bash
   # CRITICAL: Change ALL default passwords
   GRAFANA_ADMIN_PASSWORD=<secure-password>
   JWT_SECRET_KEY=<32-char-secure-key>
   API_KEY_1=<api-key>:<client-id>:<permissions>
   ```

3. **Configure TLS certificates**:
   ```bash
   # Place certificates in designated paths
   cp your-cert.pem /etc/ssl/certs/a2a.crt
   cp your-key.pem /etc/ssl/private/a2a.key
   ```

### Dependencies
```bash
# Install production dependencies
pip install -r requirements.txt

# Verify critical dependencies
python -c "import fastapi, uvicorn, jwt, pydantic; print('✅ Dependencies OK')"
```

## Deployment Steps

### 1. Build Docker Images
```bash
# Build all MCP server images
docker build -t a2a-mcp-base -f Dockerfile .

# Or use docker-compose
docker-compose -f docker/docker-compose.infrastructure.yml build
```

### 2. Deploy Infrastructure Services
```bash
# Start infrastructure (Redis, Prometheus, Grafana, etc.)
docker-compose -f docker/docker-compose.infrastructure.yml up -d

# Verify infrastructure
docker-compose -f docker/docker-compose.infrastructure.yml ps
```

### 3. Deploy MCP Servers

#### Option A: Using Service Manager
```bash
# Start all MCP servers
python -m app.a2a.mcp.service_manager start --production

# Check health
python -m app.a2a.mcp.service_manager health
```

#### Option B: Using start.sh Script
```bash
# Start entire A2A system including MCP servers
./start.sh
```

#### Option C: Individual Server Deployment
```bash
# Start specific server
python app/a2a/mcp/servers/semantic_similarity_mcp_server.py &
```

### 4. Verify Deployment
```bash
# Check all health endpoints
for port in {8101..8109}; do
  echo "Checking port $port..."
  curl -s http://localhost:$port/health | jq .
done
```

## MCP Server Inventory

| Service | Port | Tools | Purpose |
|---------|------|-------|---------|
| data_standardization | 8101 | 12 | Data format standardization |
| vector_similarity | 8102 | 5 | Vector embeddings and similarity |
| vector_ranking | 8103 | 5 | Ranking and scoring vectors |
| transport_layer | 8104 | 2 | Network transport protocols |
| reasoning_agent | 8105 | 9 | Logical reasoning and inference |
| session_management | 8106 | 3 | Session handling and auth |
| resource_streaming | 8107 | 2 | Real-time data streaming |
| confidence_calculator | 8108 | 6 | Confidence scoring |
| semantic_similarity | 8109 | 7 | Semantic analysis |

## Security Configuration

### API Authentication
```bash
# Generate API keys
python -c "import secrets; print(f'API_KEY_1={secrets.token_urlsafe(32)}:client1:read,write')"

# Set in environment
export API_KEY_1=<generated-key>:client1:read,write
```

### JWT Configuration
```bash
# Generate JWT secret
python -c "import secrets; print(f'JWT_SECRET_KEY={secrets.token_urlsafe(32)}')"

# Configure expiration
export JWT_EXPIRATION_HOURS=24
```

### Rate Limiting
```bash
# Configure rate limits
export RATE_LIMIT_REQUESTS=100
export RATE_LIMIT_WINDOW=60  # seconds
```

## Monitoring and Observability

### Prometheus Metrics
- Access: http://localhost:9090
- All MCP servers expose metrics at `/metrics`

### Grafana Dashboards
- Access: http://localhost:3000
- Default login: admin / ${GRAFANA_ADMIN_PASSWORD}
- Import dashboards from `monitoring/grafana/dashboards/`

### Log Aggregation
```bash
# View logs with correlation ID
docker logs a2a-mcp-semantic-similarity | jq 'select(.correlation_id=="<id>")'

# Monitor errors
docker logs -f a2a-mcp-reasoning | jq 'select(.level=="ERROR")'
```

### Circuit Breaker Status
```bash
# Check circuit breaker health
curl http://localhost:8105/circuit-breakers | jq .
```

## Production Best Practices

### 1. **Load Balancing**
```nginx
upstream mcp_semantic {
    server localhost:8109;
    server localhost:8209;  # Second instance
    server localhost:8309;  # Third instance
}
```

### 2. **Health Checks**
```yaml
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 8109
  initialDelaySeconds: 10
  periodSeconds: 5
```

### 3. **Resource Limits**
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### 4. **Backup Strategy**
- Use persistent volumes for stateful components
- Regular snapshots of configuration
- Disaster recovery procedures documented

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   lsof -i :8101
   # Kill if necessary
   kill -9 <PID>
   ```

2. **Authentication Failures**
   - Verify JWT_SECRET_KEY is set
   - Check API key format
   - Ensure headers are properly sent

3. **Circuit Breaker Open**
   ```bash
   # Reset circuit breakers
   curl -X POST http://localhost:8105/circuit-breakers/reset
   ```

### Debug Mode
```bash
# Enable debug logging
export MCP_LOG_LEVEL=DEBUG

# Run with verbose output
python -m app.a2a.mcp.service_manager start --verbose
```

## Maintenance

### Rolling Updates
```bash
# Update one service at a time
python -m app.a2a.mcp.service_manager stop semantic_similarity
docker pull a2a-mcp-semantic:latest
python -m app.a2a.mcp.service_manager start semantic_similarity
```

### Backup Configuration
```bash
# Backup all configuration
tar -czf mcp-config-backup-$(date +%Y%m%d).tar.gz \
  .env \
  app/a2a/mcp/servers/service_registry.json \
  docker/docker-compose.infrastructure.yml
```

## Security Audit Checklist

- [ ] All default passwords changed
- [ ] TLS enabled for production
- [ ] API keys rotated regularly
- [ ] Rate limiting configured
- [ ] CORS origins restricted
- [ ] Security headers enabled
- [ ] Logs not exposing sensitive data
- [ ] Regular security updates applied

## Support

For issues or questions:
- Check logs: `docker logs <container-name>`
- Review metrics: http://localhost:9090
- Monitor dashboards: http://localhost:3000
- Health status: `curl http://localhost:<port>/health`

## Next Steps

1. **Enable TLS**: Configure nginx with SSL certificates
2. **Set up monitoring alerts**: Configure Prometheus alerting rules
3. **Implement backup automation**: Schedule regular backups
4. **Load testing**: Verify performance under load
5. **Security scanning**: Run regular vulnerability scans

---

**WARNING**: Never deploy with default credentials or without proper authentication configured. All MCP servers must be protected in production environments.