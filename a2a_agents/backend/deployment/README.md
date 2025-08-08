# A2A Network Blue/Green Deployment

## Overview

This directory contains the Blue/Green deployment system for the A2A (Agent-to-Agent) network, enabling zero-downtime deployments with automatic rollback capabilities.

## Features

- **Zero-downtime deployments** using blue/green strategy
- **Automated health checks** and service validation
- **Traffic switching** with load balancer configuration
- **Automatic rollback** on deployment failures
- **Comprehensive testing** with smoke and integration tests
- **Docker Compose** orchestration for easy service management
- **Nginx load balancing** for traffic routing

## Architecture

### Blue/Green Strategy

The system maintains two identical environments:
- **Blue Environment**: Currently active production environment
- **Green Environment**: Standby environment for new deployments

During deployment:
1. New version is deployed to the inactive environment
2. Health checks and tests are performed
3. Traffic is switched to the new environment
4. Old environment remains available for rollback

### Port Strategy

Services use different port ranges for each environment:
- **Blue Environment**: Base ports (8001-8006, 8080, 8090)
- **Green Environment**: Base ports + 1000 (9001-9006, 9080, 9090)

## Quick Start

### 1. Deploy New Version

```bash
python deployment/deploy.py deploy
```

This will:
- Deploy services to the inactive environment
- Run health checks and integration tests
- Prepare the environment for traffic switching

### 2. Switch Traffic

```bash
python deployment/deploy.py switch
```

This activates the new deployment by switching the load balancer.

### 3. Check Status

```bash
python deployment/deploy.py status
```

View the current state of both environments.

### 4. Rollback (if needed)

```bash
python deployment/deploy.py rollback
```

Switch back to the previous stable version.

## Configuration

### Deployment Configuration (`deployment_config.yaml`)

```yaml
version: "1.0.0"
services:
  - name: "data_product_agent_0"
    image: "a2a/data-product-agent:latest"
    port: 8001
    health_endpoint: "/health"
    environment_variables:
      PROMETHEUS_PORT: "8001"
    resource_limits:
      memory: "512M"

pre_deployment_checks:
  - "docker --version"

post_deployment_tests:
  - "python3 deployment/tests/smoke_test.py"

rollback_on_failure: true
health_check_timeout: 300
traffic_switch_delay: 30
```

### Service Configuration

Each service requires:
- **name**: Unique service identifier
- **image**: Docker image name and tag
- **port**: Base port number
- **health_endpoint**: Health check URL path
- **environment_variables**: Environment vars for the container
- **resource_limits**: CPU and memory constraints

## Testing

### Smoke Tests

Basic health checks to ensure services are responding:

```bash
python deployment/tests/smoke_test.py
```

Tests:
- Service health endpoints
- Basic connectivity
- Service availability

### Integration Tests

End-to-end workflow validation:

```bash
python deployment/tests/integration_test.py
```

Tests:
- API Gateway routing
- Agent communication
- Metrics endpoints
- Basic workflows

## Load Balancer Configuration

The system automatically generates Nginx configuration for traffic routing:

```nginx
upstream a2a_agents {
    server host.docker.internal:8001;  # Blue
    server host.docker.internal:9001;  # Green (during switch)
}

server {
    location /api/ {
        proxy_pass http://a2a_gateway;
    }
    
    location /agents/ {
        proxy_pass http://a2a_agents;
    }
}
```

## Deployment Workflow

### Complete Deployment Process

1. **Preparation Phase**
   ```bash
   python deployment/deploy.py deploy
   ```
   - Run pre-deployment checks
   - Generate Docker Compose configuration
   - Deploy services to inactive environment

2. **Validation Phase**
   - Health checks on all services
   - Smoke tests for basic functionality
   - Integration tests for workflows

3. **Activation Phase**
   ```bash
   python deployment/deploy.py switch
   ```
   - Update load balancer configuration
   - Switch traffic to new environment
   - Mark new environment as active

4. **Verification Phase**
   - Monitor new deployment
   - Validate metrics and logs
   - Confirm system stability

### Rollback Process

If issues are detected:

```bash
python deployment/deploy.py rollback
```

This will:
- Switch traffic back to previous environment
- Mark failed deployment for cleanup
- Restore stable service

## Directory Structure

```
deployment/
├── blue_green_deployment.py    # Core deployment logic
├── deploy.py                   # CLI interface
├── deployment_config.yaml      # Service configuration
├── tests/
│   ├── smoke_test.py          # Basic health checks
│   └── integration_test.py     # Workflow tests
├── docker-compose.blue.yml     # Generated blue environment
├── docker-compose.green.yml    # Generated green environment
├── nginx.conf                  # Generated load balancer config
└── deployment_state.json      # Current deployment state
```

## Monitoring and Observability

### Deployment Status

Monitor deployment progress:
- Environment health status
- Service availability metrics
- Test result tracking
- Rollback triggers

### Integration with A2A Monitoring

The deployment system integrates with:
- **Prometheus metrics** for service monitoring
- **Health Dashboard** for real-time status
- **Grafana dashboards** for deployment tracking
- **AlertManager** for deployment alerts

## Advanced Usage

### Custom Pre-deployment Checks

Add custom validation scripts to `pre_deployment_checks`:

```yaml
pre_deployment_checks:
  - "python scripts/validate_database.py"
  - "curl -f http://dependency-service/health"
  - "python scripts/check_prerequisites.py"
```

### Custom Post-deployment Tests

Add comprehensive testing to `post_deployment_tests`:

```yaml
post_deployment_tests:
  - "python deployment/tests/smoke_test.py"
  - "python deployment/tests/integration_test.py"
  - "python deployment/tests/performance_test.py"
  - "python deployment/tests/security_test.py"
```

### Environment Variables

Services can access deployment context:

```bash
DEPLOYMENT_ENV=blue|green
DEPLOYMENT_VERSION=1.0.0
PORT_OFFSET=0|1000
```

## Troubleshooting

### Common Issues

**Services fail to start:**
- Check Docker images are available
- Verify port conflicts
- Review resource limits

**Health checks fail:**
- Confirm health endpoint paths
- Check service startup time
- Validate network connectivity

**Traffic switch fails:**
- Verify Nginx configuration
- Check load balancer connectivity
- Review port mappings

### Debug Commands

```bash
# Check service logs
docker-compose -f docker-compose.blue.yml logs service_name

# Test health endpoint directly
curl http://localhost:8001/health

# Validate Nginx config
nginx -t -c deployment/nginx.conf

# Check port usage
netstat -tulpn | grep :8001
```

### Recovery Procedures

**Manual rollback:**
```bash
python deployment/deploy.py rollback
```

**Force cleanup:**
```bash
docker-compose -f docker-compose.blue.yml down
docker-compose -f docker-compose.green.yml down
```

**Reset deployment state:**
```bash
rm deployment/deployment_state.json
```

## Security Considerations

- Service isolation between environments
- Resource limits to prevent resource exhaustion
- Health check endpoint security
- Load balancer configuration validation
- Container image security scanning

## Performance Optimization

- Parallel service deployment
- Optimized health check intervals
- Resource-aware scheduling
- Connection pooling for load balancer
- Metrics collection optimization