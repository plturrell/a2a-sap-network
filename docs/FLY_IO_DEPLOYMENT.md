# Fly.io Deployment Guide for A2A Platform

## Overview
This guide covers the deployment of the A2A Platform to Fly.io with full optimization for production use.

## Prerequisites
- Fly.io account with API token
- Docker Hub account (for pulling the image)
- `flyctl` CLI installed

## Deployment Steps

### 1. Set up Fly.io API Token
```bash
export FLY_API_TOKEN=your-fly-api-token
```

### 2. Deploy Using Script
```bash
./scripts/deploy-fly.sh
```

### 3. Manual Deployment (Alternative)
```bash
# Create app if needed
flyctl apps create a2a-platform --org personal

# Deploy with Docker image
flyctl deploy --app a2a-platform --image finsightintelligence/a2a:main

# Scale appropriately
flyctl scale count 1 --app a2a-platform
```

## Configuration Details

### fly.toml Configuration
- **Region**: `ord` (Chicago)
- **VM Size**: 8 CPUs, 16GB RAM (performance tier)
- **Auto-restart**: Enabled with 10 retries
- **Health checks**: Every 60s with 600s grace period
- **Main port**: 8000 (exposed via HTTPS)

### Environment Variables
- `A2A_ENVIRONMENT`: production
- `ENABLE_ALL_AGENTS`: true
- `STARTUP_MODE`: complete
- `AGENT_STARTUP_MODE`: quick (for faster startup)

### Startup Modes
1. **quick**: Starts only Agent 0 for minimal footprint
2. **backend**: Starts agents 0-5 (core agents)
3. **complete**: Starts all 18 agents + services

## Monitoring

### Health Check
```bash
curl https://a2a-platform.fly.dev/health
```

### Monitoring Dashboard
```bash
curl https://a2a-platform.fly.dev/api/v1/monitoring/dashboard
```

### Metrics (Prometheus format)
```bash
curl https://a2a-platform.fly.dev/api/v1/monitoring/metrics
```

### Agent Status
```bash
curl https://a2a-platform.fly.dev/api/v1/monitoring/agents/status
```

### System Alerts
```bash
curl https://a2a-platform.fly.dev/api/v1/monitoring/alerts
```

## Troubleshooting

### View Logs
```bash
flyctl logs --app a2a-platform
```

### SSH into Container
```bash
flyctl ssh console --app a2a-platform
```

### Check Status
```bash
flyctl status --app a2a-platform
```

### Restart App
```bash
flyctl apps restart a2a-platform
```

### Scale Resources
```bash
# Scale up
flyctl scale vm performance-2x --app a2a-platform

# Add more memory
flyctl scale memory 32768 --app a2a-platform
```

## Performance Optimization

### 1. Startup Optimization
The platform uses an optimized startup sequence:
- Critical agents (0-5) start first with retry logic
- Remaining agents start in parallel
- Health checks ensure services are ready

### 2. Resource Management
- Automatic restart on failure
- Memory monitoring via dashboard
- CPU usage tracking
- Alert system for resource issues

### 3. Network Optimization
- Single entry point via port 8000
- Internal service communication
- Health check optimization

## Security Considerations

1. **HTTPS Only**: Force HTTPS enabled
2. **Environment Isolation**: Production environment variables
3. **Non-root User**: Services run as `a2auser`
4. **Health Endpoints**: No sensitive data exposed

## Deployment Workflow

### GitHub Actions Integration
The deployment is automated via GitHub Actions:
1. Push to `main` branch triggers build
2. Docker image built and pushed to Docker Hub
3. Fly.io deployment triggered automatically
4. Health checks verify deployment

### Manual Deployment
```bash
# Build locally (optional)
docker build -t finsightintelligence/a2a:main .

# Push to Docker Hub
docker push finsightintelligence/a2a:main

# Deploy to Fly.io
./scripts/deploy-fly.sh
```

## Rollback Procedure

### Quick Rollback
```bash
flyctl releases list --app a2a-platform
flyctl deploy --app a2a-platform --image finsightintelligence/a2a:previous-tag
```

### Full Rollback
1. Identify working release: `flyctl releases list`
2. Rollback: `flyctl rollback <version>`
3. Verify: `flyctl status`

## Cost Optimization

### Recommendations
1. Use `auto_stop_machines = false` to prevent cold starts
2. Set appropriate concurrency limits
3. Monitor resource usage via dashboard
4. Scale down during low-traffic periods

### Resource Sizing
- **Development**: 2 CPUs, 4GB RAM
- **Staging**: 4 CPUs, 8GB RAM  
- **Production**: 8 CPUs, 16GB RAM

## Support

### Logs Location
- Application logs: `/app/logs/`
- Agent logs: `/app/logs/agent*.log`
- System logs: `flyctl logs`

### Debug Mode
Set environment variable: `LOG_LEVEL=DEBUG`

### Contact
For issues, check:
1. Monitoring dashboard
2. System alerts
3. Application logs
4. Fly.io status page