# Fly.io Deployment Guide for A2A Platform

## Overview

The A2A Platform can be deployed to Fly.io for global edge deployment with automatic scaling and high availability.

## Prerequisites

1. Fly.io account
2. `flyctl` CLI installed locally (for manual deployments)
3. `FLY_API_TOKEN` secret configured in GitHub repository

## Automatic Deployment

The A2A Platform automatically deploys to Fly.io on every push to the `main` branch via GitHub Actions.

### Deployment Pipeline

1. **Build**: Docker image built using Docker Build Cloud
2. **Test**: 18-step verification ensures all components work
3. **UI Test**: Complete platform UI testing  
4. **Deploy**: Automatic deployment to Fly.io

## Configuration

### fly.toml

The `fly.toml` file configures:

- **App Name**: `a2a-platform`
- **Primary Region**: `ord` (Chicago)
- **Resources**: 4 CPUs, 4GB RAM
- **Services**:
  - Port 80/443: Main application
  - Port 3000: Launch Pad UI
  - Port 4004: A2A Network API
  - Port 8000: Agent 0 (expandable to 8000-8017)

### Environment Variables

- `A2A_ENVIRONMENT`: Set to `production`
- `ENABLE_ALL_AGENTS`: Enables all 18 agents
- `A2A_NETWORK_ENABLED`: Enables network services
- `FRONTEND_ENABLED`: Enables frontend UI

## Manual Deployment

```bash
# Authenticate with Fly.io
flyctl auth login

# Deploy the application
flyctl deploy

# Check deployment status
flyctl status

# View logs
flyctl logs

# Scale resources
flyctl scale vm shared-cpu-4x --memory 4096
```

## Accessing the Deployed Platform

Once deployed, access the A2A Platform at:

- **Launch Pad**: https://a2a-platform.fly.dev:3000
- **A2A Network API**: https://a2a-platform.fly.dev:4004/api/v1
- **Agent Services**: https://a2a-platform.fly.dev:8000-8017
- **API Documentation**: https://a2a-platform.fly.dev:8000/docs

## Monitoring

### Health Checks

Fly.io performs TCP health checks every 15 seconds on configured ports.

### Metrics

Prometheus metrics available at: https://a2a-platform.fly.dev:9091/metrics

### Logs

```bash
# Stream live logs
flyctl logs -f

# View recent logs
flyctl logs -n 100
```

## Scaling

### Horizontal Scaling

```bash
# Scale to multiple instances
flyctl scale count 3

# Scale by region
flyctl regions add sin hkg nrt
```

### Vertical Scaling

```bash
# Upgrade to dedicated CPU
flyctl scale vm dedicated-cpu-2x --memory 8192
```

## Troubleshooting

### Common Issues

1. **Deployment Timeout**
   - Increase `wait-timeout` in deployment command
   - Check if all 18 steps pass in CI

2. **Memory Issues**
   - Scale up VM memory
   - Enable swap: `flyctl scale memory 8192`

3. **Port Access**
   - Ensure services are configured in fly.toml
   - Check firewall rules

### Debug Commands

```bash
# SSH into running instance
flyctl ssh console

# Check running processes
flyctl ssh console -C "ps aux"

# View environment
flyctl ssh console -C "env"
```

## Security

1. **Secrets Management**
   ```bash
   flyctl secrets set DATABASE_URL=postgres://...
   flyctl secrets set API_KEY=...
   ```

2. **Private Networking**
   - Internal services communicate via `.internal` domains
   - External access only through configured ports

3. **SSL/TLS**
   - Automatic SSL certificates for all domains
   - Force HTTPS redirects enabled

## Backup and Recovery

1. **Database Backups**
   - Configure automated PostgreSQL backups
   - Use Fly.io volumes for persistent storage

2. **Disaster Recovery**
   - Multi-region deployment for high availability
   - Automated failover between regions

## Cost Optimization

- Use shared CPU for development/staging
- Scale down during off-peak hours
- Monitor usage with `flyctl dashboard`

## CI/CD Integration

The GitHub Actions workflow handles:
1. Building with Docker Build Cloud
2. Testing with 18-step verification
3. Deploying to Fly.io automatically
4. Health check verification post-deployment

To manually trigger deployment:
```bash
gh workflow run build-and-deploy.yml -f test_type=complete
```