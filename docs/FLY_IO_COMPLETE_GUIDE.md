# Complete Fly.io Deployment Guide for A2A Platform

## Table of Contents
1. [Overview](#overview)
2. [Initial Setup](#initial-setup)
3. [Deployment Strategies](#deployment-strategies)
4. [Monitoring & Logging](#monitoring--logging)
5. [Security & Secrets](#security--secrets)
6. [Database Management](#database-management)
7. [Staging Environment](#staging-environment)
8. [Custom Domains](#custom-domains)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Overview

The A2A Platform is fully optimized for Fly.io deployment with:
- ✅ Zero-downtime deployments
- ✅ Comprehensive monitoring
- ✅ Log aggregation
- ✅ Staging environment
- ✅ Database migrations
- ✅ Custom domain support
- ✅ Automatic scaling
- ✅ Health checks

## Initial Setup

### Prerequisites
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login to Fly.io
flyctl auth login

# Set API token
export FLY_API_TOKEN=your-token
```

### Quick Deploy
```bash
# Deploy with single command
./scripts/deploy-fly.sh
```

## Deployment Strategies

### 1. Standard Deployment
```bash
flyctl deploy --app a2a-platform
```

### 2. Zero-Downtime Deployment
```bash
./scripts/deploy-fly-zero-downtime.sh
```

### 3. Staging Deployment
```bash
./scripts/deploy-staging.sh
```

### 4. GitHub Actions (Automated)
Push to `main` branch triggers automatic deployment

## Monitoring & Logging

### Health Endpoints
- **Basic Health**: `https://a2a-platform.fly.dev/health`
- **Monitoring Dashboard**: `https://a2a-platform.fly.dev/api/v1/monitoring/dashboard`
- **Metrics**: `https://a2a-platform.fly.dev/api/v1/monitoring/metrics`
- **Agent Status**: `https://a2a-platform.fly.dev/api/v1/monitoring/agents/status`
- **Alerts**: `https://a2a-platform.fly.dev/api/v1/monitoring/alerts`

### Log Aggregation
- **View Logs**: `https://a2a-platform.fly.dev/api/v1/logs/stream`
- **Error Logs**: `https://a2a-platform.fly.dev/api/v1/logs/errors`
- **Search Logs**: `https://a2a-platform.fly.dev/api/v1/logs/search?query=error`
- **Service Logs**: `https://a2a-platform.fly.dev/api/v1/logs/tail/agent0`
- **Statistics**: `https://a2a-platform.fly.dev/api/v1/logs/statistics`

### Real-time Monitoring
```bash
# View live logs
flyctl logs --app a2a-platform

# SSH into container
flyctl ssh console --app a2a-platform

# View metrics
curl https://a2a-platform.fly.dev/api/v1/monitoring/metrics
```

## Security & Secrets

### Managing Secrets
```bash
# List secrets
./scripts/manage-fly-secrets.sh list

# Set a secret
./scripts/manage-fly-secrets.sh set JWT_SECRET_KEY mysecret

# Import from .env file
./scripts/manage-fly-secrets.sh import .env.production

# Validate required secrets
./scripts/manage-fly-secrets.sh validate

# Setup production secrets
./scripts/manage-fly-secrets.sh setup-production
```

### Required Secrets
- `JWT_SECRET_KEY` - Authentication token secret
- `SESSION_SECRET_KEY` - Session encryption key
- `DATABASE_URL` - PostgreSQL connection string

### Recommended Secrets
- `REDIS_URL` - Redis connection for caching
- `OPENAI_API_KEY` - For AI features
- `SENTRY_DSN` - Error tracking
- `SAP_CLIENT_ID` - SAP integration
- `SAP_CLIENT_SECRET` - SAP authentication

## Database Management

### Running Migrations
```bash
# Check migration status
./scripts/fly-db-migrate.sh status

# Create new migration
./scripts/fly-db-migrate.sh create "add user table"

# Run migrations
./scripts/fly-db-migrate.sh migrate

# Rollback migration
./scripts/fly-db-migrate.sh rollback -1

# Create backup
./scripts/fly-db-migrate.sh backup
```

### Database Operations
```bash
# Connect to database
flyctl postgres connect --app a2a-platform-db

# Create database backup
flyctl postgres backup create --app a2a-platform-db

# List backups
flyctl postgres backup list --app a2a-platform-db
```

## Staging Environment

### Deploy to Staging
```bash
# Full staging deployment with tests
./scripts/deploy-staging.sh

# Manual staging deployment
flyctl deploy --app a2a-platform-staging --config fly.staging.toml
```

### Staging Configuration
- Smaller VM (4 CPUs, 8GB RAM)
- Auto-stop enabled to save costs
- Debug logging enabled
- Separate database

### Promotion to Production
After testing staging, the script will prompt to promote to production

## Custom Domains

### Setup Custom Domain
```bash
# Add custom domain
./scripts/setup-custom-domain.sh setup api.example.com

# Check status
./scripts/setup-custom-domain.sh status api.example.com

# Remove domain
./scripts/setup-custom-domain.sh remove api.example.com
```

### DNS Configuration
Add these records to your DNS provider:
- A record: `@` → `<fly-ipv4>`
- AAAA record: `@` → `<fly-ipv6>`

## Troubleshooting

### Common Issues

#### 1. Deployment Fails
```bash
# Check logs
flyctl logs --app a2a-platform

# SSH and debug
flyctl ssh console --app a2a-platform
cd /app
./start.sh test
```

#### 2. Health Check Failures
```bash
# Validate deployment
./scripts/validate-fly-deployment.sh

# Check specific agent
curl https://a2a-platform.fly.dev/api/v1/monitoring/agents/status
```

#### 3. Memory Issues
```bash
# Scale up memory
flyctl scale memory 32768 --app a2a-platform

# Check resource usage
flyctl status --app a2a-platform
```

#### 4. Slow Startup
```bash
# Use quick startup mode
flyctl secrets set STARTUP_MODE=quick --app a2a-platform
```

### Debug Commands
```bash
# View app info
flyctl info --app a2a-platform

# List machines
flyctl machines list --app a2a-platform

# Restart app
flyctl apps restart a2a-platform

# View releases
flyctl releases list --app a2a-platform

# Rollback
flyctl rollback <version> --app a2a-platform
```

## Best Practices

### 1. Resource Optimization
- Use `STARTUP_MODE=quick` for faster boots
- Enable `auto_stop_machines` for staging
- Monitor memory usage via dashboard
- Scale based on load

### 2. Security
- Rotate secrets regularly
- Use separate secrets for staging/production
- Enable HTTPS only
- Monitor security alerts

### 3. Deployment
- Always test in staging first
- Use zero-downtime deployments
- Monitor after deployment
- Keep rollback plan ready

### 4. Monitoring
- Check dashboard regularly
- Set up alerts for critical issues
- Monitor error patterns
- Track performance metrics

### 5. Cost Management
- Use auto-stop for non-production
- Right-size VMs based on usage
- Monitor resource utilization
- Clean up old deployments

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `deploy-fly.sh` | Standard deployment |
| `deploy-fly-zero-downtime.sh` | Zero-downtime deployment |
| `deploy-staging.sh` | Deploy to staging |
| `manage-fly-secrets.sh` | Manage secrets |
| `fly-db-migrate.sh` | Database migrations |
| `setup-custom-domain.sh` | Configure domains |
| `validate-fly-deployment.sh` | Validate deployment |
| `start-fly.sh` | Optimized startup |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `A2A_ENVIRONMENT` | Environment name | production |
| `STARTUP_MODE` | Startup mode (quick/backend/complete) | complete |
| `LOG_LEVEL` | Logging level | INFO |
| `ENABLE_ALL_AGENTS` | Start all agents | true |
| `DATABASE_TIMEOUT` | DB connection timeout | 30000 |

## Monitoring URLs

- **Main App**: https://a2a-platform.fly.dev
- **API Docs**: https://a2a-platform.fly.dev/docs
- **Monitoring**: https://a2a-platform.fly.dev/api/v1/monitoring/dashboard
- **Logs**: https://a2a-platform.fly.dev/api/v1/logs/stream
- **Metrics**: https://a2a-platform.fly.dev/api/v1/monitoring/metrics

## Support

For issues:
1. Check monitoring dashboard for alerts
2. Review logs via aggregator
3. Validate deployment health
4. Check Fly.io status page
5. Review this guide's troubleshooting section

## Conclusion

The A2A Platform is fully optimized for production deployment on Fly.io with comprehensive monitoring, logging, and deployment automation. Follow this guide for successful deployments and operations.