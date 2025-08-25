# A2A Platform Scripts Directory

This directory contains all operational scripts for the A2A Platform, organized by purpose.

## Directory Structure

```
scripts/
├── deployment/           # Deployment scripts
│   ├── fly/             # Fly.io specific deployment
│   └── local/           # Local deployment (future)
├── operations/          # Day-to-day operations
├── testing/             # Testing and verification
└── [symlinks]           # Backward compatibility symlinks
```

## 🚀 Deployment Scripts

### Fly.io Deployment (`deployment/fly/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy-fly.sh` | Standard Fly.io deployment | `./scripts/deploy-fly.sh` |
| `deploy-fly-zero-downtime.sh` | Zero-downtime deployment with blue-green strategy | `./scripts/deploy-fly-zero-downtime.sh` |
| `deploy-staging.sh` | Deploy to staging environment with tests | `./scripts/deploy-staging.sh` |
| `manage-fly-secrets.sh` | Manage Fly.io secrets and environment variables | `./scripts/manage-fly-secrets.sh [command]` |
| `fly-db-migrate.sh` | Database migration management | `./scripts/fly-db-migrate.sh [command]` |
| `setup-custom-domain.sh` | Configure custom domains with SSL | `./scripts/setup-custom-domain.sh [domain]` |
| `start-fly.sh` | Optimized startup script for Fly.io | Used internally by Docker |
| `validate-fly-deployment.sh` | Validate deployment health | `./scripts/validate-fly-deployment.sh [app-name]` |

### Quick Start - Fly.io
```bash
# First time setup
export FLY_API_TOKEN=your-token
./scripts/manage-fly-secrets.sh setup-production

# Deploy to production
./scripts/deploy-fly-zero-downtime.sh

# Deploy to staging
./scripts/deploy-staging.sh

# Check deployment
./scripts/validate-fly-deployment.sh
```

## 🔧 Operations Scripts (`operations/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `start.sh` | Start A2A platform locally | `./scripts/start.sh [mode]` |
| `stop.sh` | Stop all A2A services | `./scripts/stop.sh` |
| `status.sh` | Check service status | `./scripts/status.sh` |
| `start-all-agents.sh` | Start all 18 agents | `./scripts/start-all-agents.sh` |

### Startup Modes
- `quick` - Start minimal services (Agent 0 only)
- `backend` - Start core agents (0-5)
- `complete` - Start all services (default)
- `test` - Run verification tests
- `verify` - Run 18-step verification

### Local Development
```bash
# Start in development mode
./scripts/start.sh

# Start specific mode
./scripts/start.sh backend

# Check status
./scripts/status.sh

# Stop all services
./scripts/stop.sh
```

## 🧪 Testing Scripts (`testing/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `verify-18-steps.sh` | Verify all 18 startup steps | `./scripts/verify-18-steps.sh` |

## 📋 Common Workflows

### 1. Deploy New Feature
```bash
# 1. Deploy to staging
./scripts/deploy-staging.sh

# 2. Test staging
./scripts/validate-fly-deployment.sh a2a-platform-staging

# 3. If tests pass, promote to production
# (The staging script will prompt for this)
```

### 2. Update Secrets
```bash
# View current secrets
./scripts/manage-fly-secrets.sh list

# Add new secret
./scripts/manage-fly-secrets.sh set API_KEY "new-value"

# Import from .env file
./scripts/manage-fly-secrets.sh import .env.production
```

### 3. Database Operations
```bash
# Check migration status
./scripts/fly-db-migrate.sh status

# Create new migration
./scripts/fly-db-migrate.sh create "add feature table"

# Run migrations
./scripts/fly-db-migrate.sh migrate

# Backup database
./scripts/fly-db-migrate.sh backup
```

### 4. Setup Custom Domain
```bash
# Add domain
./scripts/setup-custom-domain.sh setup api.example.com

# Check status
./scripts/setup-custom-domain.sh status api.example.com
```

### 5. Emergency Rollback
```bash
# List releases
flyctl releases list --app a2a-platform

# Rollback to previous version
flyctl rollback --app a2a-platform
```

## 🔗 Backward Compatibility

All scripts maintain backward compatibility through symlinks in the root scripts directory. You can continue using:
```bash
./scripts/deploy-fly.sh
# Instead of
./scripts/deployment/fly/deploy-fly.sh
```

## 📚 Script Details

### Deployment Scripts

#### `deploy-fly-zero-downtime.sh`
- Uses blue-green deployment strategy
- Pre-deployment health checks
- Automatic rollback on failure
- Post-deployment validation
- Cleanup of old machines

#### `manage-fly-secrets.sh`
Commands:
- `list` - List all secrets
- `set KEY VALUE` - Set a secret
- `unset KEY` - Remove a secret
- `import FILE` - Import from .env file
- `setup-production` - Setup all production secrets
- `setup-staging` - Setup staging secrets
- `validate` - Check required secrets

#### `fly-db-migrate.sh`
Commands:
- `status` - Show migration status
- `init` - Initialize migrations
- `create NAME` - Create new migration
- `migrate` - Run pending migrations
- `rollback [N]` - Rollback N migrations
- `backup` - Create database backup
- `validate` - Validate database connection

### Operations Scripts

#### `start.sh`
The main startup script supporting multiple modes:
- Handles both local and container environments
- Manages virtual environments
- Starts services in correct order
- Performs health checks
- Supports different startup modes

#### `status.sh`
Shows status of:
- All 18 agents
- Core services (Network, Frontend)
- Database connections
- Health endpoints

### Testing Scripts

#### `verify-18-steps.sh`
Validates the complete A2A platform startup:
1. Pre-flight checks
2. Environment setup
3. Infrastructure services
4. Blockchain services
5. Core services
6. Trust system
7. Agent services (0-17)
8. MCP servers
9. API gateway
10. Health checks

## 🛠️ Maintenance

### Adding New Scripts
1. Place in appropriate subdirectory
2. Make executable: `chmod +x script.sh`
3. Add symlink if needed for backward compatibility
4. Update this README
5. Add to relevant documentation

### Script Standards
- Use `#!/bin/bash` shebang
- Add descriptive comments
- Include usage instructions
- Handle errors gracefully
- Use consistent naming (kebab-case)
- Add color output for better UX

## 🆘 Troubleshooting

### Script Not Found
```bash
# Ensure script is executable
chmod +x scripts/path/to/script.sh

# Check symlink
ls -la scripts/script-name.sh
```

### Permission Denied
```bash
# Fix permissions
find scripts -name "*.sh" -exec chmod +x {} \;
```

### Path Issues
All scripts should be run from the project root:
```bash
cd /path/to/a2a
./scripts/script-name.sh
```

## 📞 Support

For script issues:
1. Check script has execute permissions
2. Verify you're in the project root directory
3. Check environment variables are set
4. Review script output for error messages
5. Check logs in `/app/logs/` (in container) or `./logs/` (local)