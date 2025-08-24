# A2A Network Environment Configuration

This document explains how to securely configure your A2A Network environment.

## Quick Start

### Development Setup
```bash
# Option 1: Use the automated setup script
./scripts/setup-environment.sh development

# Option 2: Manual setup
cp .env.development.template .env
# Edit .env with your preferred values
```

### Production Setup
```bash
# Use the automated setup script (recommended)
./scripts/setup-environment.sh production

# This will:
# - Copy .env.template to .env
# - Generate secure random secrets
# - Set proper file permissions
# - Provide security warnings
```

## Environment Files

| File | Purpose | Status |
|------|---------|--------|
| `.env.template` | Complete configuration template for all environments | ✅ Use for production |
| `.env.development.template` | Development-specific safe defaults | ✅ Use for development |
| `.env.example` | Documentation and setup instructions | 📖 Reference only |
| `.env.production` | Production example with placeholders | 📖 Reference only |

## Security Requirements

### Development Environment
- ✅ Safe default values included
- ✅ Uses local test accounts (Anvil)
- ✅ No real secrets required
- ✅ Debug mode enabled

### Production Environment
- ⚠️ **ALL** `YOUR_*_HERE` placeholders must be replaced
- ⚠️ Generate secure secrets: `openssl rand -hex 32`
- ⚠️ Use environment-specific values
- ⚠️ Store secrets in secure secret management systems

## Key Configuration Sections

### 1. Database Configuration
```env
# Development: SQLite (local file)
DB_TYPE=sqlite
DB_CONNECTION_STRING=sqlite:./db/a2a_dev.db

# Production: SAP HANA Cloud
DB_TYPE=hana-cloud
DB_CONNECTION_STRING=your-hana-connection-string
```

### 2. Authentication & Security
```env
# CRITICAL: Replace with secure random values for production
JWT_SECRET=your-256-bit-jwt-secret-here
JWT_REFRESH_SECRET=your-256-bit-refresh-secret-here
SESSION_SECRET=your-256-bit-session-secret-here
ENCRYPTION_KEY=your-32-byte-encryption-key-here
```

### 3. Blockchain Configuration
```env
# Development: Local Anvil
A2A_RPC_URL=http://localhost:8545
A2A_CHAIN_ID=31337

# Production: Real network
A2A_RPC_URL=https://polygon-mainnet.infura.io/v3/YOUR_PROJECT_ID
A2A_CHAIN_ID=137
```

### 4. External Services
```env
# AI Services
GROK_API_KEY=your-openai-api-key-here

# Monitoring
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=your-otel-endpoint
SAP_CLOUD_ALM_TOKEN=your-sap-cloud-alm-token
```

## Security Best Practices

### ✅ DO:
- Use the automated setup script
- Generate unique secrets for each environment
- Store production secrets in secure secret managers (AWS Secrets Manager, HashiCorp Vault)
- Rotate secrets regularly
- Use environment-specific configuration
- Set proper file permissions (600) on .env files
- Monitor for exposed secrets in logs and code

### ❌ DON'T:
- Commit .env files to version control
- Use development secrets in production
- Share secrets in plain text
- Use default/example values in production
- Include secrets in log outputs
- Store secrets in application code

## Environment Variables Reference

### Core Application
| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `NODE_ENV` | Environment type | Yes | `development`, `production` |
| `PORT` | Server port | Yes | `4004` |
| `LOG_LEVEL` | Logging level | Yes | `info`, `debug` |

### Database
| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `DB_TYPE` | Database type | Yes | `sqlite`, `hana-cloud` |
| `DB_CONNECTION_STRING` | Connection string | Yes | `sqlite:./db/a2a.db` |
| `DB_POOL_SIZE` | Connection pool size | No | `10` |

### Authentication
| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `JWT_SECRET` | JWT signing secret | Yes | 32+ character string |
| `JWT_REFRESH_SECRET` | JWT refresh secret | Yes | 32+ character string |
| `SESSION_SECRET` | Session secret | Yes | 32+ character string |

### Blockchain
| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `A2A_RPC_URL` | Blockchain RPC URL | Yes | `http://localhost:8545` |
| `A2A_CHAIN_ID` | Chain ID | Yes | `31337`, `137` |
| `A2A_PRIVATE_KEY` | Agent private key | Yes | `0x...` |

## Troubleshooting

### Common Issues

1. **Missing .env file**
   ```bash
   # Run the setup script
   ./scripts/setup-environment.sh development
   ```

2. **Invalid configuration**
   ```bash
   # Check configuration validation
   npm run validate-config
   ```

3. **Permission errors**
   ```bash
   # Fix file permissions
   chmod 600 .env
   ```

4. **Database connection errors**
   - Verify `DB_CONNECTION_STRING` is correct
   - Ensure database service is running
   - Check network connectivity

5. **Blockchain connection errors**
   - Verify `A2A_RPC_URL` is accessible
   - Check `A2A_CHAIN_ID` matches the network
   - Ensure private keys are valid

### Validation Commands
```bash
# Validate environment configuration
npm run validate-config

# Test database connection
npm run test:db

# Test blockchain connection
npm run test:blockchain

# Run health checks
curl http://localhost:4004/health
```

## Migration Guide

### From Legacy Configuration
If you have existing .env files with the old format:

1. **Backup existing configuration**
   ```bash
   cp .env .env.legacy.backup
   ```

2. **Use migration script**
   ```bash
   ./scripts/migrate-config.sh
   ```

3. **Verify new configuration**
   ```bash
   npm run validate-config
   ```

## Support

For configuration issues:
1. Check this documentation
2. Review error logs in `logs/` directory
3. Run diagnostic commands above
4. Check the project GitHub issues

## Security Contacts

For security-related configuration issues:
- Email: security@a2a-network.com
- Report vulnerabilities through GitHub Security tab