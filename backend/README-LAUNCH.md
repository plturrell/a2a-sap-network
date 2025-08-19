# A2A Agents System Launcher

This system provides a unified way to run the full A2A agents implementation both locally and on SAP BTP without any code changes.

## Quick Start

### Local Development
```bash
# Use minimal setup (fastest)
npm run start:minimal

# Or use full system
npm start
```

### BTP Deployment
```bash
# Deploy to BTP
npm run deploy:btp
```

## Architecture

The system uses a **BTP Adapter** pattern that:

1. **Detects environment automatically** (local vs BTP)
2. **Loads appropriate configuration** (env vars vs VCAP_SERVICES)
3. **Injects compatibility variables** so existing code works unchanged
4. **Provides unified APIs** for database, auth, and caching

## Configuration

### Local Development
Set environment variables in `.env`:
```env
HANA_HOST=localhost
HANA_PORT=30015
HANA_USER=SYSTEM
HANA_PASSWORD=your_password
```

### BTP Deployment
Services are automatically bound via `VCAP_SERVICES`:
- **HANA HDI Container** for database
- **XSUAA** for authentication  
- **Redis** for caching (optional)

## Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | System information |
| `/health` | Health check with service status |
| `/info` | Detailed system information |
| `/config` | Configuration overview |
| `/api/agents` | Agents API |
| `/api/agents/:id/status` | Individual agent status |

## Agents Included

1. **Enhanced Calculation Agent** - Self-healing calculations with blockchain
2. **Data Processing Agent** - HANA integration with lifecycle management
3. **Reasoning Agent** - Chain-of-thought reasoning with error recovery
4. **SQL Agent** - Query optimization with caching
5. **Vector Processing Agent** - HANA vector engine integration

## Key Features

✅ **Zero code changes** required for existing A2A implementation  
✅ **Automatic environment detection** (local/BTP)  
✅ **Service binding adaptation** for BTP services  
✅ **Backward compatibility** with existing configuration  
✅ **Health monitoring** and status reporting  
✅ **Production ready** with proper error handling  

## Files Structure

```
backend/
├── launchA2aSystem.js          # Main launcher (full system)
├── srv/server-minimal.js       # Minimal server
├── config/
│   ├── btpAdapter.js           # BTP adaptation layer
│   └── minimalBtpConfig.py     # Python BTP config
├── package-full.json           # Full system dependencies
├── package-minimal.json        # Minimal dependencies
├── deploy-local.sh             # Local setup script
├── deploy-btp.sh              # BTP deployment script
└── mta-minimal.yaml           # MTA descriptor for BTP
```

## Usage Examples

### Check System Health
```bash
curl http://localhost:8080/health
```

### List Available Agents
```bash
curl http://localhost:8080/api/agents
```

### Get Agent Status
```bash
curl http://localhost:8080/api/agents/calculation-agent/status
```

## Environment Variables

The BTP adapter automatically sets these for existing code compatibility:

| Variable | Source | Description |
|----------|--------|-------------|
| `HANA_HOST` | BTP/Local | Database host |
| `HANA_PORT` | BTP/Local | Database port |
| `HANA_USER` | BTP/Local | Database user |
| `XSUAA_URL` | BTP/Local | Auth service URL |
| `BTP_ENVIRONMENT` | Auto | true/false |

## Troubleshooting

### Local Development Issues
1. Check `.env` file exists and has correct values
2. Ensure local services (HANA, Redis) are running if configured
3. Use `npm run start:minimal` for basic functionality

### BTP Deployment Issues  
1. Verify `cf login` and correct space/org
2. Check MTA build with `mbt build`
3. Review service bindings in BTP cockpit
4. Check application logs with `cf logs a2a-agents-srv`

## Integration with Existing Code

The adapter works by:
1. **Detecting BTP environment** via `VCAP_SERVICES`
2. **Loading service credentials** from BTP bindings  
3. **Injecting environment variables** that existing code expects
4. **Providing fallbacks** for local development

This means **no changes required** to existing A2A agents code - they continue to use environment variables as before, but the adapter ensures the right values are available in both environments.