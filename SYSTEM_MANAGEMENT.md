# A2A System Management Guide

This guide provides comprehensive instructions for efficiently starting, stopping, and monitoring the A2A (Agent-to-Agent) system.

## 🚀 Quick Start

```bash
# Start the complete A2A ecosystem
./start.sh complete

# Check system status
./status.sh

# Stop all services
./stop.sh
```

## 📁 Management Scripts

### `start.sh` - System Startup
Comprehensive startup script for the A2A ecosystem with 18 progressive steps.

```bash
# Usage
./start.sh [MODE] [OPTIONS]

# Modes
./start.sh complete      # Start COMPLETE ecosystem (recommended)
./start.sh local         # Start local development environment
./start.sh blockchain    # Start with blockchain integration
./start.sh enterprise    # Start enterprise production environment
./start.sh agents        # Start agents only
./start.sh network       # Start network only
./start.sh minimal       # Start minimal services
./start.sh infrastructure # Start infrastructure stack only

# Options
--no-blockchain          # Skip blockchain services
--no-agents             # Skip agent services
--no-network            # Skip network services
--help                  # Show help
```

**Startup Sequence (18 Steps):**
1. Pre-flight Checks
2. Environment Setup
3. Infrastructure Services (Redis, Prometheus)
4. Blockchain Services (Anvil + Smart Contracts)
5. Core Services (Registries, Gateways, Dashboards)
6. Trust Systems
7. MCP Servers (9 AI services)
8. Network Services (CAP/CDS)
9. Agent Services (16 A2A agents)
10. Notification System
11. Developer Tools
12. Security Services
13. Monitoring & Observability
14. Integration Testing
15. Performance Optimization
16. Final Validation
17. Health Verification
18. Startup Complete

### `stop.sh` - System Shutdown
Comprehensive shutdown script with graceful termination and cleanup.

```bash
# Usage
./stop.sh [MODE] [OPTIONS]

# Modes
./stop.sh all           # Stop all services (default)
./stop.sh agents        # Stop agents only
./stop.sh core          # Stop core services only
./stop.sh mcp           # Stop MCP servers only
./stop.sh infrastructure # Stop infrastructure only
./stop.sh blockchain    # Stop blockchain only
./stop.sh network       # Stop network services only

# Options
--force                 # Fast forceful shutdown
--graceful              # Extended graceful shutdown (30s timeout)
--clear-logs           # Archive current logs
--dry-run              # Preview what would be stopped
--help                 # Show help
```

**Shutdown Sequence (10 Steps):**
1. Pre-shutdown Assessment
2. Agent Services Shutdown
3. MCP Servers Shutdown
4. Core Services Shutdown
5. Infrastructure Shutdown
6. Process Pattern Cleanup
7. File System Cleanup
8. System Resource Cleanup
9. Verification
10. Shutdown Complete

### `status.sh` - System Monitoring
Real-time status monitoring for all A2A services.

```bash
# Usage
./status.sh [OPTIONS]

# Options
--summary               # Quick overview (default)
--detailed              # Detailed service status
--health                # Include health checks
--json                  # JSON output for automation
--help                  # Show help
```

## 🏗️ System Architecture

### Service Categories

**A2A Agents (16 services)** - Ports 8001-8015, 8888
- Agent 0: Data Product (8001)
- Agent 1: Standardization (8002)
- Agent 2: AI Preparation (8003)
- Agent 3: Vector Processing (8004)
- Agent 4: Calc Validation (8005)
- Agent 5: QA Validation (8006)
- Agent 6: Quality Control (8007)
- Reasoning Agent (8008)
- SQL Agent (8009)
- Agent Manager (8010)
- Data Manager (8011)
- Catalog Manager (8012)
- Calculation Agent (8013)
- Agent Builder (8014)
- Embedding Fine-tuner (8015)
- Unified Agent Service (8888)

**Core Services (7 services)** - Various ports
- CAP/CDS Network (4004)
- API Gateway (8080)
- A2A Registry (8090)
- ORD Registry (8091)
- Health Dashboard (8889)
- Developer Portal (3001)
- Trust System (8020)

**Infrastructure (3 services)**
- Redis Cache (6379)
- Prometheus Monitoring (9090)
- Blockchain/Anvil (8545)

**MCP Servers (10 services)** - Ports 8100-8109
- Enhanced Test Suite (8100)
- Data Standardization (8101)
- Vector Similarity (8102)
- Vector Ranking (8103)
- Transport Layer (8104)
- Reasoning Agent MCP (8105)
- Session Management (8106)
- Resource Streaming (8107)
- Confidence Calculator (8108)
- Semantic Similarity (8109)

## 🔧 Common Workflows

### Development Workflow
```bash
# 1. Check current status
./status.sh

# 2. Stop any running services
./stop.sh

# 3. Start development environment
./start.sh local

# 4. Monitor system
./status.sh --detailed --health
```

### Production Deployment
```bash
# 1. Clean shutdown
./stop.sh --graceful --clear-logs

# 2. Start production environment
./start.sh enterprise

# 3. Verify all services
./status.sh --detailed --health

# 4. Check logs if needed
tail -f logs/startup.log
```

### Testing & Debugging
```bash
# Start minimal infrastructure
./start.sh infrastructure

# Start specific service groups
./start.sh mcp           # Start MCP servers only
./start.sh agents        # Start agents only

# Preview shutdown operations
./stop.sh --dry-run

# Monitor specific services
./status.sh --json | jq '.mcp[] | select(.status=="running")'
```

### Maintenance Operations
```bash
# Restart MCP servers only
./stop.sh mcp
./start.sh mcp

# Quick restart with log archival
./stop.sh --clear-logs
./start.sh complete

# Force shutdown stuck services
./stop.sh --force
```

## 📊 Status Indicators

### System Health Levels
- **OPERATIONAL** (80-100%): System fully functional
- **PARTIAL** (50-79%): Core services running, some features degraded
- **DEGRADED** (<50%): Major issues, requires attention

### Service Status Icons
- ✅ **HEALTHY**: Service running and responding to health checks
- ⚠️ **RUNNING**: Service active but no health endpoint response
- ❌ **DOWN**: Service not running
- 🔄 **STARTING**: Service in startup process
- 🛑 **STOPPING**: Service in shutdown process

## 🗂️ Log Management

### Log Locations
- **Startup**: `logs/startup.log`
- **Shutdown**: `logs/stop.log`
- **Individual Services**: `logs/[service-name].log`
- **Debug**: `logs/debug.log`

### Log Archival
```bash
# Archive logs during shutdown
./stop.sh --clear-logs

# Manual log archive
mkdir -p logs/archive_$(date +%Y%m%d_%H%M%S)
mv logs/*.log logs/archive_*/
```

## 🔍 Troubleshooting

### Common Issues

**Port Conflicts**
```bash
# Check what's using a port
lsof -ti :8003

# Kill process on specific port
./stop.sh blockchain  # Stop specific service group
```

**Startup Failures**
```bash
# Check detailed startup log
tail -f logs/startup.log

# Run pre-flight checks only
./start.sh --help  # See available options

# Start with minimal services
./start.sh minimal
```

**Health Check Failures**
```bash
# Check service health individually
curl http://localhost:8003/health
curl http://localhost:8889/health

# View detailed status
./status.sh --detailed --health
```

### Recovery Procedures

**Full System Recovery**
```bash
# 1. Force stop everything
./stop.sh --force

# 2. Clean up temporary files
rm -rf /tmp/a2a_* /tmp/anvil_*

# 3. Fresh start
./start.sh complete

# 4. Verify status
./status.sh --detailed
```

**Partial Recovery**
```bash
# Restart specific service groups
./stop.sh mcp && ./start.sh mcp
./stop.sh agents && ./start.sh agents
```

## 🚀 Performance Optimization

### Recommended Startup Order
1. Infrastructure (Redis, Prometheus)
2. Blockchain (Anvil + Smart Contracts)
3. Core Services (Registries, Gateways)
4. MCP Servers (AI Services)
5. A2A Agents (Main Application Logic)

### Resource Monitoring
```bash
# Monitor system resources
./status.sh --json | jq '.infrastructure[] | select(.status=="running")'

# Check memory usage
ps aux | grep -E "(anvil|redis|prometheus|uvicorn)"

# Monitor port usage
netstat -tulpn | grep -E "(8[0-9]{3}|6379|9090)"
```

## 🔐 Security Considerations

- All scripts use safe shell practices (`set -euo pipefail`)
- Graceful shutdown attempts before force termination
- PID file cleanup to prevent zombie processes
- Shared memory and message queue cleanup
- No hardcoded credentials or secrets in scripts

## 📝 Script Customization

### Environment Variables
```bash
# Customize timeouts
export STARTUP_TIMEOUT=120
export SHUTDOWN_TIMEOUT=30

# Customize ports (if needed)
export BLOCKCHAIN_PORT=8545
export AGENTS_PORT=8888
```

### Adding New Services
1. Update port definitions in scripts
2. Add service name mapping in `get_service_name()`
3. Update health check logic if needed
4. Test with `--dry-run` options

## 🎯 Best Practices

1. **Always check status before operations**
   ```bash
   ./status.sh && ./stop.sh && ./start.sh complete
   ```

2. **Use dry-run for verification**
   ```bash
   ./stop.sh --dry-run  # Preview shutdown
   ```

3. **Monitor logs during operations**
   ```bash
   tail -f logs/startup.log &
   ./start.sh complete
   ```

4. **Archive logs regularly**
   ```bash
   ./stop.sh --clear-logs  # Archive during shutdown
   ```

5. **Use appropriate shutdown modes**
   ```bash
   ./stop.sh --graceful    # For production
   ./stop.sh --force       # For development/stuck processes
   ```

## 🆘 Emergency Procedures

### Emergency Shutdown
```bash
# Kill all A2A processes immediately
pkill -f "anvil|redis|prometheus|uvicorn|python.*mcp"

# Or use force shutdown
./stop.sh --force
```

### Service Recovery
```bash
# Reset to known good state
./stop.sh --force --clear-logs
sleep 5
./start.sh complete
./status.sh --detailed --health
```

---

## 🏁 Summary

The A2A system now provides enterprise-grade start/stop capabilities with:

- **Comprehensive Management**: 3 main scripts (`start.sh`, `stop.sh`, `status.sh`)
- **Selective Operations**: Target specific service groups
- **Safe Operations**: Graceful shutdown with force fallback
- **Monitoring**: Real-time status and health checks
- **Logging**: Detailed logs with archival support
- **Recovery**: Emergency procedures and troubleshooting guides

Use `./start.sh complete` for full system startup and `./status.sh --detailed --health` for comprehensive monitoring.