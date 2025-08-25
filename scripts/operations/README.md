# Operations Scripts

This directory contains scripts for day-to-day operations of the A2A Platform.

## Scripts Overview

- **`start.sh`** - Main startup script with multiple modes
- **`stop.sh`** - Stop all A2A services gracefully
- **`status.sh`** - Check status of all services
- **`start-all-agents.sh`** - Start all 18 agents individually

## Startup Modes

The `start.sh` script supports multiple modes:

| Mode | Description | Services Started |
|------|-------------|------------------|
| `quick` | Minimal startup | Agent 0 only |
| `backend` | Core services | Agents 0-5 |
| `complete` | Full platform | All 18 agents + services |
| `test` | Test mode | Runs verification |
| `verify` | Verification | 18-step validation |
| `ci-verify` | CI mode | Automated testing |

## Usage Examples

### Local Development
```bash
# Start everything (default)
./start.sh

# Quick start (Agent 0 only)
./start.sh quick

# Start backend agents only
./start.sh backend

# Check status
./status.sh

# Stop all services
./stop.sh
```

### Docker/Container
```bash
# The scripts auto-detect container environment
docker run -it a2a-platform ./scripts/operations/start.sh
```

## Service Architecture

```
A2A Platform Services:
├── Infrastructure
│   ├── Redis (6379)
│   ├── PostgreSQL (5432)
│   └── Blockchain (8545)
├── Core Services
│   ├── A2A Network API (4004)
│   ├── Frontend UI (3000)
│   └── API Gateway (8080)
└── Agents (8000-8017)
    ├── Agent 0: Data Product
    ├── Agent 1: Standardization
    ├── Agent 2: AI Preparation
    ├── Agent 3: Vector Processing
    ├── Agent 4: Calc Validation
    ├── Agent 5: QA Validation
    └── ... (12 more agents)
```

## Health Checks

The status script checks:
- Process status
- Port availability
- HTTP health endpoints
- Database connections
- Service dependencies

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STARTUP_MODE` | Default startup mode | complete |
| `A2A_ENVIRONMENT` | Environment name | development |
| `ENABLE_ALL_AGENTS` | Start all agents | true |
| `HOT_RELOAD` | Enable hot reload | true |
| `LOG_LEVEL` | Logging level | INFO |

## Troubleshooting

### Services Won't Start
```bash
# Check what's already running
./status.sh

# Stop everything and restart
./stop.sh
./start.sh
```

### Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Logs Location
- Local: `./logs/`
- Docker: `/app/logs/`
- Systemd: `journalctl -u a2a-platform`