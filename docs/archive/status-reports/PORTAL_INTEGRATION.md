# A2A Developer Portal - Complete Integration Guide

## Overview

The A2A Developer Portal is a comprehensive web-based IDE for developing, testing, and deploying A2A agents. This integration provides a unified platform that combines:

- **Agent Builder**: Visual agent creation with templates and validation
- **BPMN Workflow Designer**: Visual workflow creation and blockchain execution
- **A2A Network Integration**: Full blockchain connectivity and messaging
- **SAP BTP Services**: Enterprise authentication and authorization
- **Deployment Pipeline**: Automated testing and deployment to various platforms
- **Real-time Collaboration**: Multi-user editing and WebSocket communication

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.portal.template .env.portal

# Configure your environment variables
nano .env.portal
```

### 2. Launch Portal

```bash
# Using the integrated launcher
python portalLauncher.py

# Or using the deployment script
python deployPortalIntegration.py
```

### 3. Access Portal

- **Portal Dashboard**: http://localhost:3001
- **API Documentation**: http://localhost:3001/docs
- **Health Check**: http://localhost:3001/api/health

## Core Components

### 1. Portal Server (`portalServer.py`)

**Features**:
- FastAPI-based web server
- SAP UI5 Fiori interface
- Project management system
- Real-time collaboration via WebSockets
- Built-in testing and deployment

**Key Capabilities**:
- Multi-project workspace management
- In-browser Monaco code editor
- BPMN workflow designer integration
- Live preview and testing
- Version control integration

### 2. Enhanced Agent Builder (`agentBuilder/enhancedAgentBuilder.py`)

**Features**:
- Template-based agent generation
- Comprehensive validation system
- Skills and handlers configuration
- Jinja2-based code generation
- A2A network integration

**Agent Types Supported**:
- Data Processor
- Workflow Orchestrator
- Integration Connector
- Decision Maker
- Monitoring Agent
- Custom agents

### 3. A2A Network Integration (`api/a2aNetworkIntegration.py`)

**Features**:
- Complete blockchain connectivity
- Agent registration on blockchain
- Real-time message routing
- Reputation management
- Webhook subscriptions
- Token operations
- Governance proposal management

**API Endpoints**:
- `/api/a2a-network/connect` - Connect to A2A Network
- `/api/a2a-network/agents/register` - Register agents
- `/api/a2a-network/messages/send` - Send messages
- `/api/a2a-network/webhooks/subscribe` - Subscribe to events
- `/api/a2a-network/analytics/network` - Network analytics

### 4. SAP BTP Services

**Authentication** (`sapBtp/authApi.py`):
- XSUAA integration
- JWT token management
- Session handling

**RBAC Service** (`sapBtp/rbacService.py`):
- Role-based access control
- Permission management
- Resource protection

**Notification Service** (`sapBtp/notificationApi.py`):
- Real-time notifications
- Multi-channel delivery
- Priority-based routing

### 5. Deployment Pipeline (`deployment/deploymentPipeline.py`)

**Supported Targets**:
- Docker containers
- Kubernetes clusters
- Cloud platforms (AWS, Azure, GCP)
- SAP BTP
- Local development

**Pipeline Stages**:
- Code validation
- Testing (pytest, jest, go test)
- Security scanning
- Container building
- Deployment
- Health verification

## Configuration

### Environment Variables

Key configuration options:

```bash
# Portal Configuration
PORTAL_PORT=3001
A2A_WORKSPACE_PATH=/tmp/a2a_workspace
AUTO_DEPLOYMENT=false

# Blockchain Integration
BLOCKCHAIN_NETWORK=mainnet
BLOCKCHAIN_RPC_URL=https://mainnet.infura.io/v3/YOUR-PROJECT-ID

# SAP BTP (Optional)
XSUAA_SERVICE_URL=https://your-subdomain.authentication.us10.hana.ondemand.com
XSUAA_CLIENT_ID=your-client-id
XSUAA_CLIENT_SECRET=your-client-secret

# Security
SECRET_KEY=your-super-secret-key
JWT_ALGORITHM=HS256

# Database
DATABASE_URL=postgresql://username:password@localhost:5432/a2a_portal

# Email
EMAIL_SERVICE=smtp
SMTP_HOST=smtp.gmail.com
```

### Feature Flags

Enable/disable portal features:

```bash
FEATURE_AGENT_BUILDER=true
FEATURE_WORKFLOW_DESIGNER=true
FEATURE_DEPLOYMENT_PIPELINE=true
FEATURE_A2A_NETWORK_INTEGRATION=true
FEATURE_COLLABORATION_TOOLS=true
FEATURE_ANALYTICS_DASHBOARD=true
```

## Architecture

### Frontend (SAP UI5/Fiori)

```
a2a-developer-portal/
├── static/
│   ├── index.html              # Main portal entry
│   ├── manifest.json           # App configuration
│   ├── controller/             # UI5 controllers
│   │   ├── App.controller.js
│   │   ├── ProjectsListReport.controller.js
│   │   └── ProjectObjectPage.controller.js
│   ├── view/                   # UI5 views
│   │   ├── App.view.xml
│   │   ├── ProjectsList.view.xml
│   │   └── ProjectDetails.view.xml
│   └── utils/                  # Utility modules
│       ├── ApiClient.js
│       ├── WebSocketManager.js
│       └── BpmnDesigner.js
```

### Backend (Python/FastAPI)

```
app/a2a/developerPortal/
├── portalLauncher.py           # Main launcher (NEW)
├── portalServer.py             # Core portal server
├── api/                        # API endpoints
│   └── a2aNetworkIntegration.py
├── agentBuilder/               # Agent building
│   └── enhancedAgentBuilder.py
├── sapBtp/                     # SAP BTP services
│   ├── authApi.py
│   ├── rbacService.py
│   └── notificationApi.py
├── deployment/                 # Deployment pipeline
│   └── deploymentPipeline.py
└── config/                     # Configuration
    └── production.py
```

## API Reference

### Portal Management

```http
GET /api/health
GET /api/portal/projects
POST /api/portal/projects
GET /api/portal/projects/{id}
PUT /api/portal/projects/{id}
DELETE /api/portal/projects/{id}
```

### Agent Builder

```http
POST /api/agent-builder/generate
GET /api/agent-builder/templates
POST /api/agent-builder/validate
GET /api/agent-builder/skills
POST /api/agent-builder/skills
```

### A2A Network

```http
POST /api/a2a-network/connect
POST /api/a2a-network/agents/register
POST /api/a2a-network/messages/send
GET /api/a2a-network/agents/search
POST /api/a2a-network/webhooks/subscribe
```

### Deployment

```http
POST /api/deployment/deploy
GET /api/deployment/status/{deployment_id}
GET /api/deployment/logs/{deployment_id}
POST /api/deployment/rollback/{deployment_id}
```

## WebSocket Events

Real-time events for collaboration:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:3001/ws');

// Event types
ws.on('project_updated', (data) => {
    // Project file changed
});

ws.on('agent_deployed', (data) => {
    // Agent deployment completed
});

ws.on('a2a_message_received', (data) => {
    // New A2A network message
});

ws.on('collaboration_cursor', (data) => {
    // User cursor position in editor
});
```

## Security

### Authentication Flow

1. **Development Mode**: Simple token-based auth
2. **Production Mode**: SAP XSUAA integration with JWT

### Authorization

- Role-based access control (RBAC)
- Resource-level permissions
- API key management for A2A network

### Security Headers

Automatic security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security`
- Content Security Policy

## Deployment

### Development

```bash
# Quick development start
python portalLauncher.py
```

### Production

```bash
# Using Docker
docker build -t a2a-portal .
docker run -p 3001:3001 --env-file .env.portal a2a-portal

# Using Kubernetes
kubectl apply -f k8s/portal-deployment.yaml

# Using SAP BTP
cf push a2a-portal -f manifest.yml
```

### Environment-Specific Configurations

**Development**:
- SQLite database
- Local blockchain
- Debug mode enabled
- Hot reload

**Staging**:
- PostgreSQL database
- Testnet blockchain
- Performance monitoring
- Automated testing

**Production**:
- High-availability database
- Mainnet blockchain
- Full security headers
- Load balancing

## Monitoring

### Health Checks

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "integration_status": {
    "portal_server": true,
    "agent_builder": true,
    "network_integration": true,
    "sap_btp_services": true,
    "deployment_pipeline": true
  },
  "version": "1.0.0"
}
```

### Telemetry

- OpenTelemetry integration
- Prometheus metrics
- Grafana dashboards
- Log aggregation

## Troubleshooting

### Common Issues

**Portal won't start**:
```bash
# Check configuration
python -c "from config.production import ProductionConfig; ProductionConfig.validate()"

# Check dependencies
pip install -r requirements.txt
```

**A2A Network connection fails**:
```bash
# Verify blockchain connectivity
curl -X POST -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  $BLOCKCHAIN_RPC_URL
```

**SAP BTP authentication fails**:
```bash
# Test XSUAA connectivity
curl -X POST "$XSUAA_SERVICE_URL/oauth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=$XSUAA_CLIENT_ID&client_secret=$XSUAA_CLIENT_SECRET"
```

### Log Analysis

```bash
# Check portal logs
tail -f logs/portal.log

# Check deployment logs
tail -f logs/deployment.log

# Check A2A network logs
tail -f logs/a2a-network.log
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/a2a-portal.git
cd a2a-portal

# Setup virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Start development server
python portalLauncher.py
```

### Code Standards

- **Python**: Black formatting, Flake8 linting, MyPy type checking
- **JavaScript**: ESLint, Prettier formatting
- **Documentation**: Sphinx for Python, JSDoc for JavaScript

## Support

- **Documentation**: [Portal Docs](https://docs.a2a-network.com/portal)
- **API Reference**: http://localhost:3001/docs
- **Issues**: [GitHub Issues](https://github.com/your-org/a2a-portal/issues)
- **Community**: [Discord](https://discord.gg/a2a-network)

## License

MIT License - see [LICENSE](LICENSE) file for details.