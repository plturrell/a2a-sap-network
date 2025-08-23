# Local Testing Setup for A2A SAP Network

## Overview
Modified CI/CD pipeline to focus on local testing before SAP BTP deployment.

## What's Changed in CI/CD
- ✅ **Keeps**: Code quality, security scans, tests, build & package
- ❌ **Disabled**: SAP BTP staging/production deployments
- 🔧 **Modified**: Notifications now report build success instead of deployment

## Local Development Setup

### 1. Prerequisites
```bash
# Node.js 18+
node --version

# Python 3.9+
python --version

# Foundry (for blockchain)
forge --version

# SAP CDS CLI (for CAP development)
npm install -g @sap/cds-dk
```

### 2. Install Dependencies

#### A2A Network (Blockchain)
```bash
cd a2a_network
npm install
forge install
```

#### A2A Agents (SAP CAP Portal)
```bash
cd a2aAgents/backend/app/a2a/developer_portal/cap
npm install
```

#### Python Dependencies
```bash
cd a2aAgents/backend
pip install -r requirements.txt
```

### 3. Local Testing Commands

#### Run All Tests Locally (Same as CI/CD)
```bash
# Blockchain tests
cd a2a_network
forge test --gas-report

# CAP tests  
cd a2aAgents/backend/app/a2a/developer_portal/cap
npm test

# Python tests
cd a2aAgents/backend
python -m pytest -v

# Security scans
cd a2a_network
slither . --exclude-dependencies
```

#### Start Local Development Servers
```bash
# Start SAP CAP Portal
cd a2aAgents/backend/app/a2a/developer_portal/cap
cds watch

# Start Blockchain Local Network
cd a2a_network
anvil  # Local Ethereum node

# Deploy contracts locally
forge script script/Deploy.s.sol --rpc-url http://localhost:8545 --broadcast
```

### 4. Local Integration Testing
```bash
# Start all services
cd a2aAgents/backend
python main.py  # Start main backend

# In another terminal - start portal
cd a2aAgents/backend/app/a2a/developer_portal
python portal_server.py

# Test integration
curl http://localhost:4004/health  # CAP service
curl http://localhost:8000/health  # Main backend
```

### 5. Development Workflow

#### Make Changes → Test Locally
```bash
# Run the same tests as CI/CD
npm run lint  # Code quality
forge test    # Blockchain tests  
npm test      # CAP tests
pytest        # Python tests

# Build locally
cds build     # Build CAP
forge build   # Build contracts
```

#### Commit → CI/CD Pipeline Runs
- Code quality & security checks
- All test suites
- Build & package artifacts
- **No deployment** (until ready for SAP BTP)

## CI/CD Status
- **Current**: Build, test, package only
- **Next Phase**: Enable SAP BTP deployment when ready
- **Local First**: Perfect for development and testing

## Ready to Enable SAP BTP?
When ready, change in `.github/workflows/ci-cd.yml`:
```yaml
if: false  # Change to: if: github.ref == 'refs/heads/main'
```

This setup gives you full local development with the safety of CI/CD testing before any cloud deployment.