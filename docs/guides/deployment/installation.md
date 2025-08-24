# A2A Network Installation Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [Quick Start Installation](#quick-start-installation)
5. [Production Installation](#production-installation)
6. [Database Setup](#database-setup)
7. [Blockchain Configuration](#blockchain-configuration)
8. [Security Configuration](#security-configuration)
9. [Verification](#verification)
10. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: 
  - Linux (Ubuntu 20.04 LTS or later, RHEL 8+)
  - macOS 12.0 or later
  - Windows Server 2019 or later

- **Hardware**:
  - CPU: 4 cores (2.4 GHz or higher)
  - RAM: 8 GB minimum, 16 GB recommended
  - Storage: 50 GB available space
  - Network: Stable internet connection (100 Mbps+)

- **Runtime**:
  - Node.js 18.x or 20.x (LTS versions only)
  - npm 9.0 or later
  - Git 2.25 or later

### Production Requirements

- **Hardware**:
  - CPU: 8+ cores (3.0 GHz or higher)
  - RAM: 32 GB minimum
  - Storage: 200 GB SSD
  - Network: Dedicated 1 Gbps connection

- **Additional Services**:
  - SAP HANA Cloud instance
  - Redis 7.0+ for caching
  - Ethereum node or Infura access

## Prerequisites

### 1. SAP BTP Account Setup

```bash
# Install Cloud Foundry CLI
brew install cloudfoundry/tap/cf-cli  # macOS
# or
sudo apt-get install cf-cli            # Ubuntu
# or
choco install cloudfoundry-cli         # Windows
```

### 2. SAP Development Tools

```bash
# Install SAP CAP development kit
npm install -g @sap/cds-dk

# Verify installation
cds --version
```

### 3. Database Prerequisites

#### For HANA Cloud
- Active HANA Cloud instance
- Database user with schema creation privileges
- HDI container setup

#### For Local Development (SQLite)
```bash
# No additional setup required - included in dependencies
```

### 4. Blockchain Prerequisites

```bash
# Install Foundry for smart contract development
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Verify installation
forge --version
cast --version
```

## Installation Methods

### Method 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/sap/a2a-network.git
cd a2a-network

# Run installation script
./scripts/install.sh
```

### Method 2: Manual Installation

Follow the step-by-step instructions below.

## Quick Start Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/sap/a2a-network.git
cd a2a-network
```

### Step 2: Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install UI5 dependencies
cd app/a2a-fiori
npm install
cd ../..
```

### Step 3: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env  # or use your preferred editor
```

Required environment variables:
```env
# Application Settings
NODE_ENV=development
PORT=4004

# Database Configuration
DATABASE_TYPE=sqlite  # or 'hana' for production

# Blockchain Settings
BLOCKCHAIN_RPC_URL=http://localhost:8545
DEFAULT_PRIVATE_KEY=your-private-key-here

# Security Settings
JWT_SECRET=generate-a-secure-secret
SESSION_SECRET=another-secure-secret

# SAP BTP Settings (for production)
VCAP_SERVICES={}  # Will be auto-populated in BTP
```

### Step 4: Database Initialization

```bash
# Deploy database schema
cds deploy --to sqlite  # for local development
# or
cds deploy --to hana    # for HANA Cloud
```

### Step 5: Compile Smart Contracts

```bash
# Navigate to contracts directory
cd contracts

# Install Solidity dependencies
forge install

# Compile contracts
forge build

# Run tests
forge test
```

### Step 6: Start the Application

```bash
# Development mode with auto-reload
npm run watch

# Production mode
npm start
```

The application will be available at:
- API: http://localhost:4004
- UI: http://localhost:4004/launchpad.html

## Production Installation

### Step 1: Prepare SAP BTP Environment

```bash
# Login to Cloud Foundry
cf login -a https://api.cf.eu10.hana.ondemand.com

# Create space if needed
cf create-space a2a-prod

# Target the space
cf target -o your-org -s a2a-prod
```

### Step 2: Create Required Services

```bash
# Create HANA Cloud instance
cf create-service hana-cloud hana a2a-hana-db

# Create XSUAA instance
cf create-service xsuaa application a2a-uaa -c xs-security.json

# Create Destination service
cf create-service destination lite a2a-destination

# Create Connectivity service
cf create-service connectivity lite a2a-connectivity
```

### Step 3: Configure Production Environment

Create `manifest.yml`:
```yaml
applications:
- name: a2a-network
  memory: 2G
  instances: 3
  buildpack: nodejs_buildpack
  path: .
  services:
    - a2a-hana-db
    - a2a-uaa
    - a2a-destination
    - a2a-connectivity
  env:
    NODE_ENV: production
    OPTIMIZE_MEMORY: true
```

### Step 4: Deploy Smart Contracts

```bash
# Deploy to mainnet/testnet
cd contracts
forge script script/Deploy.s.sol:DeployScript --rpc-url $BLOCKCHAIN_RPC_URL --broadcast

# Save deployment addresses
cat broadcast/Deploy.s.sol/1/run-latest.json | jq '.receipts[].contractAddress'
```

### Step 5: Build and Deploy

```bash
# Build the application
npm run build

# Deploy to Cloud Foundry
cf push

# Check application status
cf app a2a-network
```

## Database Setup

### HANA Cloud Configuration

1. **Create HDI Container**:
```sql
CREATE SCHEMA A2A_HDI_CONTAINER;
GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA A2A_HDI_CONTAINER TO A2A_USER;
```

2. **Configure Connection**:
```json
{
  "hana": {
    "host": "your-hana-instance.hanacloud.ondemand.com",
    "port": 443,
    "user": "A2A_USER",
    "password": "secure-password",
    "encrypt": true,
    "sslValidateCertificate": true
  }
}
```

3. **Initialize Schema**:
```bash
cds deploy --to hana --vcap-file default-env.json
```

### Data Migration

For existing data:
```bash
# Export from old system
cds compile db/schema.cds --to sql > schema.sql

# Import to new system
hdbsql -n your-instance:30015 -u SYSTEM -p password -I schema.sql
```

## Blockchain Configuration

### Local Development

1. **Start Local Ethereum Node**:
```bash
# Using Anvil (from Foundry)
anvil --fork-url https://eth-mainnet.g.alchemy.com/v2/your-api-key
```

2. **Deploy Contracts Locally**:
```bash
cd contracts
forge script script/Deploy.s.sol:DeployScript --fork-url http://localhost:8545 --broadcast
```

### Production Blockchain

1. **Configure Mainnet Access**:
```env
BLOCKCHAIN_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/your-api-key
ETHERSCAN_API_KEY=your-etherscan-api-key
```

2. **Deploy with Verification**:
```bash
forge script script/Deploy.s.sol:DeployScript \
  --rpc-url $BLOCKCHAIN_RPC_URL \
  --broadcast \
  --verify \
  --etherscan-api-key $ETHERSCAN_API_KEY
```

## Security Configuration

### 1. XSUAA Configuration

Update `xs-security.json`:
```json
{
  "xsappname": "a2a-network",
  "tenant-mode": "shared",
  "scopes": [
    {
      "name": "$XSAPPNAME.Admin",
      "description": "Administrator access"
    },
    {
      "name": "$XSAPPNAME.User",
      "description": "User access"
    }
  ],
  "role-templates": [
    {
      "name": "Admin",
      "scope-references": ["$XSAPPNAME.Admin", "$XSAPPNAME.User"]
    },
    {
      "name": "User",
      "scope-references": ["$XSAPPNAME.User"]
    }
  ]
}
```

### 2. SSL/TLS Setup

```bash
# Generate certificates for local development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure in application
export HTTPS_KEY=key.pem
export HTTPS_CERT=cert.pem
```

### 3. API Key Management

```bash
# Generate API keys
node scripts/generate-api-key.js --name "Production API" --scope "read,write"

# Store securely
cf create-user-provided-service a2a-api-keys -p '{"api-key":"generated-key"}'
```

## Verification

### 1. Health Check

```bash
# Check application health
curl http://localhost:4004/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "blockchain": "connected",
    "cache": "connected"
  }
}
```

### 2. Database Verification

```bash
# Test database connection
cds eval "SELECT 1 FROM DUMMY"

# Check schema deployment
cds compile db/schema.cds --to sql | head -20
```

### 3. Blockchain Verification

```bash
# Check contract deployment
cast call $AGENT_REGISTRY_ADDRESS "getAgentCount()" --rpc-url $BLOCKCHAIN_RPC_URL

# Verify contract code
cast code $AGENT_REGISTRY_ADDRESS --rpc-url $BLOCKCHAIN_RPC_URL
```

### 4. API Testing

```bash
# Test API endpoints
npm test

# Run integration tests
npm run test:integration
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :4004  # macOS/Linux
netstat -ano | findstr :4004  # Windows

# Kill process or use different port
export PORT=4005
```

#### Database Connection Failed
```bash
# Check HANA Cloud status
cf service a2a-hana-db

# Test connection
cds eval "SELECT 1 FROM DUMMY" --vcap-file default-env.json
```

#### Smart Contract Deployment Failed
```bash
# Check account balance
cast balance $DEFAULT_ACCOUNT --rpc-url $BLOCKCHAIN_RPC_URL

# Verify RPC connection
cast client --rpc-url $BLOCKCHAIN_RPC_URL
```

#### Memory Issues
```bash
# Increase Node.js memory
export NODE_OPTIONS="--max-old-space-size=4096"

# For Cloud Foundry
cf set-env a2a-network NODE_OPTIONS "--max-old-space-size=4096"
```

### Getting Help

1. **Check Logs**:
```bash
# Application logs
npm run logs

# Cloud Foundry logs
cf logs a2a-network --recent
```

2. **Enable Debug Mode**:
```bash
export DEBUG=cds:*
npm start
```

3. **Support Channels**:
- GitHub Issues: https://github.com/sap/a2a-network/issues
- SAP Community: https://community.sap.com/a2a-network
- Email: a2a-support@sap.com

---

## Next Steps

After successful installation:

1. Review the [User Guide](./USER_GUIDE.md) for platform usage
2. Configure agents using the [Administrator Guide](./ADMIN_GUIDE.md)
3. Explore the [API Reference](./API_REFERENCE.md) for integration
4. Set up monitoring following the [Operations Guide](./operations/MONITORING.md)

---

*Last updated: November 2024 | Version 1.0.0*