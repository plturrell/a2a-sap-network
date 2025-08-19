# SAP BTP Deployment Guide for A2A Agents System

This guide provides step-by-step instructions for SAP engineers to deploy the A2A Agents system to SAP Business Technology Platform (BTP).

## Prerequisites

### 1. Required Tools
```bash
# Check if Cloud Foundry CLI is installed
cf --version
# Expected: cf version 8.x.x or higher

# Check if MBT (Cloud MTA Build Tool) is installed
mbt --version
# Expected: Cloud MTA Build Tool version 1.x.x

# Check if Node.js is installed
node --version
# Expected: v18.0.0 or higher
```

### 2. Install Missing Tools
```bash
# Install Cloud Foundry CLI
# macOS
brew install cloudfoundry/tap/cf-cli

# Windows (using Chocolatey)
choco install cloudfoundry-cli

# Linux
wget -q -O - https://packages.cloudfoundry.org/debian/cli.cloudfoundry.org.key | sudo apt-key add -
echo "deb https://packages.cloudfoundry.org/debian stable main" | sudo tee /etc/apt/sources.list.d/cloudfoundry-cli.list
sudo apt-get update
sudo apt-get install cf-cli

# Install MBT
npm install -g mbt
```

### 3. BTP Account Requirements
- Access to a BTP subaccount with Cloud Foundry environment enabled
- Space Developer role or higher
- Quota for the following services:
  - HANA Cloud (HDI container)
  - XSUAA (Application plan)
  - Redis Cache (optional)

## Step 1: Clone and Prepare the Project

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd a2aAgents/backend

# Install dependencies
npm install

# Run validation to ensure everything is ready
node validateIntegration.js
```

Expected output:
```
✅ ALL VALIDATIONS PASSED
🚀 System is ready for both local and BTP deployment
```

## Step 2: Configure BTP Connection

### Login to Cloud Foundry
```bash
# Login to your BTP Cloud Foundry environment
cf login -a https://api.cf.<region>.hana.ondemand.com

# Example for EU10:
cf login -a https://api.cf.eu10.hana.ondemand.com

# You will be prompted for:
# - Email/Username
# - Password
# - Organization (select from list)
# - Space (select from list)
```

### Verify Target
```bash
cf target
```

Expected output:
```
API endpoint:   https://api.cf.eu10.hana.ondemand.com
API version:    3.xxx.0
user:           your.email@company.com
org:            your-org-name
space:          dev
```

## Step 3: Review and Customize MTA Configuration

### Check MTA Descriptor
```bash
cat mta-minimal.yaml
```

### Optional: Customize Service Names
Edit `mta-minimal.yaml` if you need to:
- Change service instance names
- Modify memory/disk allocations
- Add custom environment variables

```yaml
modules:
  - name: a2a-agents-srv
    parameters:
      memory: 512M      # Increase if needed
      instances: 1      # Increase for high availability
```

## Step 4: Build the MTA Archive

```bash
# Build the MTA archive
mbt build -t ./mta_archives

# This creates an .mtar file in the mta_archives directory
ls -la ./mta_archives/
```

Expected output:
```
a2a-agents-minimal_1.0.0.mtar
```

## Step 5: Deploy to BTP

### Option A: Automated Deployment (Recommended)
```bash
# Use the provided deployment script
./deploy-btp.sh
```

### Option B: Manual Deployment
```bash
# Deploy the MTA archive
cf deploy ./mta_archives/a2a-agents-minimal_1.0.0.mtar

# Monitor deployment progress
# This may take 5-10 minutes
```

## Step 6: Verify Deployment

### Check Application Status
```bash
cf apps
```

Expected output:
```
name                requested state   instances   memory   disk   urls
a2a-agents-srv      started          1/1         512M     1G     a2a-agents-srv-<random>.cfapps.eu10.hana.ondemand.com
```

### Check Service Bindings
```bash
cf services
```

Expected output:
```
name                 service         plan          bound apps        last operation
a2a-agents-hana      hana           hdi-shared    a2a-agents-srv    create succeeded
a2a-agents-xsuaa     xsuaa          application   a2a-agents-srv    create succeeded
```

### Test Health Endpoint
```bash
# Get the application URL
APP_URL=$(cf app a2a-agents-srv | grep routes | awk '{print $2}')

# Test health endpoint
curl https://$APP_URL/health
```

Expected response:
```json
{
  "status": "healthy",
  "environment": "btp",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "services": {
    "database": true,
    "authentication": true,
    "cache": false
  }
}
```

## Step 7: Access Application Logs

### View Recent Logs
```bash
cf logs a2a-agents-srv --recent
```

### Stream Live Logs
```bash
cf logs a2a-agents-srv
```

## Step 8: Post-Deployment Configuration

### 1. Configure Authentication (if needed)
```bash
# Download and review XSUAA configuration
cf env a2a-agents-srv | grep -A 20 VCAP_SERVICES
```

### 2. Set Environment Variables (if needed)
```bash
# Example: Enable debug logging
cf set-env a2a-agents-srv LOG_LEVEL debug
cf restage a2a-agents-srv
```

### 3. Scale Application (if needed)
```bash
# Horizontal scaling
cf scale a2a-agents-srv -i 3

# Vertical scaling
cf scale a2a-agents-srv -m 1G
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Deployment Fails with "Insufficient Resources"
```bash
# Check organization quota
cf org <your-org> --guid
cf quota <quota-name>

# Solution: Contact BTP admin to increase quota
```

#### 2. HANA Connection Issues
```bash
# Check HANA service binding
cf env a2a-agents-srv | grep -A 30 "hana"

# Verify credentials are present
# If missing, rebind the service:
cf unbind-service a2a-agents-srv a2a-agents-hana
cf bind-service a2a-agents-srv a2a-agents-hana
cf restage a2a-agents-srv
```

#### 3. Application Crashes on Startup
```bash
# Check crash logs
cf logs a2a-agents-srv --recent | grep ERROR

# Common causes:
# - Missing npm dependencies
# - Node.js version mismatch
# - Memory limit too low
```

#### 4. Authentication Errors
```bash
# Verify XSUAA binding
cf env a2a-agents-srv | grep -A 20 "xsuaa"

# Check if JWT validation is configured correctly
```

## Maintenance Tasks

### Update Application
```bash
# Make code changes
# Rebuild MTA
mbt build -t ./mta_archives

# Deploy update
cf deploy ./mta_archives/a2a-agents-minimal_1.0.0.mtar
```

### Backup and Restore
```bash
# Export environment configuration
cf env a2a-agents-srv > env-backup.json

# Export service keys (if any)
cf service-keys a2a-agents-hana
```

### Monitor Performance
```bash
# Check application metrics
cf app a2a-agents-srv

# View detailed statistics
cf app a2a-agents-srv --guid
```

## Security Checklist

- [ ] XSUAA service is properly configured
- [ ] No hardcoded credentials in code
- [ ] Environment variables used for sensitive data
- [ ] HTTPS enforced for all endpoints
- [ ] Rate limiting configured (if applicable)
- [ ] Audit logging enabled

## Integration with SAP Services

### Connect to SAP HANA Cloud
The application automatically connects using the bound HANA service credentials.

### Integration with SAP Launchpad
1. Register the application in SAP Launchpad Service
2. Configure the app descriptor
3. Assign to appropriate role collections

### Connect to SAP Event Mesh (Optional)
```bash
# Create Event Mesh service instance
cf create-service enterprise-messaging default a2a-event-mesh

# Bind to application
cf bind-service a2a-agents-srv a2a-event-mesh
cf restage a2a-agents-srv
```

## Support and Resources

### Useful Commands Reference
```bash
# Application management
cf apps                          # List all apps
cf app a2a-agents-srv           # App details
cf ssh a2a-agents-srv           # SSH into container
cf events a2a-agents-srv        # View app events

# Service management  
cf services                      # List all services
cf service a2a-agents-hana      # Service details
cf create-service-key           # Create service key

# Troubleshooting
cf logs a2a-agents-srv --recent # Recent logs
cf env a2a-agents-srv           # Environment variables
cf restart a2a-agents-srv       # Restart app
```

### Documentation Links
- [SAP BTP Documentation](https://help.sap.com/docs/btp)
- [Cloud Foundry CLI Reference](https://cli.cloudfoundry.org/en-US/)
- [SAP HANA Cloud Documentation](https://help.sap.com/docs/hana-cloud)
- [SAP XSUAA Documentation](https://help.sap.com/docs/btp/sap-business-technology-platform/application-security)

### Getting Help
1. Check application logs: `cf logs a2a-agents-srv --recent`
2. Validate configuration: `node validateIntegration.js`
3. Review BTP Cockpit for service status
4. Contact: BTP Support Portal

---

**Deployment Time Estimate: 15-30 minutes** (including service provisioning)

**Note**: First-time deployments may take longer due to service provisioning. Subsequent updates are typically faster (5-10 minutes).