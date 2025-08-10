# GitHub Secrets Setup for A2A SAP Network CI/CD

## Required GitHub Secrets for SAP BTP Deployment

### 1. SAP BTP Cloud Foundry Credentials
Go to **Settings → Secrets and variables → Actions** in your GitHub repository and add:

#### Core SAP BTP Secrets:
```
SAP_BTP_API_ENDPOINT     # CF API endpoint (e.g., https://api.cf.us10.hana.ondemand.com)
SAP_BTP_USERNAME         # Your SAP BTP username/email
SAP_BTP_PASSWORD         # Your SAP BTP password
SAP_BTP_ORG              # Your SAP BTP organization name
SAP_BTP_SPACE_STAGING    # Staging space name (e.g., dev, staging)
SAP_BTP_SPACE_PRODUCTION # Production space name (e.g., prod, production)
```

### 2. How to Get SAP BTP Values

#### API Endpoint:
- Log in to SAP BTP Cockpit
- Navigate to your subaccount
- Look for "CF API Endpoint" in overview (usually in format: `https://api.cf.<region>.hana.ondemand.com`)

#### Organization & Spaces:
- In SAP BTP Cockpit → Cloud Foundry → Spaces
- Organization name is shown at the top
- Create/note staging and production space names

## Current CI/CD Issues Identified

### 1. **Node.js Setup Failure** ❌
- **Issue**: Setup Node.js step failing in Code Quality job
- **Cause**: Cache configuration or Node.js version mismatch
- **Fix**: Workflow expects Node.js dependencies in specific paths

### 2. **Missing CAP Project Structure** ⚠️
- **Expected Path**: `a2a-agents/backend/app/a2a/developer_portal/cap`
- **Current Status**: ✅ CAP project exists with package.json and proper structure
- **Action**: No fix needed

### 3. **Dependency Installation Paths** ❌
- **Issue**: Workflow expects dependencies at specific paths:
  - `a2a-agents/backend/app/a2a/developer_portal/cap/package.json` ✅ EXISTS
  - `a2a_network/package.json` ✅ EXISTS
  - Root `requirements.txt` ✅ EXISTS

## Quick Fix Actions

### Step 1: Set Up GitHub Secrets
```bash
# Go to: https://github.com/plturrell/a2a-sap-network/settings/secrets/actions
# Add all 6 required secrets listed above
```

### Step 2: Fix Node.js Cache Issue
The workflow uses `cache: 'npm'` but may need to specify working directory. Monitor next run.

### Step 3: Verify SAP BTP Access
Test your credentials locally:
```bash
cf api <YOUR_API_ENDPOINT>
cf login -u <YOUR_USERNAME> -p <YOUR_PASSWORD>
cf orgs  # Should show your organization
cf spaces  # Should show your spaces
```

### Step 4: Test Pipeline Again
After setting up secrets, push a small change to trigger the pipeline:
```bash
git commit --allow-empty -m "Trigger CI/CD after secrets setup"
git push origin main
```

## Expected Pipeline Flow After Fixes

1. ✅ **Code Quality** - Should pass after Node.js cache fix
2. ✅ **SAP CAP Tests** - CAP project structure is correct
3. ✅ **Blockchain Tests** - Foundry setup should work
4. ⚠️ **Deploy Staging** - Requires SAP BTP secrets
5. ⚠️ **Deploy Production** - Requires SAP BTP secrets

## Monitoring
- **Actions URL**: https://github.com/plturrell/a2a-sap-network/actions
- **Latest Run**: https://github.com/plturrell/a2a-sap-network/actions/runs/16821055925
- **Current Status**: Code Quality job failing at Node.js setup step

The pipeline infrastructure is **correctly configured** - just needs proper credentials and minor path fixes.