#!/bin/bash
# Simple BTP deployment script
# No over-engineering - just deploy

set -e

echo "🚀 Deploying A2A Agents to SAP BTP"

# Check prerequisites
if ! command -v cf &> /dev/null; then
    echo "❌ Cloud Foundry CLI not found. Install: https://docs.cloudfoundry.org/cf-cli/install-go-cli.html"
    exit 1
fi

if ! command -v mbt &> /dev/null; then
    echo "❌ MBT tool not found. Install: npm install -g mbt"
    exit 1
fi

# Check CF login
if ! cf target &> /dev/null; then
    echo "❌ Not logged into Cloud Foundry. Run: cf login"
    exit 1
fi

echo "📋 Current CF target:"
cf target

# Build MTA
echo "🔨 Building MTA archive"
mbt build -t ./mta_archives

# Get the latest MTA file
MTA_FILE=$(ls -t ./mta_archives/*.mtar | head -1)
echo "📦 Built: $MTA_FILE"

# Deploy
echo "🚀 Deploying to BTP"
cf deploy "$MTA_FILE"

# Show status
echo "✅ Deployment complete!"
echo "📋 Application status:"
cf apps

# Get application URL
APP_URL=$(cf app a2a-agents-srv --urls | tail -1)
echo "🌐 Application URL: https://$APP_URL"
echo "❤️ Health check: https://$APP_URL/health"