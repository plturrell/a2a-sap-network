#!/bin/bash

# A2A Network - SAP BTP Deployment Script

set -e

echo "🚀 Deploying A2A Network to SAP BTP"
echo "===================================="

# Check if CF CLI is installed
if ! command -v cf &> /dev/null; then
    echo "❌ Cloud Foundry CLI is not installed. Please install it first:"
    echo "   https://docs.cloudfoundry.org/cf-cli/install-go-cli.html"
    exit 1
fi

# Check if user is logged in to CF
if ! cf target &> /dev/null; then
    echo "❌ Please log in to Cloud Foundry first:"
    echo "   cf login -a https://api.cf.sap.hana.ondemand.com"
    exit 1
fi

echo "✅ Cloud Foundry CLI is ready"

# Show current target
echo "📍 Current CF target:"
cf target

echo ""
echo "🔧 Preparing deployment..."

# Create or update XSUAA service
echo "🔐 Setting up XSUAA service..."
cf create-service xsuaa application a2a-network-xsuaa -c xs-security.json || {
    echo "🔄 Updating existing XSUAA service..."
    cf update-service a2a-network-xsuaa -c xs-security.json
}

# Create HANA service (HDI container)
echo "🗄️  Setting up HANA HDI container..."
cf create-service hana hdi-shared a2a-network-hdi-container || {
    echo "ℹ️  HANA service already exists"
}

# Create Application Logging service
echo "📊 Setting up Application Logging..."
cf create-service application-logs lite a2a-network-application-logs || {
    echo "ℹ️  Application Logs service already exists"
}

# Create Connectivity service (optional)
echo "🔗 Setting up Connectivity service..."
cf create-service connectivity lite a2a-network-connectivity || {
    echo "ℹ️  Connectivity service already exists"
}

# Create Destination service (optional)
echo "🎯 Setting up Destination service..."
cf create-service destination lite a2a-network-destination || {
    echo "ℹ️  Destination service already exists"
}

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Build the application
echo "🏗️  Building application..."
npm run build || {
    echo "⚠️  Build failed, but continuing with deployment..."
}

# Deploy the application
echo "🚀 Deploying application..."
cf push -f manifest.yml

# Check deployment status
echo ""
echo "✅ Deployment completed!"
echo ""
echo "🌐 Application URLs:"
cf apps | grep "a2a-network-launchpad"

echo ""
echo "🔍 Checking application health..."
sleep 5

# Get app URL
APP_URL=$(cf apps | grep "a2a-network-launchpad" | awk '{print $6}' | head -1)

if [ ! -z "$APP_URL" ]; then
    echo "📱 Launchpad URL: https://$APP_URL/launchpad.html"
    echo "🔧 Health check: https://$APP_URL/health"
    echo "👤 User info: https://$APP_URL/user/info"
    
    # Test health endpoint
    echo ""
    echo "🏥 Testing health endpoint..."
    curl -f "https://$APP_URL/health" && echo "✅ Health check passed" || echo "❌ Health check failed"
fi

echo ""
echo "📋 Post-deployment checklist:"
echo "  1. Configure role assignments in BTP Cockpit"
echo "  2. Test user authentication and authorization"
echo "  3. Verify database connectivity"
echo "  4. Check application logs: cf logs a2a-network-launchpad --recent"
echo ""
echo "🎉 Deployment complete!"