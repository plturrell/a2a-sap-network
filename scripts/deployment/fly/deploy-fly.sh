#!/bin/bash
# Fly.io deployment script for A2A Platform

set -e

echo "🚀 Deploying A2A Platform to Fly.io"
echo "===================================="

# Check if FLY_API_TOKEN is set
if [ -z "$FLY_API_TOKEN" ]; then
    echo "❌ Error: FLY_API_TOKEN environment variable not set"
    echo "Please set it with: export FLY_API_TOKEN=your-token"
    exit 1
fi

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ Error: flyctl not found. Please install it first:"
    echo "curl -L https://fly.io/install.sh | sh"
    exit 1
fi

APP_NAME="a2a-platform"
DOCKER_IMAGE="finsightintelligence/a2a:main"

echo "📦 Using Docker image: $DOCKER_IMAGE"

# Check if app exists
if flyctl apps list | grep -q "$APP_NAME"; then
    echo "✅ App $APP_NAME already exists"
else
    echo "📱 Creating new app: $APP_NAME"
    flyctl apps create $APP_NAME --org personal || true
fi

# Deploy the app
echo "🚀 Deploying to Fly.io..."
flyctl deploy --app $APP_NAME --image $DOCKER_IMAGE --remote-only --wait-timeout 900

# Scale the app if needed
echo "⚖️ Ensuring proper scaling..."
flyctl scale count 1 --app $APP_NAME

# Show deployment status
echo ""
echo "📊 Deployment Status:"
flyctl status --app $APP_NAME

# Get the app URL
APP_URL=$(flyctl info --app $APP_NAME --json | jq -r '.Hostname' 2>/dev/null || echo "$APP_NAME.fly.dev")

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Access your A2A Platform at:"
echo "   Main App: https://$APP_URL"
echo "   API Docs: https://$APP_URL/docs"
echo "   Health: https://$APP_URL/health"
echo ""
echo "📊 Monitor with:"
echo "   flyctl logs --app $APP_NAME"
echo "   flyctl status --app $APP_NAME"
echo ""