#!/bin/bash
# Simple local development setup
# No over-engineering - just the basics

echo "🚀 Setting up A2A Agents for local development"

# Check if .env exists, create if not
if [ ! -f .env ]; then
    echo "📝 Creating local .env file"
    cat > .env << EOF
# Local Development Configuration
NODE_ENV=development
PORT=8080

# HANA Local (optional - will work without)
HANA_HOST=localhost
HANA_PORT=30015
HANA_USER=SYSTEM
HANA_PASSWORD=
HANA_SCHEMA=A2A_AGENTS

# Authentication (bypass for local dev)
BYPASS_AUTH=true
XSUAA_CLIENT_ID=local-client
XSUAA_CLIENT_SECRET=local-secret

# Redis (optional)
# REDIS_HOST=localhost
# REDIS_PORT=6379

# Logging
LOG_LEVEL=info
EOF
    echo "✅ Created .env file - edit with your local settings"
fi

# Install dependencies
echo "📦 Installing dependencies"
npm install

# Start the application
echo "🔥 Starting A2A Agents locally"
echo "🌐 Access at: http://localhost:8080"
echo "📋 Health check: http://localhost:8080/health"

npm start