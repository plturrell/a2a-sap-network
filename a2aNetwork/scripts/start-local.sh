#!/bin/bash

# A2A Network - Local Development Startup Script

set -e

echo "🚀 Starting A2A Network in Local Development Mode"
echo "=================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ and try again."
    exit 1
fi

# Check if required environment file exists
if [ ! -f ".env" ]; then
    echo "📋 Creating local development environment file from template..."
    cp .env.example .env
    
    # Update for local development
    sed -i '' 's/NODE_ENV=production/NODE_ENV=development/' .env
    sed -i '' 's/BTP_ENVIRONMENT=false/BTP_ENVIRONMENT=false/' .env
    sed -i '' 's/ENABLE_XSUAA_VALIDATION=true/ENABLE_XSUAA_VALIDATION=false/' .env
    sed -i '' 's/ALLOW_NON_BTP_AUTH=true/ALLOW_NON_BTP_AUTH=true/' .env
    
    echo "✅ Environment file created. Please review .env and update as needed."
fi

# Create data directory for SQLite
mkdir -p data

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check for required local dependencies
echo "🔍 Checking local dependencies..."
npm list sqlite3 &>/dev/null || {
    echo "📦 Installing SQLite3 for local development..."
    npm install sqlite3 --save-optional
}

npm list dotenv &>/dev/null || {
    echo "📦 Installing dotenv..."
    npm install dotenv --save
}

# Start the local development server
echo ""
echo "🏃‍♂️ Starting Local Development Server..."
echo "Environment: Development"
echo "Database: SQLite (./data/a2a-local.db)"
echo "Authentication: Disabled"
echo "Port: 4004"
echo ""

# Export environment for this session
export NODE_ENV=development
export BTP_ENVIRONMENT=false
export ENABLE_XSUAA_VALIDATION=false
export ALLOW_NON_BTP_AUTH=true

# Start the server
node app/production-launchpad-server.js

echo ""
echo "📱 Access the launchpad at: http://localhost:4004/launchpad.html"
echo "🔧 Health check: http://localhost:4004/health"
echo "👤 User info: http://localhost:4004/user/info"
echo ""
echo "Press Ctrl+C to stop the server"