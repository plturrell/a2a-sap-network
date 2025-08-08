#!/bin/bash

# A2A Network - Start with Blockchain
# This script starts Anvil blockchain and the CAP application

echo "🚀 Starting A2A Network with Blockchain..."

# Check if Anvil is installed
if ! command -v anvil &> /dev/null; then
    echo "❌ Anvil not found. Please install Foundry first:"
    echo "   curl -L https://foundry.paradigm.xyz | bash"
    echo "   foundryup"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start Anvil in the background
echo "⛓️  Starting Anvil blockchain..."
anvil --port 8545 --accounts 10 --block-time 1 > anvil.log 2>&1 &
ANVIL_PID=$!

# Wait for Anvil to start
echo "⏳ Waiting for Anvil to start..."
sleep 3

# Check if Anvil is running
if ! nc -z localhost 8545; then
    echo "❌ Anvil failed to start. Check anvil.log for details."
    exit 1
fi

echo "✅ Anvil started (PID: $ANVIL_PID)"

# Deploy contracts if not already deployed
if [ ! -f "broadcast/Deploy.s.sol/31337/run-latest.json" ]; then
    echo "📄 Deploying contracts..."
    cd ../
    forge script script/Deploy.s.sol:DeployScript --rpc-url http://localhost:8545 --broadcast
    cd a2a_network
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    
    # Kill Anvil
    if [ ! -z "$ANVIL_PID" ]; then
        kill $ANVIL_PID 2>/dev/null
        echo "✅ Anvil stopped"
    fi
    
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup INT TERM

# Start CAP application
echo "🌐 Starting CAP application..."
npm start &
CAP_PID=$!

echo ""
echo "✅ A2A Network is running!"
echo ""
echo "📍 Services available at:"
echo "   - CAP Application: http://localhost:4004"
echo "   - OData Service:   http://localhost:4004/odata/v4/a2a/"
echo "   - Fiori App:       http://localhost:4004/app/a2a-fiori/webapp/index.html"
echo "   - Blockchain RPC:  http://localhost:8545"
echo ""
echo "🔑 Default account: $DEFAULT_ACCOUNT"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for CAP process
wait $CAP_PID