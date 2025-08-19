#!/bin/bash

# Start Local Blockchain for A2A Testing
# Starts Anvil with deployed contracts

echo "🚀 Starting local blockchain for A2A testing..."

# Navigate to a2aNetwork directory
cd /Users/apple/projects/a2a/a2aNetwork

# Check if anvil is already running
if lsof -Pi :8545 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8545 is already in use. Stopping existing process..."
    pkill -f anvil
    sleep 2
fi

# Start anvil with specific configuration
echo "Starting Anvil blockchain..."
anvil \
    --port 8545 \
    --accounts 10 \
    --balance 10000 \
    --mnemonic "test test test test test test test test test test test junk" \
    --chain-id 31337 \
    --block-time 2 \
    --state /tmp/anvil-a2a-state \
    &

# Wait for anvil to start
echo "Waiting for Anvil to start..."
sleep 3

# Deploy contracts
echo "Deploying A2A contracts..."
forge script script/Deploy.s.sol:Deploy \
    --rpc-url http://localhost:8545 \
    --broadcast \
    --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
    --legacy

echo "✅ Blockchain started and contracts deployed!"
echo ""
echo "Contract addresses:"
echo "  AgentRegistry: 0x5FbDB2315678afecb367f032d93F642f64180aa3"
echo "  MessageRouter: 0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
echo ""
echo "RPC URL: http://localhost:8545"
echo "Chain ID: 31337"
echo ""
echo "To stop the blockchain, run: pkill -f anvil"