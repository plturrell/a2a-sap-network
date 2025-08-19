#!/bin/bash
# Local blockchain startup script

echo "🔗 Starting local Ethereum blockchain..."

# Check if ganache is installed
if ! command -v ganache &> /dev/null; then
    echo "Installing ganache-cli..."
    npm install -g ganache-cli
fi

# Start ganache with deterministic accounts
ganache-cli \
  --deterministic \
  --accounts 20 \
  --host 0.0.0.0 \
  --port 8545 \
  --gasLimit 12000000 \
  --quiet &

echo "✅ Local blockchain started on http://localhost:8545"
echo "📝 Using deterministic accounts for testing"
