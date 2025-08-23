#!/bin/bash
# Consolidated blockchain operations script for A2A Network

set -e

COMMAND=${1:-"help"}
ARGS="${@:2}"

case $COMMAND in
  "local")
    echo "Starting local blockchain..."
    sh scripts/start-local-blockchain.sh $ARGS
    ;;
  "compile")
    echo "Compiling smart contracts..."
    cd contracts && forge build $ARGS
    ;;
  "test")
    echo "Running blockchain tests..."
    cd contracts && forge test $ARGS
    ;;
  "deploy:local")
    echo "Deploying contracts to local blockchain..."
    node scripts/deployTestContracts.js $ARGS
    ;;
  "deploy:network")
    echo "Deploying contracts to network..."
    cd contracts && forge script script/Deploy.s.sol --broadcast $ARGS
    ;;
  "all")
    echo "Running full blockchain setup..."
    echo "1. Compiling contracts..."
    cd contracts && forge build
    echo "2. Running tests..."
    cd contracts && forge test
    echo "3. Starting local blockchain..."
    sh scripts/start-local-blockchain.sh &
    sleep 10
    echo "4. Deploying contracts..."
    node scripts/deployTestContracts.js $ARGS
    ;;
  *)
    echo "Available blockchain commands:"
    echo "  local         - Start local blockchain"
    echo "  compile       - Compile smart contracts" 
    echo "  test          - Run blockchain tests"
    echo "  deploy:local  - Deploy to local blockchain"
    echo "  deploy:network - Deploy to network"
    echo "  all           - Full blockchain setup"
    echo "Usage: npm run blockchain [command] [options]"
    ;;
esac