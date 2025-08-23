#!/bin/bash
# Consolidated test script for A2A Network

set -e

COMMAND=${1:-"all"}
ARGS="${@:2}"

case $COMMAND in
  "all")
    echo "Running all tests..."
    jest --config jest.config.js $ARGS
    ;;
  "unit")
    echo "Running unit tests..."
    jest --testPathPattern=unit $ARGS
    ;;
  "integration") 
    echo "Running integration tests..."
    jest --testPathPattern=integration $ARGS
    ;;
  "coverage")
    echo "Running tests with coverage..."
    jest --coverage $ARGS
    ;;
  "watch")
    echo "Running tests in watch mode..."
    jest --watch $ARGS
    ;;
  "quick")
    echo "Running quick test suite..."
    jest --maxWorkers=4 $ARGS
    ;;
  "memory")
    echo "Running memory tests..."
    node --expose-gc --inspect test/memory-test.js $ARGS
    ;;
  "glean")
    case ${2:-"all"} in
      "integration")
        node test/integration/gleanIntegrationTest.js ${@:3}
        ;;
      "verbose")
        node test/integration/gleanIntegrationTest.js --verbose ${@:3}
        ;;
      "validate")
        node test/integration/phase1ValidationTest.js ${@:3}
        ;;
      "ast")
        node test/ast-parsing-test.js ${@:3}
        ;;
      "codebase")
        node test/codebase-parsing-test.js ${@:3}
        ;;
      "cap")
        node test/cap-integration-test.js ${@:3}
        ;;
      *)
        echo "Available Glean test commands: integration, verbose, validate, ast, codebase, cap"
        ;;
    esac
    ;;
  *)
    echo "Available test commands: all, unit, integration, coverage, watch, quick, memory, glean"
    echo "Usage: npm run test [command] [options]"
    ;;
esac