#!/bin/bash
# Consolidated linting script for A2A Network

set -e

COMMAND=${1:-"check"}
ARGS="${@:2}"

case $COMMAND in
  "check")
    echo "Running linting checks..."
    eslint srv/ app/ --ext .js,.ts $ARGS
    ;;
  "fix")
    echo "Running linting with auto-fix..."
    eslint srv/ app/ --ext .js,.ts --fix $ARGS
    ;;
  "quick")
    echo "Running quick lint with fixes..."
    eslint srv/ app/ --fix --quiet $ARGS
    ;;
  "format")
    echo "Formatting code with Prettier..."
    prettier --write "**/*.{js,ts,json,md}" $ARGS
    ;;
  "all")
    echo "Running full linting and formatting..."
    eslint srv/ app/ --ext .js,.ts --fix $ARGS
    prettier --write "**/*.{js,ts,json,md}"
    ;;
  *)
    echo "Available lint commands: check, fix, quick, format, all"
    echo "Usage: npm run lint [command] [options]"
    ;;
esac