#!/bin/bash
# Consolidated security operations script for A2A Network

set -e

COMMAND=${1:-"help"}
ARGS="${@:2}"

case $COMMAND in
  "audit")
    echo "Running security audit..."
    npm audit $ARGS
    node scripts/security-scan.js
    ;;
  "scan")
    echo "Running security scan..."
    node scripts/security-scan.js $ARGS
    ;;
  "validate")
    echo "Running security validation..."
    node scripts/validateSecurity.js $ARGS
    ;;
  "compliance")
    echo "Running compliance check..."
    node scripts/compliance-check.js $ARGS
    ;;
  "enterprise:validate")
    echo "Validating enterprise configuration..."
    node scripts/validate-enterprise-config.js $ARGS
    ;;
  "enterprise:setup")
    echo "Setting up enterprise security..."
    node scripts/setup-enterprise.js $ARGS
    ;;
  "all")
    echo "Running comprehensive security check..."
    echo "1. Running npm audit..."
    npm audit --audit-level moderate
    echo "2. Running security scan..."
    node scripts/security-scan.js
    echo "3. Validating security configuration..."
    node scripts/validateSecurity.js
    echo "4. Running compliance check..."
    node scripts/compliance-check.js $ARGS
    ;;
  *)
    echo "Available security commands:"
    echo "  audit                - Run npm audit and security scan"
    echo "  scan                 - Run security scan"
    echo "  validate             - Validate security configuration"
    echo "  compliance           - Run compliance check"
    echo "  enterprise:validate  - Validate enterprise config"
    echo "  enterprise:setup     - Setup enterprise security"
    echo "  all                  - Run comprehensive security check"
    echo "Usage: npm run security [command] [options]"
    ;;
esac