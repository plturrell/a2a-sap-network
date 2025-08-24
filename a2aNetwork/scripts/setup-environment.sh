#!/bin/bash

# A2A Network Environment Setup Script
# This script helps developers set up their environment securely

set -e

echo "🔧 A2A Network Environment Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -f ".env.template" ]; then
    print_error "Please run this script from the A2A Network root directory"
    exit 1
fi

print_info "Setting up A2A Network environment..."

# Determine environment
ENV=${1:-development}
if [ "$ENV" != "development" ] && [ "$ENV" != "staging" ] && [ "$ENV" != "production" ]; then
    print_error "Invalid environment: $ENV. Use: development, staging, or production"
    exit 1
fi

print_info "Environment: $ENV"

# Check if .env already exists
if [ -f ".env" ]; then
    print_warning ".env file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled. Your existing .env file is unchanged."
        exit 0
    fi
    print_warning "Backing up existing .env to .env.backup"
    cp .env .env.backup
fi

# Copy appropriate template
if [ "$ENV" = "development" ]; then
    if [ -f ".env.development.template" ]; then
        cp .env.development.template .env
        print_status "Copied development template to .env"
    else
        print_error "Development template not found"
        exit 1
    fi
elif [ "$ENV" = "production" ] || [ "$ENV" = "staging" ]; then
    cp .env.template .env
    print_status "Copied production template to .env"
    
    print_warning "SECURITY WARNING: You must replace ALL placeholder values!"
    print_warning "Generate secure secrets using: openssl rand -hex 32"
fi

# Generate secure secrets for production/staging
if [ "$ENV" = "production" ] || [ "$ENV" = "staging" ]; then
    echo ""
    print_info "Generating secure secrets..."
    
    # Generate JWT secrets
    JWT_SECRET=$(openssl rand -hex 32)
    JWT_REFRESH_SECRET=$(openssl rand -hex 32)
    SESSION_SECRET=$(openssl rand -hex 32)
    ENCRYPTION_KEY=$(openssl rand -hex 32)
    
    # Replace placeholders in .env file
    if command -v gsed >/dev/null 2>&1; then
        SED_CMD=gsed
    else
        SED_CMD=sed
    fi
    
    $SED_CMD -i "s/YOUR_JWT_SECRET_256_BITS_MINIMUM_32_CHARACTERS/$JWT_SECRET/g" .env
    $SED_CMD -i "s/YOUR_JWT_REFRESH_SECRET_256_BITS_MINIMUM_32_CHARACTERS/$JWT_REFRESH_SECRET/g" .env
    $SED_CMD -i "s/YOUR_SESSION_SECRET_256_BITS_MINIMUM/$SESSION_SECRET/g" .env
    $SED_CMD -i "s/YOUR_32_BYTE_ENCRYPTION_KEY_HERE/$ENCRYPTION_KEY/g" .env
    
    print_status "Generated secure JWT secrets"
    print_warning "You still need to update database passwords, API keys, and other service credentials!"
fi

# Set appropriate file permissions
chmod 600 .env
print_status "Set secure file permissions (600) on .env"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    print_info "Installing dependencies..."
    npm install
    print_status "Dependencies installed"
fi

# Verify required directories exist
mkdir -p db logs temp
print_status "Created required directories"

# Final instructions
echo ""
echo "🎉 Environment setup complete!"
echo ""

if [ "$ENV" = "development" ]; then
    print_info "Development environment ready:"
    echo "  • Database: SQLite (./db/a2a_dev.db)"
    echo "  • Blockchain: Local Anvil (localhost:8545)"
    echo "  • Debug mode: Enabled"
    echo "  • Rate limiting: Disabled for easier testing"
    echo ""
    print_info "To start the development server:"
    echo "  npm run dev"
else
    print_warning "Production environment requires additional configuration:"
    echo "  • Update database connection string"
    echo "  • Set blockchain RPC URL and private keys"
    echo "  • Configure monitoring endpoints"
    echo "  • Set up SSL certificates"
    echo "  • Update all YOUR_*_HERE placeholders"
    echo ""
    print_error "DO NOT use this configuration in production without updating secrets!"
fi

echo ""
print_info "Environment variables are stored in .env (ignored by git)"
print_info "Template files are available for reference"

# Security reminder
echo ""
echo "🔒 SECURITY REMINDERS:"
echo "  • Never commit .env files to version control"
echo "  • Use different secrets for each environment"
echo "  • Store production secrets in secure secret management"
echo "  • Rotate secrets regularly"
echo "  • Monitor for exposed secrets in logs"

exit 0