#!/bin/bash
# Fly.io secrets management for A2A Platform

set -e

APP_NAME="${FLY_APP_NAME:-a2a-platform}"
COMMAND="${1:-list}"

echo "🔐 Fly.io Secrets Management"
echo "==========================="
echo "App: $APP_NAME"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if flyctl is available
check_flyctl() {
    if ! command -v flyctl &> /dev/null; then
        echo -e "${RED}❌ Error: flyctl not found${NC}"
        echo "Please install flyctl first: curl -L https://fly.io/install.sh | sh"
        exit 1
    fi
}

# Function to set a secret
set_secret() {
    local key=$1
    local value=$2
    
    echo -n "Setting secret $key... "
    if flyctl secrets set "$key=$value" --app "$APP_NAME" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Done${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
        return 1
    fi
}

# Function to import secrets from env file
import_secrets() {
    local env_file=$1
    
    if [ ! -f "$env_file" ]; then
        echo -e "${RED}❌ Error: File $env_file not found${NC}"
        exit 1
    fi
    
    echo "Importing secrets from $env_file..."
    echo ""
    
    # Read file and set secrets
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Remove quotes from value
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        
        set_secret "$key" "$value"
    done < "$env_file"
}

# Main logic
check_flyctl

case "$COMMAND" in
    list|ls)
        echo "📋 Current secrets:"
        flyctl secrets list --app "$APP_NAME"
        ;;
        
    set)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo -e "${RED}❌ Error: Usage: $0 set KEY VALUE${NC}"
            exit 1
        fi
        set_secret "$2" "$3"
        ;;
        
    unset|remove|rm)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Usage: $0 unset KEY${NC}"
            exit 1
        fi
        echo -n "Removing secret $2... "
        if flyctl secrets unset "$2" --app "$APP_NAME" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Done${NC}"
        else
            echo -e "${RED}❌ Failed${NC}"
        fi
        ;;
        
    import)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Usage: $0 import FILE${NC}"
            exit 1
        fi
        import_secrets "$2"
        ;;
        
    setup-production)
        echo "🚀 Setting up production secrets..."
        echo ""
        
        # Database secrets
        set_secret "DATABASE_URL" "${DATABASE_URL:-postgresql://user:pass@host:5432/a2a}"
        set_secret "REDIS_URL" "${REDIS_URL:-redis://localhost:6379}"
        
        # Authentication secrets
        set_secret "JWT_SECRET_KEY" "${JWT_SECRET_KEY:-$(openssl rand -hex 32)}"
        set_secret "SESSION_SECRET_KEY" "${SESSION_SECRET_KEY:-$(openssl rand -hex 32)}"
        
        # API Keys
        set_secret "OPENAI_API_KEY" "${OPENAI_API_KEY:-}"
        set_secret "SAP_CLIENT_ID" "${SAP_CLIENT_ID:-}"
        set_secret "SAP_CLIENT_SECRET" "${SAP_CLIENT_SECRET:-}"
        
        # Blockchain
        set_secret "ETH_PRIVATE_KEY" "${ETH_PRIVATE_KEY:-}"
        set_secret "ETH_RPC_URL" "${ETH_RPC_URL:-}"
        
        # Monitoring
        set_secret "SENTRY_DSN" "${SENTRY_DSN:-}"
        set_secret "DATADOG_API_KEY" "${DATADOG_API_KEY:-}"
        
        echo ""
        echo -e "${GREEN}✅ Production secrets configured${NC}"
        echo "Note: Empty values need to be set manually"
        ;;
        
    setup-staging)
        echo "🚀 Setting up staging secrets..."
        echo ""
        
        # Use staging-specific values
        set_secret "DATABASE_URL" "${STAGING_DATABASE_URL:-postgresql://user:pass@host:5432/a2a_staging}"
        set_secret "REDIS_URL" "${STAGING_REDIS_URL:-redis://localhost:6379/1}"
        set_secret "A2A_ENVIRONMENT" "staging"
        
        # Copy other secrets from production
        echo "Copying other secrets from production app..."
        flyctl secrets list --app "$APP_NAME" --json | \
            jq -r '.[] | select(.Name != "DATABASE_URL" and .Name != "REDIS_URL" and .Name != "A2A_ENVIRONMENT") | .Name' | \
            while read secret_name; do
                echo "Copying $secret_name to staging..."
            done
        
        echo ""
        echo -e "${GREEN}✅ Staging secrets configured${NC}"
        ;;
        
    validate)
        echo "🔍 Validating required secrets..."
        echo ""
        
        # Get current secrets
        secrets=$(flyctl secrets list --app "$APP_NAME" --json | jq -r '.[].Name' 2>/dev/null || echo "")
        
        # Required secrets
        required_secrets=(
            "JWT_SECRET_KEY"
            "SESSION_SECRET_KEY"
            "DATABASE_URL"
        )
        
        # Recommended secrets
        recommended_secrets=(
            "REDIS_URL"
            "OPENAI_API_KEY"
            "SENTRY_DSN"
            "SAP_CLIENT_ID"
            "SAP_CLIENT_SECRET"
        )
        
        missing_required=0
        missing_recommended=0
        
        echo "Required secrets:"
        for secret in "${required_secrets[@]}"; do
            if echo "$secrets" | grep -q "^$secret$"; then
                echo -e "  ${GREEN}✅ $secret${NC}"
            else
                echo -e "  ${RED}❌ $secret (MISSING)${NC}"
                ((missing_required++))
            fi
        done
        
        echo ""
        echo "Recommended secrets:"
        for secret in "${recommended_secrets[@]}"; do
            if echo "$secrets" | grep -q "^$secret$"; then
                echo -e "  ${GREEN}✅ $secret${NC}"
            else
                echo -e "  ${YELLOW}⚠️  $secret (optional)${NC}"
                ((missing_recommended++))
            fi
        done
        
        echo ""
        if [ $missing_required -eq 0 ]; then
            echo -e "${GREEN}✅ All required secrets are set${NC}"
        else
            echo -e "${RED}❌ Missing $missing_required required secrets${NC}"
            exit 1
        fi
        
        if [ $missing_recommended -gt 0 ]; then
            echo -e "${YELLOW}ℹ️  Missing $missing_recommended recommended secrets${NC}"
        fi
        ;;
        
    export)
        output_file="${2:-.env.fly}"
        echo "📤 Exporting secrets to $output_file..."
        
        # Export secrets to file (values are not included for security)
        flyctl secrets list --app "$APP_NAME" --json | \
            jq -r '.[] | "# \(.Name)\n\(.Name)="' > "$output_file"
        
        echo -e "${GREEN}✅ Exported secret names to $output_file${NC}"
        echo "Note: Secret values are not exported for security reasons"
        ;;
        
    *)
        echo "Usage: $0 {list|set|unset|import|setup-production|setup-staging|validate|export} [args]"
        echo ""
        echo "Commands:"
        echo "  list                    - List all secrets"
        echo "  set KEY VALUE          - Set a secret"
        echo "  unset KEY              - Remove a secret"
        echo "  import FILE            - Import secrets from .env file"
        echo "  setup-production       - Set up production secrets"
        echo "  setup-staging          - Set up staging secrets"
        echo "  validate               - Validate required secrets"
        echo "  export [FILE]          - Export secret names to file"
        echo ""
        echo "Examples:"
        echo "  $0 set JWT_SECRET_KEY mysecretkey"
        echo "  $0 import .env.production"
        echo "  $0 setup-production"
        echo "  $0 validate"
        exit 1
        ;;
esac