#!/bin/bash
# Set up custom domain for Fly.io app

set -e

APP_NAME="${FLY_APP_NAME:-a2a-platform}"
DOMAIN="${1}"

echo "🌐 Custom Domain Setup for Fly.io"
echo "================================"
echo "App: $APP_NAME"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check prerequisites
if ! command -v flyctl &> /dev/null; then
    echo -e "${RED}❌ Error: flyctl not found${NC}"
    exit 1
fi

if [ -z "$DOMAIN" ]; then
    echo -e "${RED}❌ Error: Domain name required${NC}"
    echo "Usage: $0 <domain>"
    echo "Example: $0 api.example.com"
    exit 1
fi

# Function to add certificate
add_certificate() {
    local domain=$1
    
    echo -e "${BLUE}🔐 Adding SSL certificate for $domain${NC}"
    
    if flyctl certs add "$domain" --app "$APP_NAME"; then
        echo -e "${GREEN}✅ Certificate requested${NC}"
    else
        echo -e "${RED}❌ Failed to add certificate${NC}"
        return 1
    fi
}

# Function to check certificate status
check_certificate() {
    local domain=$1
    
    echo -e "${BLUE}🔍 Checking certificate status${NC}"
    
    flyctl certs show "$domain" --app "$APP_NAME"
}

# Function to get DNS records
get_dns_records() {
    local domain=$1
    
    echo -e "${BLUE}📋 Required DNS Records${NC}"
    echo "======================"
    
    # Get the app's IPv4 and IPv6 addresses
    APP_INFO=$(flyctl info --app "$APP_NAME" --json)
    IPV4=$(echo "$APP_INFO" | jq -r '.IPAddresses[] | select(.Type == "v4") | .Address' | head -1)
    IPV6=$(echo "$APP_INFO" | jq -r '.IPAddresses[] | select(.Type == "v6") | .Address' | head -1)
    
    if [ -z "$IPV4" ] && [ -z "$IPV6" ]; then
        echo -e "${YELLOW}⚠️  No IP addresses found. Allocating...${NC}"
        
        # Allocate IPs
        flyctl ips allocate-v4 --app "$APP_NAME"
        flyctl ips allocate-v6 --app "$APP_NAME"
        
        # Get IPs again
        APP_INFO=$(flyctl info --app "$APP_NAME" --json)
        IPV4=$(echo "$APP_INFO" | jq -r '.IPAddresses[] | select(.Type == "v4") | .Address' | head -1)
        IPV6=$(echo "$APP_INFO" | jq -r '.IPAddresses[] | select(.Type == "v6") | .Address' | head -1)
    fi
    
    echo ""
    echo "Add these DNS records to your domain:"
    echo ""
    
    if [ -n "$IPV4" ]; then
        echo "A Record:"
        echo "  Name: @ (or $domain)"
        echo "  Value: $IPV4"
        echo "  TTL: 300"
        echo ""
    fi
    
    if [ -n "$IPV6" ]; then
        echo "AAAA Record:"
        echo "  Name: @ (or $domain)"
        echo "  Value: $IPV6"
        echo "  TTL: 300"
        echo ""
    fi
    
    # For subdomains
    if [[ "$domain" == *.* ]] && [[ "$domain" != www.* ]]; then
        echo "Alternative: CNAME Record"
        echo "  Name: ${domain%%.*}"
        echo "  Value: ${APP_NAME}.fly.dev"
        echo "  TTL: 300"
    fi
}

# Function to wait for DNS propagation
wait_for_dns() {
    local domain=$1
    local max_attempts=30
    local attempt=0
    
    echo -e "${BLUE}⏳ Waiting for DNS propagation${NC}"
    echo "This may take a few minutes..."
    
    while [ $attempt -lt $max_attempts ]; do
        if nslookup "$domain" 8.8.8.8 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ DNS is resolving${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 10
        ((attempt++))
    done
    
    echo ""
    echo -e "${YELLOW}⚠️  DNS not fully propagated yet${NC}"
    echo "Certificate validation may fail. You can check status later with:"
    echo "  $0 status $domain"
}

# Function to validate SSL
validate_ssl() {
    local domain=$1
    
    echo -e "${BLUE}🔒 Validating SSL certificate${NC}"
    
    if curl -s -I "https://$domain" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ SSL is working${NC}"
        
        # Show certificate details
        echo ""
        echo "Certificate details:"
        echo "==================="
        openssl s_client -connect "$domain:443" -servername "$domain" </dev/null 2>/dev/null | \
            openssl x509 -noout -dates -subject -issuer 2>/dev/null || \
            echo "Unable to fetch certificate details"
    else
        echo -e "${YELLOW}⚠️  SSL not ready yet${NC}"
        echo "Check certificate status with: flyctl certs show $domain --app $APP_NAME"
    fi
}

# Function to update app configuration
update_app_config() {
    local domain=$1
    
    echo -e "${BLUE}⚙️  Updating app configuration${NC}"
    
    # Update environment variables
    flyctl secrets set \
        CUSTOM_DOMAIN="$domain" \
        ALLOWED_HOSTS="$domain,${APP_NAME}.fly.dev,localhost" \
        --app "$APP_NAME"
    
    echo -e "${GREEN}✅ App configuration updated${NC}"
}

# Main flow based on command
COMMAND="${1:-setup}"
DOMAIN="${2:-$1}"

case "$COMMAND" in
    setup)
        echo -e "${BLUE}Setting up custom domain: $DOMAIN${NC}"
        echo ""
        
        # Step 1: Get DNS records
        get_dns_records "$DOMAIN"
        
        echo ""
        echo -e "${YELLOW}⚠️  Please add the DNS records above to your domain provider${NC}"
        echo "Press Enter when DNS records are added..."
        read -r
        
        # Step 2: Wait for DNS
        wait_for_dns "$DOMAIN"
        
        # Step 3: Add certificate
        echo ""
        add_certificate "$DOMAIN"
        
        # Step 4: Update app config
        echo ""
        update_app_config "$DOMAIN"
        
        # Step 5: Check certificate status
        echo ""
        check_certificate "$DOMAIN"
        
        # Step 6: Validate SSL
        echo ""
        sleep 30  # Give certificate time to provision
        validate_ssl "$DOMAIN"
        
        echo ""
        echo -e "${GREEN}✅ Custom domain setup complete!${NC}"
        echo ""
        echo "Your app is now available at:"
        echo "  https://$DOMAIN"
        echo "  https://${APP_NAME}.fly.dev (backup)"
        ;;
        
    status)
        if [ -z "$DOMAIN" ]; then
            echo -e "${BLUE}📋 Current domains and certificates:${NC}"
            flyctl certs list --app "$APP_NAME"
        else
            check_certificate "$DOMAIN"
            echo ""
            validate_ssl "$DOMAIN"
        fi
        ;;
        
    remove)
        if [ -z "$DOMAIN" ]; then
            echo -e "${RED}❌ Error: Domain name required${NC}"
            echo "Usage: $0 remove <domain>"
            exit 1
        fi
        
        echo -e "${YELLOW}⚠️  Removing custom domain: $DOMAIN${NC}"
        
        if flyctl certs remove "$DOMAIN" --app "$APP_NAME" --yes; then
            echo -e "${GREEN}✅ Domain removed${NC}"
        else
            echo -e "${RED}❌ Failed to remove domain${NC}"
        fi
        ;;
        
    *)
        echo "Usage: $0 {setup|status|remove} [domain]"
        echo ""
        echo "Commands:"
        echo "  setup <domain>    - Set up a new custom domain"
        echo "  status [domain]   - Check domain/certificate status"
        echo "  remove <domain>   - Remove a custom domain"
        echo ""
        echo "Examples:"
        echo "  $0 setup api.example.com"
        echo "  $0 status api.example.com"
        echo "  $0 remove api.example.com"
        exit 1
        ;;
esac