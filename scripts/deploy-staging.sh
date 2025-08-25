#!/bin/bash
# Deploy to Fly.io staging environment

set -e

STAGING_APP="a2a-platform-staging"
PROD_APP="a2a-platform"
DOCKER_IMAGE="finsightintelligence/a2a"

echo "🚀 A2A Platform Staging Deployment"
echo "================================="
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

if [ -z "$FLY_API_TOKEN" ]; then
    echo -e "${RED}❌ Error: FLY_API_TOKEN not set${NC}"
    exit 1
fi

# Function to create staging app if needed
create_staging_app() {
    if ! flyctl apps list | grep -q "$STAGING_APP"; then
        echo -e "${BLUE}📱 Creating staging app: $STAGING_APP${NC}"
        flyctl apps create "$STAGING_APP" --org personal
        
        # Set up staging secrets
        echo -e "${BLUE}🔐 Setting up staging secrets${NC}"
        ./scripts/manage-fly-secrets.sh setup-staging
    else
        echo -e "${GREEN}✅ Staging app already exists${NC}"
    fi
}

# Function to build staging image
build_staging_image() {
    echo -e "${BLUE}🔨 Building staging image${NC}"
    
    # Get current git hash
    GIT_HASH=$(git rev-parse --short HEAD)
    STAGING_TAG="staging-$GIT_HASH"
    
    echo "Building image with tag: $STAGING_TAG"
    
    # Build and push staging image
    docker build -t "$DOCKER_IMAGE:$STAGING_TAG" .
    docker tag "$DOCKER_IMAGE:$STAGING_TAG" "$DOCKER_IMAGE:staging"
    
    echo -e "${BLUE}📤 Pushing staging image${NC}"
    docker push "$DOCKER_IMAGE:$STAGING_TAG"
    docker push "$DOCKER_IMAGE:staging"
    
    echo -e "${GREEN}✅ Staging image built: $DOCKER_IMAGE:staging${NC}"
}

# Function to deploy to staging
deploy_staging() {
    echo -e "${BLUE}🚀 Deploying to staging${NC}"
    
    # Deploy using staging config
    flyctl deploy \
        --app "$STAGING_APP" \
        --config fly.staging.toml \
        --image "$DOCKER_IMAGE:staging" \
        --wait-timeout 300
    
    echo -e "${GREEN}✅ Deployed to staging${NC}"
}

# Function to run staging tests
run_staging_tests() {
    echo -e "${BLUE}🧪 Running staging tests${NC}"
    
    # Wait for deployment to stabilize
    sleep 30
    
    # Run validation
    if ./scripts/validate-fly-deployment.sh "$STAGING_APP"; then
        echo -e "${GREEN}✅ Staging validation passed${NC}"
        return 0
    else
        echo -e "${RED}❌ Staging validation failed${NC}"
        return 1
    fi
}

# Function to promote to production
promote_to_production() {
    echo ""
    echo -e "${YELLOW}📋 Promotion Checklist${NC}"
    echo "===================="
    echo "✓ Staging deployment successful"
    echo "✓ All tests passing"
    echo "✓ No critical alerts"
    echo ""
    
    read -p "Promote staging to production? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🚀 Promoting to production${NC}"
        
        # Tag staging image as production
        docker pull "$DOCKER_IMAGE:staging"
        docker tag "$DOCKER_IMAGE:staging" "$DOCKER_IMAGE:main"
        docker push "$DOCKER_IMAGE:main"
        
        # Deploy to production with zero-downtime
        ./scripts/deploy-fly-zero-downtime.sh
        
        echo -e "${GREEN}✅ Promoted to production${NC}"
    else
        echo "Promotion cancelled"
    fi
}

# Main deployment flow
echo -e "${BLUE}1️⃣ Setting up staging environment${NC}"
create_staging_app

echo ""
echo -e "${BLUE}2️⃣ Building staging image${NC}"
build_staging_image

echo ""
echo -e "${BLUE}3️⃣ Deploying to staging${NC}"
deploy_staging

echo ""
echo -e "${BLUE}4️⃣ Testing staging deployment${NC}"
if run_staging_tests; then
    echo ""
    echo -e "${GREEN}✅ Staging deployment successful!${NC}"
    echo ""
    echo "Staging URL: https://${STAGING_APP}.fly.dev"
    echo ""
    echo "Next steps:"
    echo "1. Test the staging environment"
    echo "2. Monitor logs: flyctl logs --app $STAGING_APP"
    echo "3. Check metrics: https://${STAGING_APP}.fly.dev/api/v1/monitoring/dashboard"
    echo ""
    
    # Ask about promotion
    promote_to_production
else
    echo ""
    echo -e "${RED}❌ Staging deployment failed${NC}"
    echo "Check logs: flyctl logs --app $STAGING_APP"
    exit 1
fi