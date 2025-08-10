#!/bin/bash

# FinSight CIB ORD Registry - Staging Deployment Script
# This script deploys the dual-database ORD registry to staging environment

set -e  # Exit on any error

echo "🚀 Starting FinSight CIB ORD Registry Staging Deployment"
echo "========================================================"

# Configuration
STAGING_ENV_FILE="config/staging.env"
DOCKER_IMAGE="finsight-ord-registry:staging-latest"
CONTAINER_NAME="ord-registry-staging"
HEALTH_ENDPOINT="http://localhost:8080/health"

# Check if staging environment file exists
if [ ! -f "$STAGING_ENV_FILE" ]; then
    echo "❌ Error: Staging environment file not found at $STAGING_ENV_FILE"
    echo "Please ensure all environment variables are configured."
    exit 1
fi

echo "✅ Environment configuration found"

# Stop existing containers (if any)
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.staging.yml down --remove-orphans || true

# Pull latest base images
echo "📦 Pulling latest base images..."
docker pull python:3.11-slim
docker pull redis:7-alpine
docker pull prom/prometheus:latest
docker pull grafana/grafana:latest

# Build the ORD Registry image
echo "🔨 Building ORD Registry Docker image..."
docker build -f Dockerfile.staging -t $DOCKER_IMAGE .

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs data monitoring/grafana/dashboards monitoring/grafana/datasources

# Run tests before deployment
echo "🧪 Running tests before deployment..."
python3 test_ord_dual_database.py || {
    echo "❌ Tests failed! Deployment aborted."
    exit 1
}

echo "✅ All tests passed (100% success rate expected)"

# Deploy using Docker Compose
echo "🚀 Deploying to staging environment..."
docker-compose -f docker-compose.staging.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
for i in {1..10}; do
    if curl -f $HEALTH_ENDPOINT > /dev/null 2>&1; then
        echo "✅ Health check passed!"
        break
    else
        echo "⏳ Health check attempt $i/10 failed, retrying in 10 seconds..."
        sleep 10
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ Health check failed after 10 attempts!"
        echo "📋 Container logs:"
        docker logs $CONTAINER_NAME --tail 50
        exit 1
    fi
done

# Verify all tests pass in staging
echo "🧪 Running full test suite in staging environment..."
docker exec $CONTAINER_NAME python3 test_ord_dual_database.py || {
    echo "❌ Staging tests failed!"
    echo "📋 Container logs:"
    docker logs $CONTAINER_NAME --tail 50
    exit 1
}

# Display deployment status
echo ""
echo "🎉 ORD Registry Staging Deployment Complete!"
echo "=============================================="
echo "📍 API Endpoint: http://localhost:8080"
echo "📊 Metrics: http://localhost:9090 (Prometheus)"
echo "📈 Dashboard: http://localhost:3000 (Grafana)"
echo "🏥 Health Check: $HEALTH_ENDPOINT"
echo ""
echo "🔧 Useful Commands:"
echo "  View logs: docker logs $CONTAINER_NAME -f"
echo "  Stop: docker-compose -f docker-compose.staging.yml down"
echo "  Restart: docker-compose -f docker-compose.staging.yml restart"
echo ""
echo "✅ Your dual-database ORD registry is now running in staging!"
