#!/bin/bash
# Docker Build Cloud setup script for finsightintelligence/a2a

set -e

# Configuration
DOCKER_HUB_NAMESPACE="finsightintelligence"
DOCKER_CLOUD_ORG="${DOCKER_HUB_NAMESPACE}"
BUILDER_NAME="a2a-cloud-builder"
VERSION="${VERSION:-latest}"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "🚀 Setting up Docker Build Cloud for A2A Platform"
echo "📦 Repository: ${DOCKER_HUB_NAMESPACE}/a2a"
echo "🏷️  Version: ${VERSION}"

# Create Docker Build Cloud builder
echo "Creating Docker Build Cloud builder..."
docker buildx create --driver cloud finsightintelligence/a22 \
  --name ${BUILDER_NAME} \
  --use || {
    echo "Builder already exists, using existing one..."
    docker buildx use ${BUILDER_NAME}
}

# Build multi-platform images with Docker Build Cloud
echo "🔨 Building multi-platform images..."

# Build main application image
echo "Building main A2A platform image..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --builder ${BUILDER_NAME} \
  --tag ${DOCKER_HUB_NAMESPACE}/a2a:${VERSION} \
  --tag ${DOCKER_HUB_NAMESPACE}/a2a:latest \
  --label "org.opencontainers.image.created=${BUILD_DATE}" \
  --label "org.opencontainers.image.revision=${GIT_COMMIT}" \
  --label "org.opencontainers.image.version=${VERSION}" \
  --push \
  .

# Build CDS/CAP network service
echo "Building CDS/CAP network service..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --builder ${BUILDER_NAME} \
  --tag ${DOCKER_HUB_NAMESPACE}/a2a:network-${VERSION} \
  --tag ${DOCKER_HUB_NAMESPACE}/a2a:network-latest \
  --push \
  ./a2aNetwork

# Build frontend
echo "Building frontend..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --builder ${BUILDER_NAME} \
  --tag ${DOCKER_HUB_NAMESPACE}/a2a:frontend-${VERSION} \
  --tag ${DOCKER_HUB_NAMESPACE}/a2a:frontend-latest \
  --push \
  ./a2aAgents/frontend

echo "✅ Build complete!"
echo ""
echo "📋 Images pushed:"
echo "  - ${DOCKER_HUB_NAMESPACE}/a2a:${VERSION}"
echo "  - ${DOCKER_HUB_NAMESPACE}/a2a:latest"
echo "  - ${DOCKER_HUB_NAMESPACE}/a2a:network-${VERSION}"
echo "  - ${DOCKER_HUB_NAMESPACE}/a2a:network-latest"
echo "  - ${DOCKER_HUB_NAMESPACE}/a2a:frontend-${VERSION}"
echo "  - ${DOCKER_HUB_NAMESPACE}/a2a:frontend-latest"