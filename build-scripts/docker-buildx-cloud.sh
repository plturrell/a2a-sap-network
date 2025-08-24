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

# Build unified A2A SAP Network platform image
echo "Building unified A2A SAP Network platform image..."
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --builder ${BUILDER_NAME} \
  --tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} \
  --tag ${REGISTRY}/${IMAGE_NAME}:latest \
  --label "org.opencontainers.image.created=${BUILD_DATE}" \
  --label "org.opencontainers.image.revision=${GIT_COMMIT}" \
  --label "org.opencontainers.image.version=${VERSION}" \
  --label "org.opencontainers.image.title=A2A SAP Network Platform" \
  --label "org.opencontainers.image.description=Complete A2A SAP Network with 16 agents, SAP Fiori UI, and blockchain integration" \
  --push \
  .

echo "✅ Build complete!"
echo ""
echo "📋 Images pushed:"
echo "  - ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "  - ${REGISTRY}/${IMAGE_NAME}:latest"
echo ""
echo "🚀 Deployment commands:"
echo "  docker pull ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "  docker run -d --name a2a-platform -p 3000:3000 -p 4004:4004 -p 8000-8017:8000-8017 ${REGISTRY}/${IMAGE_NAME}:${VERSION}"