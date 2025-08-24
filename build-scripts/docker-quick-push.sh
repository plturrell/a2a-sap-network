#!/bin/bash
# Quick push to finsightintelligence/a2a repository

set -e

DOCKER_HUB_NAMESPACE="finsightintelligence"
TAG="${1:-latest}"

echo "🚀 Quick push to ${DOCKER_HUB_NAMESPACE}/a2a:${TAG}"

# Simple Docker push commands as requested
echo "📦 Available push commands:"
echo ""

cat << EOF
# Push specific version:
docker push ${DOCKER_HUB_NAMESPACE}/a2a:${TAG}

# Push latest:
docker push ${DOCKER_HUB_NAMESPACE}/a2a:latest

# Push all variants:
docker push ${DOCKER_HUB_NAMESPACE}/a2a:platform-${TAG}
docker push ${DOCKER_HUB_NAMESPACE}/a2a:network-${TAG}
docker push ${DOCKER_HUB_NAMESPACE}/a2a:frontend-${TAG}

# Push with category tags:
docker push ${DOCKER_HUB_NAMESPACE}/a2a:agent-${TAG}
docker push ${DOCKER_HUB_NAMESPACE}/a2a:service-${TAG}
docker push ${DOCKER_HUB_NAMESPACE}/a2a:ui-${TAG}
EOF

echo ""
echo "🏷️  To create and push a new category tag:"
echo ""

cat << EOF
# Example: Create and push a 'production' category
docker tag ${DOCKER_HUB_NAMESPACE}/a2a:latest ${DOCKER_HUB_NAMESPACE}/a2a:production
docker push ${DOCKER_HUB_NAMESPACE}/a2a:production

# Example: Create and push a version with category
docker tag ${DOCKER_HUB_NAMESPACE}/a2a:latest ${DOCKER_HUB_NAMESPACE}/a2a:v1.0-stable
docker push ${DOCKER_HUB_NAMESPACE}/a2a:v1.0-stable
EOF

if [ "$1" ]; then
    echo ""
    echo "⚡ Executing push for tag: ${TAG}"
    docker push ${DOCKER_HUB_NAMESPACE}/a2a:${TAG}
    echo "✅ Push complete!"
fi