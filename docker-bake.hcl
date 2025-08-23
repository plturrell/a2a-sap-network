# Docker Bake configuration for building all services
# This allows building all images in parallel with proper caching

variable "REGISTRY" {
  default = "docker.io/yourusername"
}

variable "TAG" {
  default = "latest"
}

group "default" {
  targets = ["a2a-agents", "a2a-network", "a2a-unified"]
}

group "all" {
  targets = [
    "registry-server",
    "agent0-data-product",
    "agent1-standardization",
    "agent2-ai-preparation",
    "agent3-vector-processing",
    "agent4-calc-validation",
    "agent5-qa-validation",
    "agent6-quality-control",
    "reasoning-agent",
    "sql-agent",
    "agent-manager",
    "data-manager",
    "catalog-manager",
    "calculation-agent",
    "agent-builder",
    "embedding-finetuner",
    "unified-service",
    "network-service",
    "notification-system"
  ]
}

# Base target for all agent images
target "agent-base" {
  context = "./a2aAgents/backend"
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  cache-from = ["type=registry,ref=${REGISTRY}/a2a-cache:buildcache"]
  cache-to = ["type=registry,ref=${REGISTRY}/a2a-cache:buildcache,mode=max"]
}

# Individual agent targets
target "registry-server" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "registry-server"
    AGENT_PORT = "8000"
  }
  tags = ["${REGISTRY}/a2a-registry-server:${TAG}"]
}

target "agent0-data-product" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent0"
    AGENT_PORT = "8001"
  }
  tags = ["${REGISTRY}/a2a-agent0:${TAG}"]
}

target "agent1-standardization" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent1"
    AGENT_PORT = "8002"
  }
  tags = ["${REGISTRY}/a2a-agent1:${TAG}"]
}

target "agent2-ai-preparation" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent2"
    AGENT_PORT = "8003"
  }
  tags = ["${REGISTRY}/a2a-agent2:${TAG}"]
}

target "agent3-vector-processing" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent3"
    AGENT_PORT = "8004"
  }
  tags = ["${REGISTRY}/a2a-agent3:${TAG}"]
}

target "agent4-calc-validation" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent4"
    AGENT_PORT = "8005"
  }
  tags = ["${REGISTRY}/a2a-agent4:${TAG}"]
}

target "agent5-qa-validation" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent5"
    AGENT_PORT = "8006"
  }
  tags = ["${REGISTRY}/a2a-agent5:${TAG}"]
}

target "agent6-quality-control" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent6"
    AGENT_PORT = "8007"
  }
  tags = ["${REGISTRY}/a2a-agent6:${TAG}"]
}

target "reasoning-agent" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "reasoning-agent"
    AGENT_PORT = "8008"
  }
  tags = ["${REGISTRY}/a2a-reasoning:${TAG}"]
}

target "sql-agent" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "sql-agent"
    AGENT_PORT = "8009"
  }
  tags = ["${REGISTRY}/a2a-sql:${TAG}"]
}

target "agent-manager" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent-manager"
    AGENT_PORT = "8010"
  }
  tags = ["${REGISTRY}/a2a-agent-manager:${TAG}"]
}

target "data-manager" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "data-manager"
    AGENT_PORT = "8011"
  }
  tags = ["${REGISTRY}/a2a-data-manager:${TAG}"]
}

target "catalog-manager" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "catalog-manager"
    AGENT_PORT = "8012"
  }
  tags = ["${REGISTRY}/a2a-catalog-manager:${TAG}"]
}

target "calculation-agent" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "calculation-agent"
    AGENT_PORT = "8013"
  }
  tags = ["${REGISTRY}/a2a-calculation:${TAG}"]
}

target "agent-builder" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "agent-builder"
    AGENT_PORT = "8014"
  }
  tags = ["${REGISTRY}/a2a-agent-builder:${TAG}"]
}

target "embedding-finetuner" {
  inherits = ["agent-base"]
  args = {
    AGENT_NAME = "embedding-finetuner"
    AGENT_PORT = "8015"
  }
  tags = ["${REGISTRY}/a2a-embedding:${TAG}"]
}

target "unified-service" {
  context = "./a2aAgents/backend"
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["${REGISTRY}/a2a-unified:${TAG}"]
  cache-from = ["type=registry,ref=${REGISTRY}/a2a-cache:unified"]
  cache-to = ["type=registry,ref=${REGISTRY}/a2a-cache:unified,mode=max"]
}

target "network-service" {
  context = "./a2aNetwork"
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["${REGISTRY}/a2a-network:${TAG}"]
  cache-from = ["type=registry,ref=${REGISTRY}/a2a-cache:network"]
  cache-to = ["type=registry,ref=${REGISTRY}/a2a-cache:network,mode=max"]
}

target "notification-system" {
  context = "./a2aNetwork"
  dockerfile = "Dockerfile"
  args = {
    SERVICE_NAME = "notification"
  }
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["${REGISTRY}/a2a-notification:${TAG}"]
  cache-from = ["type=registry,ref=${REGISTRY}/a2a-cache:notification"]
  cache-to = ["type=registry,ref=${REGISTRY}/a2a-cache:notification,mode=max"]
}

# Grouped targets for easier building
target "a2a-agents" {
  inherits = ["agent-base"]
  tags = ["${REGISTRY}/a2a-agents:${TAG}"]
}

target "a2a-network" {
  inherits = ["network-service"]
}

target "a2a-unified" {
  inherits = ["unified-service"]
}