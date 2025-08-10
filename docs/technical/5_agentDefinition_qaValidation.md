# A2A Compliant Dynamic Testing Agent Specification
## ORD-Integrated Factuality Testing Service with Dynamic Question Generation

### Overview

**Service Name**: A2A Dynamic Testing Agent with ORD Integration  
**Purpose**: A2A compliant agent for automated factuality testing using dynamically generated questions from ORD registry metadata  
**Integration**: Agent-to-Agent protocol service with ORD registry discovery and SimpleQA methodology  
**Compliance**: 
- A2A Protocol v0.2.9+ compliant
- ORD Specification v1.0+ compatible
- SimpleQA methodology implementation
- Dublin Core ISO 15836 metadata support

---

## Service Architecture

### Core Capabilities

- **A2A Protocol Implementation**: Full compliance with Agent-to-Agent communication standards
- **ORD Registry Discovery**: Automated discovery and metadata extraction from ORD endpoints
- **Dynamic Test Generation**: SimpleQA-style question generation from structured metadata
- **Vector Embedding Integration**: Semantic search and metadata-driven retrieval
- **Ground Truth Validation**: Automated verification against ORD-registered data products
- **Streaming Results**: Real-time test execution and progress reporting

### Supported Resource Types with Dynamic Testing

| Resource Type | ORD Source | Test Generation Method | Example Tests |
|---------------|------------|------------------------|---------------|
| `dataProducts` | Data product metadata | Dublin Core + technical specs | Title, description, ownership questions |
| `apis` | API definitions | Endpoint + schema analysis | Protocol, version, capability questions |
| `events` | Event definitions | Event schema + relationships | Type, structure, dependency questions |
| `entityTypes` | Entity schemas | Type definitions + constraints | Schema, validation, relationship questions |

### A2A Integration Levels

| Integration Level | Features | Use Case |
|------------------|----------|----------|
| **Discovery** | Agent card publishing | Basic A2A registration |
| **Task Execution** | JSON-RPC 2.0 testing workflows | Automated test generation |
| **Streaming** | Real-time progress updates | Long-running test suites |

---

## A2A Agent Card Specification

### Agent Card Structure
```json
{
  "name": "ORD-Dynamic-Testing-Agent",
  "description": "A2A compliant agent for dynamic factuality testing using ORD registry data",
  "version": "1.0.0",
  "protocolVersion": "0.2.9",
  "provider": {
    "name": "Testing Framework Inc",
    "url": "https://testing-framework.com",
    "contact": "support@testing-framework.com"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "stateHistory": true,
    "longRunningTasks": true
  },
  "skills": [
    {
      "name": "dynamic_test_generation",
      "description": "Generate SimpleQA-style tests from ORD metadata",
      "inputModes": ["application/json"],
      "outputModes": ["application/json", "text/plain"],
      "parameters": {
        "ord_endpoints": {
          "type": "array",
          "items": {"type": "string"},
          "description": "List of ORD registry endpoints"
        },
        "test_methodology": {
          "type": "string",
          "enum": ["simpleqa", "factuality_check", "comprehensive"],
          "default": "simpleqa"
        },
        "metadata_schemas": {
          "type": "array",
          "items": {"type": "string"},
          "default": ["dublin_core", "ord_native"]
        }
      }
    },
    {
      "name": "ord_discovery",
      "description": "Discover and process ORD data products",
      "parameters": {
        "namespace_filter": {"type": "string"},
        "resource_types": {
          "type": "array",
          "items": {"type": "string"},
          "enum": ["dataProducts", "apis", "events", "entityTypes"]
        }
      }
    }
  ],
  "securitySchemes": {
    "bearer": {
      "type": "http",
      "scheme": "bearer",
      "bearerFormat": "JWT"
    }
  },
  "serviceEndpoint": "https://api.testing-agent.com/a2a"
}
```

---

## API Specification

### A2A Task Execution Endpoints

#### Execute Dynamic Testing Task
```http
POST /a2a/tasks
Content-Type: application/json
Authorization: Bearer {token}

{
  "method": "executeTask",
  "params": {
    "taskId": "task_${GENERATED_ID}",
    "contextId": "ctx_${CONTEXT_ID}",
    "skill": "dynamic_test_generation",
    "parameters": {
      "ord_endpoints": ["${ORD_ENDPOINT_1}", "${ORD_ENDPOINT_2}"],
      "namespace_filter": "${NAMESPACE_PATTERN}",
      "test_config": {
        "methodology": "simpleqa",
        "difficulty_levels": ["easy", "medium", "hard"],
        "coverage_threshold": 0.8,
        "max_tests_per_product": 50
      }
    }
  }
}
```

**Response:**
```json
{
  "id": "task_${GENERATED_ID}",
  "result": {
    "status": "accepted",
    "taskId": "task_${GENERATED_ID}",
    "estimatedDuration": "${DURATION_SECONDS}",
    "streamingEndpoint": "wss://api.testing-agent.com/a2a/stream/${TASK_ID}"
  }
}
```

#### Get Task Status
```http
GET /a2a/tasks/{taskId}/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "taskId": "task_${TASK_ID}",
  "status": "working",
  "progress": {
    "stage": "generating_tests",
    "percentage": 65,
    "current_operation": "Processing data product: ${PRODUCT_ID}",
    "tests_generated": 127,
    "products_processed": 8
  },
  "results": {
    "partial_report": "${PARTIAL_RESULTS_URL}"
  }
}
```

#### Stream Task Results
```
WebSocket: wss://api.testing-agent.com/a2a/stream/{taskId}
```

**Stream Messages:**
```json
{
  "type": "progress",
  "taskId": "task_${TASK_ID}",
  "data": {
    "stage": "discovery",
    "message": "Discovered ${COUNT} data products from ORD registry",
    "percentage": 25
  }
}

{
  "type": "test_generated",
  "taskId": "task_${TASK_ID}",
  "data": {
    "product_id": "${PRODUCT_ID}",
    "question": "What is the title of data product ${PRODUCT_ID}?",
    "answer": "${EXPECTED_ANSWER}",
    "difficulty": "easy",
    "metadata_source": "dublin_core.title"
  }
}

{
  "type": "completed",
  "taskId": "task_${TASK_ID}",
  "data": {
    "summary": {
      "total_products": 15,
      "total_tests": 342,
      "average_accuracy": 0.87,
      "coverage_score": 0.92
    },
    "report_url": "${FULL_REPORT_URL}"
  }
}
```

---

## Enhanced Data Model

### ORD-Based Test Case Structure

```json
{
  "test_id": "test_${GENERATED_ID}",
  "source_product": {
    "ord_id": "${ORD_ID}",
    "title": "${PRODUCT_TITLE}",
    "namespace": "${NAMESPACE}",
    "registry_endpoint": "${ORD_ENDPOINT}"
  },
  "test_case": {
    "question": "${GENERATED_QUESTION}",
    "answer": "${EXPECTED_ANSWER}",
    "type": "factual|reverse_lookup|enumeration|relationship",
    "difficulty": "easy|medium|hard",
    "methodology": "simpleqa"
  },
  "metadata_source": {
    "element": "dublin_core.title|technical.version|relationship.depends_on",
    "extraction_method": "direct|derived|inferred",
    "confidence": 0.95
  },
  "validation": {
    "ground_truth_source": "${ORD_ENDPOINT}/products/${PRODUCT_ID}",
    "verification_method": "metadata_lookup|api_call|semantic_match",
    "last_verified": "${TIMESTAMP}"
  },
  "embedding_data": {
    "question_vector": [0.1, 0.2, ...],
    "context_vector": [0.3, 0.4, ...],
    "similarity_threshold": 0.8
  }
}
```

### Dynamic Test Suite Structure

```json
{
  "suite_id": "suite_${GENERATED_ID}",
  "created_at": "${TIMESTAMP}",
  "configuration": {
    "ord_discovery": {
      "endpoints": ["${ENDPOINT_LIST}"],
      "namespace_filter": "${FILTER_PATTERN}",
      "resource_types": ["dataProducts", "apis"]
    },
    "test_generation": {
      "methodology": "simpleqa",
      "max_tests_per_product": 50,
      "difficulty_distribution": {
        "easy": 0.4,
        "medium": 0.4,
        "hard": 0.2
      }
    }
  },
  "discovered_products": [
    {
      "ord_id": "${PRODUCT_ID}",
      "metadata": {
        "dublin_core": {
          "title": "${TITLE}",
          "description": "${DESCRIPTION}",
          "creator": "${CREATOR}",
          "subject": ["${TAG_LIST}"]
        },
        "technical": {
          "version": "${VERSION}",
          "apis": ["${API_LIST}"],
          "dependencies": ["${DEPENDENCY_LIST}"]
        }
      },
      "generated_tests": [
        {
          "test_id": "test_${TEST_ID}",
          "question": "${QUESTION}",
          "answer": "${ANSWER}",
          "metadata_source": "${SOURCE_ELEMENT}"
        }
      ]
    }
  ],
  "execution_results": {
    "total_tests": ${TOTAL_COUNT},
    "passed": ${PASSED_COUNT},
    "failed": ${FAILED_COUNT},
    "accuracy": ${ACCURACY_SCORE},
    "coverage_metrics": {
      "dublin_core_coverage": 0.95,
      "technical_coverage": 0.78,
      "relationship_coverage": 0.83
    }
  }
}
```

---

## ORD Integration Specification

### Discovery Workflow

```yaml
ord_discovery:
  steps:
    1. well_known_discovery:
        endpoint: "{ord_base}/.well-known/open-resource-discovery"
        extract: ["openResourceDiscoveryV1", "baseUrl"]
    
    2. configuration_retrieval:
        endpoint: "{baseUrl}/open-resource-discovery/v1/documents"
        extract: ["documents"]
    
    3. document_processing:
        for_each: documents
        extract:
          - dataProducts
          - apiResources  
          - events
          - entityTypes
    
    4. metadata_enrichment:
        dublin_core_mapping:
          title: "dataProduct.title"
          description: "dataProduct.description" 
          creator: "dataProduct.responsible"
          subject: "dataProduct.tags"
          type: "Dataset"
          identifier: "dataProduct.ordId"
```

### Test Generation Rules

```yaml
test_generation_rules:
  dublin_core_elements:
    title:
      question_templates:
        - "What is the title of data product {ord_id}?"
        - "Which data product is titled '{title}'?"
      difficulty: "easy"
      
    description:
      question_templates:
        - "What is the description of {title}?"
        - "Describe the purpose of data product {ord_id}"
      difficulty: "medium"
      
    subject:
      question_templates:
        - "What are the subject tags for {title}?"
        - "Which data products are tagged with '{tag}'?"
      difficulty: "medium"
      
  technical_metadata:
    version:
      question_templates:
        - "What is the current version of {title}?"
        - "When was version {version} of {title} released?"
      difficulty: "easy"
      
    apis:
      question_templates:
        - "What APIs does {title} expose?"
        - "What protocol does the {api_name} API use?"
      difficulty: "medium"
      
  relationships:
    dependencies:
      question_templates:
        - "What data products does {title} depend on?"
        - "Which data products consume data from {title}?"
      difficulty: "hard"
```

---

## Vector Embedding Strategy

### Hierarchical Embedding Structure

```yaml
embedding_levels:
  1_full_context:
    description: "Complete metadata embedding"
    input: "Data Product: {title} | ID: {ord_id} | Description: {description} | Domain: {domain}"
    model: "sentence-transformers/all-mpnet-base-v2"
    
  2_dublin_core:
    description: "Dublin Core specific embedding"
    input: "Title: {dc_title} | Creator: {dc_creator} | Subject: {dc_subject} | Type: {dc_type}"
    model: "sentence-transformers/all-mpnet-base-v2"
    
  3_technical:
    description: "Technical metadata embedding"
    input: "Version: {version} | APIs: {api_list} | Format: {format} | Protocol: {protocol}"
    model: "sentence-transformers/all-mpnet-base-v2"
    
  4_relationships:
    description: "Relationship embedding"
    input: "Dependencies: {dependencies} | Consumers: {consumers} | Related: {related_products}"
    model: "sentence-transformers/all-mpnet-base-v2"
```

---

## Configuration Management

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `A2A_AGENT_PORT` | Service port | No | `${DEFAULT_PORT}` |
| `A2A_PROTOCOL_VERSION` | A2A protocol version | No | `0.2.9` |
| `ORD_DISCOVERY_ENDPOINTS` | ORD registry endpoints | Yes | - |
| `VECTOR_DB_URL` | Vector database connection | Yes | - |
| `EMBEDDING_MODEL` | Sentence transformer model | No | `all-mpnet-base-v2` |
| `TEST_GENERATION_MAX_BATCH` | Max tests per batch | No | `100` |
| `STREAMING_ENABLED` | Enable WebSocket streaming | No | `true` |
| `AUTH_PROVIDER_URL` | Authentication service | Yes | - |
| `GROUND_TRUTH_CACHE_TTL` | Cache TTL for ORD data | No | `3600` |

### Database Schema

```sql
-- A2A Task Management
CREATE TABLE a2a_tasks (
    task_id VARCHAR(${ID_LENGTH}) PRIMARY KEY,
    context_id VARCHAR(${ID_LENGTH}),
    skill_name VARCHAR(100) NOT NULL,
    status ENUM('pending', 'working', 'completed', 'failed') DEFAULT 'pending',
    parameters JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_a2a_tasks_status (status),
    INDEX idx_a2a_tasks_created (created_at)
);

-- ORD Product Discovery Cache
CREATE TABLE ord_products_cache (
    product_id VARCHAR(${ID_LENGTH}) PRIMARY KEY,
    ord_id VARCHAR(200) UNIQUE NOT NULL,
    registry_endpoint VARCHAR(500) NOT NULL,
    metadata JSON NOT NULL,
    dublin_core JSON,
    technical_metadata JSON,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    INDEX idx_ord_cache_endpoint (registry_endpoint),
    INDEX idx_ord_cache_expires (expires_at)
);

-- Dynamic Test Cases
CREATE TABLE dynamic_test_cases (
    test_id VARCHAR(${ID_LENGTH}) PRIMARY KEY,
    suite_id VARCHAR(${ID_LENGTH}) NOT NULL,
    source_product_id VARCHAR(${ID_LENGTH}) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    test_type ENUM('factual', 'reverse_lookup', 'enumeration', 'relationship'),
    difficulty ENUM('easy', 'medium', 'hard'),
    metadata_source VARCHAR(200),
    confidence_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_product_id) REFERENCES ord_products_cache(product_id),
    INDEX idx_dynamic_tests_suite (suite_id),
    INDEX idx_dynamic_tests_type (test_type, difficulty)
);

-- Vector Embeddings
CREATE TABLE test_embeddings (
    embedding_id VARCHAR(${ID_LENGTH}) PRIMARY KEY,
    test_id VARCHAR(${ID_LENGTH}) NOT NULL,
    embedding_type ENUM('question', 'context', 'full'),
    vector_data BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (test_id) REFERENCES dynamic_test_cases(test_id),
    INDEX idx_embeddings_test (test_id),
    INDEX idx_embeddings_type (embedding_type)
);
```

---

## Security & Compliance

### A2A Security Features

1. **Authentication & Authorization**
   - **Bearer Token Authentication**: JWT-based agent authentication
   - **OAuth2 Integration**: Support for enterprise OAuth2 providers
   - **Role-Based Access**: Configurable permissions for test execution

2. **ORD Registry Security**
   - **TLS Encryption**: All ORD communications over HTTPS
   - **API Key Management**: Secure storage of ORD registry credentials
   - **Rate Limiting**: Respectful API usage with configurable limits

3. **Data Privacy**
   - **Metadata Anonymization**: Optional PII removal from test questions
   - **Audit Logging**: Complete audit trail of all test generation
   - **Data Retention**: Configurable retention policies for test data

---

## Monitoring & Analytics

### A2A Agent Metrics

```yaml
metrics:
  a2a_protocol:
    task_execution_time: "${EXECUTION_TIME_P95}"
    streaming_latency: "${STREAM_LATENCY_P95}"
    task_success_rate: "${SUCCESS_RATE_PERCENTAGE}"
    concurrent_tasks: "${ACTIVE_TASK_COUNT}"
  
  ord_integration:
    discovery_latency: "${DISCOVERY_TIME_P95}"
    metadata_extraction_rate: "${EXTRACTION_RATE}"
    cache_hit_ratio: "${CACHE_HIT_PERCENTAGE}"
    registry_availability: "${AVAILABILITY_PERCENTAGE}"
    
  test_generation:
    tests_per_product: "${AVERAGE_TESTS_PER_PRODUCT}"
    generation_speed: "${TESTS_PER_SECOND}"
    accuracy_score: "${AVERAGE_ACCURACY}"
    coverage_score: "${AVERAGE_COVERAGE}"
    
  vector_embeddings:
    embedding_generation_time: "${EMBEDDING_TIME_P95}"
    similarity_search_latency: "${SEARCH_LATENCY_P95}"
    vector_db_performance: "${VECTOR_DB_OPS_PER_SEC}"
```

---

## Deployment Architecture

### Service Components

```yaml
a2a_testing_agent:
  components:
    - a2a_protocol_handler:
        features: [json-rpc-2.0, streaming, task-management]
    - ord_discovery_service:
        features: [registry-discovery, metadata-extraction, caching]
    - test_generation_engine:
        features: [dynamic-questions, simpleqa-methodology, difficulty-scoring]
    - vector_embedding_service:
        features: [hierarchical-embeddings, semantic-search, similarity-matching]
    - ground_truth_validator:
        features: [metadata-verification, api-validation, confidence-scoring]
  
  dependencies:
    - vector_database: "${VECTOR_DB_TYPE}"
    - cache_store: "${REDIS_TYPE}"
    - embedding_model: "${TRANSFORMER_MODEL}"
    - ord_registries: "${ORD_ENDPOINT_LIST}"

  scaling:
    replicas: "${SERVICE_REPLICA_COUNT}"
    auto_scaling:
      min_replicas: "${MIN_REPLICAS}"
      max_replicas: "${MAX_REPLICAS}"
      cpu_threshold: "${CPU_SCALE_THRESHOLD}"
      memory_threshold: "${MEMORY_SCALE_THRESHOLD}"
```

---

## Health Check & Status

### Enhanced Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "a2a_protocol": "healthy",
    "ord_discovery": "healthy", 
    "vector_database": "healthy",
    "embedding_service": "healthy"
  },
  "metrics": {
    "active_tasks": ${ACTIVE_TASK_COUNT},
    "cached_products": ${CACHED_PRODUCT_COUNT},
    "total_tests_generated": ${TOTAL_TESTS},
    "average_accuracy": ${AVERAGE_ACCURACY}
  },
  "ord_registries": {
    "total_configured": ${REGISTRY_COUNT},
    "available": ${AVAILABLE_COUNT},
    "last_discovery": "${LAST_DISCOVERY_TIME}"
  },
  "a2a_compliance": {
    "protocol_version": "0.2.9",
    "capabilities": ["streaming", "longRunningTasks"],
    "agent_card_published": true
  },
  "timestamp": "${HEALTH_CHECK_TIMESTAMP}"
}
```

This specification provides a complete framework for building an A2A compliant agent that dynamically generates SimpleQA-style tests from ORD registry metadata, including all necessary APIs, data models, security considerations, and deployment guidance.