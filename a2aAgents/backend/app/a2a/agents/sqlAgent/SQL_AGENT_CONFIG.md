# SQL Agent Configuration Guide

## Environment Variables

### Core Configuration
```bash
# Agent Provider Information
export AGENT_PROVIDER_NAME="Your Organization Name"
export AGENT_PROVIDER_URL="https://your-domain.com"
export AGENT_PROVIDER_CONTACT="support@your-domain.com"

# Agent Identification
export SQL_AGENT_ID="sql_agent"
export SQL_AGENT_NAME="A2A SQL Agent"
export SQL_AGENT_DESCRIPTION="Agent for natural language to SQL and SQL to natural language conversion with HANA support"
export SQL_AGENT_VERSION="2.0.0"

# A2A Communication Timeouts (in seconds)
export SQL_AGENT_HTTP_TIMEOUT="30.0"
export A2A_DATA_PRODUCT_TIMEOUT="30.0"
export A2A_QA_VALIDATION_TIMEOUT="30.0"
export A2A_DATA_MANAGER_TIMEOUT="30.0"

# Performance Monitoring
export SQL_AGENT_METRICS_PORT="8007"
export SQL_AGENT_CPU_THRESHOLD="70.0"
export SQL_AGENT_MEMORY_THRESHOLD="70.0"
export SQL_AGENT_RESPONSE_THRESHOLD="5000.0"
export SQL_AGENT_ERROR_THRESHOLD="0.05"
export SQL_AGENT_QUEUE_THRESHOLD="50"

# File System
export SQL_OUTPUT_DIR="/path/to/sql/results"
```

### NLP Model Configuration
```bash
# spaCy Configuration
export SPACY_MODEL="en_core_web_lg"  # or en_core_web_sm, en_core_web_md

# Transformers Configuration  
export TEXT2SQL_MODEL="t5-base"  # or any text2text-generation model
export QA_MODEL="roberta-base-squad2"  # or any question-answering model
export NLP_DEVICE="0"  # GPU device (-1 for CPU, 0+ for GPU)
```

### Database Defaults
```bash
# Vector Database Defaults
export DEFAULT_VECTOR_TABLE="embeddings_table"
export DEFAULT_VECTOR_COLUMN="embedding_vector"
export DEFAULT_TEXT_COLUMN="text_content"

# Spatial Database Defaults  
export DEFAULT_SPATIAL_COORDINATES="52.5200, 13.4050"  # Berlin coordinates
```

### Query Processing
```bash
# Cache Configuration
export SQL_CACHE_SIZE="5000"  # Number of cached queries

# Query Limits
export SQL_DEFAULT_LIMIT="100"  # Default LIMIT for SELECT * queries
```

### GrokClient Configuration
```bash
# AI Validation
export GROK_API_KEY="your-api-key"
export GROK_MODEL="gpt-4"  # or claude-3-sonnet, etc.
export GROK_BASE_URL="https://api.openai.com/v1"
```

## Production Configuration Example

```bash
# Production environment variables
export AGENT_PROVIDER_NAME="Enterprise SQL Analytics"
export AGENT_PROVIDER_URL="https://sql-analytics.enterprise.com"
export AGENT_PROVIDER_CONTACT="sql-support@enterprise.com"

export SQL_AGENT_ID="enterprise_sql_agent"
export SQL_AGENT_NAME="Enterprise SQL Analytics Agent"
export SQL_AGENT_VERSION="2.1.0"

export SQL_AGENT_HTTP_TIMEOUT="45.0"
export A2A_DATA_PRODUCT_TIMEOUT="60.0"
export A2A_QA_VALIDATION_TIMEOUT="45.0"
export A2A_DATA_MANAGER_TIMEOUT="90.0"

export SPACY_MODEL="en_core_web_lg"
export TEXT2SQL_MODEL="facebook/bart-large-cnn"
export QA_MODEL="deepset/roberta-base-squad2"
export NLP_DEVICE="0"

export DEFAULT_VECTOR_TABLE="product_embeddings"
export DEFAULT_VECTOR_COLUMN="description_vector"
export DEFAULT_TEXT_COLUMN="product_description"

export SQL_CACHE_SIZE="10000"
export SQL_DEFAULT_LIMIT="500"

export GROK_API_KEY="${OPENAI_API_KEY}"
export GROK_MODEL="gpt-4-turbo-preview"

export SQL_AGENT_METRICS_PORT="9007"
export SQL_AGENT_CPU_THRESHOLD="80.0"
export SQL_AGENT_MEMORY_THRESHOLD="85.0"
export SQL_OUTPUT_DIR="/opt/sql-agent/results"
```

## Development Configuration Example

```bash
# Development environment variables
export AGENT_PROVIDER_NAME="Dev SQL Agent"
export AGENT_PROVIDER_URL="http://localhost:8007"
export AGENT_PROVIDER_CONTACT="dev@localhost"

export SPACY_MODEL="en_core_web_sm"
export NLP_DEVICE="-1"  # CPU only

export DEFAULT_VECTOR_TABLE="test_documents"
export DEFAULT_SPATIAL_COORDINATES="37.7749, -122.4194"  # San Francisco

export SQL_CACHE_SIZE="100"
export SQL_DEFAULT_LIMIT="10"

# Use local fallback (no API key needed)
# export GROK_API_KEY=""  # Empty = use local analysis
```