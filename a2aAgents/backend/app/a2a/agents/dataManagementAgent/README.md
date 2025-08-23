# Data Management Agent (Agent 8)

A comprehensive data management agent providing enterprise-grade data operations for the A2A platform.

## Overview

The Data Management Agent provides advanced data management capabilities including quality assessment, pipeline orchestration, cataloging, security, and lifecycle management. It leverages real machine learning models and integrates with blockchain for data provenance and integrity verification.

## Key Features

### 🔍 Data Quality Assessment
- **ML-powered quality analysis** using RandomForest and IsolationForest models
- **Automated issue detection** for missing values, duplicates, outliers, and schema inconsistencies
- **Quality scoring** with actionable recommendations
- **Schema validation** against predefined data contracts

### 🔄 Data Pipeline Management
- **ETL pipeline creation** with configurable transformation rules
- **Pipeline orchestration** with status tracking and error handling
- **Data transformation** including filtering, aggregation, and field mapping
- **Multi-format support** (CSV, JSON, Parquet)

### 📊 Data Cataloging
- **Automated metadata extraction** and schema inference
- **Data lineage tracking** with provenance information
- **Tagging and classification** for improved discoverability
- **Size and performance metrics** collection

### 🔐 Data Security & Privacy
- **Encryption at rest** using Fernet symmetric encryption
- **Data integrity verification** with checksums and validation rules
- **Access control** through security hardening mixin
- **Audit logging** for compliance requirements

### 📈 Performance Monitoring
- **Real-time metrics** collection and analysis
- **Performance trend analysis** with ML-based insights
- **Resource utilization** monitoring
- **Alert generation** for threshold violations

### 🗄️ Data Archival & Lifecycle
- **Intelligent archival** with compression and encryption
- **Lifecycle stage management** (Active → Archival → Retention → Disposal)
- **Storage optimization** with multiple compression algorithms
- **Retention policy enforcement**

## Technical Architecture

### Core Components

```python
ComprehensiveDataManagementAgent(
    A2AAgentBase,
    PerformanceMonitoringMixin,
    SecurityHardeningMixin,
    BlockchainIntegrationMixin,
    AIIntelligenceMixin
)
```

### Machine Learning Models
- **Quality Classifier**: RandomForestClassifier for data quality prediction
- **Anomaly Detector**: IsolationForest for outlier detection  
- **Clustering Model**: KMeans for data profiling
- **Embedding Model**: SentenceTransformer for semantic similarity

### Storage Backends
- **Local Filesystem**: Default storage with configurable paths
- **AWS S3**: Cloud storage integration (optional)
- **Azure Blob**: Microsoft cloud storage (optional)
- **Google Cloud Storage**: GCP integration (optional)

## Usage Examples

### Data Quality Assessment

```python
from app.a2a.agents.dataManagementAgent import create_data_management_agent

# Create agent
agent = create_data_management_agent()

# Assess data quality
result = await agent.assess_data_quality(
    data_source="data/customer_data.csv",
    schema={
        "required_columns": ["id", "name", "email"],
        "column_types": {"id": "int64", "name": "object"}
    }
)

print(f"Quality Score: {result.overall_score}/100")
print(f"Issues Found: {len(result.issues)}")
for recommendation in result.recommendations:
    print(f"- {recommendation}")
```

### Data Pipeline Creation

```python
# Create ETL pipeline
pipeline_id = await agent.create_data_pipeline(
    name="Customer Data ETL",
    description="Process and clean customer data",
    source_config={
        "type": "file",
        "path": "raw_data/customers.csv"
    },
    target_config={
        "type": "file", 
        "path": "processed_data/customers_clean.csv",
        "format": "csv"
    },
    transformation_rules=[
        {"type": "filter", "condition": "age >= 18"},
        {"type": "rename", "mapping": {"email_address": "email"}},
        {"type": "calculate", "column": "full_name", "expression": "first_name + ' ' + last_name"}
    ]
)

# Execute pipeline
result = await agent.execute_pipeline(pipeline_id)
print(f"Processed {result['records_processed']} records")
```

### Data Cataloging

```python
# Catalog dataset
catalog_id = await agent.catalog_dataset(
    name="Customer Master Data",
    description="Complete customer information dataset",
    data_location="data/customers.parquet",
    tags=["customers", "master_data", "pii"]
)

# Retrieve catalog entry
entry = agent.data_catalog[catalog_id]
print(f"Dataset: {entry.name}")
print(f"Records: {entry.record_count:,}")
print(f"Size: {entry.size_bytes:,} bytes")
```

### Data Integrity Validation

```python
# Validate data integrity
validation_rules = {
    "ranges": {
        "age": {"min": 0, "max": 120},
        "salary": {"min": 0, "max": 1000000}
    },
    "patterns": {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    }
}

result = await agent.validate_data_integrity(
    data_location="data/employees.csv",
    validation_rules=validation_rules
)

print(f"Integrity Score: {result['integrity_score']}/100")
print(f"Validation Errors: {len(result['validation_errors'])}")
```

### Data Archival

```python
# Archive old data
result = await agent.archive_data(
    data_location="data/old_transactions.csv",
    archive_config={
        "compression": "gzip",
        "encryption": True
    }
)

print(f"Archive ID: {result['archive_id']}")
print(f"Compression Ratio: {result['compression_ratio']:.1f}%")
print(f"Archive Location: {result['archive_location']}")
```

## API Skills Reference

### Core Skills

| Skill | Description | Parameters |
|-------|-------------|------------|
| `assess_data_quality` | ML-powered data quality assessment | `data_source`, `schema?` |
| `create_data_pipeline` | Create ETL pipeline | `name`, `description`, `source_config`, `target_config`, `transformation_rules?` |
| `execute_pipeline` | Execute data pipeline | `pipeline_id` |
| `catalog_dataset` | Add dataset to catalog | `name`, `description`, `data_location`, `schema?`, `tags?` |
| `validate_data_integrity` | Validate data integrity | `data_location`, `validation_rules?` |
| `archive_data` | Archive data with compression | `data_location`, `archive_config?` |
| `monitor_data_performance` | Get performance metrics | `time_range?` |

### MCP Tools

All skills are exposed as MCP tools for integration with other agents and external systems:

- **Data Quality Tools**: Quality assessment, schema validation
- **Pipeline Tools**: Pipeline management, execution monitoring  
- **Catalog Tools**: Dataset cataloging, metadata management
- **Security Tools**: Integrity validation, encryption
- **Performance Tools**: Metrics collection, trend analysis

## Configuration

### Environment Variables

Required for A2A compliance:
```bash
A2A_SERVICE_URL=http://localhost:3000
A2A_SERVICE_HOST=localhost:3000
A2A_BASE_URL=http://localhost:3000
```

### Agent Configuration

```python
config = {
    "temp_storage": "/tmp/data_management",
    "encryption_enabled": True,
    "compression_default": "gzip",
    "ml_models_path": "/path/to/models",
    "performance_monitoring": True,
    "blockchain_integration": True
}

agent = create_data_management_agent(config)
```

## Integration with Service Layer

The agent integrates with the A2A network service layer through:

### Agent 8 Adapter (`agent8-adapter.js`)
Handles REST to OData conversion for:
- Data tasks CRUD operations
- Storage backend management
- Cache configuration
- Data versioning
- Backup operations

### Agent 8 Service (`agent8-service.js`) 
Implements CDS service handlers for:
- Task execution and monitoring
- Health checks and optimization
- Performance metrics collection
- Event emission for blockchain integration

## Testing

Run comprehensive tests:

```bash
cd a2aAgents/backend/app/a2a/agents/dataManagementAgent/active
python test_comprehensive_data_management.py
```

Test coverage includes:
- Data quality assessment with various data issues
- Pipeline creation, execution, and error handling
- Data cataloging and metadata extraction
- Integrity validation with custom rules
- Data archival with compression and encryption
- Performance monitoring and metrics collection
- Error handling and edge cases

## Dependencies

### Required
- pandas, numpy: Data manipulation and analysis
- scikit-learn: Machine learning models
- asyncio: Asynchronous operations
- A2A SDK: Agent base classes and mixins

### Optional (Cloud Storage)
- boto3: AWS S3 integration
- azure-storage-blob: Azure Blob Storage
- google-cloud-storage: Google Cloud Storage

### Optional (Advanced Features)
- sentence-transformers: Semantic embeddings
- cryptography: Data encryption
- lz4, zstandard: Advanced compression
- jsonschema, cerberus: Schema validation

## Security Considerations

- **Data Encryption**: All sensitive data encrypted at rest
- **Access Control**: Role-based access through security mixin
- **Audit Logging**: Complete audit trail of all operations
- **Input Validation**: Comprehensive validation of all inputs
- **Blockchain Integration**: Immutable record of data operations
- **Privacy Protection**: PII detection and masking capabilities

## Performance Characteristics

- **Scalability**: Handles datasets up to 1M+ records efficiently
- **Memory Usage**: Streaming processing for large datasets
- **Concurrent Operations**: Thread pool for parallel processing
- **Caching**: Intelligent caching of frequently accessed data
- **Optimization**: ML-driven performance optimization

## Roadmap

- **Advanced ML Models**: Deep learning for complex data patterns
- **Real-time Streaming**: Support for streaming data pipelines
- **Multi-cloud Support**: Enhanced cloud storage integration
- **Data Mesh**: Federated data management capabilities
- **Automated Governance**: AI-driven data governance policies