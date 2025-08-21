# Agent 0: Data Product Agent

## Overview
The Data Product Agent (Agent 0) serves as the primary entry point for data products into the A2A Network. It creates, manages, and enriches data products with Dublin Core metadata standards, ensuring proper cataloging and lifecycle management.

## Purpose
- Create and register new data products in the A2A Network
- Manage data product lifecycle from ingestion to transformation
- Enrich data with Dublin Core metadata standards
- Perform initial quality control checks
- Prepare data for downstream processing

## Key Features
- **Data Product Creation**: Creates new data products with unique identifiers
- **Data Ingestion**: Handles various data sources and formats
- **Data Transformation**: Initial data transformations and preparations
- **Quality Control**: Basic quality checks before processing
- **Metadata Management**: Dublin Core metadata enrichment and management

## Technical Details
- **Agent Type**: `dataProductAgent`
- **Agent Number**: 0
- **Default Port**: 8000
- **Blockchain Address**: `0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266`
- **Registration Block**: 3

## Capabilities
- `data_product_creation`
- `data_ingestion`
- `data_transformation`
- `quality_control`
- `metadata_management`

## Input/Output
- **Input**: Raw data from various sources (files, APIs, databases, streams)
- **Output**: Registered data products with metadata, ready for standardization

## Integration Points
- Sends data products to Agent 1 (Data Standardization) for processing
- Registers products with Agent 12 (Catalog Manager)
- Reports to Agent 7 (Agent Manager) for lifecycle management
- Stores metadata with Agent 8 (Data Manager)

## Dublin Core Metadata
The agent enriches data products with standard Dublin Core elements:
- Title, Creator, Subject, Description
- Publisher, Contributor, Date, Type
- Format, Identifier, Source, Language
- Relation, Coverage, Rights

## Configuration
```yaml
dataProductAgent:
  metadata:
    required_fields: ["title", "creator", "date", "type"]
    validation_level: "strict"
  quality_control:
    min_quality_score: 0.7
    auto_reject_threshold: 0.3
  ingestion:
    supported_formats: ["json", "xml", "csv", "parquet"]
    max_size_mb: 1000
```

## Usage Example
```python
from a2aNetwork.sdk import Agent

# Initialize Data Product Agent
data_product_agent = Agent(
    agent_type="dataProductAgent",
    endpoint="http://localhost:8000"
)

# Create a new data product
new_product = data_product_agent.create_product({
    "data": raw_data,
    "metadata": {
        "title": "Financial Report Q4 2024",
        "creator": "Finance Department",
        "subject": "Quarterly Financial Data",
        "type": "Dataset",
        "format": "application/json"
    },
    "quality_checks": ["completeness", "validity", "consistency"]
})

print(f"Product created: {new_product['product_id']}")
print(f"Quality score: {new_product['quality_score']}")
```

## Workflow
1. **Ingestion**: Receives raw data from external sources
2. **Validation**: Performs initial validation and quality checks
3. **Metadata Enrichment**: Adds Dublin Core metadata
4. **Registration**: Registers product in blockchain and catalog
5. **Routing**: Routes to Agent 1 for standardization

## Error Codes
- `DP001`: Invalid data format
- `DP002`: Missing required metadata fields
- `DP003`: Quality score below threshold
- `DP004`: Registration failed
- `DP005`: Ingestion size limit exceeded

## Monitoring
- Product creation rate
- Quality score distribution
- Metadata completeness metrics
- Ingestion success/failure rates
- Processing time per product

## Dependencies
- Dublin Core metadata library
- Data validation frameworks
- Blockchain registration modules
- Quality assessment tools