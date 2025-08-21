# Agent 11: SQL Agent

## Overview
The SQL Agent (Agent 11) provides natural language to SQL translation and advanced database operations for the A2A Network. It enables users to query data using natural language and handles complex database interactions across multiple data sources.

## Purpose
- Convert natural language queries to SQL
- Execute complex database operations
- Optimize query performance
- Extract and transform data from databases
- Manage database schemas and migrations

## Key Features
- **SQL Query Execution**: Execute queries across multiple databases
- **Database Operations**: CRUD operations, transactions, migrations
- **Query Optimization**: Automatic query optimization and indexing
- **Data Extraction**: Efficient data extraction and transformation
- **Schema Management**: Database schema creation and updates

## Technical Details
- **Agent Type**: `sqlAgent`
- **Agent Number**: 11
- **Default Port**: 8011
- **Blockchain Address**: `0x71bE63f3384f5fb98995898A86B02Fb2426c5788`
- **Registration Block**: 14

## Capabilities
- `sql_query_execution`
- `database_operations`
- `query_optimization`
- `data_extraction`
- `schema_management`

## Input/Output
- **Input**: Natural language queries, SQL statements, schema definitions
- **Output**: Query results, execution status, optimization suggestions

## Database Configuration
```yaml
sqlAgent:
  databases:
    primary:
      type: "hana"
      connection_pool:
        min: 5
        max: 20
    secondary:
      - type: "postgresql"
        read_replicas: 3
      - type: "sqlite"
        path: "/data/local.db"
  nl_to_sql:
    model: "gpt-4-sql"
    context_aware: true
    schema_learning: true
  optimization:
    auto_indexing: true
    query_cache: true
    explain_analyze: true
```

## Usage Example
```python
from a2aNetwork.sdk import Agent

# Initialize SQL Agent
sql_agent = Agent(
    agent_type="sqlAgent",
    endpoint="http://localhost:8011"
)

# Natural language query
result = sql_agent.query({
    "natural_language": "Show me top 10 customers by revenue last quarter",
    "context": {
        "database": "sales_db",
        "fiscal_calendar": "standard"
    },
    "options": {
        "optimize": True,
        "explain": True
    }
})

print(f"Generated SQL: {result['sql']}")
print(f"Results: {result['data']}")
print(f"Execution time: {result['execution_time_ms']}ms")

# Direct SQL execution
sql_result = sql_agent.execute_sql({
    "sql": "SELECT * FROM transactions WHERE amount > :amount",
    "parameters": {"amount": 10000},
    "database": "financial_db",
    "timeout": 30
})

# Schema management
schema_update = sql_agent.update_schema({
    "operation": "add_column",
    "table": "customers",
    "column": {
        "name": "risk_score",
        "type": "DECIMAL(3,2)",
        "default": 0.5
    }
})
```

## Natural Language Processing
```json
{
  "nl_processing": {
    "steps": [
      "intent_recognition",
      "entity_extraction",
      "schema_mapping",
      "sql_generation",
      "validation"
    ],
    "context_retention": true,
    "ambiguity_resolution": "interactive"
  }
}
```

## Query Optimization
1. **Index Analysis**: Automatic index recommendations
2. **Query Rewriting**: Optimize query structure
3. **Execution Plans**: Analyze and improve
4. **Caching Strategy**: Smart result caching
5. **Partitioning**: Data partitioning suggestions

## Supported SQL Dialects
- SAP HANA SQL
- PostgreSQL
- MySQL/MariaDB
- SQLite
- Oracle SQL
- Microsoft SQL Server

## Error Codes
- `SQL001`: Invalid SQL syntax
- `SQL002`: Connection timeout
- `SQL003`: Permission denied
- `SQL004`: Natural language parsing failed
- `SQL005`: Query optimization failed

## Security Features
- SQL injection prevention
- Parameter binding
- Query whitelisting
- Access control integration
- Audit logging

## Dependencies
- NL to SQL models
- Database drivers
- Query optimization engines
- Schema migration tools
- SQL parsing libraries