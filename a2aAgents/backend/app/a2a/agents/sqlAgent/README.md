# SQL Agent

## Overview

The SQL Agent is a powerful service that translates natural language questions into SQL queries, executes them against a database, and returns the results in a user-friendly format. It acts as a natural language interface to relational databases. Its primary functions include:

-   **Natural Language to SQL**: Utilizes advanced NLP models to convert user questions into executable SQL queries.
-   **Query Execution**: Securely connects to a target database to run the generated SQL query.
-   **Result Formatting**: Formats the query results into a structured format (e.g., JSON) for consumption by other agents or applications.
-   **Schema Awareness**: Maintains an awareness of the database schema to generate accurate queries.

This agent makes database interaction accessible to non-technical users and enables automated data retrieval based on natural language commands.

## API Specification

This agent exposes a RESTful API for translating natural language to SQL and executing queries.

```yaml
openapi: 3.0.0
info:
  title: SQL Agent API
  description: API for translating natural language questions into SQL queries and executing them against data products.
  version: 1.0.0

servers:
  - url: http://localhost:8014
    description: Local development server

paths:
  /agents/sqlAgent/jobs:
    post:
      summary: Submit a natural language query job
      operationId: submitQueryJob
      tags:
        - SQL Query
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SqlQueryRequest'
      responses:
        '202':
          description: SQL query job accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SqlJobStatus'

  /agents/sqlAgent/jobs/{job_id}:
    get:
      summary: Get the status of a SQL query job
      operationId: getQueryJobStatus
      tags:
        - SQL Query
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Job status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SqlJobStatus'

  /agents/sqlAgent/jobs/{job_id}/results:
    get:
      summary: Get the results of a completed SQL query job
      operationId: getQueryJobResults
      tags:
        - SQL Query
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Query results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QueryResult'

components:
  schemas:
    SqlQueryRequest:
      type: object
      required:
        - data_product_id
        - natural_language_question
      properties:
        data_product_id:
          type: string
          description: The ID of the data product to query.
        natural_language_question:
          type: string
          description: The question to be translated into a SQL query.

    SqlJobStatus:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [pending, generating_sql, executing, completed, failed]
        natural_language_question:
          type: string
        generated_sql:
          type: string
          description: The SQL query generated from the natural language question.
        created_at:
          type: string
          format: date-time

    QueryResult:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [completed]
        columns:
          type: array
          items:
            type: string
        rows:
          type: array
          items:
            type: object
            additionalProperties: true
```

## Configuration

This agent is configured through environment variables. A detailed configuration guide is also available at `SQL_AGENT_CONFIG.md`.

### Core Configuration
-   `SQL_AGENT_HOST`: The host for the agent to listen on (default: `0.0.0.0`).
-   `SQL_AGENT_PORT`: The port for the agent to listen on (default: `8014`).
-   `SQL_AGENT_LOG_LEVEL`: Logging level (e.g., `INFO`, `DEBUG`).

### NLP Model Configuration
-   `NLP_MODEL_PROVIDER`: The NLP model provider (e.g., `openai`, `huggingface`).
-   `NLP_MODEL_NAME`: The specific model to use for NL-to-SQL translation.
-   `OPENAI_API_KEY`: API key if using OpenAI models.

### Database Configuration
-   `DB_TYPE`: The type of the target database (e.g., `postgresql`, `mysql`).
-   `DB_HOST`: The database host.
-   `DB_PORT`: The database port.
-   `DB_USER`: The database username.
-   `DB_PASSWORD`: The database password.
-   `DB_NAME`: The name of the database.

### Query Processing
-   `MAX_QUERY_ROWS`: Maximum number of rows to return from a query (default: `1000`).
-   `QUERY_TIMEOUT`: Timeout for query execution in seconds (default: `60`).

## Usage

To run this agent using its SDK implementation:

```bash
python launch_sql_agent_sdk.py
```

The agent will be available at `http://localhost:8014`.
