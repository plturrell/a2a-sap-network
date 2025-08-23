# Data Manager

## Overview

The Data Manager agent is responsible for the physical storage and retrieval of data products within the A2A Network. It provides a unified interface to interact with various underlying storage systems, abstracting the complexity from other agents. Its key functions include:

-   **Data Storage**: Manages the writing of data products to persistent storage, such as a database, data lake, or object store.
-   **Data Retrieval**: Provides an API to read data products from storage, handling aspects like access control and data format conversion.
-   **Lifecycle Management**: Implements data retention policies, including archiving and deletion of data products.

This agent acts as the primary custodian of data within the network, ensuring its availability, integrity, and security.

## API Specification

This agent exposes a RESTful API for data storage and retrieval.

```yaml
openapi: 3.0.0
info:
  title: Data Manager API
  description: API for ingesting, storing, and retrieving data product payloads.
  version: 1.0.0

servers:
  - url: http://localhost:8011
    description: Local development server

paths:
  /agents/dataManager/ingest:
    post:
      summary: Ingest data from a source
      operationId: ingestData
      tags:
        - Data Ingestion
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/IngestionRequest'
      responses:
        '202':
          description: Ingestion job started
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/IngestionJob'

  /agents/dataManager/ingest/{job_id}:
    get:
      summary: Get ingestion job status
      operationId: getIngestionStatus
      tags:
        - Data Ingestion
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Ingestion job status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/IngestionJob'

  /agents/dataManager/data/{data_product_id}:
    get:
      summary: Retrieve a data product payload
      operationId: getDataPayload
      tags:
        - Data Access
      parameters:
        - name: data_product_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Data product payload
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
            application/json:
              schema:
                type: object
            text/csv:
              schema:
                type: string

components:
  schemas:
    IngestionRequest:
      type: object
      required:
        - data_product_id
        - source
      properties:
        data_product_id:
          type: string
          description: The ID of the data product this payload belongs to.
        source:
          type: object
          properties:
            type:
              type: string
              enum: [s3, database, http, file_upload]
            uri:
              type: string
              format: uri
            credentials_secret:
              type: string
              description: Secret key for accessing the data source.

    IngestionJob:
      type: object
      properties:
        job_id:
          type: string
        data_product_id:
          type: string
        status:
          type: string
          enum: [pending, running, completed, failed]
        message:
          type: string
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Data Manager are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_data_manager_sdk.py
```

The agent will be available at `http://localhost:8001`.
