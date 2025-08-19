# Agent 0: Data Product Registration

## Overview

The Data Product Registration Agent (Agent 0) is the entry point for new data products into the A2A Network. Its primary responsibilities are:

-   **Registration**: Receives and registers new data products, capturing essential metadata.
-   **Validation**: Performs initial validation checks on the data product's schema and metadata against predefined rules.
-   **Cataloging**: Creates an initial entry in the data catalog for the newly registered product.

This agent ensures that all data products entering the network meet a baseline level of quality and are properly documented from the start.

## API Specification

This agent exposes a RESTful API for registering and managing data products.

```yaml
openapi: 3.0.0
info:
  title: Agent 0 - Data Product Registration API
  description: Handles initial data product registration, validation, and catalog entry
  version: 1.0.0
  contact:
    name: A2A Platform Team
    email: a2a-platform@company.com

servers:
  - url: http://localhost:8001
    description: Local development server

paths:
  /agents/agent0/register:
    post:
      summary: Register new data product
      operationId: registerDataProduct
      tags:
        - Data Product Registration
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DataProductRegistration'
            examples:
              financialDataProduct:
                value:
                  product_name: "Treasury Rates Dataset"
                  product_type: "financial_time_series"
                  source_system: "BLOOMBERG"
                  data_format: "JSON"
                  schema_version: "2.0"
                  metadata:
                    asset_class: "RATES"
                    frequency: "DAILY"
                    start_date: "2020-01-01"
                    end_date: "2024-12-31"
                  validation_rules:
                    - rule_type: "COMPLETENESS"
                      threshold: 0.95
                    - rule_type: "ACCURACY"
                      threshold: 0.99
      responses:
        '201':
          description: Data product successfully registered
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RegistrationResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          $ref: '#/components/responses/Conflict'
        '500':
          $ref: '#/components/responses/InternalError'

  /agents/agent0/validate:
    post:
      summary: Validate data product without registration
      operationId: validateDataProduct
      tags:
        - Data Product Validation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DataProductValidation'
      responses:
        '200':
          description: Validation results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ValidationResult'

  /agents/agent0/products/{productId}:
    get:
      summary: Get data product details
      operationId: getDataProduct
      tags:
        - Data Product Management
      parameters:
        - name: productId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Data product details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DataProduct'
        '404':
          $ref: '#/components/responses/NotFound'

components:
  schemas:
    DataProductRegistration:
      type: object
      required:
        - product_name
        - product_type
        - source_system
        - data_format
        - metadata
      properties:
        product_name:
          type: string
          minLength: 3
          maxLength: 255
        product_type:
          type: string
          enum: [financial_time_series, reference_data, transaction_data, market_data]
        source_system:
          type: string
          enum: [BLOOMBERG, REUTERS, INTERNAL, SAP_S4, CUSTOM]
        data_format:
          type: string
          enum: [JSON, XML, CSV, PARQUET, AVRO]
        schema_version:
          type: string
          pattern: '^\d+\.\d+$'
        metadata:
          type: object
          additionalProperties: true
        validation_rules:
          type: array
          items:
            $ref: '#/components/schemas/ValidationRule'
    
    ValidationRule:
      type: object
      required:
        - rule_type
        - threshold
      properties:
        rule_type:
          type: string
          enum: [COMPLETENESS, ACCURACY, TIMELINESS, CONSISTENCY, UNIQUENESS]
        threshold:
          type: number
          minimum: 0
          maximum: 1
        parameters:
          type: object
          additionalProperties: true
    
    RegistrationResponse:
      type: object
      properties:
        product_id:
          type: string
          format: uuid
        registration_timestamp:
          type: string
          format: date-time
        catalog_entry_id:
          type: string
        validation_score:
          type: number
          minimum: 0
          maximum: 1
        status:
          type: string
          enum: [REGISTERED, PENDING_APPROVAL, REJECTED]
        next_agent:
          type: string
          description: Next agent in the processing pipeline
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Agent 0 are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_agent0_sdk.py
```

The agent will be available at `http://localhost:8001`.
